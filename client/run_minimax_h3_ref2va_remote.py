"""Remote MiniMax H3 Ref2VA client for a LAN ComfyUI server.

Uploads 1-9 local reference images, runs MiniMax H3's Ref2VA task on the
server (joint video+audio diffusion, native ComfyUI nodes from
comfy_extras/nodes_minimax_h3.py), downloads the muxed MP4 back.

Reference images ride through every sampling step and steer identity; they
are not keyframes (no fixed frame position). Tag them in --prompt as
<Picture 1>, <Picture 2>, ... in upload order -- the node's tokenizer
presentation uses the same 1-based ordinals.

The Ref2VA node's optional inputs (ref_images/ref_videos/ref_video_audios/
ref_audios) use ComfyUI's V3 "autogrow" dynamic input: each individual slot
is addressed in the flat API graph as "<input_id>.<prefix><index>", e.g.
the first two reference images are "ref_images.ref_image_0" and
"ref_images.ref_image_1" (0-indexed). This script only wires ref_images;
video/audio references aren't exposed here.

Serial by design: ComfyUI runs one prompt at a time, and this script waits
for the job to finish before returning.
"""
from __future__ import annotations

import argparse
import json
import random
import uuid
from datetime import datetime
from pathlib import Path

from comfyui_remote_client import (
    add_server_args,
    connect_from_args,
    unique_remote_name,
)

UNET = "minimax_h3_ref2va_pruned_int8_convrot.safetensors"
TEXT_ENCODER = "qwen3vl_32b_minimax_h3_int8_convrot.safetensors"
VIDEO_VAE = "minimax_h3_video_vae_fp16.safetensors"
AUDIO_VAE = "minimax_h3_audio_vae_fp32.safetensors"

DEFAULT_PROMPT_TEMPLATE = "A video steered by {refs}, in a consistent, coherent scene."

FPS = 24
MAX_REF_IMAGES = 9
VIDEO_SUFFIXES = {".mp4", ".webm", ".mkv", ".avi", ".mov"}

# class_type -> which model list the name has to appear in, for the pre-flight check
MODEL_SLOTS = {
    "unet": ("UNETLoader", "unet_name"),
    "text_encoder": ("CLIPLoader", "clip_name"),
    "video_vae": ("VAELoader", "vae_name"),
    "audio_vae": ("VAELoader", "vae_name"),
}

REQUIRED_NODES = [
    "UNETLoader",
    "CLIPLoader",
    "VAELoader",
    "LoadImage",
    "MiniMaxH3SigmaShift",
    "MiniMaxH3ReferenceToVideo",
    "ConditioningZeroOut",
    "KSampler",
    "VAEDecode",
    "VAEDecodeAudio",
    "VHS_VideoCombine",
]


def align_frame_count(n: int) -> int:
    while n % 17 != 5:
        n += 1
    return n


def snap_length(duration: float, fps: int = FPS) -> int:
    return align_frame_count(max(5, round(duration * fps)))


def build_prompt(
    ref_image_names: list[str],
    *,
    prompt: str,
    width: int,
    height: int,
    length: int,
    ref_image_size: str,
    seed: int,
    steps: int,
    cfg: float,
    shift_video: float,
    shift_audio: float,
    sampler_name: str,
    scheduler: str,
    filename_prefix: str,
    models: dict[str, str],
) -> dict:
    g: dict = {}

    g["1"] = {"class_type": "UNETLoader",
              "inputs": {"unet_name": models["unet"], "weight_dtype": "default"}}
    g["2"] = {"class_type": "CLIPLoader",
              "inputs": {"clip_name": models["text_encoder"], "type": "minimax"}}
    g["3"] = {"class_type": "VAELoader", "inputs": {"vae_name": models["video_vae"]}}
    g["4"] = {"class_type": "VAELoader", "inputs": {"vae_name": models["audio_vae"]}}
    g["5"] = {"class_type": "MiniMaxH3SigmaShift",
              "inputs": {"model": ["1", 0], "shift_video": shift_video, "shift_audio": shift_audio}}

    ref_links: dict[str, list] = {}
    for i, name in enumerate(ref_image_names):
        node_id = f"img{i}"
        g[node_id] = {"class_type": "LoadImage", "inputs": {"image": name}}
        ref_links[f"ref_images.ref_image_{i}"] = [node_id, 0]

    g["8"] = {"class_type": "MiniMaxH3ReferenceToVideo",
              "inputs": {
                  "clip": ["2", 0], "vae": ["3", 0], "audio_vae": ["4", 0],
                  "prompt": prompt, "width": width, "height": height, "length": length,
                  "ref_image_size": ref_image_size,
                  **ref_links,
              }}
    g["9"] = {"class_type": "ConditioningZeroOut", "inputs": {"conditioning": ["8", 0]}}
    g["10"] = {"class_type": "KSampler",
               "inputs": {"model": ["5", 0], "positive": ["8", 0], "negative": ["9", 0],
                          "latent_image": ["8", 1], "seed": seed, "steps": steps, "cfg": cfg,
                          "sampler_name": sampler_name, "scheduler": scheduler, "denoise": 1.0}}
    g["11"] = {"class_type": "VAEDecode", "inputs": {"samples": ["10", 0], "vae": ["3", 0]}}
    g["12"] = {"class_type": "VAEDecodeAudio", "inputs": {"samples": ["10", 0], "vae": ["4", 0]}}
    g["13"] = {"class_type": "VHS_VideoCombine",
               "inputs": {"images": ["11", 0], "audio": ["12", 0], "frame_rate": float(FPS),
                          "loop_count": 0, "filename_prefix": filename_prefix,
                          "format": "video/h264-mp4", "pingpong": False, "save_output": True}}
    return g


# ------------------------------------------------------------------ pre-flight
def _combo_options(spec) -> list[str]:
    if not isinstance(spec, list) or not spec:
        return []
    if isinstance(spec[0], list):
        return [o for o in spec[0] if isinstance(o, str)]
    opts = spec[1].get("options") if len(spec) > 1 and isinstance(spec[1], dict) else None
    return [o for o in (opts or []) if isinstance(o, str)]


def check_nodes(client) -> list[str]:
    return [c for c in REQUIRED_NODES if not client.http_json("GET", f"/object_info/{c}", timeout=30)]


def resolve_models(client, requested: dict[str, str]) -> dict[str, str]:
    resolved = dict(requested)
    seen_fields: set[str] = set()
    for slot, (class_type, field) in MODEL_SLOTS.items():
        key = f"{class_type}.{field}.{requested[slot]}"
        if key in seen_fields:
            continue
        seen_fields.add(key)
        info = client.http_json("GET", f"/object_info/{class_type}", timeout=30)
        node = info.get(class_type)
        if not node:
            continue
        spec = (node["input"].get("required") or {}).get(field)
        options = _combo_options(spec)
        want = requested[slot]
        if not options or want in options:
            continue
        stem = want.rsplit("-", 1)[0].rsplit("_", 1)[0][:24]
        near = [o for o in options if o.startswith(stem)]
        if near:
            print(f"note: {class_type}.{field} '{want}' not on server, using '{near[0]}'")
            resolved[slot] = near[0]
            continue
        raise SystemExit(
            f"{class_type}.{field}: '{want}' is not installed on {client.base}.\n"
            f"  available: {options}\n"
            "  Download it on the server or pass an explicit --unet / --text-encoder / "
            "--video-vae / --audio-vae."
        )
    return resolved


# ------------------------------------------------------------------ CLI helpers
def load_text(inline: str | None, file_path: str | None, label: str, default: str) -> str:
    if file_path:
        p = Path(file_path)
        if not p.exists():
            raise SystemExit(f"--{label}-file not found: {p}")
        text = p.read_text(encoding="utf-8").strip()
        if not text:
            raise SystemExit(f"--{label}-file is empty: {p}")
        return text
    if inline is not None and inline.strip():
        return inline.strip()
    return default


def pick_primary_video(saved: list[Path]) -> Path | None:
    vids = [p for p in saved if p.suffix.lower() in VIDEO_SUFFIXES]
    return vids[0] if vids else (saved[0] if saved else None)


def main() -> None:
    ap = argparse.ArgumentParser(
        description="Remote MiniMax H3 Ref2VA via LAN ComfyUI (upload 1-9 ref images, download MP4)"
    )
    add_server_args(ap)
    ap.add_argument("--ref-image", dest="ref_images", action="append", required=True,
                    help=f"Local reference image; repeat for more (1-{MAX_REF_IMAGES}). "
                         "Tag them in --prompt as <Picture 1>, <Picture 2>, ... in this order.")
    ap.add_argument("--prompt", type=str, default=None)
    ap.add_argument("--prompt-file", type=str, default=None)
    ap.add_argument("--width", type=int, default=1344)
    ap.add_argument("--height", type=int, default=768)
    ap.add_argument("--duration", type=float, default=5.0, help="target seconds (default 5)")
    ap.add_argument("--ref-image-size", choices=["match", "max"], default="match",
                    help="'match': scale refs down to the generation's pixel area (default). "
                         "'max': 2048px short edge for best identity fidelity, several times slower.")
    ap.add_argument("--seed", type=int, default=None, help="default: random")
    ap.add_argument("--steps", type=int, default=50)
    ap.add_argument("--cfg", type=float, default=1.0)
    ap.add_argument("--shift-video", type=float, default=12.0)
    ap.add_argument("--shift-audio", type=float, default=3.0)
    ap.add_argument("--sampler", type=str, default="euler")
    ap.add_argument("--scheduler", type=str, default="simple")
    ap.add_argument("--unet", type=str, default=UNET)
    ap.add_argument("--text-encoder", type=str, default=TEXT_ENCODER)
    ap.add_argument("--video-vae", type=str, default=VIDEO_VAE)
    ap.add_argument("--audio-vae", type=str, default=AUDIO_VAE)
    ap.add_argument("--output", type=str, default=None, help="Local output MP4 path")
    ap.add_argument("--timeout", type=float, default=7200)
    ap.add_argument("--skip-model-check", action="store_true",
                    help="do not verify node classes / model names before queueing")
    ap.add_argument("--dump", metavar="FILE",
                    help="write the API prompt to FILE and exit (no server needed)")
    args = ap.parse_args()

    if len(args.ref_images) > MAX_REF_IMAGES:
        raise SystemExit(f"--ref-image given {len(args.ref_images)} times, max is {MAX_REF_IMAGES}")

    default_prompt = DEFAULT_PROMPT_TEMPLATE.format(
        refs=", ".join(f"<Picture {i + 1}>" for i in range(len(args.ref_images)))
    )
    prompt = load_text(args.prompt, args.prompt_file, "prompt", default_prompt)
    seed = args.seed if args.seed is not None else random.randrange(0, 2**32)
    run_id = uuid.uuid4().hex[:8]
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    length = snap_length(args.duration)
    duration = round(length / FPS, 2)
    if duration != args.duration:
        print(f"note: {args.duration}s -> {duration}s ({length} frames; "
              "MiniMax H3 frame counts land on the 17k+5 grid)")

    models = {
        "unet": args.unet,
        "text_encoder": args.text_encoder,
        "video_vae": args.video_vae,
        "audio_vae": args.audio_vae,
    }

    if args.dump:
        placeholders = [f"PLACEHOLDER_{i}.png" for i in range(len(args.ref_images))]
        graph = build_prompt(
            placeholders, prompt=prompt, width=args.width, height=args.height, length=length,
            ref_image_size=args.ref_image_size, seed=seed, steps=args.steps, cfg=args.cfg,
            shift_video=args.shift_video, shift_audio=args.shift_audio,
            sampler_name=args.sampler, scheduler=args.scheduler,
            filename_prefix=f"video/minimax_h3_ref2va_{run_id}", models=models,
        )
        Path(args.dump).write_text(json.dumps(graph, indent=2, ensure_ascii=False), encoding="utf-8")
        print(f"wrote {args.dump} ({len(graph)} nodes, length={length} frames)")
        return

    local_imgs = [Path(p) for p in args.ref_images]
    for p in local_imgs:
        if not p.is_file():
            raise SystemExit(f"reference image not found: {p}")

    client = connect_from_args(args)

    if not args.skip_model_check:
        missing = check_nodes(client)
        if missing:
            raise SystemExit(
                f"server {client.base} is missing node classes: {', '.join(missing)}. "
                "Update ComfyUI on the server (MiniMax H3 needs a recent build)."
            )
        models = resolve_models(client, models)

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    remote_names = [
        client.upload_file(p, remote_name=unique_remote_name(p, prefix=f"h3ref_in_{run_id}_{i}"),
                            overwrite=True)["name"]
        for i, p in enumerate(local_imgs)
    ]

    prefix = f"remote/minimax_h3_ref2va_{run_id}"
    graph = build_prompt(
        remote_names, prompt=prompt, width=args.width, height=args.height, length=length,
        ref_image_size=args.ref_image_size, seed=seed, steps=args.steps, cfg=args.cfg,
        shift_video=args.shift_video, shift_audio=args.shift_audio,
        sampler_name=args.sampler, scheduler=args.scheduler,
        filename_prefix=prefix, models=models,
    )

    print(f"[minimax_h3_ref2va] {args.width}x{args.height} {duration}s @{FPS}fps "
          f"({length} frames, {len(local_imgs)} ref images, steps={args.steps}) seed={seed}")

    meta: dict = {
        "mode": "remote_minimax_h3_ref2va",
        "server": client.server,
        "run_id": run_id,
        "seed": seed,
        "prompt": prompt,
        "width": args.width,
        "height": args.height,
        "duration_sec": duration,
        "fps": FPS,
        "length_frames": length,
        "steps": args.steps,
        "cfg": args.cfg,
        "ref_image_size": args.ref_image_size,
        "models": models,
        "input_images": [str(p.resolve()) for p in local_imgs],
        "remote_image_names": remote_names,
        "ok": False,
    }
    meta_path = out_dir / f"minimax_h3_ref2va_meta_{stamp}_{run_id}.json"

    try:
        entry, elapsed, prompt_id = client.queue_and_wait(
            graph, tag="minimax_h3_ref2va", timeout=args.timeout
        )
        saved = client.download_history_outputs(entry, out_dir)
        primary = pick_primary_video(saved)
        if primary is None:
            raise RuntimeError("no video downloaded from history outputs")

        if args.output:
            dest = Path(args.output)
            if dest.suffix.lower() not in VIDEO_SUFFIXES:
                dest = dest.with_suffix(".mp4")
        else:
            dest = out_dir / f"minimax_h3_ref2va_{args.width}x{args.height}_{duration}s_{stamp}.mp4"
        if dest.resolve() != primary.resolve():
            dest.parent.mkdir(parents=True, exist_ok=True)
            dest.write_bytes(primary.read_bytes())

        meta.update(
            ok=True,
            prompt_id=prompt_id,
            elapsed_sec=round(elapsed, 2),
            output_video_file=str(dest.resolve()),
            downloaded=[str(p) for p in saved],
        )
        print(f"[minimax_h3_ref2va] OUT {dest}")
    except Exception as e:
        meta["error"] = str(e)
        meta_path.write_text(json.dumps(meta, indent=2, ensure_ascii=False), encoding="utf-8")
        print(f"[minimax_h3_ref2va] FAILED: {e}")
        print("meta written:", meta_path)
        raise SystemExit(1)

    meta_path.write_text(json.dumps(meta, indent=2, ensure_ascii=False), encoding="utf-8")
    print("meta written:", meta_path)


if __name__ == "__main__":
    main()
