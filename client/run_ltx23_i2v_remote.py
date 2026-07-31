"""Remote LTX-2.3 image-to-video client for a LAN ComfyUI server.

Uploads a local image, runs LTX-2.3 I2V on the server, downloads the MP4 back.

The graph here is ComfyUI's bundled `video_ltx2_3_i2v.json` template flattened to
API format. The template wraps everything in a *UI* subgraph, and the backend
only expands node-level dynamic subgraphs -- not UI subgraph definitions -- so
/prompt cannot take that file as-is. Node ids are kept identical to the template
(276..328, plus 269 LoadImage and 75 SaveVideo from its top level) so the two can
be diffed side by side.

Three deliberate deviations from the template:

  * Text encoder defaults to gemma_3_12B_it_fp8_scaled instead of the template's
    gemma_3_12B_it_fp4_mixed. NVFP4 dispatches through comfy_kitchen, whose only
    native backend is CUDA, and Intel's llm-scaler-omni guide recommends fp8
    throughout on Arc. Override with --text-encoder.
  * The Gemma prompt-enhancer branch (LoraLoader + TextGenerateLTX2Prompt +
    ComfySwitchNode) is dropped, so --prompt is used verbatim. The template
    defaults that switch off anyway; enabling it costs an extra LLM pass per run.
  * The template's helper nodes (PrimitiveInt / ComfyMathExpression) are constant
    folded: base latent is (width/2, height/2), upsampled x2 by the spatial
    upscaler in the refine pass, and the frame count comes from duration * fps
    snapped to the 8k+1 grid LTX actually produces.

Serial by design: ComfyUI runs one prompt at a time, and this script waits for
the job to finish before returning.
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

CKPT = "ltx-2.3-22b-dev-fp8.safetensors"
TEXT_ENCODER = "gemma_3_12B_it_fp8_scaled.safetensors"
DISTILL_LORA = "ltx_2.3_22b_distilled_1.1_lora_dynamic_fro09_avg_rank_111_bf16.safetensors"
UPSCALER = "ltx-2.3-spatial-upscaler-x2-1.1.safetensors"

# Sigma schedules lifted verbatim from the template: stage 1 runs 8 steps from
# 1.0, stage 2 refines the upsampled latent in 3 steps from 0.85.
SIGMAS_BASE = "1.0, 0.99375, 0.9875, 0.98125, 0.975, 0.909375, 0.725, 0.421875, 0.0"
SIGMAS_REFINE = "0.85, 0.7250, 0.4219, 0.0"

DEFAULT_PROMPT = (
    "The scene comes alive with subtle, natural motion. The camera performs a "
    "single smooth push-in, cinematic lighting, photorealistic detail, "
    "consistent subject, no cuts."
)
DEFAULT_NEGATIVE = "pc game, console game, video game, cartoon, childish, ugly"

VIDEO_SUFFIXES = {".mp4", ".webm", ".mkv", ".avi", ".mov"}

# class_type -> which model list the name has to appear in, for the pre-flight check
MODEL_SLOTS = {
    "ckpt": ("CheckpointLoaderSimple", "ckpt_name"),
    "text_encoder": ("LTXAVTextEncoderLoader", "text_encoder"),
    "distill_lora": ("LoraLoaderModelOnly", "lora_name"),
    "upscaler": ("LatentUpscaleModelLoader", "model_name"),
}

# Nodes the graph needs beyond ComfyUI core; missing ones mean the server is
# running an older ComfyUI or is missing the LTX-2.3 nodes entirely.
REQUIRED_NODES = [
    "CheckpointLoaderSimple",
    "LTXVAudioVAELoader",
    "LTXAVTextEncoderLoader",
    "LatentUpscaleModelLoader",
    "ResizeImageMaskNode",
    "ResizeImagesByLongerEdge",
    "LTXVPreprocess",
    "LTXVImgToVideoInplace",
    "LTXVEmptyLatentAudio",
    "LTXVConcatAVLatent",
    "LTXVSeparateAVLatent",
    "LTXVLatentUpsampler",
    "LTXVCropGuides",
    "LTXVAudioVAEDecode",
    "ManualSigmas",
    "SamplerCustomAdvanced",
]


def build_prompt(
    image_name: str,
    *,
    positive: str,
    negative: str,
    width: int,
    height: int,
    duration: float,
    fps: int,
    seed: int,
    refine: bool,
    filename_prefix: str,
    models: dict[str, str],
    distill_strength: float = 0.5,
    img_compression: int = 18,
) -> dict:
    """Flatten video_ltx2_3_i2v.json into an API-format prompt graph."""
    base_w, base_h = width // 2, height // 2
    length = snap_length(duration, fps)
    g: dict = {}

    # ---- loaders -------------------------------------------------------
    g["316"] = {"class_type": "CheckpointLoaderSimple", "inputs": {"ckpt_name": models["ckpt"]}}
    g["279"] = {"class_type": "LTXVAudioVAELoader", "inputs": {"ckpt_name": models["ckpt"]}}
    g["317"] = {"class_type": "LTXAVTextEncoderLoader",
                "inputs": {"text_encoder": models["text_encoder"], "ckpt_name": models["ckpt"],
                           "device": "default"}}
    g["285"] = {"class_type": "LoraLoaderModelOnly",
                "inputs": {"model": ["316", 0], "lora_name": models["distill_lora"],
                           "strength_model": distill_strength}}
    g["311"] = {"class_type": "LatentUpscaleModelLoader", "inputs": {"model_name": models["upscaler"]}}

    # ---- input image ---------------------------------------------------
    # 290 fits the upload to the requested frame size (centre crop), 286 then
    # caps the longer edge at 1536 so the VAE encode stays cheap, and 289 is the
    # compression pass LTX expects on conditioning frames.
    g["269"] = {"class_type": "LoadImage", "inputs": {"image": image_name}}
    # resize_type is a V3 dynamic combo: the selected option's sub-inputs are
    # namespaced under it ("resize_type.width"), not flattened to the top level.
    g["290"] = {"class_type": "ResizeImageMaskNode",
                "inputs": {"input": ["269", 0], "resize_type": "scale dimensions",
                           "resize_type.width": width, "resize_type.height": height,
                           "resize_type.crop": "center", "scale_method": "lanczos"}}
    g["286"] = {"class_type": "ResizeImagesByLongerEdge",
                "inputs": {"images": ["290", 0], "longer_edge": 1536}}
    g["289"] = {"class_type": "LTXVPreprocess",
                "inputs": {"image": ["286", 0], "img_compression": img_compression}}

    # ---- conditioning --------------------------------------------------
    g["303"] = {"class_type": "CLIPTextEncode", "inputs": {"text": positive, "clip": ["317", 0]}}
    g["313"] = {"class_type": "CLIPTextEncode", "inputs": {"text": negative, "clip": ["317", 0]}}
    g["304"] = {"class_type": "LTXVConditioning",
                "inputs": {"positive": ["303", 0], "negative": ["313", 0], "frame_rate": float(fps)}}

    # ---- stage 1: base sample at half resolution -----------------------
    # bypass=False is what makes this image-to-video rather than text-to-video:
    # the preprocessed frame is written into the empty latent at strength 0.7.
    g["295"] = {"class_type": "EmptyLTXVLatentVideo",
                "inputs": {"width": base_w, "height": base_h, "length": length, "batch_size": 1}}
    g["296"] = {"class_type": "LTXVImgToVideoInplace",
                "inputs": {"vae": ["316", 2], "image": ["289", 0], "latent": ["295", 0],
                           "strength": 0.7, "bypass": False}}
    g["305"] = {"class_type": "LTXVEmptyLatentAudio",
                "inputs": {"frames_number": length, "frame_rate": float(fps), "batch_size": 1,
                           "audio_vae": ["279", 0]}}
    g["318"] = {"class_type": "LTXVConcatAVLatent",
                "inputs": {"video_latent": ["296", 0], "audio_latent": ["305", 0]}}
    g["314"] = {"class_type": "CFGGuider",
                "inputs": {"model": ["285", 0], "positive": ["304", 0],
                           "negative": ["304", 1], "cfg": 1}}
    g["277"] = {"class_type": "RandomNoise", "inputs": {"noise_seed": seed}}
    g["291"] = {"class_type": "KSamplerSelect", "inputs": {"sampler_name": "euler"}}
    g["306"] = {"class_type": "ManualSigmas", "inputs": {"sigmas": SIGMAS_BASE}}
    g["283"] = {"class_type": "SamplerCustomAdvanced",
                "inputs": {"noise": ["277", 0], "guider": ["314", 0], "sampler": ["291", 0],
                           "sigmas": ["306", 0], "latent_image": ["318", 0]}}
    g["307"] = {"class_type": "LTXVSeparateAVLatent", "inputs": {"av_latent": ["283", 0]}}

    if refine:
        # ---- stage 2: x2 latent upsample, then a short refine pass -----
        g["287"] = {"class_type": "LTXVLatentUpsampler",
                    "inputs": {"samples": ["307", 0], "upscale_model": ["311", 0],
                               "vae": ["316", 2]}}
        g["288"] = {"class_type": "LTXVImgToVideoInplace",
                    "inputs": {"vae": ["316", 2], "image": ["289", 0], "latent": ["287", 0],
                               "strength": 1, "bypass": False}}
        g["278"] = {"class_type": "LTXVConcatAVLatent",
                    "inputs": {"video_latent": ["288", 0], "audio_latent": ["307", 1]}}
        g["284"] = {"class_type": "LTXVCropGuides",
                    "inputs": {"positive": ["304", 0], "negative": ["304", 1],
                               "latent": ["307", 0]}}
        g["282"] = {"class_type": "CFGGuider",
                    "inputs": {"model": ["285", 0], "positive": ["284", 0],
                               "negative": ["284", 1], "cfg": 1}}
        g["276"] = {"class_type": "RandomNoise", "inputs": {"noise_seed": 42}}
        g["280"] = {"class_type": "KSamplerSelect", "inputs": {"sampler_name": "euler"}}
        g["281"] = {"class_type": "ManualSigmas", "inputs": {"sigmas": SIGMAS_REFINE}}
        g["308"] = {"class_type": "SamplerCustomAdvanced",
                    "inputs": {"noise": ["276", 0], "guider": ["282", 0], "sampler": ["280", 0],
                               "sigmas": ["281", 0], "latent_image": ["278", 0]}}
        g["309"] = {"class_type": "LTXVSeparateAVLatent", "inputs": {"av_latent": ["308", 0]}}
        final = "309"
    else:
        final = "307"

    # ---- decode + mux --------------------------------------------------
    g["315"] = {"class_type": "VAEDecodeTiled",
                "inputs": {"samples": [final, 0], "vae": ["316", 2], "tile_size": 768,
                           "overlap": 64, "temporal_size": 4096, "temporal_overlap": 4}}
    g["297"] = {"class_type": "LTXVAudioVAEDecode",
                "inputs": {"samples": [final, 1], "audio_vae": ["279", 0]}}
    g["310"] = {"class_type": "CreateVideo",
                "inputs": {"images": ["315", 0], "audio": ["297", 0], "fps": float(fps)}}
    g["75"] = {"class_type": "SaveVideo",
               "inputs": {"video": ["310", 0], "filename_prefix": filename_prefix,
                          "format": "auto", "codec": "auto"}}
    return g


# ------------------------------------------------------------------ pre-flight
def _combo_options(spec) -> list[str]:
    """Pull the option list out of either COMBO spelling in /object_info."""
    if not isinstance(spec, list) or not spec:
        return []
    if isinstance(spec[0], list):  # legacy: [[...options...], {...}]
        return [o for o in spec[0] if isinstance(o, str)]
    opts = spec[1].get("options") if len(spec) > 1 and isinstance(spec[1], dict) else None
    return [o for o in (opts or []) if isinstance(o, str)]


def check_nodes(client) -> list[str]:
    return [c for c in REQUIRED_NODES if not client.http_json("GET", f"/object_info/{c}", timeout=30)]


def resolve_models(client, requested: dict[str, str]) -> dict[str, str]:
    """Verify each model name exists on the server; fall back to a near match."""
    resolved = dict(requested)
    for slot, (class_type, field) in MODEL_SLOTS.items():
        info = client.http_json("GET", f"/object_info/{class_type}", timeout=30)
        node = info.get(class_type)
        if not node:
            continue  # check_nodes() already reports a missing class
        spec = (node["input"].get("required") or {}).get(field)
        options = _combo_options(spec)
        want = requested[slot]
        if not options or want in options:
            continue
        # e.g. the host has ...upscaler-x2-1.0 but not 1.1: take the closest name
        stem = want.rsplit("-", 1)[0].rsplit("_", 1)[0][:24]
        near = [o for o in options if o.startswith(stem)]
        if near:
            print(f"note: {class_type}.{field} '{want}' not on server, using '{near[0]}'")
            resolved[slot] = near[0]
            continue
        raise SystemExit(
            f"{class_type}.{field}: '{want}' is not installed on {client.base}.\n"
            f"  available: {options}\n"
            "  Download it on the server or pass an explicit --ckpt / --text-encoder / "
            "--distill-lora / --upscaler."
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


def snap64(v: int) -> int:
    """Round a requested dimension to the nearest multiple of 64 (min 64)."""
    return max(64, int(round(v / 64)) * 64)


def snap_length(duration: float, fps: int) -> int:
    """Frame count LTX will actually produce for `duration` seconds.

    Temporal compression is 8, so the length lands on 8k+1 and anything else is
    floored: 5s @25fps is 126 frames on paper but comes back as 121 (4.84s).
    Compute the real number up front so the log, filename and meta match.
    """
    return max(1, int(duration * fps) // 8) * 8 + 1


def pick_primary_video(saved: list[Path]) -> Path | None:
    vids = [p for p in saved if p.suffix.lower() in VIDEO_SUFFIXES]
    return vids[0] if vids else (saved[0] if saved else None)


def main() -> None:
    ap = argparse.ArgumentParser(
        description="Remote LTX-2.3 image-to-video via LAN ComfyUI (upload image, download MP4)"
    )
    add_server_args(ap)
    ap.add_argument("--image", type=str, required=True, help="Local input image (first frame)")
    ap.add_argument("--prompt", type=str, default=None)
    ap.add_argument("--prompt-file", type=str, default=None)
    ap.add_argument("--negative-prompt", type=str, default=None)
    ap.add_argument("--negative-prompt-file", type=str, default=None)
    ap.add_argument("--width", type=int, default=1280, help="final width (base pass is half)")
    ap.add_argument("--height", type=int, default=720, help="final height (base pass is half)")
    ap.add_argument("--duration", type=float, default=5.0, help="target seconds (default 5)")
    ap.add_argument("--fps", type=int, default=25)
    ap.add_argument("--seed", type=int, default=None, help="default: random")
    ap.add_argument("--no-refine", action="store_true",
                    help="stop after the base pass: half resolution, much faster")
    ap.add_argument("--img-compression", type=int, default=18,
                    help="LTXVPreprocess strength on the input frame (default 18)")
    ap.add_argument("--distill-strength", type=float, default=0.5,
                    help="distilled LoRA strength (default 0.5)")
    ap.add_argument("--ckpt", type=str, default=CKPT)
    ap.add_argument("--text-encoder", type=str, default=TEXT_ENCODER)
    ap.add_argument("--distill-lora", type=str, default=DISTILL_LORA)
    ap.add_argument("--upscaler", type=str, default=UPSCALER)
    ap.add_argument("--output", type=str, default=None, help="Local output MP4 path")
    ap.add_argument("--timeout", type=float, default=7200)
    ap.add_argument("--skip-model-check", action="store_true",
                    help="do not verify node classes / model names before queueing")
    ap.add_argument("--dump", metavar="FILE",
                    help="write the API prompt to FILE and exit (no server needed)")
    args = ap.parse_args()

    positive = load_text(args.prompt, args.prompt_file, "prompt", DEFAULT_PROMPT)
    negative = load_text(args.negative_prompt, args.negative_prompt_file,
                         "negative-prompt", DEFAULT_NEGATIVE)
    seed = args.seed if args.seed is not None else random.randrange(0, 2**32)
    run_id = uuid.uuid4().hex[:8]
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    refine = not args.no_refine

    # LTX floors the base latent to a multiple of 32 px, and the base pass runs
    # at half the requested size, so anything not a multiple of 64 silently
    # shrinks (1280x720 comes back as 1280x704). Snap up front instead, so the
    # centre crop, the log line, the filename and the meta all agree with the
    # file that actually lands.
    width, height = snap64(args.width), snap64(args.height)
    if (width, height) != (args.width, args.height):
        print(f"note: {args.width}x{args.height} -> {width}x{height} "
              "(LTX needs multiples of 64; the base pass is half resolution)")
    # Same story along the time axis: temporal compression is 8, so the frame
    # count lands on 8k+1 and 5s @25fps really means 121 frames / 4.84s.
    length = snap_length(args.duration, args.fps)
    duration = round(length / args.fps, 2)  # what the muxed file actually plays
    if duration != args.duration:
        print(f"note: {args.duration}s -> {duration}s ({length} frames; "
              "LTX frame counts are 8k+1)")
    models = {
        "ckpt": args.ckpt,
        "text_encoder": args.text_encoder,
        "distill_lora": args.distill_lora,
        "upscaler": args.upscaler,
    }

    if args.dump:
        graph = build_prompt(
            "PLACEHOLDER.png", positive=positive, negative=negative, width=width, height=height,
            duration=duration, fps=args.fps, seed=seed, refine=refine,
            filename_prefix=f"video/ltx23_i2v_{run_id}", models=models,
            distill_strength=args.distill_strength, img_compression=args.img_compression,
        )
        Path(args.dump).write_text(json.dumps(graph, indent=2, ensure_ascii=False), encoding="utf-8")
        print(f"wrote {args.dump} ({len(graph)} nodes, length={length} frames)")
        return

    local_img = Path(args.image)
    if not local_img.is_file():
        raise SystemExit(f"input image not found: {local_img}")

    client = connect_from_args(args)

    if not args.skip_model_check:
        missing = check_nodes(client)
        if missing:
            raise SystemExit(
                f"server {client.base} is missing node classes: {', '.join(missing)}. "
                "Update ComfyUI on the server (LTX-2.3 needs a recent build)."
            )
        models = resolve_models(client, models)

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    remote_name = unique_remote_name(local_img, prefix=f"ltx_in_{run_id}")
    up = client.upload_file(local_img, remote_name=remote_name, overwrite=True)

    prefix = f"remote/ltx23_i2v_{run_id}"
    graph = build_prompt(
        up["name"], positive=positive, negative=negative, width=width, height=height,
        duration=duration, fps=args.fps, seed=seed, refine=refine,
        filename_prefix=prefix, models=models,
        distill_strength=args.distill_strength, img_compression=args.img_compression,
    )

    out_w, out_h = (width, height) if refine else (width // 2, height // 2)
    print(f"[ltx23_i2v] {out_w}x{out_h} {duration}s @{args.fps}fps "
          f"({length} frames, refine={'on' if refine else 'off'}) seed={seed}")

    meta: dict = {
        "mode": "remote_ltx23_i2v",
        "server": client.server,
        "run_id": run_id,
        "seed": seed,
        "prompt": positive,
        "negative_prompt": negative,
        "width": out_w,
        "height": out_h,
        "duration_sec": duration,
        "fps": args.fps,
        "length_frames": length,
        "refine": refine,
        "models": models,
        "input_image": str(local_img.resolve()),
        "remote_image_name": up["name"],
        "ok": False,
    }
    meta_path = out_dir / f"ltx23_i2v_meta_{stamp}_{run_id}.json"

    try:
        entry, elapsed, prompt_id = client.queue_and_wait(
            graph, tag="ltx23_i2v", timeout=args.timeout
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
            dest = out_dir / f"ltx23_i2v_{out_w}x{out_h}_{duration}s_{stamp}.mp4"
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
        print(f"[ltx23_i2v] OUT {dest}")
    except Exception as e:
        meta["error"] = str(e)
        meta_path.write_text(json.dumps(meta, indent=2, ensure_ascii=False), encoding="utf-8")
        print(f"[ltx23_i2v] FAILED: {e}")
        print("meta written:", meta_path)
        raise SystemExit(1)

    meta_path.write_text(json.dumps(meta, indent=2, ensure_ascii=False), encoding="utf-8")
    print("meta written:", meta_path)


if __name__ == "__main__":
    main()
