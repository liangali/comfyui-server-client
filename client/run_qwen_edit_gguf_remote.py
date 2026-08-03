"""Remote Qwen-Image-Edit client using a GGUF-quantized UNet.

Same TextEncodeQwenImageEditPlus pipeline as run_qwen_edit_remote.py, but the
diffusion model is a GGUF quant (city96 ComfyUI-GGUF tooling) instead of the
single-file "Rapid-AIO" merge. GGUF quants of this model only cover the UNet
tensors (img_in/transformer_blocks/txt_in/proj_out -- confirmed by inspecting
the tensor names directly), so CLIP and VAE have to be loaded separately from
the standard split Comfy-Org/Qwen-Image_ComfyUI files:
  - text_encoders/qwen_2.5_vl_7b_fp8_scaled.safetensors
  - vae/qwen_image_vae.safetensors

Requires the ComfyUI-GGUF custom node (city96/ComfyUI-GGUF) for UnetLoaderGGUF.

Source workflow: https://hf-mirror.com/Phr00t/Qwen-Image-Edit-Rapid-AIO/blob/main/Qwen-Rapid-AIO.json
GGUF quants:      https://hf-mirror.com/Novice25/Qwen-Image-Edit-Rapid-AIO-GGUF
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

DEFAULT_UNET_GGUF = "Qwen-Rapid-AIO-NSFW-v19_Q4_K.gguf"
DEFAULT_CLIP = "qwen_2.5_vl_7b_fp8_scaled.safetensors"
DEFAULT_VAE = "qwen_image_vae.safetensors"

DEFAULT_NEGATIVE = ""
DEFAULT_SAMPLER = "sa_solver"
DEFAULT_SCHEDULER = "beta"


def build_prompt(
    image_names: list[str],
    *,
    prompt: str,
    negative: str,
    width: int,
    height: int,
    steps: int,
    cfg: float,
    sampler_name: str,
    scheduler: str,
    seed: int,
    unet_gguf: str,
    clip_name: str,
    vae_name: str,
    filename_prefix: str,
) -> dict:
    g: dict = {}
    g["unet"] = {"class_type": "UnetLoaderGGUF", "inputs": {"unet_name": unet_gguf}}
    g["clip"] = {
        "class_type": "CLIPLoader",
        "inputs": {"clip_name": clip_name, "type": "qwen_image"},
    }
    g["vae"] = {"class_type": "VAELoader", "inputs": {"vae_name": vae_name}}

    load_ids = []
    for i, name in enumerate(image_names, start=1):
        nid = f"img{i}"
        g[nid] = {"class_type": "LoadImage", "inputs": {"image": name}}
        load_ids.append(nid)

    pos_inputs = {"clip": ["clip", 0], "vae": ["vae", 0], "prompt": prompt}
    for i, nid in enumerate(load_ids, start=1):
        pos_inputs[f"image{i}"] = [nid, 0]
    g["pos"] = {"class_type": "TextEncodeQwenImageEditPlus", "inputs": pos_inputs}
    g["neg"] = {
        "class_type": "TextEncodeQwenImageEditPlus",
        "inputs": {"clip": ["clip", 0], "vae": ["vae", 0], "prompt": negative},
    }

    g["latent"] = {
        "class_type": "EmptyLatentImage",
        "inputs": {"width": width, "height": height, "batch_size": 1},
    }
    g["sampler"] = {
        "class_type": "KSampler",
        "inputs": {
            "model": ["unet", 0],
            "positive": ["pos", 0],
            "negative": ["neg", 0],
            "latent_image": ["latent", 0],
            "seed": seed,
            "steps": steps,
            "cfg": cfg,
            "sampler_name": sampler_name,
            "scheduler": scheduler,
            "denoise": 1.0,
        },
    }
    g["decode"] = {"class_type": "VAEDecode", "inputs": {"samples": ["sampler", 0], "vae": ["vae", 0]}}
    g["save"] = {
        "class_type": "SaveImage",
        "inputs": {"images": ["decode", 0], "filename_prefix": filename_prefix},
    }
    return g


def check_nodes(client) -> list[str]:
    required = [
        "UnetLoaderGGUF", "CLIPLoader", "VAELoader", "LoadImage",
        "TextEncodeQwenImageEditPlus", "EmptyLatentImage", "KSampler",
        "VAEDecode", "SaveImage",
    ]
    return [c for c in required if not client.http_json("GET", f"/object_info/{c}", timeout=30)]


def check_files(client, unet_gguf: str, clip_name: str, vae_name: str) -> None:
    checks = [
        ("UnetLoaderGGUF", "unet_name", unet_gguf),
        ("CLIPLoader", "clip_name", clip_name),
        ("VAELoader", "vae_name", vae_name),
    ]
    for node, field, value in checks:
        info = client.http_json("GET", f"/object_info/{node}", timeout=30)
        spec = info[node]["input"]["required"][field]
        options = spec[0] if isinstance(spec[0], list) else []
        if value not in options:
            raise SystemExit(
                f"{field} '{value}' is not installed on {client.base}.\n"
                f"  available: {options}"
            )


def load_text(inline: str | None, file_path: str | None, default: str) -> str:
    if file_path:
        p = Path(file_path)
        if not p.exists():
            raise SystemExit(f"file not found: {p}")
        text = p.read_text(encoding="utf-8").strip()
        if not text:
            raise SystemExit(f"file is empty: {p}")
        return text
    if inline is not None and inline.strip():
        return inline.strip()
    return default


def main() -> None:
    ap = argparse.ArgumentParser(
        description="Remote Qwen-Image-Edit (GGUF unet) via LAN ComfyUI (upload 1-3 images, download PNG)"
    )
    add_server_args(ap)
    ap.add_argument("--image1", type=str, required=True, help="Primary reference image")
    ap.add_argument("--image2", type=str, default=None, help="Optional second reference image")
    ap.add_argument("--image3", type=str, default=None, help="Optional third reference image")
    ap.add_argument("--prompt", type=str, required=True)
    ap.add_argument("--prompt-file", type=str, default=None)
    ap.add_argument("--negative-prompt", type=str, default=None)
    ap.add_argument("--negative-prompt-file", type=str, default=None)
    ap.add_argument("--width", type=int, default=768)
    ap.add_argument("--height", type=int, default=768)
    ap.add_argument("--steps", type=int, default=4, help="author recommends 4 for v1")
    ap.add_argument("--cfg", type=float, default=1.0, help="author recommends 1 for v1")
    ap.add_argument("--sampler", type=str, default=DEFAULT_SAMPLER)
    ap.add_argument("--scheduler", type=str, default=DEFAULT_SCHEDULER)
    ap.add_argument("--seed", type=int, default=None, help="default: random")
    ap.add_argument("--unet-gguf", type=str, default=DEFAULT_UNET_GGUF)
    ap.add_argument("--clip", type=str, default=DEFAULT_CLIP)
    ap.add_argument("--vae", type=str, default=DEFAULT_VAE)
    ap.add_argument("--output", type=str, default=None, help="Local output PNG path")
    ap.add_argument("--timeout", type=float, default=1800)
    ap.add_argument("--skip-model-check", action="store_true")
    ap.add_argument("--dump", metavar="FILE", help="write the API prompt to FILE and exit")
    args = ap.parse_args()

    prompt = load_text(args.prompt, args.prompt_file, "")
    if not prompt:
        raise SystemExit("--prompt (or --prompt-file) is required and cannot be empty")
    negative = load_text(args.negative_prompt, args.negative_prompt_file, DEFAULT_NEGATIVE)
    seed = args.seed if args.seed is not None else random.randrange(0, 2**64)
    run_id = uuid.uuid4().hex[:8]
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    if args.dump:
        graph = build_prompt(
            ["PLACEHOLDER.png"] * (1 + bool(args.image2) + bool(args.image3)),
            prompt=prompt, negative=negative, width=args.width, height=args.height,
            steps=args.steps, cfg=args.cfg, sampler_name=args.sampler,
            scheduler=args.scheduler, seed=seed, unet_gguf=args.unet_gguf,
            clip_name=args.clip, vae_name=args.vae,
            filename_prefix=f"qwen_edit_gguf_{run_id}",
        )
        Path(args.dump).write_text(json.dumps(graph, indent=2, ensure_ascii=False), encoding="utf-8")
        print(f"wrote {args.dump} ({len(graph)} nodes)")
        return

    local_images = [Path(p) for p in (args.image1, args.image2, args.image3) if p]
    for p in local_images:
        if not p.is_file():
            raise SystemExit(f"input image not found: {p}")

    client = connect_from_args(args)

    if not args.skip_model_check:
        missing = check_nodes(client)
        if missing:
            raise SystemExit(f"server {client.base} is missing node classes: {', '.join(missing)}")
        check_files(client, args.unet_gguf, args.clip, args.vae)

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    remote_names = []
    for i, p in enumerate(local_images, start=1):
        remote_name = unique_remote_name(p, prefix=f"qwen_gguf_in{i}_{run_id}")
        up = client.upload_file(p, remote_name=remote_name, overwrite=True)
        remote_names.append(up["name"])

    prefix = f"remote/qwen_edit_gguf_{run_id}"
    graph = build_prompt(
        remote_names, prompt=prompt, negative=negative, width=args.width, height=args.height,
        steps=args.steps, cfg=args.cfg, sampler_name=args.sampler, scheduler=args.scheduler,
        seed=seed, unet_gguf=args.unet_gguf, clip_name=args.clip, vae_name=args.vae,
        filename_prefix=prefix,
    )

    print(f"[qwen_edit_gguf] {args.width}x{args.height} steps={args.steps} cfg={args.cfg} "
          f"{args.sampler}/{args.scheduler} seed={seed} images={len(remote_names)}")

    meta: dict = {
        "mode": "remote_qwen_edit_gguf",
        "server": client.server,
        "run_id": run_id,
        "seed": seed,
        "prompt": prompt,
        "negative_prompt": negative,
        "width": args.width,
        "height": args.height,
        "steps": args.steps,
        "cfg": args.cfg,
        "sampler": args.sampler,
        "scheduler": args.scheduler,
        "unet_gguf": args.unet_gguf,
        "clip": args.clip,
        "vae": args.vae,
        "input_images": [str(p.resolve()) for p in local_images],
        "remote_image_names": remote_names,
        "ok": False,
    }
    meta_path = out_dir / f"qwen_edit_gguf_meta_{stamp}_{run_id}.json"

    try:
        entry, elapsed, prompt_id = client.queue_and_wait(graph, tag="qwen_edit_gguf", timeout=args.timeout)
        saved = client.download_history_outputs(entry, out_dir)
        primary = saved[0] if saved else None
        if primary is None:
            raise RuntimeError("no image downloaded from history outputs")

        if args.output:
            dest = Path(args.output)
            if dest.suffix.lower() not in {".png", ".jpg", ".jpeg", ".webp"}:
                dest = dest.with_suffix(".png")
        else:
            dest = out_dir / f"qwen_edit_gguf_{args.width}x{args.height}_{stamp}.png"
        if dest.resolve() != primary.resolve():
            dest.parent.mkdir(parents=True, exist_ok=True)
            dest.write_bytes(primary.read_bytes())

        meta.update(ok=True, prompt_id=prompt_id, elapsed_sec=round(elapsed, 2),
                    output_image_file=str(dest.resolve()), downloaded=[str(p) for p in saved])
        print(f"[qwen_edit_gguf] OUT {dest}")
    except Exception as e:
        meta["error"] = str(e)
        meta_path.write_text(json.dumps(meta, indent=2, ensure_ascii=False), encoding="utf-8")
        print(f"[qwen_edit_gguf] FAILED: {e}")
        print("meta written:", meta_path)
        raise SystemExit(1)

    meta_path.write_text(json.dumps(meta, indent=2, ensure_ascii=False), encoding="utf-8")
    print("meta written:", meta_path)


if __name__ == "__main__":
    main()
