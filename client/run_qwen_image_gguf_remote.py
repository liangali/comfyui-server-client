"""Remote Qwen-Image text-to-image using the on-disk GGUF quant.

run_qwen_image_remote.py expects qwen_image_distill_full_fp8_e4m3fn.safetensors in
models/unet. This variant drives the GGUF quant that is already installed
(the same file run_qwen_edit_gguf_remote.py uses), so text-to-image needs no
extra download and no second model load if an edit job ran just before.

Qwen-Image is a flow-matching model with a 16-channel latent, so this uses
EmptySD3LatentImage plus ModelSamplingAuraFlow for the sigma shift. The
"Rapid-AIO" merges are distilled: ~8 steps at cfg 1.0 is the sweet spot.
"""
from __future__ import annotations

import argparse
import json
import random
import uuid
from datetime import datetime
from pathlib import Path

from comfyui_remote_client import add_server_args, connect_from_args

DEFAULT_UNET_GGUF = "Qwen-Rapid-AIO-NSFW-v19_Q4_K.gguf"
DEFAULT_CLIP = "qwen_2.5_vl_7b_fp8_scaled.safetensors"
DEFAULT_VAE = "qwen_image_vae.safetensors"

DEFAULT_NEGATIVE = "blurry, low quality, watermark, text, deformed"
DEFAULT_SHIFT = 3.1
DEFAULT_SAMPLER = "euler"
DEFAULT_SCHEDULER = "simple"


def build_prompt(
    *,
    prompt: str,
    negative: str,
    width: int,
    height: int,
    steps: int,
    cfg: float,
    shift: float,
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
    g["sampling"] = {
        "class_type": "ModelSamplingAuraFlow",
        "inputs": {"model": ["unet", 0], "shift": shift},
    }
    g["pos"] = {"class_type": "CLIPTextEncode", "inputs": {"clip": ["clip", 0], "text": prompt}}
    g["neg"] = {"class_type": "CLIPTextEncode", "inputs": {"clip": ["clip", 0], "text": negative}}
    g["latent"] = {
        "class_type": "EmptySD3LatentImage",
        "inputs": {"width": width, "height": height, "batch_size": 1},
    }
    g["sampler"] = {
        "class_type": "KSampler",
        "inputs": {
            "model": ["sampling", 0],
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


def main() -> None:
    ap = argparse.ArgumentParser(
        description="Remote Qwen-Image text-to-image via the on-disk GGUF quant"
    )
    add_server_args(ap)
    ap.add_argument("--prompt", required=True)
    ap.add_argument("--prompt-file", default=None)
    ap.add_argument("--negative", default=DEFAULT_NEGATIVE)
    ap.add_argument("--width", type=int, default=1328)
    ap.add_argument("--height", type=int, default=1328)
    ap.add_argument("--steps", type=int, default=8)
    ap.add_argument("--cfg", type=float, default=1.0)
    ap.add_argument("--shift", type=float, default=DEFAULT_SHIFT)
    ap.add_argument("--sampler", default=DEFAULT_SAMPLER)
    ap.add_argument("--scheduler", default=DEFAULT_SCHEDULER)
    ap.add_argument("--seed", type=int, default=None, help="default: random")
    ap.add_argument("--unet-gguf", default=DEFAULT_UNET_GGUF)
    ap.add_argument("--clip", default=DEFAULT_CLIP)
    ap.add_argument("--vae", default=DEFAULT_VAE)
    ap.add_argument("--output", default=None, help="Local output PNG path")
    ap.add_argument("--timeout", type=float, default=1800)
    args = ap.parse_args()

    prompt = args.prompt
    if args.prompt_file:
        prompt = Path(args.prompt_file).read_text(encoding="utf-8").strip()
    if not prompt.strip():
        raise SystemExit("--prompt (or --prompt-file) cannot be empty")

    seed = args.seed if args.seed is not None else random.randrange(0, 2**63)
    run_id = uuid.uuid4().hex[:8]
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    client = connect_from_args(args)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    graph = build_prompt(
        prompt=prompt, negative=args.negative, width=args.width, height=args.height,
        steps=args.steps, cfg=args.cfg, shift=args.shift, sampler_name=args.sampler,
        scheduler=args.scheduler, seed=seed, unet_gguf=args.unet_gguf,
        clip_name=args.clip, vae_name=args.vae,
        filename_prefix=f"remote/qwen_image_gguf_{run_id}",
    )
    print(f"[qwen_gguf] {args.width}x{args.height} steps={args.steps} cfg={args.cfg} "
          f"shift={args.shift} {args.sampler}/{args.scheduler} seed={seed}")

    meta: dict = {
        "mode": "remote_qwen_image_gguf", "server": client.server, "run_id": run_id,
        "seed": seed, "prompt": prompt, "negative_prompt": args.negative,
        "width": args.width, "height": args.height, "steps": args.steps, "cfg": args.cfg,
        "shift": args.shift, "sampler": args.sampler, "scheduler": args.scheduler,
        "unet_gguf": args.unet_gguf, "ok": False,
    }
    meta_path = out_dir / f"qwen_image_gguf_meta_{stamp}_{run_id}.json"

    try:
        entry, elapsed, prompt_id = client.queue_and_wait(graph, tag="qwen_gguf", timeout=args.timeout)
        saved = client.download_history_outputs(entry, out_dir)
        if not saved:
            raise RuntimeError("no image downloaded from history outputs")
        dest = Path(args.output) if args.output else out_dir / f"qwen_image_gguf_{stamp}.png"
        if dest.suffix.lower() not in {".png", ".jpg", ".jpeg", ".webp"}:
            dest = dest.with_suffix(".png")
        if dest.resolve() != saved[0].resolve():
            dest.parent.mkdir(parents=True, exist_ok=True)
            dest.write_bytes(saved[0].read_bytes())
        meta.update(ok=True, prompt_id=prompt_id, elapsed_sec=round(elapsed, 2),
                    output_image_file=str(dest.resolve()))
        print(f"[qwen_gguf] OUT {dest}")
    except Exception as e:
        meta["error"] = str(e)
        meta_path.write_text(json.dumps(meta, indent=2, ensure_ascii=False), encoding="utf-8")
        print(f"[qwen_gguf] FAILED: {e}")
        raise SystemExit(1)

    meta_path.write_text(json.dumps(meta, indent=2, ensure_ascii=False), encoding="utf-8")
    print("meta written:", meta_path)


if __name__ == "__main__":
    main()
