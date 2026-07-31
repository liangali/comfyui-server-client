"""Remote Qwen-Image (distill) client for LAN ComfyUI on B70.

Runs on any machine that can reach the B70 ComfyUI server.
Uploads nothing (text-to-image); downloads the generated PNG locally.
"""
from __future__ import annotations

import argparse
import json
import uuid
from datetime import datetime
from pathlib import Path

from comfyui_remote_client import add_server_args, connect_from_args

DEFAULT_PROMPT = (
    'A realistic Plato statue is drinking beer, showing an extremely comfortable feeling.\n\n'
    'On the left is a piece of minimalist text:\n'
    'First line: "Qwen-Image distilled version"\n'
    'Second line: "Faster inference"\n'
    "Bold font, sans-serif, white.\n\n"
    "Golden scarf, black sunglasses, statue, Klein blue background, master, artwork, "
    "ultra-high definition, 32k, simple."
)


def build_qwen_prompt(
    text: str,
    *,
    width: int = 1024,
    height: int = 1024,
    steps: int = 10,
    cfg: float = 1.0,
    seed: int = 42,
    filename_prefix: str = "remote/qwen_image",
) -> dict:
    return {
        "37": {
            "class_type": "UNETLoader",
            "inputs": {
                "unet_name": "qwen_image_distill_full_fp8_e4m3fn.safetensors",
                "weight_dtype": "default",
            },
        },
        "38": {
            "class_type": "CLIPLoader",
            "inputs": {
                "clip_name": "qwen_2.5_vl_7b_fp8_scaled.safetensors",
                "type": "qwen_image",
                "device": "default",
            },
        },
        "39": {
            "class_type": "VAELoader",
            "inputs": {"vae_name": "qwen_image_vae.safetensors"},
        },
        "66": {
            "class_type": "ModelSamplingAuraFlow",
            "inputs": {"model": ["37", 0], "shift": 3.0},
        },
        "6": {
            "class_type": "CLIPTextEncode",
            "inputs": {"clip": ["38", 0], "text": text},
        },
        "7": {
            "class_type": "CLIPTextEncode",
            "inputs": {"clip": ["38", 0], "text": ""},
        },
        "72": {
            "class_type": "EmptySD3LatentImage",
            "inputs": {"width": width, "height": height, "batch_size": 1},
        },
        "3": {
            "class_type": "KSampler",
            "inputs": {
                "model": ["66", 0],
                "positive": ["6", 0],
                "negative": ["7", 0],
                "latent_image": ["72", 0],
                "seed": seed,
                "steps": steps,
                "cfg": cfg,
                "sampler_name": "res_multistep",
                "scheduler": "simple",
                "denoise": 1.0,
            },
        },
        "8": {
            "class_type": "VAEDecode",
            "inputs": {"samples": ["3", 0], "vae": ["39", 0]},
        },
        "60": {
            "class_type": "SaveImage",
            "inputs": {"images": ["8", 0], "filename_prefix": filename_prefix},
        },
    }


def main() -> None:
    ap = argparse.ArgumentParser(
        description="Remote Qwen-Image distill via LAN ComfyUI (upload N/A, download PNG)"
    )
    add_server_args(ap)
    ap.add_argument("--prompt", type=str, default=None, help="Positive text prompt")
    ap.add_argument("--prompt-file", type=str, default=None, help="UTF-8 file with prompt")
    ap.add_argument("--width", type=int, default=1024)
    ap.add_argument("--height", type=int, default=1024)
    ap.add_argument("--steps", type=int, default=10)
    ap.add_argument("--cfg", type=float, default=1.0)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument(
        "--output",
        type=str,
        default=None,
        help="Local output PNG path (default: <out-dir>/qwen_image_<stamp>.png)",
    )
    ap.add_argument(
        "--timeout",
        type=float,
        default=1800,
        help="Max seconds to wait for generation (default: 1800)",
    )
    args = ap.parse_args()

    if args.prompt_file:
        text = Path(args.prompt_file).read_text(encoding="utf-8").strip()
        if not text:
            raise SystemExit(f"empty prompt file: {args.prompt_file}")
    elif args.prompt:
        text = args.prompt.strip()
    else:
        text = DEFAULT_PROMPT

    client = connect_from_args(args)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    run_id = uuid.uuid4().hex[:8]
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    prefix = f"remote/qwen_image_{run_id}"
    prompt = build_qwen_prompt(
        text,
        width=args.width,
        height=args.height,
        steps=args.steps,
        cfg=args.cfg,
        seed=args.seed,
        filename_prefix=prefix,
    )

    print(f"submitting Qwen-Image {args.width}x{args.height} steps={args.steps} seed={args.seed}")
    entry, elapsed, prompt_id = client.queue_and_wait(
        prompt, tag="qwen_image", timeout=args.timeout
    )

    meta_path = out_dir / f"qwen_image_{stamp}_{run_id}_meta.json"
    meta_path.write_text(
        json.dumps(
            {
                "prompt_id": prompt_id,
                "elapsed_sec": round(elapsed, 2),
                "server": client.server,
                "width": args.width,
                "height": args.height,
                "steps": args.steps,
                "seed": args.seed,
                "prompt_text": text,
                "outputs": entry.get("outputs"),
            },
            indent=2,
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )

    saved = client.download_history_outputs(entry, out_dir)
    if not saved:
        raise SystemExit("no image outputs in history; check server logs")

    final = saved[0]
    if args.output:
        dest = Path(args.output)
        dest.parent.mkdir(parents=True, exist_ok=True)
        dest.write_bytes(final.read_bytes())
        print(f"copied primary output -> {dest}")
        final = dest
    else:
        # Rename first PNG to a stable friendly name
        friendly = out_dir / f"qwen_image_{args.width}x{args.height}_{stamp}.png"
        if final.suffix.lower() in {".png", ".jpg", ".jpeg", ".webp"} and final != friendly:
            friendly.write_bytes(final.read_bytes())
            print(f"primary output alias -> {friendly}")
            final = friendly

    print("=" * 60)
    print(f"OK  elapsed={elapsed:.1f}s")
    print(f"OUT {final}")
    print(f"META {meta_path}")


if __name__ == "__main__":
    main()
