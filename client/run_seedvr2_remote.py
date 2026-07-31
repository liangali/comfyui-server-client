"""Remote SeedVR2 video upscaler client for LAN ComfyUI on B70.

Uploads a local input video to the B70 server, runs SeedVR2 (default 1080p),
downloads the upscaled MP4 back to the client machine.
"""
from __future__ import annotations

import argparse
import json
import uuid
from datetime import datetime
from pathlib import Path

from comfyui_remote_client import (
    add_server_args,
    connect_from_args,
    unique_remote_name,
)


def seedvr2_prompt(
    video_name: str,
    *,
    resolution: int = 1080,
    seed: int = 42,
    filename_prefix: str = "remote/seedvr2_1080p",
    batch_size: int = 33,
    blocks_to_swap: int = 32,
) -> dict:
    return {
        "load": {"class_type": "LoadVideo", "inputs": {"file": video_name}},
        "comp": {"class_type": "GetVideoComponents", "inputs": {"video": ["load", 0]}},
        "dit": {
            "class_type": "SeedVR2LoadDiTModel",
            "inputs": {
                "model": "seedvr2_ema_3b_fp8_e4m3fn.safetensors",
                "device": "xpu",
                "blocks_to_swap": blocks_to_swap,
                "swap_io_components": False,
                "offload_device": "cpu",
                "cache_model": False,
                "attention_mode": "sdpa",
            },
        },
        "vae": {
            "class_type": "SeedVR2LoadVAEModel",
            "inputs": {
                "model": "ema_vae_fp16.safetensors",
                "device": "xpu",
                "encode_tiled": True,
                "encode_tile_size": 1024,
                "encode_tile_overlap": 128,
                "decode_tiled": True,
                "decode_tile_size": 768,
                "decode_tile_overlap": 128,
                "tile_debug": "false",
                "offload_device": "cpu",
                "cache_model": False,
            },
        },
        "up": {
            "class_type": "SeedVR2VideoUpscaler",
            "inputs": {
                "image": ["comp", 0],
                "dit": ["dit", 0],
                "vae": ["vae", 0],
                "seed": seed,
                "resolution": resolution,
                "max_resolution": 0,
                "batch_size": batch_size,
                "uniform_batch_size": True,
                "color_correction": "lab",
                "temporal_overlap": 3,
                "prepend_frames": 0,
                "input_noise_scale": 0.0,
                "latent_noise_scale": 0.0,
                "offload_device": "cpu",
                "enable_debug": False,
            },
        },
        "vid": {
            "class_type": "CreateVideo",
            "inputs": {"images": ["up", 0], "audio": ["comp", 1], "fps": ["comp", 2]},
        },
        "save": {
            "class_type": "SaveVideo",
            "inputs": {
                "video": ["vid", 0],
                "filename_prefix": filename_prefix,
                "format": "auto",
                "codec": "auto",
            },
        },
    }


def pick_primary_video(saved: list[Path]) -> Path | None:
    vids = [p for p in saved if p.suffix.lower() in {".mp4", ".webm", ".mkv", ".avi", ".mov"}]
    return vids[0] if vids else (saved[0] if saved else None)


def main() -> None:
    ap = argparse.ArgumentParser(
        description="Remote SeedVR2 upscale via LAN ComfyUI (upload MP4, download MP4)"
    )
    add_server_args(ap)
    ap.add_argument(
        "--video",
        type=str,
        required=True,
        help="Local input video path (e.g. 480p MP4 from Wan)",
    )
    ap.add_argument("--resolution", type=int, default=1080, help="SeedVR2 target height (default 1080)")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--batch-size", type=int, default=33)
    ap.add_argument("--blocks-to-swap", type=int, default=32)
    ap.add_argument(
        "--output",
        type=str,
        default=None,
        help="Local output MP4 path (default: <out-dir>/seedvr2_<res>p_<stamp>.mp4)",
    )
    ap.add_argument("--timeout", type=float, default=7200)
    ap.add_argument(
        "--skip-node-check",
        action="store_true",
        help="Do not query /object_info/SeedVR2VideoUpscaler before running",
    )
    args = ap.parse_args()

    local_video = Path(args.video)
    if not local_video.is_file():
        raise SystemExit(f"input video not found: {local_video}")
    if local_video.suffix.lower() not in {".mp4", ".webm", ".mkv", ".avi", ".mov", ".gif"}:
        print(f"warning: unusual video extension {local_video.suffix!r}, uploading anyway")

    client = connect_from_args(args)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    if not args.skip_node_check:
        info = client.http_json("GET", "/object_info/SeedVR2VideoUpscaler", timeout=60)
        if "SeedVR2VideoUpscaler" not in info:
            raise SystemExit(
                "SeedVR2 nodes not loaded on the remote ComfyUI. "
                "Install ComfyUI-SeedVR2_VideoUpscaler + XPU patch on the B70 machine."
            )
        print("SeedVR2 nodes OK on remote server")

    run_id = uuid.uuid4().hex[:8]
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    remote_name = unique_remote_name(local_video, prefix=f"seedvr2_in_{run_id}")
    up = client.upload_file(local_video, remote_name=remote_name, overwrite=True)
    remote_video_name = up["name"]

    prefix = f"remote/seedvr2_{args.resolution}p_{run_id}"
    prompt = seedvr2_prompt(
        remote_video_name,
        resolution=args.resolution,
        seed=args.seed,
        filename_prefix=prefix,
        batch_size=args.batch_size,
        blocks_to_swap=args.blocks_to_swap,
    )

    print(
        f"submitting SeedVR2 resolution={args.resolution} "
        f"batch_size={args.batch_size} input={local_video.name}"
    )
    entry, elapsed, prompt_id = client.queue_and_wait(
        prompt, tag="seedvr2", timeout=args.timeout
    )

    saved = client.download_history_outputs(entry, out_dir)
    primary = pick_primary_video(saved)
    if primary is None:
        raise SystemExit("no video outputs in history; check server logs")

    if args.output:
        dest = Path(args.output)
        if dest.suffix.lower() not in {".mp4", ".webm", ".mkv", ".avi", ".mov"}:
            dest = dest.with_suffix(".mp4")
        dest.parent.mkdir(parents=True, exist_ok=True)
        dest.write_bytes(primary.read_bytes())
        primary = dest
    else:
        friendly = out_dir / f"seedvr2_{args.resolution}p_{stamp}.mp4"
        friendly.write_bytes(primary.read_bytes())
        primary = friendly

    meta = {
        "tag": "seedvr2_remote",
        "prompt_id": prompt_id,
        "elapsed_sec": round(elapsed, 2),
        "server": client.server,
        "input_video": str(local_video.resolve()),
        "remote_video_name": remote_video_name,
        "output_video": str(primary.resolve()),
        "downloaded": [str(p) for p in saved],
        "settings": {
            "dit": "seedvr2_ema_3b_fp8_e4m3fn.safetensors",
            "vae": "ema_vae_fp16.safetensors",
            "device": "xpu",
            "resolution": args.resolution,
            "batch_size": args.batch_size,
            "blocks_to_swap": args.blocks_to_swap,
            "attention_mode": "sdpa",
            "seed": args.seed,
        },
        "outputs": entry.get("outputs"),
    }
    meta_path = out_dir / f"seedvr2_remote_meta_{stamp}_{run_id}.json"
    meta_path.write_text(json.dumps(meta, indent=2, ensure_ascii=False), encoding="utf-8")

    print("=" * 60)
    print(f"OK  elapsed={elapsed:.1f}s")
    print(f"OUT {primary}")
    print(f"META {meta_path}")


if __name__ == "__main__":
    main()
