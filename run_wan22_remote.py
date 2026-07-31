"""Remote Wan2.2 video client for LAN ComfyUI on B70.

Supports:
  - 5B TI2V  (needs --image)
  - 14B T2V  (text only)
  - 14B I2V  (needs --image)

Uploads local input images to the B70 server, runs inference, downloads MP4 back.
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

NEG = (
    "色调艳丽，过曝，静态，细节模糊不清，字幕，风格，作品，画作，画面，静止，整体发灰，"
    "最差质量，低质量，JPEG压缩残留，丑陋的，残缺的，多余的手指，画得不好的手部，"
    "画得不好的脸部，畸形的，毁容的，形态畸形的肢体，手指融合，静止不动的画面，"
    "杂乱的背景，三条腿，背景人很多，倒着走"
)


def wan_length(duration_sec: float, fps: float = 16.0) -> int:
    n = int(round(duration_sec * fps)) + 1
    while n % 4 != 1:
        n += 1
    return n


def prompt_5b_ti2v(image_name: str, seed: int, filename_prefix: str, positive: str | None = None) -> dict:
    width, height, length = 832, 480, 49
    pos = positive or (
        "A cinematic camera slowly pans across a rainy neon city street at dusk, "
        "reflections on wet asphalt, subtle motion, high detail, smooth animation."
    )
    return {
        "37": {
            "class_type": "UNETLoader",
            "inputs": {"unet_name": "wan2.2_ti2v_5B_fp16.safetensors", "weight_dtype": "default"},
        },
        "38": {
            "class_type": "CLIPLoader",
            "inputs": {
                "clip_name": "umt5_xxl_fp8_e4m3fn_scaled.safetensors",
                "type": "wan",
                "device": "default",
            },
        },
        "39": {"class_type": "VAELoader", "inputs": {"vae_name": "wan2.2_vae.safetensors"}},
        "48": {"class_type": "ModelSamplingSD3", "inputs": {"model": ["37", 0], "shift": 8.0}},
        "6": {"class_type": "CLIPTextEncode", "inputs": {"clip": ["38", 0], "text": pos}},
        "7": {"class_type": "CLIPTextEncode", "inputs": {"clip": ["38", 0], "text": NEG}},
        "56": {"class_type": "LoadImage", "inputs": {"image": image_name}},
        "55": {
            "class_type": "Wan22ImageToVideoLatent",
            "inputs": {
                "vae": ["39", 0],
                "start_image": ["56", 0],
                "width": width,
                "height": height,
                "length": length,
                "batch_size": 1,
            },
        },
        "3": {
            "class_type": "KSampler",
            "inputs": {
                "model": ["48", 0],
                "positive": ["6", 0],
                "negative": ["7", 0],
                "latent_image": ["55", 0],
                "seed": seed,
                "steps": 15,
                "cfg": 5.0,
                "sampler_name": "uni_pc",
                "scheduler": "simple",
                "denoise": 1.0,
            },
        },
        "8": {"class_type": "VAEDecode", "inputs": {"samples": ["3", 0], "vae": ["39", 0]}},
        "57": {"class_type": "CreateVideo", "inputs": {"images": ["8", 0], "fps": 16.0}},
        "58": {
            "class_type": "SaveVideo",
            "inputs": {
                "video": ["57", 0],
                "filename_prefix": filename_prefix,
                "format": "auto",
                "codec": "auto",
            },
        },
    }


def prompt_14b_t2v(seed: int, filename_prefix: str, positive: str | None = None) -> dict:
    width, height, length = 640, 640, 81
    pos = positive or (
        "Beautiful young European woman with honey blonde hair gracefully turning her head "
        "back over shoulder, gentle smile, bright eyes looking at camera. Hair flowing in "
        "slow motion as she turns. Soft natural lighting, clean background, cinematic slow-motion portrait."
    )
    return {
        "clip": {
            "class_type": "CLIPLoader",
            "inputs": {
                "clip_name": "umt5_xxl_fp8_e4m3fn_scaled.safetensors",
                "type": "wan",
                "device": "default",
            },
        },
        "vae": {"class_type": "VAELoader", "inputs": {"vae_name": "wan_2.1_vae.safetensors"}},
        "unet_high": {
            "class_type": "UNETLoader",
            "inputs": {
                "unet_name": "wan2.2_t2v_high_noise_14B_fp8_scaled.safetensors",
                "weight_dtype": "default",
            },
        },
        "unet_low": {
            "class_type": "UNETLoader",
            "inputs": {
                "unet_name": "wan2.2_t2v_low_noise_14B_fp8_scaled.safetensors",
                "weight_dtype": "default",
            },
        },
        "lora_high": {
            "class_type": "LoraLoaderModelOnly",
            "inputs": {
                "model": ["unet_high", 0],
                "lora_name": "wan2.2_t2v_lightx2v_4steps_lora_v1.1_high_noise.safetensors",
                "strength_model": 1.0,
            },
        },
        "lora_low": {
            "class_type": "LoraLoaderModelOnly",
            "inputs": {
                "model": ["unet_low", 0],
                "lora_name": "wan2.2_t2v_lightx2v_4steps_lora_v1.1_low_noise.safetensors",
                "strength_model": 1.0,
            },
        },
        "ms_high": {"class_type": "ModelSamplingSD3", "inputs": {"model": ["lora_high", 0], "shift": 5.0}},
        "ms_low": {"class_type": "ModelSamplingSD3", "inputs": {"model": ["lora_low", 0], "shift": 5.0}},
        "pos": {"class_type": "CLIPTextEncode", "inputs": {"clip": ["clip", 0], "text": pos}},
        "neg": {"class_type": "CLIPTextEncode", "inputs": {"clip": ["clip", 0], "text": NEG}},
        "latent": {
            "class_type": "EmptyHunyuanLatentVideo",
            "inputs": {"width": width, "height": height, "length": length, "batch_size": 1},
        },
        "ks_high": {
            "class_type": "KSamplerAdvanced",
            "inputs": {
                "model": ["ms_high", 0],
                "positive": ["pos", 0],
                "negative": ["neg", 0],
                "latent_image": ["latent", 0],
                "add_noise": "enable",
                "noise_seed": seed,
                "steps": 4,
                "cfg": 1.0,
                "sampler_name": "euler",
                "scheduler": "simple",
                "start_at_step": 0,
                "end_at_step": 2,
                "return_with_leftover_noise": "enable",
            },
        },
        "ks_low": {
            "class_type": "KSamplerAdvanced",
            "inputs": {
                "model": ["ms_low", 0],
                "positive": ["pos", 0],
                "negative": ["neg", 0],
                "latent_image": ["ks_high", 0],
                "add_noise": "disable",
                "noise_seed": 0,
                "steps": 4,
                "cfg": 1.0,
                "sampler_name": "euler",
                "scheduler": "simple",
                "start_at_step": 2,
                "end_at_step": 4,
                "return_with_leftover_noise": "disable",
            },
        },
        "decode": {"class_type": "VAEDecode", "inputs": {"samples": ["ks_low", 0], "vae": ["vae", 0]}},
        "video": {"class_type": "CreateVideo", "inputs": {"images": ["decode", 0], "fps": 16.0}},
        "save": {
            "class_type": "SaveVideo",
            "inputs": {
                "video": ["video", 0],
                "filename_prefix": filename_prefix,
                "format": "auto",
                "codec": "auto",
            },
        },
    }


def prompt_14b_i2v(
    image_name: str,
    *,
    positive: str | None,
    negative: str | None,
    width: int,
    height: int,
    length: int,
    seed: int,
    filename_prefix: str,
) -> dict:
    pos = positive or (
        "The scene gently comes alive with subtle camera motion and natural movement, "
        "cinematic lighting, smooth animation, highly detailed."
    )
    neg = negative if negative is not None and negative.strip() else NEG
    return {
        "clip": {
            "class_type": "CLIPLoader",
            "inputs": {
                "clip_name": "umt5_xxl_fp8_e4m3fn_scaled.safetensors",
                "type": "wan",
                "device": "default",
            },
        },
        "vae": {"class_type": "VAELoader", "inputs": {"vae_name": "wan_2.1_vae.safetensors"}},
        "unet_high": {
            "class_type": "UNETLoader",
            "inputs": {
                "unet_name": "wan2.2_i2v_high_noise_14B_fp8_scaled.safetensors",
                "weight_dtype": "default",
            },
        },
        "unet_low": {
            "class_type": "UNETLoader",
            "inputs": {
                "unet_name": "wan2.2_i2v_low_noise_14B_fp8_scaled.safetensors",
                "weight_dtype": "default",
            },
        },
        "lora_high": {
            "class_type": "LoraLoaderModelOnly",
            "inputs": {
                "model": ["unet_high", 0],
                "lora_name": "wan2.2_i2v_lightx2v_4steps_lora_v1_high_noise.safetensors",
                "strength_model": 1.0,
            },
        },
        "lora_low": {
            "class_type": "LoraLoaderModelOnly",
            "inputs": {
                "model": ["unet_low", 0],
                "lora_name": "wan2.2_i2v_lightx2v_4steps_lora_v1_low_noise.safetensors",
                "strength_model": 1.0,
            },
        },
        "ms_high": {"class_type": "ModelSamplingSD3", "inputs": {"model": ["lora_high", 0], "shift": 5.0}},
        "ms_low": {"class_type": "ModelSamplingSD3", "inputs": {"model": ["lora_low", 0], "shift": 5.0}},
        "pos": {"class_type": "CLIPTextEncode", "inputs": {"clip": ["clip", 0], "text": pos}},
        "neg": {"class_type": "CLIPTextEncode", "inputs": {"clip": ["clip", 0], "text": neg}},
        "img": {"class_type": "LoadImage", "inputs": {"image": image_name}},
        "i2v": {
            "class_type": "WanImageToVideo",
            "inputs": {
                "positive": ["pos", 0],
                "negative": ["neg", 0],
                "vae": ["vae", 0],
                "start_image": ["img", 0],
                "width": width,
                "height": height,
                "length": length,
                "batch_size": 1,
            },
        },
        "ks_high": {
            "class_type": "KSamplerAdvanced",
            "inputs": {
                "model": ["ms_high", 0],
                "positive": ["i2v", 0],
                "negative": ["i2v", 1],
                "latent_image": ["i2v", 2],
                "add_noise": "enable",
                "noise_seed": seed,
                "steps": 4,
                "cfg": 1.0,
                "sampler_name": "euler",
                "scheduler": "simple",
                "start_at_step": 0,
                "end_at_step": 2,
                "return_with_leftover_noise": "enable",
            },
        },
        "ks_low": {
            "class_type": "KSamplerAdvanced",
            "inputs": {
                "model": ["ms_low", 0],
                "positive": ["i2v", 0],
                "negative": ["i2v", 1],
                "latent_image": ["ks_high", 0],
                "add_noise": "disable",
                "noise_seed": 0,
                "steps": 4,
                "cfg": 1.0,
                "sampler_name": "euler",
                "scheduler": "simple",
                "start_at_step": 2,
                "end_at_step": 4,
                "return_with_leftover_noise": "disable",
            },
        },
        "decode": {"class_type": "VAEDecode", "inputs": {"samples": ["ks_low", 0], "vae": ["vae", 0]}},
        "video": {"class_type": "CreateVideo", "inputs": {"images": ["decode", 0], "fps": 16.0}},
        "save": {
            "class_type": "SaveVideo",
            "inputs": {
                "video": ["video", 0],
                "filename_prefix": filename_prefix,
                "format": "auto",
                "codec": "auto",
            },
        },
    }


def load_text(inline: str | None, file_path: str | None, label: str) -> str | None:
    if file_path:
        p = Path(file_path)
        if not p.exists():
            raise FileNotFoundError(f"--{label}-file not found: {p}")
        text = p.read_text(encoding="utf-8").strip()
        if not text:
            raise ValueError(f"--{label}-file is empty: {p}")
        return text
    if inline is not None and inline.strip():
        return inline.strip()
    return None


def pick_primary_video(saved: list[Path]) -> Path | None:
    vids = [p for p in saved if p.suffix.lower() in {".mp4", ".webm", ".mkv", ".avi", ".mov"}]
    return vids[0] if vids else (saved[0] if saved else None)


def main() -> None:
    ap = argparse.ArgumentParser(
        description="Remote Wan2.2 via LAN ComfyUI (upload image, download MP4)"
    )
    add_server_args(ap)
    ap.add_argument("--only", choices=["5b", "t2v", "i2v", "all"], default="i2v")
    ap.add_argument("--image", type=str, default=None, help="Local input image for 5B TI2V / 14B I2V")
    ap.add_argument("--prompt", type=str, default=None)
    ap.add_argument("--prompt-file", type=str, default=None)
    ap.add_argument("--negative-prompt", type=str, default=None)
    ap.add_argument("--negative-prompt-file", type=str, default=None)
    ap.add_argument("--width", type=int, default=640, help="I2V width (default 640)")
    ap.add_argument("--height", type=int, default=640, help="I2V height (default 640)")
    ap.add_argument("--duration", type=float, default=5.0, help="I2V target seconds @16fps")
    ap.add_argument("--seed", type=int, default=None)
    ap.add_argument(
        "--output",
        type=str,
        default=None,
        help="Local output MP4 path (multi-job: stem gets _model suffix)",
    )
    ap.add_argument("--timeout", type=float, default=7200)
    args = ap.parse_args()

    positive = load_text(args.prompt, args.prompt_file, "prompt")
    negative = load_text(args.negative_prompt, args.negative_prompt_file, "negative-prompt")
    seed = args.seed if args.seed is not None else random.randrange(0, 2**32)
    run_id = uuid.uuid4().hex[:8]
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    needs_image = args.only in ("5b", "i2v", "all")
    if needs_image and not args.image:
        raise SystemExit(f"--image is required for --only {args.only}")

    client = connect_from_args(args)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    remote_image_name = None
    if args.image:
        local_img = Path(args.image)
        if not local_img.is_file():
            raise SystemExit(f"input image not found: {local_img}")
        remote_name = unique_remote_name(local_img, prefix=f"wan_in_{run_id}")
        up = client.upload_file(local_img, remote_name=remote_name, overwrite=True)
        remote_image_name = up["name"]

    length = wan_length(args.duration)
    jobs: list[tuple[str, dict, str, int, int, float]] = []
    # tag, prompt_graph, model_name, w, h, duration

    if args.only in ("5b", "all"):
        assert remote_image_name
        prefix = f"remote/wan22_5b_ti2v_{run_id}"
        jobs.append(
            (
                "wan22_5b_ti2v",
                prompt_5b_ti2v(remote_image_name, seed, prefix, positive=positive),
                "wan22_5b_ti2v",
                832,
                480,
                3.0,
            )
        )
    if args.only in ("t2v", "all"):
        prefix = f"remote/wan22_14b_t2v_{run_id}"
        jobs.append(
            (
                "wan22_14b_t2v",
                prompt_14b_t2v(seed, prefix, positive=positive),
                "wan22_14b_t2v",
                640,
                640,
                5.0,
            )
        )
    if args.only in ("i2v", "all"):
        assert remote_image_name
        prefix = f"remote/wan22_14b_i2v_{run_id}"
        jobs.append(
            (
                "wan22_14b_i2v",
                prompt_14b_i2v(
                    remote_image_name,
                    positive=positive,
                    negative=negative,
                    width=args.width,
                    height=args.height,
                    length=length,
                    seed=seed,
                    filename_prefix=prefix,
                ),
                "wan22_14b_i2v",
                args.width,
                args.height,
                args.duration,
            )
        )

    summary: dict = {
        "mode": "remote_wan22",
        "server": client.server,
        "run_id": run_id,
        "seed": seed,
        "input_image": str(Path(args.image).resolve()) if args.image else None,
        "remote_image_name": remote_image_name,
        "runs": [],
    }

    multi = len(jobs) > 1
    for tag, prompt, model_name, w, h, dur in jobs:
        print("=" * 60, tag)
        try:
            entry, elapsed, prompt_id = client.queue_and_wait(
                prompt, tag=tag, timeout=args.timeout
            )
            saved = client.download_history_outputs(entry, out_dir)
            primary = pick_primary_video(saved)
            if primary is None:
                raise RuntimeError("no video downloaded from history outputs")

            if args.output:
                dest = Path(args.output)
                if multi:
                    dest = dest.with_name(f"{dest.stem}_{model_name}{dest.suffix or '.mp4'}")
                if dest.suffix.lower() not in {".mp4", ".webm", ".mkv", ".avi", ".mov"}:
                    dest = dest.with_suffix(".mp4")
                dest.parent.mkdir(parents=True, exist_ok=True)
                dest.write_bytes(primary.read_bytes())
                primary = dest
            else:
                friendly = out_dir / f"{model_name}_{w}x{h}_{dur}s_{stamp}.mp4"
                friendly.write_bytes(primary.read_bytes())
                primary = friendly

            print(f"[{tag}] OUT {primary}")
            summary["runs"].append(
                {
                    "tag": tag,
                    "ok": True,
                    "prompt_id": prompt_id,
                    "elapsed_sec": round(elapsed, 2),
                    "output_video_file": str(primary.resolve()),
                    "downloaded": [str(p) for p in saved],
                }
            )
        except Exception as e:
            print(f"[{tag}] FAILED: {e}")
            summary["runs"].append({"tag": tag, "ok": False, "error": str(e)})

    summary_path = out_dir / f"wan22_remote_summary_{stamp}_{run_id}.json"
    summary_path.write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
    print("SUMMARY", json.dumps(summary, indent=2, ensure_ascii=False))
    print("summary written:", summary_path)
    if not summary["runs"] or not all(r.get("ok") for r in summary["runs"]):
        raise SystemExit(1)


if __name__ == "__main__":
    main()
