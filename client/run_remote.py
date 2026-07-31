"""Single entry point for every remote ComfyUI job: pick the task, it runs that model.

One ComfyUI server can execute any graph, so routing belongs on the client. This
dispatcher maps a task name to the matching run_*_remote.py module and forwards
all remaining arguments to it untouched -- every flag documented by the
individual scripts still works.

  python run_remote.py qwen      --prompt "..."                  # text -> image
  python run_remote.py qwen-edit --image1 char.png --image2 bg.png --prompt "..."
  python run_remote.py ltx       --image first_frame.png --prompt "..."

A storyboard (分镜图) is just qwen-edit called once per panel: same --image1
(character) and --image2 (background), varying only the camera direction in
--prompt.

Server address resolution (shared by all tasks): --server > $COMFYUI_SERVER >
127.0.0.1:8188. Set it once per shell:

  export COMFYUI_SERVER=127.0.0.1:8189

Use `python run_remote.py list` to see the tasks, and
`python run_remote.py <task> --help` for a task's own flags.
"""
from __future__ import annotations

import importlib
import os
import subprocess
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
REPO_ROOT = HERE.parent

# task name -> (module, aliases, one-line description)
TASKS: dict[str, tuple[str, tuple[str, ...], str]] = {
    "qwen": (
        "run_qwen_image_remote", ("qwen-image", "t2i"),
        "Qwen-Image text -> image (needs qwen_image_distill_full_fp8 in models/unet)",
    ),
    "qwen-gguf": (
        "run_qwen_image_gguf_remote", ("t2i-gguf",),
        "Qwen-Image text -> image using the on-disk GGUF quant (no extra download)",
    ),
    "qwen-edit": (
        "run_qwen_edit_gguf_remote", ("edit", "qwen-edit-gguf"),
        "Qwen-Image-Edit GGUF: 1-3 reference images + prompt -> edited image",
    ),
    "qwen-edit-aio": (
        "run_qwen_edit_remote", ("edit-aio",),
        "Qwen-Image-Edit via the single-file Rapid-AIO checkpoint",
    ),
    "ltx": (
        "run_ltx23_i2v_remote", ("ltx23", "i2v"),
        "LTX-2.3 22B image -> video (mp4), optional x2 latent refine pass",
    ),
    "wan": (
        "run_wan22_remote", ("wan22",),
        "Wan 2.2 image/text -> video",
    ),
    "seedvr2": (
        "run_seedvr2_remote", ("upscale-video",),
        "SeedVR2 video restoration / upscale",
    ),
}

ALIASES = {alias: name for name, (_, aliases, _) in TASKS.items() for alias in aliases}


def resolve_task(token: str) -> str:
    key = token.strip().lower()
    if key in TASKS:
        return key
    if key in ALIASES:
        return ALIASES[key]
    raise SystemExit(
        f"unknown task {token!r}.\nAvailable: {', '.join(sorted(TASKS))}\n"
        "Run `python run_remote.py list` for descriptions."
    )


def print_tasks() -> None:
    server = os.environ.get("COMFYUI_SERVER", "").strip() or "127.0.0.1:8188 (default)"
    print(f"COMFYUI_SERVER = {server}\n")
    print("tasks:")
    width = max(len(n) for n in TASKS)
    for name in sorted(TASKS):
        _, aliases, desc = TASKS[name]
        alias_txt = f"  (aliases: {', '.join(aliases)})" if aliases else ""
        print(f"  {name:<{width}}  {desc}{alias_txt}")
    print("\nserve  start the ComfyUI server itself (server/start_comfyui_remote.sh)")
    print("\nper-task flags:  python run_remote.py <task> --help")


def do_serve(rest: list[str]) -> int:
    """Launch the ComfyUI server via the repo's start script, forwarding extra flags."""
    script = REPO_ROOT / "server" / "start_comfyui_remote.sh"
    if not script.is_file():
        raise SystemExit(f"start script not found: {script}")
    if not os.access(script, os.X_OK):
        cmd = ["bash", str(script), *rest]
    else:
        cmd = [str(script), *rest]
    port = os.environ.get("COMFY_PORT", "8188")
    print(f"launching {script} (port {port}) ...")
    return subprocess.call(cmd)


def main(argv: list[str] | None = None) -> int:
    argv = list(sys.argv[1:] if argv is None else argv)
    if not argv or argv[0] in {"-h", "--help", "help"}:
        print(__doc__)
        print_tasks()
        return 0
    token, rest = argv[0], argv[1:]
    if token.lower() in {"list", "tasks", "--list"}:
        print_tasks()
        return 0
    if token.lower() == "serve":
        return do_serve(rest)

    task = resolve_task(token)
    module_name, _, desc = TASKS[task]
    if str(HERE) not in sys.path:
        sys.path.insert(0, str(HERE))

    try:
        module = importlib.import_module(module_name)
    except ImportError as e:
        raise SystemExit(f"task {task!r} -> module {module_name!r} could not be imported: {e}")
    if not hasattr(module, "main"):
        raise SystemExit(f"module {module_name!r} has no main()")

    print(f"[run_remote] task={task} -> {module_name}  ({desc})")
    # Rewrite argv so the target script's argparse sees a sensible prog name and
    # exactly the flags the user passed after the task token.
    sys.argv = [f"{Path(sys.argv[0]).name} {task}", *rest]
    try:
        result = module.main()
    except SystemExit as e:
        return e.code if isinstance(e.code, int) else (0 if e.code is None else 1)
    return result if isinstance(result, int) else 0


if __name__ == "__main__":
    sys.exit(main())
