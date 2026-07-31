#!/usr/bin/env bash
# Start ComfyUI listening on all interfaces so LAN clients can call the API.
# Linux counterpart of start_comfyui_remote.bat (Windows / portable python).
#
# Usage (on the ComfyUI host):
#   server/start_comfyui_remote.sh
#   COMFY_PORT=8189 server/start_comfyui_remote.sh --reserve-vram 2
#
# Clients should use:  http://<this-machine-LAN-IP>:8188
# Example: python run_ltx23_i2v_remote.py --server 192.168.1.50:8188 --image f.png
#
# Everything is overridable by environment variable:
#   COMFYUI_DIR   ComfyUI checkout (default: first of <repo>/../ComfyUI or the
#                 llm-scaler portable layout used by the .bat)
#   COMFY_PORT    listen port (default 8188)
#   CONDA_ENV     conda env to activate if present (default: comfyui)
#   PYTHON        interpreter to use when no conda env is activated
#   COMFY_FLAGS   flags passed to main.py (default: --disable-smart-memory,
#                 same as the .bat; e.g. "--reserve-vram 2" on a 32 GB card)
# Extra ComfyUI flags can also be appended on the command line.
#
# NOTE: no `set -u`. Some torch-xpu envs ship a conda activate.d hook (oneCCL /
# MPI setvars) that reads unset variables, and `set -u` aborts activation there.
set -eo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PORT="${COMFY_PORT:-8188}"
CONDA_ENV="${CONDA_ENV:-comfyui}"
PYTHON="${PYTHON:-python}"

# ---- locate ComfyUI ---------------------------------------------------------
if [ -z "${COMFYUI_DIR:-}" ]; then
    for candidate in \
        "$REPO_ROOT/../ComfyUI" \
        "$REPO_ROOT/../llm-scaler/omni/comfyui_windows_setup/ComfyUI" \
        "$REPO_ROOT/llm-scaler/omni/comfyui_windows_setup/ComfyUI"; do
        if [ -f "$candidate/main.py" ]; then
            COMFYUI_DIR="$(cd "$candidate" && pwd)"
            break
        fi
    done
fi
if [ ! -f "${COMFYUI_DIR:-}/main.py" ]; then
    echo "ERROR: ComfyUI main.py not found." >&2
    echo "Set COMFYUI_DIR to your checkout, e.g.:" >&2
    echo "  COMFYUI_DIR=/path/to/ComfyUI server/start_comfyui_remote.sh" >&2
    exit 1
fi

# ---- conda env (optional) ---------------------------------------------------
for conda_sh in \
    "${CONDA_PREFIX:-}/etc/profile.d/conda.sh" \
    "$HOME/miniforge3/etc/profile.d/conda.sh" \
    "$HOME/miniconda3/etc/profile.d/conda.sh" \
    "$HOME/anaconda3/etc/profile.d/conda.sh"; do
    if [ -f "$conda_sh" ]; then
        # shellcheck disable=SC1090
        source "$conda_sh"
        if conda env list | awk '{print $1}' | grep -qx "$CONDA_ENV"; then
            conda activate "$CONDA_ENV"
            PYTHON=python
        fi
        break
    fi
done

# Local requests must not go through a corporate proxy.
export no_proxy="localhost,127.0.0.1,::1${no_proxy:+,$no_proxy}"

echo "============================================================"
echo " ComfyUI REMOTE / LAN mode"
echo " Listen: 0.0.0.0:${PORT}  (all interfaces)"
echo " ComfyUI: ${COMFYUI_DIR}"
echo " Python:  $(command -v "$PYTHON")"
echo " Local GUI:  http://127.0.0.1:${PORT}"
echo "============================================================"
echo
echo "LAN addresses on this machine (pick one for --server):"
hostname -I 2>/dev/null | tr ' ' '\n' | grep -E '^[0-9]' | sed "s|\$|:${PORT}|;s|^|  |"
echo
echo "If LAN clients cannot connect, allow inbound TCP ${PORT}:"
echo "  sudo ufw allow ${PORT}/tcp                        # ufw"
echo "  sudo firewall-cmd --add-port=${PORT}/tcp --permanent && sudo firewall-cmd --reload"
echo
echo "Press Ctrl+C to stop the server."
echo "============================================================"

cd "$COMFYUI_DIR"
# shellcheck disable=SC2086  # COMFY_FLAGS is intentionally word-split
exec "$PYTHON" main.py --listen 0.0.0.0 --port "$PORT" \
    ${COMFY_FLAGS---disable-smart-memory} "$@"
