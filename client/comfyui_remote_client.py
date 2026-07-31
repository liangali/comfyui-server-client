"""Shared ComfyUI LAN client: upload inputs, queue prompts, download outputs.

Used by run_*_remote.py scripts on a client machine that talks to a B70
ComfyUI server listening on 0.0.0.0:8188.
"""
from __future__ import annotations

import argparse
import json
import mimetypes
import os
import time
import urllib.error
import urllib.parse
import urllib.request
import uuid
from pathlib import Path
from typing import Any

DEFAULT_PORT = 8188
DEFAULT_OUT_DIR = Path(__file__).resolve().parent / "remote_outputs"

# LAN ComfyUI must be reached directly. Corporate HTTP_PROXY (e.g. proxy-ir.intel.com)
# often returns a 403 HTML policy page for private IPs like 10.x, which urllib would
# otherwise honor via getproxies() / env. Empty ProxyHandler disables that.
_NO_PROXY_OPENER = urllib.request.build_opener(urllib.request.ProxyHandler({}))


def _proxy_hint(base: str) -> str:
    proxies = urllib.request.getproxies()
    http_proxy = proxies.get("http") or proxies.get("HTTP") or ""
    if not http_proxy:
        return ""
    return (
        f" Detected HTTP proxy {http_proxy}; these scripts bypass it for LAN. "
        f'For curl use: curl --noproxy "*" {base}/system_stats'
    )


class ComfyRemoteClient:
    """HTTP client for a remote ComfyUI server (scheme A / LAN)."""

    def __init__(self, server: str, client_id: str | None = None):
        server = (server or "").strip()
        if not server:
            raise ValueError("server address is empty")
        # Allow http://host:port or host:port
        if server.startswith("http://"):
            server = server[len("http://") :]
        elif server.startswith("https://"):
            raise ValueError("HTTPS is not supported by these scripts; use http://host:port")
        server = server.rstrip("/")
        if ":" not in server:
            server = f"{server}:{DEFAULT_PORT}"
        self.server = server
        self.client_id = client_id or str(uuid.uuid4())
        self.base = f"http://{self.server}"
        self._opener = _NO_PROXY_OPENER

    # ------------------------------------------------------------------ HTTP
    def _open(self, req: urllib.request.Request | str, timeout: float):
        return self._opener.open(req, timeout=timeout)

    @staticmethod
    def _parse_json_body(raw: bytes, *, method: str, path: str) -> dict:
        if not raw:
            return {}
        text = raw.decode("utf-8", errors="replace")
        try:
            return json.loads(text)
        except json.JSONDecodeError as e:
            snippet = text[:300].replace("\n", " ")
            raise RuntimeError(
                f"Expected JSON from {method} {path}, got non-JSON "
                f"({len(raw)} bytes, starts with: {snippet!r}). "
                "Often caused by an HTTP proxy returning an HTML block page; "
                "this client bypasses proxies — check firewall / --server IP."
            ) from e

    def http_json(self, method: str, path: str, data: Any = None, timeout: float = 3600) -> dict:
        url = f"{self.base}{path}"
        body = None if data is None else json.dumps(data).encode("utf-8")
        req = urllib.request.Request(url, data=body, method=method)
        if body is not None:
            req.add_header("Content-Type", "application/json")
        try:
            with self._open(req, timeout=timeout) as resp:
                return self._parse_json_body(resp.read(), method=method, path=path)
        except urllib.error.HTTPError as e:
            detail = e.read().decode("utf-8", errors="replace")[:2000]
            raise RuntimeError(f"HTTP {e.code} {method} {path}: {detail}") from e
        except urllib.error.URLError as e:
            raise RuntimeError(
                f"Cannot reach ComfyUI at {self.base}{path}: {e.reason}. "
                f"Check --server, firewall, and that start_comfyui_remote.bat is running."
            ) from e

    def http_bytes(self, path: str, timeout: float = 3600) -> bytes:
        url = f"{self.base}{path}"
        try:
            with self._open(url, timeout=timeout) as resp:
                return resp.read()
        except urllib.error.HTTPError as e:
            detail = e.read().decode("utf-8", errors="replace")[:1000]
            raise RuntimeError(f"HTTP {e.code} GET {path}: {detail}") from e
        except urllib.error.URLError as e:
            raise RuntimeError(f"Cannot reach ComfyUI at {url}: {e.reason}") from e

    def wait_ready(self, timeout: float = 120) -> bool:
        t0 = time.time()
        last_err: Exception | None = None
        while time.time() - t0 < timeout:
            try:
                self.http_json("GET", "/system_stats", timeout=5)
                return True
            except Exception as e:
                last_err = e
                time.sleep(2)
        if last_err is not None:
            self._last_ready_error = last_err
        return False

    def system_stats(self) -> dict:
        return self.http_json("GET", "/system_stats", timeout=30)

    # ----------------------------------------------------------- upload/download
    def upload_file(
        self,
        local_path: str | Path,
        *,
        overwrite: bool = True,
        remote_name: str | None = None,
        file_type: str = "input",
        subfolder: str = "",
    ) -> dict:
        """Upload a local image/video to ComfyUI input (or temp) via POST /upload/image.

        ComfyUI uses the form field name ``image`` for both images and videos.
        Returns ``{"name", "subfolder", "type"}`` — use ``name`` in LoadImage / LoadVideo.
        """
        path = Path(local_path)
        if not path.is_file():
            raise FileNotFoundError(f"upload source not found: {path}")

        filename = remote_name or path.name
        # Avoid path separators in remote name
        filename = Path(filename).name
        mime = mimetypes.guess_type(filename)[0] or "application/octet-stream"
        data = path.read_bytes()
        boundary = f"----ComfyRemote{uuid.uuid4().hex}"

        def part(name: str, value: bytes, filename_field: str | None = None, content_type: str | None = None) -> bytes:
            hdr = [f"--{boundary}", f'Content-Disposition: form-data; name="{name}"']
            if filename_field is not None:
                hdr[1] += f'; filename="{filename_field}"'
            if content_type:
                hdr.append(f"Content-Type: {content_type}")
            hdr.append("")
            return ("\r\n".join(hdr) + "\r\n").encode("utf-8") + value + b"\r\n"

        body = b"".join(
            [
                part("image", data, filename_field=filename, content_type=mime),
                part("overwrite", b"true" if overwrite else b"false"),
                part("type", file_type.encode("utf-8")),
                part("subfolder", subfolder.encode("utf-8")),
                f"--{boundary}--\r\n".encode("utf-8"),
            ]
        )
        req = urllib.request.Request(
            f"{self.base}/upload/image",
            data=body,
            method="POST",
            headers={"Content-Type": f"multipart/form-data; boundary={boundary}"},
        )
        try:
            with self._open(req, timeout=600) as resp:
                result = json.loads(resp.read().decode("utf-8"))
        except urllib.error.HTTPError as e:
            detail = e.read().decode("utf-8", errors="replace")[:2000]
            raise RuntimeError(f"upload failed HTTP {e.code}: {detail}") from e
        except urllib.error.URLError as e:
            raise RuntimeError(f"upload failed: {e.reason}") from e

        print(
            f"uploaded {path} ({path.stat().st_size} bytes) -> "
            f"server name={result.get('name')} type={result.get('type')} "
            f"subfolder={result.get('subfolder')!r}"
        )
        return result

    def download_file(
        self,
        filename: str,
        dest: str | Path,
        *,
        subfolder: str = "",
        folder_type: str = "output",
    ) -> Path:
        """Download one file via GET /view and write it to ``dest``."""
        dest = Path(dest)
        dest.parent.mkdir(parents=True, exist_ok=True)
        qs = urllib.parse.urlencode(
            {"filename": filename, "subfolder": subfolder or "", "type": folder_type}
        )
        data = self.http_bytes(f"/view?{qs}", timeout=3600)
        dest.write_bytes(data)
        print(f"downloaded {filename} ({len(data)} bytes) -> {dest}")
        return dest

    def collect_output_refs(self, history_entry: dict) -> list[dict]:
        """Collect image/video file refs from a history entry's outputs."""
        refs: list[dict] = []
        for node_id, node_out in (history_entry.get("outputs") or {}).items():
            for key in ("images", "gifs", "videos"):
                for item in node_out.get(key) or []:
                    refs.append(
                        {
                            "node_id": node_id,
                            "key": key,
                            "filename": item["filename"],
                            "subfolder": item.get("subfolder") or "",
                            "type": item.get("type") or "output",
                        }
                    )
        return refs

    def download_history_outputs(
        self,
        history_entry: dict,
        out_dir: str | Path,
        *,
        name_map: dict[str, str] | None = None,
    ) -> list[Path]:
        """Download all image/video outputs from a completed prompt into ``out_dir``.

        ``name_map`` optionally remaps remote filename -> local filename.
        """
        out_dir = Path(out_dir)
        out_dir.mkdir(parents=True, exist_ok=True)
        saved: list[Path] = []
        for ref in self.collect_output_refs(history_entry):
            local_name = (name_map or {}).get(ref["filename"], ref["filename"])
            dest = out_dir / local_name
            # Avoid clobber if same basename from different subfolders
            if dest.exists():
                stem, suf = dest.stem, dest.suffix
                dest = out_dir / f"{stem}_{uuid.uuid4().hex[:6]}{suf}"
            self.download_file(
                ref["filename"],
                dest,
                subfolder=ref["subfolder"],
                folder_type=ref["type"],
            )
            saved.append(dest)
        return saved

    # ---------------------------------------------------------------- queue
    def queue_prompt(self, prompt: dict) -> dict:
        result = self.http_json(
            "POST",
            "/prompt",
            {"prompt": prompt, "client_id": self.client_id},
            timeout=120,
        )
        if result.get("error") or result.get("node_errors"):
            raise RuntimeError(f"prompt submit failed: {json.dumps(result)[:2000]}")
        return result

    def get_history(self, prompt_id: str) -> dict:
        return self.http_json("GET", f"/history/{prompt_id}", timeout=120)

    def free_memory(self, unload_models: bool = False) -> dict:
        return self.http_json(
            "POST",
            "/free",
            {"unload_models": unload_models, "free_memory": True},
            timeout=120,
        )

    def queue_and_wait(
        self,
        prompt: dict,
        *,
        tag: str = "job",
        timeout: float = 7200,
        poll_interval: float = 3.0,
    ) -> tuple[dict, float, str]:
        """Submit prompt and poll /history until outputs appear.

        Returns ``(history_entry, wall_sec, prompt_id)``.
        """
        t0 = time.time()
        result = self.queue_prompt(prompt)
        prompt_id = result["prompt_id"]
        print(f"[{tag}] prompt_id={prompt_id}")

        while True:
            try:
                hist = self.get_history(prompt_id)
            except Exception as e:
                print(f"[{tag}] history poll error: {e}")
                time.sleep(poll_interval)
                continue

            if prompt_id in hist:
                entry = hist[prompt_id]
                status = entry.get("status") or {}
                if entry.get("outputs"):
                    elapsed = time.time() - t0
                    print(f"[{tag}] DONE in {elapsed:.1f}s")
                    return entry, elapsed, prompt_id
                if status.get("completed") and status.get("status_str") == "error":
                    raise RuntimeError(f"[{tag}] execution error: {json.dumps(status)[:2000]}")

            if time.time() - t0 > timeout:
                raise TimeoutError(f"[{tag}] timed out after {timeout}s (prompt_id={prompt_id})")
            if int(time.time() - t0) % 30 < poll_interval + 0.5:
                print(f"[{tag}] ... running {time.time() - t0:.0f}s")
            time.sleep(poll_interval)


def resolve_server(cli_server: str | None = None) -> str:
    """Resolve server from CLI > COMFYUI_SERVER env > default localhost."""
    if cli_server and cli_server.strip():
        return cli_server.strip()
    env = os.environ.get("COMFYUI_SERVER", "").strip()
    if env:
        return env
    return f"127.0.0.1:{DEFAULT_PORT}"


def add_server_args(ap: argparse.ArgumentParser) -> None:
    ap.add_argument(
        "--server",
        type=str,
        default=None,
        help=(
            "ComfyUI host:port on the B70 machine, e.g. 192.168.1.50:8188. "
            "Overrides env COMFYUI_SERVER. Default: 127.0.0.1:8188"
        ),
    )
    ap.add_argument(
        "--out-dir",
        type=str,
        default=str(DEFAULT_OUT_DIR),
        help=f"Local directory to save downloaded outputs (default: {DEFAULT_OUT_DIR})",
    )
    ap.add_argument(
        "--ready-timeout",
        type=float,
        default=60,
        help="Seconds to wait for /system_stats before giving up (default: 60)",
    )


def connect_from_args(args: argparse.Namespace) -> ComfyRemoteClient:
    server = resolve_server(getattr(args, "server", None))
    client = ComfyRemoteClient(server)
    print(f"ComfyUI server: {client.base}")
    if not client.wait_ready(timeout=getattr(args, "ready_timeout", 60)):
        detail = getattr(client, "_last_ready_error", None)
        extra = f" Last error: {detail}" if detail else ""
        proxy_note = _proxy_hint(client.base)
        raise SystemExit(
            f"ComfyUI not reachable at {client.base}. "
            "On the B70 machine run start_comfyui_remote.bat and allow TCP 8188 in firewall."
            f"{proxy_note}{extra}"
        )
    try:
        stats = client.system_stats()
        devices = stats.get("devices") or []
        if devices:
            d0 = devices[0]
            name = d0.get("name") or d0.get("index")
            vram = d0.get("vram_total")
            vram_gb = f"{vram / 1024**3:.1f} GB" if isinstance(vram, (int, float)) else "?"
            print(f"remote device: {name}, vram_total≈{vram_gb}")
    except Exception as e:
        print(f"warning: system_stats parse failed: {e}")
    return client


def unique_remote_name(local_path: str | Path, prefix: str = "remote") -> str:
    path = Path(local_path)
    rid = uuid.uuid4().hex[:8]
    return f"{prefix}_{rid}{path.suffix.lower() or '.bin'}"
