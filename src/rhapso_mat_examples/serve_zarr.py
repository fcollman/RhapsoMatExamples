"""
Serve a Zarr store over HTTP with CORS and Range support, for Neuroglancer.

    serve-zarr fused.zarr --port 9102

Neuroglancer runs on a different origin than this server, so the browser needs
CORS headers; `python -m http.server` sends none. Range requests matter too:
an unsharded store is read with plain GETs, but a SHARDED Zarr v3 store reads
byte ranges inside each shard, and SimpleHTTPRequestHandler answers every
request with the whole file -- which quietly turns each chunk read into a
full-shard download, or breaks the read outright. This handler implements
single-range GET (206 Partial Content) so both layouts work.

Binds to 127.0.0.1 by default. Pass --bind 0.0.0.0 to expose it on the
network, which also serves the directory to anyone who can reach the port.
"""

from __future__ import annotations

import argparse
import functools
import http.server
import os
import re
import socket
from pathlib import Path

RANGE_RE = re.compile(r"^bytes=(\d*)-(\d*)$")


class CORSRangeHandler(http.server.SimpleHTTPRequestHandler):
    """Static handler with CORS headers and single-range GET support."""

    def end_headers(self):
        self.send_header("Access-Control-Allow-Origin", "*")
        self.send_header("Access-Control-Allow-Headers", "*")
        self.send_header("Access-Control-Expose-Headers", "*")
        self.send_header("Accept-Ranges", "bytes")
        self.send_header("Cache-Control", "no-store")
        super().end_headers()

    def do_OPTIONS(self):
        self.send_response(204)
        self.end_headers()

    def send_head(self):
        """Serve a byte range when asked, otherwise defer to the base class."""
        header = self.headers.get("Range")
        if not header:
            return super().send_head()

        match = RANGE_RE.match(header.strip())
        if not match:
            return super().send_head()

        path = self.translate_path(self.path)
        if os.path.isdir(path):
            return super().send_head()

        try:
            size = os.path.getsize(path)
            handle = open(path, "rb")
        except OSError:
            self.send_error(404, "File not found")
            return None

        start_text, end_text = match.groups()
        if start_text:
            start = int(start_text)
            end = int(end_text) if end_text else size - 1
        else:
            # "bytes=-N" means the final N bytes.
            if not end_text:
                handle.close()
                self.send_error(400, "Invalid Range header")
                return None
            start = max(0, size - int(end_text))
            end = size - 1

        end = min(end, size - 1)

        if start > end or start >= size:
            handle.close()
            self.send_response(416)
            self.send_header("Content-Range", f"bytes */{size}")
            self.send_header("Content-Length", "0")
            self.end_headers()
            return None

        handle.seek(start)
        self._range_remaining = end - start + 1

        self.send_response(206)
        self.send_header("Content-Type", self.guess_type(path))
        self.send_header("Content-Range", f"bytes {start}-{end}/{size}")
        self.send_header("Content-Length", str(self._range_remaining))
        self.end_headers()
        return _LimitedReader(handle, self._range_remaining)

    def log_message(self, fmt, *args):
        """Stay quiet for 2xx/3xx; report anything that failed."""
        status = str(args[1]) if len(args) > 1 else ""
        if not status.startswith(("2", "3")):
            super().log_message(fmt, *args)


class _LimitedReader:
    """File wrapper that stops after N bytes, for copyfile()."""

    def __init__(self, handle, remaining: int):
        self._handle = handle
        self._remaining = remaining

    def read(self, size=-1):
        if self._remaining <= 0:
            return b""
        if size is None or size < 0:
            size = self._remaining
        data = self._handle.read(min(size, self._remaining))
        self._remaining -= len(data)
        return data

    def close(self):
        self._handle.close()


def free_port(preferred: int, bind: str) -> int:
    """Return `preferred` if it is free, otherwise the next free port."""
    for port in range(preferred, preferred + 50):
        with socket.socket() as probe:
            try:
                probe.bind((bind, port))
                return port
            except OSError:
                continue
    raise SystemExit(f"no free port in {preferred}..{preferred + 49}")


def main(argv=None) -> None:
    parser = argparse.ArgumentParser(
        description="Serve a Zarr store with CORS and Range support."
    )
    parser.add_argument("root", help="Directory to serve (the store's parent).")
    parser.add_argument("--port", type=int, default=9102,
                        help="Port; the next free one is used if taken.")
    parser.add_argument("--bind", default="127.0.0.1",
                        help="Address to bind. 0.0.0.0 exposes it on the LAN.")
    args = parser.parse_args(argv)

    root = Path(args.root).expanduser().resolve()
    if not root.is_dir():
        raise SystemExit(f"{root} is not a directory")

    port = free_port(args.port, args.bind)
    print(f"Serving {root}")
    print(f"  http://{args.bind}:{port}/")
    for entry in sorted(root.iterdir()):
        if entry.name.endswith(".zarr"):
            print(f"  zarr://http://{args.bind}:{port}/{entry.name}")
    print("Ctrl-C to stop.")

    http.server.ThreadingHTTPServer(
        (args.bind, port),
        functools.partial(CORSRangeHandler, directory=str(root)),
    ).serve_forever()


if __name__ == "__main__":
    main()
