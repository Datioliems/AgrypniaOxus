from __future__ import annotations

import argparse
from http.server import ThreadingHTTPServer, SimpleHTTPRequestHandler
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


class QuietHandler(SimpleHTTPRequestHandler):
    def log_message(self, format: str, *args) -> None:
        print(f"{self.address_string()} - {format % args}")


def parse_args():
    parser = argparse.ArgumentParser(description="Serve the iPhone PWA demo on the local network.")
    parser.add_argument("--host", default="0.0.0.0")
    parser.add_argument("--port", default=8502, type=int)
    return parser.parse_args()


def main():
    args = parse_args()
    handler = lambda *handler_args, **handler_kwargs: QuietHandler(
        *handler_args,
        directory=str(ROOT),
        **handler_kwargs,
    )
    server = ThreadingHTTPServer((args.host, args.port), handler)
    print(f"Serving iPhone PWA from: {ROOT}")
    print(f"Open on laptop: http://127.0.0.1:{args.port}/web/iphone-pwa/")
    print("Open on iPhone: http://YOUR_LAPTOP_IP:{}/web/iphone-pwa/".format(args.port))
    print("Press Ctrl+C to stop.")
    server.serve_forever()


if __name__ == "__main__":
    main()
