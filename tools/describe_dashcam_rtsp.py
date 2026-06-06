import argparse
import socket
from pathlib import Path


DEFAULT_URLS = [
    "rtsp://192.168.25.1:8080",
    "rtsp://192.168.25.1:8080/",
    "rtsp://192.168.25.1:8080/live",
    "rtsp://192.168.25.1:8080/live.sdp",
    "rtsp://192.168.25.1:8080/stream",
    "rtsp://192.168.25.1:8080/stream1",
    "rtsp://192.168.25.1:8080/stream2",
    "rtsp://192.168.25.1:8080/h264",
    "rtsp://192.168.25.1:8080/h264.sdp",
    "rtsp://192.168.25.1:8080/av0_0",
    "rtsp://192.168.25.1:8080/av0_1",
    "rtsp://192.168.25.1:8080/ch0_0.h264",
]


def parse_rtsp_url(url):
    if not url.startswith("rtsp://"):
        raise ValueError("URL must start with rtsp://")
    rest = url[len("rtsp://") :]
    host_port, _, path = rest.partition("/")
    host, _, port = host_port.partition(":")
    return host, int(port or "554"), "/" + path if path else ""


def describe(url, timeout):
    host, port, _ = parse_rtsp_url(url)
    request = (
        f"DESCRIBE {url} RTSP/1.0\r\n"
        "CSeq: 1\r\n"
        "Accept: application/sdp\r\n"
        "User-Agent: DrowsyDriverAndroid-Probe/1.0\r\n"
        "\r\n"
    ).encode("ascii", errors="ignore")
    try:
        with socket.create_connection((host, port), timeout=timeout) as sock:
            sock.settimeout(timeout)
            sock.sendall(request)
            chunks = []
            while True:
                chunk = sock.recv(4096)
                if not chunk:
                    break
                chunks.append(chunk)
                if len(b"".join(chunks)) > 65536:
                    break
            return b"".join(chunks).decode("latin-1", errors="ignore")
    except OSError as exc:
        return f"ERROR: {exc}"


def main():
    parser = argparse.ArgumentParser(description="Send RTSP DESCRIBE to candidate dashcam URLs.")
    parser.add_argument("--url", action="append", default=[], help="RTSP URL to test.")
    parser.add_argument("--timeout", type=float, default=3.0)
    parser.add_argument("--out", default="outputs/dashcam_probe/rtsp_describe.txt")
    args = parser.parse_args()

    urls = args.url or DEFAULT_URLS
    lines = []
    for index, url in enumerate(urls, start=1):
        print(f"[{index}/{len(urls)}] DESCRIBE {url}")
        response = describe(url, args.timeout)
        first_line = response.splitlines()[0] if response.splitlines() else "(empty)"
        print(f"  {first_line}")
        if "m=video" in response or "a=control" in response:
            print("  SDP video/control info found")
        lines.extend([f"===== {url} =====", response, ""])

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text("\n".join(lines), encoding="utf-8", errors="ignore")
    print(f"Saved: {out_path.resolve()}")


if __name__ == "__main__":
    main()
