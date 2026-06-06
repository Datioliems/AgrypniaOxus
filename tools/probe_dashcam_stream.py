import argparse
from http.client import BadStatusLine, IncompleteRead
import socket
import subprocess
import sys
from pathlib import Path
from urllib.error import URLError
from urllib.request import Request, urlopen


COMMON_PORTS = [80, 81, 88, 554, 1935, 5000, 8000, 8080, 8081, 8554, 8888, 8899, 9000]

HTTP_PATHS = [
    "/",
    "/video",
    "/video.mjpg",
    "/videostream.cgi",
    "/mjpeg",
    "/mjpg/video.mjpg",
    "/stream",
    "/live",
    "/?action=stream",
]

RTSP_PATHS = [
    "",
    "/",
    "/live",
    "/live.sdp",
    "/stream1",
    "/stream2",
    "/stream",
    "/video",
    "/video1",
    "/video2",
    "/media",
    "/media/video1",
    "/h264",
    "/h264.sdp",
    "/h265",
    "/h265.sdp",
    "/av0_0",
    "/av0_1",
    "/av1_0",
    "/av1_1",
    "/ch0_0.h264",
    "/ch0_1.h264",
    "/ch1_0.h264",
    "/ch1_1.h264",
    "/cam/realmonitor?channel=1&subtype=0",
    "/cam/realmonitor?channel=1&subtype=1",
    "/Streaming/Channels/101",
    "/Streaming/Channels/102",
    "/11",
    "/12",
    "/1",
    "/user=admin_password=_channel=1_stream=0.sdp",
    "/user=admin_password=_channel=1_stream=1.sdp",
    "/user=admin&password=&channel=1&stream=0.sdp",
    "/user=admin&password=admin&channel=1&stream=0.sdp",
]


def parse_args():
    parser = argparse.ArgumentParser(
        description="Probe common HTTP/RTSP dashcam stream endpoints."
    )
    parser.add_argument(
        "--ip",
        required=True,
        help="Dashcam IP address, usually the Default Gateway after connecting to the camera Wi-Fi.",
    )
    parser.add_argument(
        "--timeout",
        type=float,
        default=2.0,
        help="Connection timeout in seconds.",
    )
    return parser.parse_args()


def is_port_open(ip, port, timeout):
    try:
        with socket.create_connection((ip, port), timeout=timeout):
            return True
    except OSError:
        return False


def check_http(url, timeout):
    try:
        request = Request(url, headers={"User-Agent": "DrowsyDriverAndroid-Probe/1.0"})
        with urlopen(request, timeout=timeout) as response:
            content_type = response.headers.get("Content-Type", "")
            status = getattr(response, "status", 200)
            return status, content_type
    except BadStatusLine as exc:
        return "NON_HTTP", str(exc).strip()
    except IncompleteRead as exc:
        return "INCOMPLETE", str(exc).strip()
    except URLError:
        return None
    except OSError:
        return None


def check_ffprobe_available():
    try:
        completed = subprocess.run(
            ["ffprobe", "-version"],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            timeout=3,
        )
        return completed.returncode == 0
    except (OSError, subprocess.SubprocessError):
        return False


def check_rtsp(url, timeout):
    try:
        completed = subprocess.run(
            [
                "ffprobe",
                "-v",
                "error",
                "-rtsp_transport",
                "tcp",
                "-stimeout",
                str(int(timeout * 1_000_000)),
                "-i",
                url,
            ],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            timeout=max(5, int(timeout) + 3),
        )
        return completed.returncode == 0
    except (OSError, subprocess.SubprocessError):
        return False


def check_rtsp_options(ip, port, path, timeout):
    url = f"rtsp://{ip}:{port}{path}"
    request = (
        f"OPTIONS {url} RTSP/1.0\r\n"
        "CSeq: 1\r\n"
        "User-Agent: DrowsyDriverAndroid-Probe/1.0\r\n"
        "\r\n"
    ).encode("ascii", errors="ignore")
    try:
        with socket.create_connection((ip, port), timeout=timeout) as sock:
            sock.settimeout(timeout)
            sock.sendall(request)
            response = sock.recv(512).decode("latin-1", errors="ignore").strip()
            if response:
                return response.splitlines()[0]
    except OSError:
        return None
    return None


def main():
    args = parse_args()
    ip = args.ip.strip()
    timeout = args.timeout

    print(f"Probing dashcam at {ip}")
    print("Open ports:")
    open_ports = []
    for port in COMMON_PORTS:
        if is_port_open(ip, port, timeout):
            open_ports.append(port)
            print(f"  - {port}")

    if not open_ports:
        print("  None of the common ports are open or reachable.")
        return 1

    print("\nHTTP candidates:")
    http_hits = []
    for port in [port for port in open_ports if port != 554 and port != 8554]:
        for path in HTTP_PATHS:
            url = f"http://{ip}:{port}{path}"
            result = check_http(url, timeout)
            if result:
                status, content_type = result
                http_hits.append(url)
                print(f"  - {url}  status={status}  content-type={content_type}")

    print("\nRTSP candidates:")
    ffprobe_available = check_ffprobe_available()
    if not ffprobe_available:
        print("  ffprobe was not found, so RTSP URLs are listed but not validated.")
        print("  Install FFmpeg or paste these URLs into VLC: Media > Open Network Stream.")

    rtsp_hits = []
    for port in open_ports:
        for path in RTSP_PATHS:
            url = f"rtsp://{ip}:{port}{path}"
            options_response = check_rtsp_options(ip, port, path, timeout)
            if options_response and "RTSP" in options_response.upper():
                print(f"  - {url}  OPTIONS={options_response}")
            if ffprobe_available:
                if check_rtsp(url, timeout):
                    rtsp_hits.append(url)
                    print(f"  - {url}  OK")
            elif port in {554, 8554, 8080, 8081}:
                print(f"  - {url}")

    output_dir = Path("outputs/dashcam_probe")
    output_dir.mkdir(parents=True, exist_ok=True)
    report = output_dir / "dashcam_probe_urls.txt"
    report.write_text(
        "\n".join(
            [
                f"Dashcam IP: {ip}",
                "",
                "Open ports:",
                *[str(port) for port in open_ports],
                "",
                "HTTP hits:",
                *http_hits,
                "",
                "RTSP hits or candidates:",
                *(rtsp_hits if rtsp_hits else []),
            ]
        ),
        encoding="utf-8",
    )
    print(f"\nSaved report: {report.resolve()}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
