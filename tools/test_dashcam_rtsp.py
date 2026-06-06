import argparse
import os
import time
from pathlib import Path


DEFAULT_URLS = [
    "rtsp://192.168.25.1:8080",
    "rtsp://192.168.25.1:8080/",
    "rtsp://192.168.25.1:8080/live",
    "rtsp://192.168.25.1:8080/live.sdp",
    "rtsp://192.168.25.1:8080/stream",
    "rtsp://192.168.25.1:8080/stream1",
    "rtsp://192.168.25.1:8080/stream2",
    "rtsp://192.168.25.1:8080/video",
    "rtsp://192.168.25.1:8080/video1",
    "rtsp://192.168.25.1:8080/video2",
    "rtsp://192.168.25.1:8080/media",
    "rtsp://192.168.25.1:8080/media/video1",
    "rtsp://192.168.25.1:8080/h264",
    "rtsp://192.168.25.1:8080/h264.sdp",
    "rtsp://192.168.25.1:8080/h265",
    "rtsp://192.168.25.1:8080/h265.sdp",
    "rtsp://192.168.25.1:8080/av0_0",
    "rtsp://192.168.25.1:8080/av0_1",
    "rtsp://192.168.25.1:8080/av1_0",
    "rtsp://192.168.25.1:8080/av1_1",
    "rtsp://192.168.25.1:8080/ch0_0.h264",
    "rtsp://192.168.25.1:8080/ch0_1.h264",
    "rtsp://192.168.25.1:8080/ch1_0.h264",
    "rtsp://192.168.25.1:8080/ch1_1.h264",
    "rtsp://192.168.25.1:8080/cam/realmonitor?channel=1&subtype=0",
    "rtsp://192.168.25.1:8080/cam/realmonitor?channel=1&subtype=1",
    "rtsp://192.168.25.1:8080/Streaming/Channels/101",
    "rtsp://192.168.25.1:8080/Streaming/Channels/102",
    "rtsp://192.168.25.1:8080/11",
    "rtsp://192.168.25.1:8080/12",
    "rtsp://192.168.25.1:8080/1",
    "rtsp://192.168.25.1:8080/user=admin_password=_channel=1_stream=0.sdp",
    "rtsp://192.168.25.1:8080/user=admin_password=_channel=1_stream=1.sdp",
    "rtsp://192.168.25.1:8080/user=admin&password=&channel=1&stream=0.sdp",
    "rtsp://192.168.25.1:8080/user=admin&password=admin&channel=1&stream=0.sdp",
    "rtsp://192.168.25.1:8081",
    "rtsp://192.168.25.1:8081/",
    "rtsp://192.168.25.1:8081/live",
    "rtsp://192.168.25.1:8081/live.sdp",
    "rtsp://192.168.25.1:8081/stream",
    "rtsp://192.168.25.1:8081/stream1",
    "rtsp://192.168.25.1:8081/stream2",
    "rtsp://192.168.25.1:8081/video",
    "rtsp://192.168.25.1:8081/h264",
    "rtsp://192.168.25.1:8081/av0_0",
    "rtsp://192.168.25.1:8081/ch0_0.h264",
]


def parse_args():
    parser = argparse.ArgumentParser(description="Try RTSP dashcam URLs and save a preview frame.")
    parser.add_argument(
        "--url",
        action="append",
        default=[],
        help="Specific RTSP URL to test. Can be passed multiple times.",
    )
    parser.add_argument(
        "--out",
        default="outputs/dashcam_probe",
        help="Folder for captured preview images.",
    )
    parser.add_argument(
        "--seconds",
        type=float,
        default=4.0,
        help="Seconds to wait for frames from each URL.",
    )
    parser.add_argument(
        "--transport",
        choices=["auto", "tcp", "udp", "udp_multicast", "http"],
        default="auto",
        help="RTSP transport to request through OpenCV/FFmpeg.",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    try:
        import cv2
    except ImportError:
        print("OpenCV is not installed.")
        print("Install it while you have Internet:")
        print("  python -m pip install opencv-python")
        return 2

    urls = args.url or DEFAULT_URLS
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    if args.transport != "auto":
        os.environ["OPENCV_FFMPEG_CAPTURE_OPTIONS"] = (
            f"rtsp_transport;{args.transport}|"
            "stimeout;4000000|"
            "max_delay;500000"
        )
        print(f"Using OpenCV FFmpeg options: {os.environ['OPENCV_FFMPEG_CAPTURE_OPTIONS']}")

    for index, url in enumerate(urls, start=1):
        print(f"[{index}/{len(urls)}] Testing {url}")
        cap = cv2.VideoCapture()
        try:
            cap.set(cv2.CAP_PROP_OPEN_TIMEOUT_MSEC, int(args.seconds * 1000))
            cap.set(cv2.CAP_PROP_READ_TIMEOUT_MSEC, int(args.seconds * 1000))
        except AttributeError:
            pass
        cap.open(url, cv2.CAP_FFMPEG)
        start = time.time()
        success = False
        frame = None
        while time.time() - start < args.seconds:
            ok, candidate = cap.read()
            if ok and candidate is not None and candidate.size > 0:
                success = True
                frame = candidate
                break
            time.sleep(0.1)
        cap.release()

        if success:
            image_path = out_dir / f"dashcam_rtsp_frame_{index}.jpg"
            cv2.imwrite(str(image_path), frame)
            print(f"SUCCESS: {url}")
            print(f"Saved preview frame: {image_path.resolve()}")
            return 0

        print("  no frame")

    print("No playable RTSP stream was found from the tested URLs.")
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
