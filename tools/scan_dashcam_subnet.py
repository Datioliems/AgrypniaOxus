import argparse
import ipaddress
import socket
from concurrent.futures import ThreadPoolExecutor, as_completed


DEFAULT_PORTS = [80, 81, 88, 554, 1935, 5000, 8000, 8080, 8081, 8554, 8888, 8899, 9000]


def parse_args():
    parser = argparse.ArgumentParser(description="Scan a local subnet for possible dashcam services.")
    parser.add_argument(
        "--subnet",
        required=True,
        help="Subnet to scan, for example: 192.168.25.0/24",
    )
    parser.add_argument(
        "--ports",
        default=",".join(str(port) for port in DEFAULT_PORTS),
        help="Comma-separated TCP ports to check.",
    )
    parser.add_argument(
        "--timeout",
        type=float,
        default=0.6,
        help="Connection timeout in seconds.",
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=96,
        help="Number of concurrent checks.",
    )
    return parser.parse_args()


def is_open(ip, port, timeout):
    try:
        with socket.create_connection((str(ip), port), timeout=timeout):
            return True
    except OSError:
        return False


def main():
    args = parse_args()
    network = ipaddress.ip_network(args.subnet, strict=False)
    ports = [int(port.strip()) for port in args.ports.split(",") if port.strip()]
    tasks = {}
    hits = []
    hosts = list(network.hosts())

    print(f"Scanning {network} ports {ports}")
    with ThreadPoolExecutor(max_workers=args.workers) as executor:
        for ip in hosts:
            for port in ports:
                future = executor.submit(is_open, ip, port, args.timeout)
                tasks[future] = (ip, port)

        for future in as_completed(tasks):
            if future.result():
                ip, port = tasks[future]
                hits.append((str(ip), port))

    if not hits:
        print("No open common dashcam ports were found.")
        return 1

    print("Possible services:")
    for ip, port in sorted(hits):
        scheme = "rtsp" if port in {554, 8554} else "http"
        print(f"  - {scheme}://{ip}:{port}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
