#!/usr/bin/env python3
"""Script to run all services locally for development/demo."""
from __future__ import annotations

import argparse
import os
import signal
import subprocess
import sys
import time
from typing import List, Optional

SERVICES = {
    "mock_apis": {
        "module": "mock_apis.internal_apis.app:app",
        "port": int(os.getenv("MOCK_APIS_PORT", "8001")),
        "env_var": "TOOL_API_BASE",
        "health_path": "/health",
    },
    "safety_check": {
        "module": "services.guardrails.safety_check.app:app",
        "port": 8010,
        "env_var": "SAFETY_CHECK_URL",
        "health_path": "/healthz",
    },
    "relevance_check": {
        "module": "services.guardrails.relevance_check.app:app",
        "port": 8011,
        "env_var": "RELEVANCE_CHECK_URL",
        "health_path": "/healthz",
    },
    "agent": {
        "module": "services.agent.app:app",
        "port": 8012,
        "env_var": "AGENT_URL",
        "health_path": "/healthz",
    },
    "orchestrator": {
        "module": "services.orchestrator.app:app",
        "port": 8000,
        "env_var": None,
        "health_path": "/healthz",
    },
}

OPTIONAL_SERVICES = {"mock_apis"}


def start_service(
    name: str,
    config: dict,
    host: str = "127.0.0.1",
    reload: bool = True,
) -> subprocess.Popen:
    """Start a single service."""
    port = config["port"]
    module = config["module"]

    env = os.environ.copy()
    env["TOOL_API_BASE"] = f"http://{host}:{SERVICES['mock_apis']['port']}"
    env["SAFETY_CHECK_URL"] = f"http://{host}:8010"
    env["RELEVANCE_CHECK_URL"] = f"http://{host}:8011"
    env["AGENT_URL"] = f"http://{host}:8012"

    cmd = [
        sys.executable,
        "-m",
        "uvicorn",
        module,
        "--host",
        host,
        "--port",
        str(port),
    ]
    if reload:
        cmd.append("--reload")

    print(f"Starting {name} on {host}:{port}...")
    return subprocess.Popen(cmd, env=env, start_new_session=True)


def is_port_open(host: str, port: int, timeout: float = 0.25) -> bool:
    """Return True if something is already listening on host:port."""
    import socket

    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.settimeout(timeout)
        return sock.connect_ex((host, port)) == 0


def wait_for_service(
    host: str,
    port: int,
    health_path: str,
    timeout: float = 30.0,
) -> bool:
    """Wait for a service to become available."""
    import httpx

    candidate_paths = [health_path]
    if health_path != "/healthz":
        candidate_paths.append("/healthz")
    if health_path != "/health":
        candidate_paths.append("/health")

    start = time.time()

    while time.time() - start < timeout:
        try:
            for path in candidate_paths:
                url = f"http://{host}:{port}{path}"
                response = httpx.get(url, timeout=1.0)
                if response.status_code == 200:
                    return True
        except Exception:
            pass
        time.sleep(0.5)

    return False


def main() -> int:
    parser = argparse.ArgumentParser(description="Run all services locally")
    parser.add_argument("--host", default="127.0.0.1", help="Host to bind to")
    parser.add_argument("--no-reload", action="store_true", help="Disable auto-reload")
    parser.add_argument("--services", nargs="+", choices=list(SERVICES.keys()),
                        help="Specific services to run (default: all)")
    args = parser.parse_args()

    services_to_run = args.services or list(SERVICES.keys())
    processes: List[subprocess.Popen] = []
    started: List[tuple[str, subprocess.Popen]] = []

    def cleanup(signum=None, frame=None):
        print("\nShutting down services...")
        for p in processes:
            if p.poll() is None:
                try:
                    os.killpg(p.pid, signal.SIGTERM)
                except ProcessLookupError:
                    pass
        for p in processes:
            try:
                p.wait(timeout=5)
            except subprocess.TimeoutExpired:
                try:
                    os.killpg(p.pid, signal.SIGKILL)
                except ProcessLookupError:
                    pass
                try:
                    p.wait(timeout=5)
                except subprocess.TimeoutExpired:
                    pass
        sys.exit(0)

    signal.signal(signal.SIGINT, cleanup)
    signal.signal(signal.SIGTERM, cleanup)

    try:
        for name in services_to_run:
            config = SERVICES[name]
            if is_port_open(args.host, config["port"]):
                print(
                    f"Skipping {name}: port {config['port']} already in use on {args.host}"
                )
                continue

            p = start_service(name, config, args.host, reload=not args.no_reload)
            time.sleep(0.5)
            rc = p.poll()
            if rc is not None:
                print(f"Service '{name}' failed to start (exit code {rc}).")
                if name in OPTIONAL_SERVICES:
                    continue
                cleanup()
            processes.append(p)
            started.append((name, p))
            time.sleep(0.5)

        print("\nWaiting for services to be ready...")
        for name in services_to_run:
            config = SERVICES[name]
            if is_port_open(args.host, config["port"]):
                # If it was skipped, it's already running.
                if wait_for_service(
                    args.host,
                    config["port"],
                    config.get("health_path", "/healthz"),
                ):
                    print(f"  {name}: ready at http://{args.host}:{config['port']}")
                else:
                    print(f"  {name}: running on port but health check failed")
                continue

            if wait_for_service(
                args.host,
                config["port"],
                config.get("health_path", "/healthz"),
            ):
                print(f"  {name}: ready at http://{args.host}:{config['port']}")
            else:
                print(f"  {name}: FAILED to start")

        print("\n" + "=" * 60)
        print("All services running. Press Ctrl+C to stop.")
        print("=" * 60)
        print(f"\nOrchestrator API: http://{args.host}:8000/v1/query")
        print(f"API Docs: http://{args.host}:8000/docs")
        print()

        while True:
            for name, p in list(started):
                rc = p.poll()
                if rc is not None:
                    if name in OPTIONAL_SERVICES:
                        print(f"Optional service '{name}' exited with code {rc}. Continuing...")
                        started.remove((name, p))
                        continue
                    print(f"Service '{name}' exited with code {rc}. Shutting down...")
                    cleanup()
            time.sleep(1)

    except KeyboardInterrupt:
        cleanup()

    return 0


if __name__ == "__main__":
    sys.exit(main())
