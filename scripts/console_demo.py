from __future__ import annotations

import argparse
import json
import os
import sys
import uuid
from datetime import datetime
from typing import Any, Dict, Optional

import httpx


def _now_iso() -> str:
    return datetime.utcnow().replace(microsecond=0).isoformat() + "Z"


def _get_env(name: str, default: str) -> str:
    value = os.getenv(name)
    return value if value else default


def _print_kv(title: str, value: Any) -> None:
    if value is None:
        return
    if isinstance(value, (dict, list)):
        rendered = json.dumps(value, indent=2, ensure_ascii=False)
        print(f"{title}:\n{rendered}")
        return
    print(f"{title}: {value}")


def _normalize_url(url: str) -> str:
    return url.rstrip("/")


def _post_query(
    client: httpx.Client,
    orchestrator_url: str,
    question: str,
    user_id: Optional[str],
    session_id: Optional[str],
    metadata: Optional[Dict[str, Any]],
) -> Dict[str, Any]:
    payload: Dict[str, Any] = {"question": question}
    if user_id:
        payload["user_id"] = user_id
    if session_id:
        payload["session_id"] = session_id
    if metadata:
        payload["metadata"] = metadata

    url = f"{_normalize_url(orchestrator_url)}/v1/query"
    resp = client.post(url, json=payload)
    resp.raise_for_status()
    return resp.json()


def _extract_answer(response_json: Dict[str, Any]) -> str:
    answer = response_json.get("answer")
    if isinstance(answer, str) and answer.strip():
        return answer
    status = response_json.get("status")
    if isinstance(status, str) and status.strip():
        return f"(no answer) status={status}"
    return "(no answer)"


def _print_response(response_json: Dict[str, Any], verbose: bool) -> None:
    request_id = response_json.get("request_id")
    status = response_json.get("status")
    safety = response_json.get("safety")
    relevance = response_json.get("relevance")

    _print_kv("request_id", request_id)
    _print_kv("status", status)

    if safety is not None:
        _print_kv("safety", safety)
    if relevance is not None:
        _print_kv("relevance", relevance)

    print("\nanswer:")
    print(_extract_answer(response_json))

    if verbose:
        citations = response_json.get("citations")
        agent_trace = response_json.get("agent_trace")
        if citations is not None:
            print("\n")
            _print_kv("citations", citations)
        if agent_trace is not None:
            print("\n")
            _print_kv("agent_trace", agent_trace)


def _print_help() -> None:
    print("Commands:")
    print("  /help                 Show this help")
    print("  /exit                 Quit")
    print("  /newsession           Start a new session_id")
    print("  /session              Print current session_id")
    print("  /set user_id <value>  Set user_id")
    print("  /set session_id <v>   Set session_id")
    print("  /set meta <json>      Set metadata JSON (replaces existing)")
    print("  /clear meta           Clear metadata")


def main(argv: Optional[list[str]] = None) -> int:
    parser = argparse.ArgumentParser(add_help=True)
    parser.add_argument(
        "--orchestrator-url",
        default=_get_env("ORCHESTRATOR_URL", "http://127.0.0.1:8000"),
    )
    parser.add_argument("--user-id", default=os.getenv("USER_ID"))
    parser.add_argument("--session-id", default=os.getenv("SESSION_ID"))
    parser.add_argument(
        "--timeout",
        type=float,
        default=float(os.getenv("CLIENT_TIMEOUT", "60")),
    )
    parser.add_argument("--verbose", action="store_true")
    args = parser.parse_args(argv)

    orchestrator_url = _normalize_url(args.orchestrator_url)
    user_id: Optional[str] = args.user_id
    session_id: str = args.session_id or f"console-{uuid.uuid4().hex}"  # noqa: S105
    metadata: Dict[str, Any] = {
        "client": "console_demo",
        "started_at": _now_iso(),
    }

    print("\nConsole Demo")
    print(f"Orchestrator: {orchestrator_url}")
    print(f"session_id: {session_id}")
    if user_id:
        print(f"user_id: {user_id}")
    print("Type /help for commands.\n")

    timeout = httpx.Timeout(args.timeout)
    with httpx.Client(timeout=timeout) as client:
        while True:
            try:
                raw = input("> ").strip()
            except (EOFError, KeyboardInterrupt):
                print()
                return 0

            if not raw:
                continue

            if raw.startswith("/"):
                parts = raw.split(maxsplit=2)
                cmd = parts[0].lower()

                if cmd in {"/exit", "/quit"}:
                    return 0

                if cmd == "/help":
                    _print_help()
                    continue

                if cmd == "/newsession":
                    session_id = f"console-{uuid.uuid4().hex}"  # noqa: S105
                    print(f"session_id: {session_id}")
                    continue

                if cmd == "/session":
                    print(f"session_id: {session_id}")
                    continue

                if cmd == "/set":
                    if len(parts) < 3:
                        print("usage: /set <user_id|session_id|meta> <value>")
                        continue
                    key = parts[1].lower()
                    value = parts[2]
                    if key == "user_id":
                        user_id = value
                        print(f"user_id: {user_id}")
                        continue
                    if key == "session_id":
                        session_id = value
                        print(f"session_id: {session_id}")
                        continue
                    if key == "meta":
                        try:
                            parsed = json.loads(value)
                            if not isinstance(parsed, dict):
                                raise ValueError("metadata must be a JSON object")
                            metadata = parsed
                            print("metadata set")
                        except Exception as e:
                            print(f"Invalid JSON: {e}")
                        continue

                    print("Unknown key. Use: user_id, session_id, meta")
                    continue

                if cmd == "/clear":
                    if len(parts) < 2:
                        print("usage: /clear meta")
                        continue
                    key = parts[1].lower()
                    if key == "meta":
                        metadata = {}
                        print("metadata cleared")
                        continue
                    print("Unknown key. Use: meta")
                    continue

                print("Unknown command. Type /help.")
                continue

            try:
                response_json = _post_query(
                    client=client,
                    orchestrator_url=orchestrator_url,
                    question=raw,
                    user_id=user_id,
                    session_id=session_id,
                    metadata=metadata,
                )
            except httpx.HTTPStatusError as e:
                body = None
                try:
                    body = e.response.json()
                except Exception:
                    body = e.response.text
                print(f"HTTP {e.response.status_code}: {body}")
                continue
            except httpx.RequestError as e:
                print(f"Request error: {e}")
                continue
            except Exception as e:
                print(f"Unexpected error: {e}")
                continue

            print()
            _print_response(response_json, verbose=args.verbose)
            print()


if __name__ == "__main__":
    raise SystemExit(main())
