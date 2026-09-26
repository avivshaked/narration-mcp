"""Spike (j): a raw stdio client that speaks JSON-RPC lines to the spike server and records the wire.

It starts ``spike_server.py`` as a subprocess, as an MCP host does, and runs four sessions:

- ``legacy``: the 2025-11-25 ``initialize`` handshake, then every behaviour the spike checks;
- ``legacy-offer-2026``: an ``initialize`` that proposes 2026-07-28 (the server counter-offers);
- ``modern``: the 2026-07-28 stateless form, every request carrying its own ``_meta`` envelope,
  opened with ``server/discover``;
- ``modern-first-no-envelope``: a first request with no envelope, which opens a legacy connection.

Every line sent and received goes to ``results/wire-<session>.jsonl``; the expectations and what was
observed go to ``results/wire-checks.json``. The exit status is 1 if any check failed.

Run: ``uv run python spikes/j-mcp-sdk/wire_probe.py``.
"""

from __future__ import annotations

import json
import os
import queue
import subprocess
import sys
import threading
import time
from pathlib import Path
from typing import Any

HERE = Path(__file__).resolve().parent
SERVER = HERE / "spike_server.py"
RESULTS = HERE / "results"

MODERN = "2026-07-28"
LEGACY = "2025-11-25"
PROTOCOL_VERSION_KEY = "io.modelcontextprotocol/protocolVersion"
CLIENT_INFO_KEY = "io.modelcontextprotocol/clientInfo"
CLIENT_CAPABILITIES_KEY = "io.modelcontextprotocol/clientCapabilities"
SERVER_INFO_KEY = "io.modelcontextprotocol/serverInfo"
SUBSCRIPTION_ID_KEY = "io.modelcontextprotocol/subscriptionId"
CLIENT_INFO = {"name": "wire-probe", "version": "0"}

GOOD_VOICE = {"path": "C:/voices/narrator.wav", "sha256": "5b" * 32, "transcript": "Good bread asks for patience."}
GOOD_LINE = {"voice": GOOD_VOICE, "text": "Before dawn,   the reef belongs to the Ossavine shrimp."}


def envelope(**extra: Any) -> dict[str, Any]:
    """The 2026-07-28 per-request ``_meta`` envelope."""
    return {PROTOCOL_VERSION_KEY: MODERN, CLIENT_INFO_KEY: CLIENT_INFO, CLIENT_CAPABILITIES_KEY: {}, **extra}


class Wire:
    """One stdio connection to a fresh spike-server process, with a log of every line both ways."""

    def __init__(self, name: str) -> None:
        self.name = name
        self.log: list[dict[str, Any]] = []
        self.stderr: list[str] = []
        self._inbox: queue.Queue[dict[str, Any]] = queue.Queue()
        self._next_id = 0
        self.proc = subprocess.Popen(
            [sys.executable, str(SERVER)],
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            env={**os.environ, "PYTHONUTF8": "1"},
        )
        threading.Thread(target=self._read_stdout, daemon=True).start()
        threading.Thread(target=self._read_stderr, daemon=True).start()

    def _read_stdout(self) -> None:
        assert self.proc.stdout is not None
        for raw in self.proc.stdout:
            line = raw.decode("utf-8").strip()
            if line:
                try:
                    self._inbox.put(json.loads(line))
                except json.JSONDecodeError:
                    self._inbox.put({"unparsed": line})

    def _read_stderr(self) -> None:
        assert self.proc.stderr is not None
        for raw in self.proc.stderr:
            self.stderr.append(raw.decode("utf-8", errors="replace").rstrip())

    def send(self, message: dict[str, Any]) -> None:
        assert self.proc.stdin is not None
        self.log.append({"dir": "->", "msg": message})
        self.proc.stdin.write((json.dumps(message, ensure_ascii=False) + "\n").encode("utf-8"))
        self.proc.stdin.flush()

    def request(self, method: str, params: dict[str, Any] | None = None) -> int:
        self._next_id += 1
        message: dict[str, Any] = {"jsonrpc": "2.0", "id": self._next_id, "method": method}
        if params is not None:
            message["params"] = params
        self.send(message)
        return self._next_id

    def notify(self, method: str, params: dict[str, Any] | None = None) -> None:
        message: dict[str, Any] = {"jsonrpc": "2.0", "method": method}
        if params is not None:
            message["params"] = params
        self.send(message)

    def recv(self, timeout: float) -> dict[str, Any] | None:
        try:
            message = self._inbox.get(timeout=timeout)
        except queue.Empty:
            return None
        self.log.append({"dir": "<-", "msg": message})
        return message

    def response(self, request_id: int, timeout: float = 15.0) -> tuple[dict[str, Any], list[dict[str, Any]]]:
        """Wait for the answer to ``request_id``; return it and the notifications seen before it."""
        seen: list[dict[str, Any]] = []
        deadline = time.monotonic() + timeout
        while (left := deadline - time.monotonic()) > 0:
            message = self.recv(left)
            if message is None:
                break
            if message.get("id") == request_id and ("result" in message or "error" in message):
                return message, seen
            seen.append(message)
        raise TimeoutError(f"{self.name}: no answer to request {request_id} within {timeout} s")

    def call(self, method: str, params: dict[str, Any] | None = None) -> dict[str, Any]:
        return self.response(self.request(method, params))[0]

    def drain(self, quiet_s: float) -> list[dict[str, Any]]:
        """Receive until nothing arrives for ``quiet_s`` seconds."""
        out: list[dict[str, Any]] = []
        while (message := self.recv(quiet_s)) is not None:
            out.append(message)
        return out

    def close(self) -> int | None:
        assert self.proc.stdin is not None
        self.proc.stdin.close()
        try:
            return self.proc.wait(timeout=15)
        except subprocess.TimeoutExpired:
            self.proc.kill()
            return self.proc.wait(timeout=15)

    def save(self) -> None:
        path = RESULTS / f"wire-{self.name}.jsonl"
        with path.open("w", encoding="utf-8", newline="\n") as out:
            for entry in self.log:
                out.write(json.dumps(entry, ensure_ascii=False) + "\n")


CHECKS: list[dict[str, Any]] = []


def check(session: str, name: str, ok: bool, observed: Any) -> None:
    CHECKS.append({"session": session, "check": name, "ok": bool(ok), "observed": observed})


def error_code(message: dict[str, Any]) -> int | None:
    error = message.get("error")
    return error.get("code") if isinstance(error, dict) else None


def result_of(message: dict[str, Any]) -> dict[str, Any]:
    result = message.get("result")
    return result if isinstance(result, dict) else {}


def check_tool_calls(w: Wire, era: str, meta: dict[str, Any] | None) -> None:
    """The tool behaviours both eras must share (design sections 5, 7.2 and 14)."""

    def params(name: str, arguments: Any, **meta_extra: Any) -> dict[str, Any]:
        p: dict[str, Any] = {"name": name, "arguments": arguments}
        if meta is not None or meta_extra:
            p["_meta"] = {**(meta or {}), **meta_extra}
        return p

    listed = w.call("tools/list", {"_meta": meta} if meta is not None else None)
    tools = result_of(listed).get("tools", [])
    text = json.dumps(tools)
    check(era, "tools/list publishes our schemas with no $ref", '"$ref"' not in text, [t["name"] for t in tools])
    roots = {t["name"]: t.get("outputSchema", {}).get("type") for t in tools if "outputSchema" in t}
    check(era, "every outputSchema root is type object", set(roots.values()) == {"object"}, roots)
    ours = next(t for t in tools if t["name"] == "check_line")
    check(
        era,
        "inputSchema published exactly as written (additionalProperties false kept, nested voice inlined)",
        ours["inputSchema"].get("additionalProperties") is False
        and ours["inputSchema"]["properties"]["voice"].get("additionalProperties") is False,
        ours["inputSchema"].get("additionalProperties"),
    )
    extras = sorted(k for k in result_of(listed) if k != "tools")
    check(era, "tools/list result keys besides tools", True, extras)

    good = result_of(w.call("tools/call", params("check_line", GOOD_LINE)))
    content = good.get("content", [])
    copy_ok = bool(content) and json.loads(content[0].get("text", "null")) == good.get("structuredContent")
    check(
        era,
        "success: structuredContent + text copy + resource_link, isError absent or false",
        not good.get("isError") and copy_ok and [c.get("type") for c in content] == ["text", "resource_link"],
        {"isError": good.get("isError"), "content_types": [c.get("type") for c in content]},
    )

    bad = result_of(w.call("tools/call", params("check_line", {**GOOD_LINE, "instruct": "calm and warm"})))
    error = (bad.get("structuredContent") or {}).get("error", {})
    content = bad.get("content", [])
    copy_ok = bool(content) and json.loads(content[0].get("text", "null")) == bad.get("structuredContent")
    check(
        era,
        "unknown field instruct: a tool result, isError true, INVALID_ARGUMENT, field instruct, hint, text copy",
        bad.get("isError") is True
        and error.get("code") == "INVALID_ARGUMENT"
        and error.get("field") == "instruct"
        and bool(error.get("hint"))
        and copy_ok,
        error,
    )

    wrong = result_of(w.call("tools/call", params("check_line", {**GOOD_LINE, "takes": "3"})))
    error = (wrong.get("structuredContent") or {}).get("error", {})
    check(
        era,
        "wrong type: isError with field takes",
        wrong.get("isError") is True and error.get("field") == "takes",
        error,
    )

    unknown = w.call("tools/call", params("no_such_tool", {}))
    check(era, "unknown tool: JSON-RPC -32602", error_code(unknown) == -32602, unknown.get("error"))

    not_object = w.call("tools/call", params("check_line", [1, 2]))
    check(
        era,
        "arguments not an object: JSON-RPC -32602 from the SDK's params model",
        error_code(not_object) == -32602,
        not_object.get("error"),
    )

    crash = w.call("tools/call", params("raise_demo", {}))
    check(era, "uncaught handler exception: what the SDK sends", "error" in crash, crash.get("error"))

    # Progress: the request carries _meta.progressToken; the server reports while it waits.
    rid = w.request("tools/call", params("wait_job", {"job_id": "j1", "wait_s": 1.0}, progressToken="p-1"))
    answer, seen = w.response(rid)
    progress = [m["params"] for m in seen if m.get("method") == "notifications/progress"]
    values = [p["progress"] for p in progress]
    check(
        era,
        "progress: notifications/progress with our token, monotonic, then the result",
        len(progress) >= 2
        and all(p["progressToken"] == "p-1" for p in progress)
        and values == sorted(values)
        and result_of(answer).get("structuredContent", {}).get("status") == "completed",
        {"count": len(progress), "first": progress[:1], "last": progress[-1:]},
    )

    # Cancellation: notifications/cancelled ends the in-flight call; it is never answered.
    rid = w.request("tools/call", params("wait_job", {"job_id": "j2", "wait_s": 5.0}, progressToken="p-2"))
    first = None
    while first is None:
        message = w.recv(10)
        if message is None:
            break
        if message.get("method") == "notifications/progress":
            first = message
    w.notify("notifications/cancelled", {"requestId": rid, "reason": "wire probe cancels the wait"})
    after = w.drain(1.5)
    answered = [m for m in after if m.get("id") == rid]
    late_progress = [m for m in after if m.get("method") == "notifications/progress"]
    alive = w.call("tools/list", {"_meta": meta} if meta is not None else None)
    check(
        era,
        "cancellation: no answer for the cancelled id, progress stops, the connection stays usable",
        first is not None and not answered and len(late_progress) <= 1 and "result" in alive,
        {"answered": answered, "progress_after_cancel": len(late_progress)},
    )


def check_resources_and_prompts(w: Wire, era: str, meta: dict[str, Any] | None) -> None:
    def p(**kw: Any) -> dict[str, Any]:
        return {**kw, "_meta": meta} if meta is not None else kw

    listed = result_of(w.call("resources/list", p()))
    templates = result_of(w.call("resources/templates/list", p()))
    check(
        era,
        "resources/templates/list returns our template",
        [t.get("uriTemplate") for t in templates.get("resourceTemplates", [])] == ["spike://jobs/{job_id}"],
        {"resources": listed, "templates": templates},
    )
    job = result_of(w.call("resources/read", p(uri="spike://jobs/j9")))
    check(
        era,
        "resources/read through the template: contents, and (2026-07-28 only) ttlMs + cacheScope",
        bool(job.get("contents")),
        {k: v for k, v in job.items() if k != "contents"},
    )
    missing = w.call("resources/read", p(uri="spike://nothing/here"))
    check(era, "missing resource: JSON-RPC -32602", error_code(missing) == -32602, missing.get("error"))
    prompts = result_of(w.call("prompts/list", p()))
    got = result_of(w.call("prompts/get", p(name="narrate_script", arguments={"voice_path": "C:/voices/a.wav"})))
    check(
        era,
        "prompts/list and prompts/get",
        [x["name"] for x in prompts.get("prompts", [])] == ["narrate_script"] and bool(got.get("messages")),
        {"prompt": prompts.get("prompts"), "messages": got.get("messages")},
    )


def legacy_session() -> None:
    era = "legacy"
    w = Wire(era)
    try:
        init = w.call(
            "initialize",
            {"protocolVersion": LEGACY, "capabilities": {}, "clientInfo": {"name": "wire-probe", "version": "0"}},
        )
        r = result_of(init)
        check(era, "initialize 2025-11-25 is answered", r.get("protocolVersion") == LEGACY, r)
        w.notify("notifications/initialized")
        check_tool_calls(w, era, None)
        check_resources_and_prompts(w, era, None)
        enveloped = w.call("tools/list", {"_meta": envelope()})
        check(
            era,
            "a 2026 envelope on a legacy connection: -32600",
            error_code(enveloped) == -32600,
            enveloped.get("error"),
        )
        sub = w.call("resources/subscribe", {"uri": "spike://jobs/j1"})
        check(
            era, "resources/subscribe with no handler registered: -32601", error_code(sub) == -32601, sub.get("error")
        )
        nope = w.call("no/such/method")
        check(era, "unknown method: -32601", error_code(nope) == -32601, nope.get("error"))
        ping = w.call("ping")
        check(era, "ping answered", "result" in ping, ping)
    finally:
        code = w.close()
        stderr_notes = [line for line in w.stderr if line.startswith("spike_server:")]
        check(era, "server saw the cancellation and exited cleanly", bool(stderr_notes) and code == 0, stderr_notes)
        w.save()


def legacy_offer_2026_session() -> None:
    era = "legacy-offer-2026"
    w = Wire(era)
    try:
        init = w.call(
            "initialize",
            {"protocolVersion": MODERN, "capabilities": {}, "clientInfo": {"name": "wire-probe", "version": "0"}},
        )
        r = result_of(init)
        check(
            era,
            "initialize proposing 2026-07-28: the server counter-offers the newest handshake version",
            r.get("protocolVersion") == LEGACY,
            r.get("protocolVersion"),
        )
    finally:
        w.close()
        w.save()


def modern_session() -> None:
    era = "modern"
    w = Wire(era)
    meta = envelope()
    try:
        discover = result_of(w.call("server/discover", {"_meta": meta}))
        check(
            era,
            "server/discover: 2026-07-28, capabilities, instructions, resultType, ttlMs, cacheScope, serverInfo",
            discover.get("supportedVersions") == [MODERN]
            and "resultType" in discover
            and "ttlMs" in discover
            and discover.get("cacheScope") in ("private", "public")
            and SERVER_INFO_KEY in discover.get("_meta", {}),
            discover,
        )
        check_tool_calls(w, era, meta)
        check_resources_and_prompts(w, era, meta)

        # subscriptions/listen: acknowledged first, then events tagged with the listen request's id.
        listen_id = w.request(
            "subscriptions/listen",
            {"_meta": meta, "notifications": {"resourceSubscriptions": ["spike://jobs/j-listen"]}},
        )
        ack = w.recv(10)
        wait_id = w.request(
            "tools/call", {"_meta": meta, "name": "wait_job", "arguments": {"job_id": "j-listen", "wait_s": 0.25}}
        )
        done, before = w.response(wait_id)
        events = [m for m in [*before, *w.drain(1.0)] if m.get("method") == "notifications/resources/updated"]
        check(
            era,
            "subscriptions/listen: acknowledged, then notifications/resources/updated tagged with the listen id",
            ack is not None
            and ack.get("method") == "notifications/subscriptions/acknowledged"
            and "result" in done
            and len(events) == 1
            and events[0]["params"]["uri"] == "spike://jobs/j-listen"
            and events[0]["params"]["_meta"][SUBSCRIPTION_ID_KEY] == listen_id,
            {"ack": ack, "events": events},
        )
        w.notify("notifications/cancelled", {"requestId": listen_id, "reason": "wire probe ends the listen"})
        w.drain(0.5)

        no_caps = w.call("tools/list", {"_meta": {PROTOCOL_VERSION_KEY: MODERN}})
        check(era, "envelope without clientCapabilities: -32602", error_code(no_caps) == -32602, no_caps.get("error"))
        no_meta = w.call("tools/list")
        check(
            era,
            "a request with no envelope on a modern connection: -32602",
            error_code(no_meta) == -32602,
            no_meta.get("error"),
        )
        future = w.call("tools/list", {"_meta": {**meta, PROTOCOL_VERSION_KEY: "2027-01-01"}})
        check(
            era,
            "an unknown protocol version: -32022 with the supported list",
            error_code(future) == -32022 and future["error"].get("data", {}).get("supported") == [MODERN],
            future.get("error"),
        )
        late_init = w.call(
            "initialize",
            {"protocolVersion": LEGACY, "capabilities": {}, "clientInfo": {"name": "wire-probe", "version": "0"}},
        )
        check(era, "initialize on a modern connection: -32022", error_code(late_init) == -32022, late_init.get("error"))
        ping = w.call("ping", {"_meta": meta})
        check(era, "ping (removed in 2026-07-28)", True, ping.get("error") or ping.get("result"))
    finally:
        code = w.close()
        stderr_notes = [line for line in w.stderr if line.startswith("spike_server:")]
        check(era, "server saw the cancellation and exited cleanly", bool(stderr_notes) and code == 0, stderr_notes)
        w.save()


def modern_first_without_envelope_session() -> None:
    era = "modern-first-no-envelope"
    w = Wire(era)
    try:
        first = w.call("tools/list")
        check(
            era,
            "a first request with no envelope opens a legacy connection, which then wants initialize",
            error_code(first) == -32602,
            first.get("error"),
        )
    finally:
        w.close()
        w.save()


def main() -> int:
    RESULTS.mkdir(exist_ok=True)
    legacy_session()
    legacy_offer_2026_session()
    modern_session()
    modern_first_without_envelope_session()
    failed = [c for c in CHECKS if not c["ok"]]
    summary = {"checks": CHECKS, "passed": len(CHECKS) - len(failed), "failed": len(failed)}
    out = RESULTS / "wire-checks.json"
    out.write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8", newline="\n")
    for c in CHECKS:
        print(f"{'ok  ' if c['ok'] else 'FAIL'} [{c['session']}] {c['check']}")
    print(f"{summary['passed']} passed, {summary['failed']} failed; details in {out.name}")
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
