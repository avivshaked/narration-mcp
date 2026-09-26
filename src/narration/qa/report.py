"""A job's report, from its assembled ``get_results`` object (design sections 7.5 and 11.1).

``report_md`` writes ``report.md``: every flag of every take, replaced attempts included, and each cue's
received → engine text. ``report_json`` gives the same content as data (``names.REPORT_SCHEMA``). Both read
the ``get_results`` layout, whose output schema fixes it, and tolerate a partial object (a failed or cancelled
job returns the segments it finished). Neither changes anything: the report records what QA found.
"""

from __future__ import annotations

import re
from collections import Counter
from collections.abc import Iterable, Mapping, Sequence
from typing import Any

from narration.contracts.names import REPORT_SCHEMA

__all__ = ["report_json", "report_md"]

_MD_SPECIAL = re.compile(r"([\\`*_\[\]<>|])")


def _esc(text: object) -> str:
    """``text`` with Markdown's inline markup characters escaped, so a cue's words print as sent."""
    return _MD_SPECIAL.sub(r"\\\1", str(text))


def _map(value: Any) -> Mapping[str, Any]:
    return value if isinstance(value, Mapping) else {}


def _list(value: Any) -> Sequence[Any]:
    return value if isinstance(value, (list, tuple)) else ()


def _num(value: Any, digits: int = 3) -> str:
    return "n/a" if not isinstance(value, (int, float)) or isinstance(value, bool) else f"{value:.{digits}f}"


def _take_flags(take: Mapping[str, Any]) -> list[Mapping[str, Any]]:
    """Every flag of a take, once: its QA flags (which include the alignment's), then its own flags."""
    out: list[Mapping[str, Any]] = []
    seen: set[tuple[Any, ...]] = set()
    sources: Iterable[Any] = (
        *_list(_map(take.get("qa")).get("flags")),
        *_list(_map(take.get("alignment")).get("flags")),
        *_list(take.get("flags")),
    )
    for flag in sources:
        f = _map(flag)
        key = (f.get("code"), f.get("severity"), f.get("cue"), f.get("message"))
        if key not in seen:
            seen.add(key)
            out.append(f)
    return out


def _flag_line(flag: Mapping[str, Any]) -> str:
    cue = f" (cue {flag['cue']})" if flag.get("cue") is not None else ""
    return f"**{_esc(flag.get('severity', '?'))}** `{flag.get('code', '?')}`{cue}: {_esc(flag.get('message', ''))}"


def report_md(results: Mapping[str, Any]) -> str:
    """``report.md`` for a job, from its ``get_results`` object."""
    job = _map(results.get("job"))
    segments = [_map(s) for s in _list(results.get("segments"))]
    title = job.get("label") or job.get("job_id") or "narration job"
    lines: list[str] = [f"# Narration report: {_esc(title)}", ""]

    facts = [f"Job `{job.get('job_id', 'n/a')}`", f"status {job.get('status', 'n/a')}"]
    if job.get("outcome"):
        facts.append(f"outcome {job['outcome']}")
    lines.append(" · ".join(facts))
    voice, engine, measurement = (
        _map(results.get("voice")),
        _map(results.get("engine_profile")),
        _map(results.get("measurement")),
    )
    if voice:
        lines.append(f"Voice `{voice.get('voice_hash', 'n/a')}` (clip sha256 `{voice.get('clip_sha256', 'n/a')}`)")
    if engine:
        lines.append(f"Engine profile `{engine.get('id', 'n/a')}` (`{engine.get('hash', 'n/a')}`)")
    if measurement:
        limit = measurement.get("max_segment_chars")
        seconds = measurement.get("max_segment_seconds")
        reliable = f"reliable up to {limit} spoken characters" if limit is not None else "no reliable length"
        if seconds is not None:
            reliable += f" ({_num(seconds, 1)} s)"
        lines.append(f"Measurement: {reliable}, measured {measurement.get('measured_at', 'n/a')}")
    lines.append("")

    # ---- summary
    verdicts: Counter[str] = Counter()
    n_takes = 0
    for segment in segments:
        for take in _list(segment.get("takes")):
            n_takes += 1
            verdicts[str(_map(_map(take).get("qa")).get("verdict", "not scored"))] += 1
    lines += ["## Summary", ""]
    counted = ", ".join(f"{k} {verdicts[k]}" for k in ("pass", "warn", "fail", "not scored") if verdicts[k])
    lines.append(f"- {len(segments)} segment(s), {n_takes} take(s)" + (f": {counted}" if counted else ""))
    consistency = _map(results.get("consistency"))
    if consistency:
        outliers = ", ".join(f"`{o}`" for o in _list(consistency.get("outliers"))) or "none"
        lines.append(
            f"- Consistency of the suggested takes: min {_num(consistency.get('min'), 4)}, "
            f"median {_num(consistency.get('median'), 4)}; outliers: {outliers} (a report, never a verdict)"
        )
    lines.append("")

    # ---- listen first
    items = [_map(i) for i in _list(results.get("listen_first"))]
    lines += ["## Listen first", ""]
    if not items:
        lines.append("Nothing: QA found nothing to listen to first.")
    for n, item in enumerate(items, 1):
        where = [str(item.get("segment_id", "?"))]
        if item.get("take_id"):
            where.append(f"`{item['take_id']}`")
        if item.get("cue") is not None:
            where.append(f"cue {item['cue']}")
        if item.get("from_s") is not None and item.get("to_s") is not None:
            where.append(f"{_num(item['from_s'], 2)}–{_num(item['to_s'], 2)} s")
        lines.append(f"{n}. {' · '.join(where)}: {_esc(item.get('reason', ''))}")
    lines.append("")

    # ---- segments
    lines += ["## Segments", ""]
    for segment in segments:
        sid = segment.get("segment_id", "?")
        lines.append(f"### {_esc(sid)} ({segment.get('status', 'n/a')})")
        lines.append("")
        suggestion = _map(segment.get("suggestion"))
        if segment.get("suggested_take_id"):
            lines.append(
                f"Suggested take `{segment['suggested_take_id']}`"
                + (f" (tier {suggestion.get('tier')}: {_esc(suggestion.get('reason', ''))})" if suggestion else "")
                + ". The suggestion is advice; the choice is the caller's."
            )
        else:
            lines.append("No suggested take.")
        lines.append("")
        text = _map(segment.get("text"))
        lines.append(f"Text ({text.get('spoken_chars', 'n/a')} spoken characters), received → engine:")
        lines.append("")
        for cue in (_map(c) for c in _list(text.get("cues"))):
            received, engine_text = cue.get("received", ""), cue.get("engine", "")
            same = " (the engine got the same text)" if received == engine_text else ""
            lines.append(f"- cue {cue.get('index', '?')}: “{_esc(received)}”{same}")
            if not same:
                lines.append(f"  → “{_esc(engine_text)}”")
            for warning in (_map(w) for w in _list(cue.get("warnings"))):
                lines.append(f"  - {_flag_line(warning)}")
        segment_flags = [_map(f) for f in _list(segment.get("flags"))]
        if segment_flags:
            lines += ["", "Segment flags:"]
            lines += [f"- {_flag_line(f)}" for f in segment_flags]
        lines += ["", "Takes (every take rendered or found for this request, replaced attempts included):", ""]
        takes = [_map(t) for t in _list(segment.get("takes"))]
        if not takes:
            lines.append("- none")
        for take in sorted(takes, key=lambda t: (t.get("attempt", 0), str(t.get("take_id", "")))):
            qa = _map(take.get("qa"))
            pace, expected = _map(qa.get("pace")), _map(qa.get("pace_expected"))
            mark = " (suggested)" if take.get("take_id") == segment.get("suggested_take_id") else ""
            parts = [f"attempt {take.get('attempt', '?')}", f"`{take.get('take_id', '?')}`{mark}"]
            if qa:
                parts += [
                    f"**{qa.get('verdict', '?')}**",
                    f"wer_adj {_num(qa.get('wer_adj'))} (raw {_num(qa.get('wer_raw'))})",
                    f"similarity {_num(qa.get('spk_sim_anchor'), 4)}",
                    f"{_num(pace.get('spoken_wpm'), 0)} spoken wpm (expected {_num(expected.get('spoken_wpm'), 0)})",
                    f"exact spans {'ok' if qa.get('exact_ok', True) else 'NOT ok'}",
                ]
            else:
                parts.append("not scored")
            lines.append("- " + " · ".join(parts))
            flags = _take_flags(take)
            lines += [f"  - {_flag_line(f)}" for f in flags] or ["  - no flags"]
        lines.append("")
    return "\n".join(lines).rstrip() + "\n"


def report_json(results: Mapping[str, Any]) -> dict[str, Any]:
    """The report as data: a summary, and per segment its text echo, suggestion and every take's flags."""
    segments_out: list[dict[str, Any]] = []
    verdicts: Counter[str] = Counter()
    flag_counts: Counter[str] = Counter()
    for segment in (_map(s) for s in _list(results.get("segments"))):
        takes_out: list[dict[str, Any]] = []
        for take in (_map(t) for t in _list(segment.get("takes"))):
            verdict = _map(take.get("qa")).get("verdict")
            verdicts[str(verdict or "not_scored")] += 1
            flags = [dict(f) for f in _take_flags(take)]
            flag_counts.update(str(f.get("code")) for f in flags)
            takes_out.append(
                {
                    "take_id": take.get("take_id"),
                    "attempt": take.get("attempt"),
                    "suggested": take.get("take_id") == segment.get("suggested_take_id"),
                    "verdict": verdict,
                    "flags": flags,
                }
            )
        segment_flags = [dict(_map(f)) for f in _list(segment.get("flags"))]
        flag_counts.update(str(f.get("code")) for f in segment_flags)
        cues_out = []
        for cue in (_map(c) for c in _list(_map(segment.get("text")).get("cues"))):
            warnings = [dict(_map(w)) for w in _list(cue.get("warnings"))]
            flag_counts.update(str(w.get("code")) for w in warnings)
            cues_out.append(
                {
                    "index": cue.get("index"),
                    "received": cue.get("received"),
                    "engine": cue.get("engine"),
                    "warnings": warnings,
                }
            )
        segments_out.append(
            {
                "segment_id": segment.get("segment_id"),
                "status": segment.get("status"),
                "suggested_take_id": segment.get("suggested_take_id"),
                "suggestion": dict(_map(segment.get("suggestion"))) or None,
                "cues": cues_out,
                "flags": segment_flags,
                "takes": takes_out,
            }
        )
    return {
        "schema": REPORT_SCHEMA,
        "job": dict(_map(results.get("job"))),
        "summary": {
            "segments": len(segments_out),
            "takes": sum(len(s["takes"]) for s in segments_out),
            "verdicts": dict(sorted(verdicts.items())),
            "flags": dict(sorted(flag_counts.items())),
        },
        "segments": segments_out,
        "consistency": dict(_map(results.get("consistency"))) or None,
        "listen_first": [dict(_map(i)) for i in _list(results.get("listen_first"))],
    }
