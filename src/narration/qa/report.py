"""A job's report, from its assembled ``get_results`` object (design sections 7.5 and 11.1).

``report_md`` writes ``report.md``: every flag of every take, replaced attempts included, and each cue's
received → engine text. Its "Failures" section gathers every attempt that failed QA or that a retake replaced,
with its fail and warn flags and the take that finally filled its slot (plan.md WP48). ``report_json`` gives the
same content as data (``names.REPORT_SCHEMA``). Both read the ``get_results`` layout, whose output schema fixes
it, and tolerate a partial object (a failed or cancelled job returns the segments it finished). Neither changes
anything: the report records what QA found. Neither holds a file path (the tool texts promise that:
``narration.mcp.descriptions.REPORT_HOLDS``).

``replacements`` reads which retake replaced which attempt from the job's ``RETAKEN`` flags; the operator's
``narration-admin failures`` uses it too, so both views agree.
"""

from __future__ import annotations

import re
from collections import Counter
from collections.abc import Iterable, Mapping, Sequence
from typing import Any

from narration.contracts import codes
from narration.contracts.names import REPORT_SCHEMA

__all__ = ["replacements", "report_json", "report_md"]

_REASONS = ("fail", "warn")
"""The severities that say why a take failed or was retaken (a retake trigger is a fail flag, or a warn
``CUE_UNALIGNED`` or ``HEAD_INSERTION``: ``codes.is_retake_trigger``)."""

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


def _int(value: Any) -> int | None:
    return value if isinstance(value, int) and not isinstance(value, bool) else None


def replacements(attempts: Iterable[tuple[int, Iterable[Mapping[str, Any]]]]) -> dict[int, int]:
    """Which attempt finally filled the take slot of each attempt a retake replaced, within one segment.

    ``attempts`` gives each attempt's number with its flags as JSON (a ``get_results`` take's ``flags``, or a
    ``JobAttempt``'s). The job writes ``RETAKEN`` on every retake, and its ``details.replaced`` lists every
    earlier attempt of the slot (``narration.jobs.record``). So the retake that lists the most is the slot's
    last attempt. The result maps each replaced attempt to that last attempt. An attempt no retake lists was
    not replaced, and is not in it.
    """
    best: dict[int, tuple[int, int]] = {}
    for number, flags in attempts:
        for flag in flags:
            if flag.get("code") != codes.RETAKEN:
                continue
            replaced = [_int(_map(entry).get("attempt")) for entry in _list(_map(flag.get("details")).get("replaced"))]
            for earlier in replaced:
                if earlier is None:
                    continue
                known = best.get(earlier)
                if known is None or len(replaced) > known[0]:
                    best[earlier] = (len(replaced), number)
    return {earlier: last for earlier, (_, last) in best.items()}


def _failures(segments: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    """Every take that failed QA or that a retake replaced, per segment in request order, by attempt: its
    fail and warn flags, and the take that finally filled its slot (itself when nothing replaced it)."""
    out: list[dict[str, Any]] = []
    for segment in segments:
        takes = [_map(t) for t in _list(segment.get("takes"))]
        numbered = [(n, t) for t in takes if (n := _int(t.get("attempt"))) is not None]
        last_of = replacements((n, [_map(f) for f in _list(t.get("flags"))]) for n, t in numbered)
        by_attempt = dict(numbered)
        for number, take in sorted(numbered, key=lambda nt: (nt[0], str(nt[1].get("take_id", "")))):
            verdict = _map(take.get("qa")).get("verdict")
            if verdict != "fail" and number not in last_of:
                continue
            final_number = last_of.get(number, number)
            final = by_attempt.get(final_number)
            out.append(
                {
                    "segment_id": segment.get("segment_id"),
                    "attempt": number,
                    "take_id": take.get("take_id"),
                    "verdict": verdict,
                    "replaced": number in last_of,
                    "final_take": {
                        "attempt": final_number,
                        "take_id": final.get("take_id") if final is not None else None,
                        "verdict": _map(final.get("qa")).get("verdict") if final is not None else None,
                    },
                    "flags": [dict(f) for f in _take_flags(take) if f.get("severity") in _REASONS],
                }
            )
    return out


def _failures_md(failures: Sequence[Mapping[str, Any]]) -> list[str]:
    lines = ["## Failures", ""]
    if not failures:
        return [*lines, "None: no take failed QA, and no retake replaced one.", ""]
    lines += [
        f"Every take that failed QA or that a retake replaced ({len(failures)}), with its fail and warn flags "
        "and the take that finally filled its slot:",
        "",
    ]
    for failure in failures:
        final = _map(failure.get("final_take"))
        if failure.get("replaced"):
            made = f"`{final['take_id']}`" if final.get("take_id") else "(not made)"
            filled = (
                f"replaced: the slot was filled by attempt {final.get('attempt', '?')} {made} "
                f"({final.get('verdict') or 'not scored'})"
            )
        else:
            filled = "not replaced: the last take of its slot"
        verdict = failure.get("verdict") or "not scored"
        lines.append(
            f"- {_esc(failure.get('segment_id', '?'))} · attempt {failure.get('attempt', '?')} · "
            f"`{failure.get('take_id', '?')}` · **{verdict}** · {filled}"
        )
        flags = [_map(f) for f in _list(failure.get("flags"))]
        lines += [f"  - {_flag_line(f)}" for f in flags] or ["  - no fail or warn flag"]
    lines.append("")
    return lines


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

    # ---- failures (plan.md WP48): every failed or replaced take, with why
    lines += _failures_md(_failures(segments))

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
    """The report as data: a summary, per segment its text echo, suggestion and every take's flags, and the
    failures (every take that failed QA or that a retake replaced, as ``report_md``'s "Failures" lists them)."""
    segments = [_map(s) for s in _list(results.get("segments"))]
    segments_out: list[dict[str, Any]] = []
    verdicts: Counter[str] = Counter()
    flag_counts: Counter[str] = Counter()
    for segment in segments:
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
        "failures": _failures(segments),
    }
