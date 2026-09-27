"""The words the MCP surface publishes: tool titles and descriptions, resource and prompt texts (section 7).

Every tool description states that the service keeps no caller state, that a paragraph's length is the
caller's decision, and that a respelling is a hint (section 7.1). A tool that returns or takes a handle (a
job, design or take) states the retention period (sections 5, 15).

A tool that can return a retryable error also states the backoff rule (DC-2, revision 5.2): the tools in
``RETRYABLE_TOOLS``, an explicit set rather than one derived from ``read_only``. The retryable codes
(``DAEMON_UNAVAILABLE``, ``STORE_FULL``, ``QUEUE_FULL``, ``RATE_LIMITED``) come from writing to the store or
commanding the daemon, which the tools that are not read-only do, and from ``get_job``: it is read-only, but
when no daemon serves an active job it starts one, and answers ``DAEMON_UNAVAILABLE`` when it cannot.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Final

from narration.config import RetentionConfig
from narration.contracts.names import TOOL_NAMES

NO_CALLER_STATE: Final = (
    "The service keeps no caller state: it records nothing of your script, choices, voices or pronunciations, "
    "so every call carries what it needs."
)
LENGTH_IS_YOURS: Final = (
    "A paragraph's length is your decision: a segment longer than the voice's reliable length is warned about "
    "(SEGMENT_TOO_LONG), never refused or split."
)
RESPELLING_IS_A_HINT: Final = "A pronunciation respelling is a hint to the engine, never a guarantee."
BACKOFF_RULE: Final = (
    "On a retryable error, wait at least its retry_after_s, add your own jitter, then send the identical "
    "request again; it is deduplicated. The service suggests; you decide."
)


def retention_clause(retention: RetentionConfig) -> str:
    """Section 15's retention rule with the configured periods."""
    return (
        f"Jobs, designs, takes and their files are kept for {retention.retention_days} days after they were last "
        f"used, measurements for {retention.measurement_retention_days} days; copy what you keep. After that, the "
        "same request is answered again from whatever is still cached."
    )


@dataclass(frozen=True, slots=True)
class ToolText:
    """A tool's title, what it does, and which of the conditional clauses apply to it."""

    title: str
    does: str
    retention: bool
    """It returns or reads a job, design or take, so it states the retention period."""


TOOL_TEXTS: Final[dict[str, ToolText]] = {
    "get_server_status": ToolText(
        "Server status",
        "The daemon's state and current job; the GPU (in use, by which model, when it unloads); the queue, "
        "threads, limits and versions; the engine profiles; the capabilities (the pace, context and instruct "
        "controls are all unsupported); the measured alignment error; and admission: whether a submission "
        "would be accepted now and when to come back. Check it before heavy jobs (measure_voice, design_voice).",
        retention=False,
    ),
    "release_gpu": ToolText(
        "Release the GPU",
        "Unload an idle model from the GPU now. While a job runs it changes nothing and names that job (busy_job).",
        retention=False,
    ),
    "get_job": ToolText(
        "Get a job",
        "A job's status, phase, progress, eta_s, queue_position and poll_after_s (the earliest poll worth "
        "making). wait_s (up to 55) waits for the job to change; if the request carries a progressToken, "
        "progress notifications are sent while it waits. Cancelling this request ends only the wait, never "
        "the job (use cancel_job). A failed job is still a successful call: the job's error is in error."
        " If the daemon died, get_job starts it again, so the job resumes.",
        retention=True,
    ),
    "get_results": ToolText(
        "Get a job's results",
        "A finished job's result. For a narration job, per segment: the text echo (received, spoken and "
        "engine text), every take with its delivery file (path and sha256; audio is never inlined), cue and "
        "word times, alignment, QA verdict and flags, and the suggested take with its reason (advice; you "
        "choose); then consistency, listen_first and report_md. For a design, measurement, profile or "
        "audition job, its result. A failed or cancelled job returns what it finished, and job.error.",
        retention=True,
    ),
    "cancel_job": ToolText(
        "Cancel a job",
        "Cancel a job cooperatively. Finished renders, takes and analyses stay in the cache. A job that has "
        "already ended gives JOB_NOT_CANCELLABLE.",
        retention=True,
    ),
    "design_voice": ToolText(
        "Design a voice",
        "Design voice candidates with Qwen VoiceDesign from a positive-only description (name the qualities "
        "you want, not those you do not) and takes seeds. Each candidate has a clip (path and sha256), its "
        "exact transcript, its seed and a profile. A description with a negation is still rendered, and lint "
        "lists what it found. Returns a job: read it with get_job, then get_results. You listen and choose; "
        "the service records no choice.",
        retention=True,
    ),
    "profile_voice": ToolText(
        "Profile a voice",
        "Measure a clip or take (pitch, speaking rate, pauses, loudness, brightness, HNR, CPPS) and draw its "
        "spectrogram and pitch contour, on the CPU in seconds. Returns a job; get_results gives the "
        "measurements and the pictures' paths.",
        retention=True,
    ),
    "measure_voice": ToolText(
        "Measure a voice",
        "Measure a voice once per clip and engine: a transcript check, a calibration set and a length ladder, "
        "giving max_segment_chars and max_segment_seconds, the pace curve and the similarity baseline. A heavy "
        "GPU job (20 to 50 minutes): check get_server_status first. If the voice is already measured under "
        "this engine, the measurement comes back at once.",
        retention=True,
    ),
    "check_text": ToolText(
        "Check text",
        "Renders nothing. Per cue: the received, spoken and engine text, the hints applied, text warnings "
        "(digits, symbols, unit-like tokens) and exact spans. With a measured voice, each segment's spoken "
        "length against the voice's reliable length and an estimated duration. Resolve text warnings by "
        "editing the text; the service speaks text as sent and never rewords it.",
        retention=False,
    ),
    "audition_pronunciation": ToolText(
        "Audition a pronunciation",
        "Render a term with up to four respelling variants, optionally inside a carrier sentence, and report "
        "what the ASR heard for each. The voice need not be measured. Whoever owns the text decides by ear; "
        "the service records no choice. Returns a job.",
        retention=True,
    ),
    "submit_job": ToolText(
        "Narrate segments",
        "Render and analyse a request's segments with a measured voice: the takes asked for, automatic retakes "
        "of failing takes, cue alignment and QA. With dry_run, only the plan (what is cached, what must be "
        "rendered, estimates) and the text echo. Text is spoken as sent (text_mode 'spoken'): write numbers "
        "and symbols as words. The same request gives the same job or the cached result; to hear new "
        "deliveries, name new attempts. Refusals are tool errors with a hint (VOICE_NOT_MEASURED: run "
        "measure_voice first). Returns a job.",
        retention=True,
    ),
}
assert tuple(TOOL_TEXTS) == TOOL_NAMES, "one text per tool, in the section 7.1 order"


def tool_description(name: str, retention: RetentionConfig) -> str:
    """The published description of a tool: what it does, then the rules every caller must know."""
    text = TOOL_TEXTS[name]
    parts = [text.does, NO_CALLER_STATE, LENGTH_IS_YOURS, RESPELLING_IS_A_HINT]
    if text.retention:
        parts.append(retention_clause(retention))
    if can_return_retryable(name):
        parts.append(BACKOFF_RULE)
    return " ".join(parts)


RETRYABLE_TOOLS: Final = frozenset(
    {
        "release_gpu",
        "cancel_job",
        "design_voice",
        "profile_voice",
        "measure_voice",
        "audition_pronunciation",
        "submit_job",
        "get_job",
    }
)
"""The tools that can return a retryable error, so their descriptions state the backoff rule: every tool that
writes, and ``get_job`` (see the module docstring)."""
assert RETRYABLE_TOOLS.issubset(TOOL_NAMES), "only published tools"


def can_return_retryable(name: str) -> bool:
    """Whether a tool can return a retryable error (``RETRYABLE_TOOLS``; see the module docstring)."""
    return name in RETRYABLE_TOOLS


SERVER_TITLE: Final = "Narration"
SERVER_INSTRUCTIONS: Final = (
    "narration-mcp designs voices, measures them, and narrates paragraphs of cues in a chosen voice, returning "
    "QA'd, cue-aligned takes as local WAV files (path and sha256). The flow: design_voice, then a person "
    "listens and chooses a clip; measure_voice once per clip; check_text; submit_job; get_job with wait_s; "
    "get_results. " + NO_CALLER_STATE + " " + LENGTH_IS_YOURS + " " + RESPELLING_IS_A_HINT + " Heavy jobs use "
    "a shared GPU: read get_server_status (admission) before submitting one. " + BACKOFF_RULE
)

RESOURCE_DESCRIPTIONS: Final[dict[str, str]] = {
    "status": "The server's status, as get_server_status returns it.",
    "design": "A design's candidates: clips, exact transcripts, seeds and profiles.",
    "measurement": "A voice's measurements, one per engine profile.",
    "job": "A job, as get_job returns it. Subscribable (2026-07-28 clients, through subscriptions/listen).",
    "job report": "A finished job's report, in Markdown.",
    "take": "A take: its delivery file, its render and its analyses.",
}


@dataclass(frozen=True, slots=True)
class PromptText:
    """A prompt's title, description, argument descriptions and message template (section 7.8)."""

    title: str
    description: str
    arguments: dict[str, str]
    template: str


PROMPT_TEXTS: Final[dict[str, PromptText]] = {
    "narrate_script": PromptText(
        "Narrate a script",
        "Narrate a script's paragraphs with a voice: measure, check the text, submit, wait, collect.",
        {"voice_path": "the absolute path of the voice's WAV clip"},
        "Narrate a script with the voice whose clip is at {voice_path}. The service keeps no caller state, so "
        "send the voice (path, sha256 and exact transcript) and your pronunciation hints with every call.\n"
        "1. If the clip has no measurement under the current engine, run measure_voice. It is a heavy GPU "
        "job: call get_server_status first and read admission.\n"
        "2. Run check_text on every paragraph's cues. The cues must already be the words to be spoken, with "
        "numbers and symbols written as words.\n"
        "3. Resolve every text warning by editing the text; the service never rewords it. A paragraph's "
        "length is your decision: an over-long segment is warned about, not refused.\n"
        "4. Run submit_job with the takes you want and your hints. A respelling is a hint to the engine, "
        "never a guarantee.\n"
        "5. Call get_job with wait_s until the job has ended, then get_results. On a retryable error, wait "
        "at least retry_after_s, add your own jitter, and send the identical request again.\n"
        "6. Copy the takes you keep and check their sha256; record the take and analysis ids; report "
        "listen_first to the person, who listens and decides.",
    ),
    "resolve_flags": PromptText(
        "Resolve flags",
        "Handle each flagged take or cue of a job with a hint, new attempts, or a text edit.",
        {"job_id": "the job whose flags to resolve"},
        "Resolve the flags of job {job_id}. Read it with get_results. For each flagged take or cue, choose "
        "one remedy:\n"
        "- a pronunciation hint: a respelling is a hint to the engine, never a guarantee; try variants with "
        "audition_pronunciation;\n"
        "- new attempts for the segment: name attempts not rendered yet, such as [3, 4];\n"
        "- an edit of the text: the service speaks text as sent.\n"
        "Then send submit_job again with the changes; cached work is reused, and the service keeps no "
        "caller state, so send the whole request. Tell the person what you changed and what to listen to "
        "first (listen_first).",
    ),
    "design_narrator_voice": PromptText(
        "Design a narrator voice",
        "Write positive-only voice descriptions, design candidates, shortlist them, and ask a person to choose.",
        {"brief": "what the narrator should sound like, and for what"},
        "Design a narrator voice for this brief: {brief}\n"
        "1. Write two or three voice descriptions that name only the qualities wanted. A negation such as "
        "'not gravelly' tends to come out as that quality.\n"
        "2. Run design_voice with takes 3 for each description. It uses the shared GPU: call "
        "get_server_status first.\n"
        "3. Read each candidate's profile (pitch, speaking rate, brightness, the pictures) to shortlist.\n"
        "4. Ask the person to listen to the shortlisted clips and choose; you cannot hear them, and the "
        "service records no choice. Copy the chosen clip and keep its path, sha256 and exact transcript: "
        "every later call needs them.",
    ),
    "add_pronunciation": PromptText(
        "Add a pronunciation",
        "Draft respelling hints for a term and audition them.",
        {"term": "the word or words as they appear in the text"},
        "Help pronounce '{term}'.\n"
        "1. Draft up to four respellings (for example 'Oss-a-veen' for 'Ossavine').\n"
        "2. Run audition_pronunciation with them and the voice, in a carrier sentence from the text if there "
        "is one.\n"
        "3. Report what the ASR heard for each variant. A respelling is a hint to the engine, never a "
        "guarantee, and whoever owns the text decides by ear; the service records no choice.\n"
        "4. Use the chosen respelling as a hint in check_text and submit_job.",
    ),
}
