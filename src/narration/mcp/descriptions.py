"""The words the MCP surface publishes: tool titles and descriptions, the server's instructions, resource and
prompt texts (section 7).

Every tool description states that the service keeps no caller state, that a paragraph's length is the
caller's decision, and that a respelling is a hint (section 7.1). A tool that returns or takes a handle (a
job, design or take) states the retention period (sections 5, 15).

A tool that can return a retryable error also states the backoff rule (DC-2, revision 5.2): the tools in
``RETRYABLE_TOOLS``, an explicit set rather than one derived from ``read_only``. The retryable codes
(``DAEMON_UNAVAILABLE``, ``STORE_FULL``, ``QUEUE_FULL``, ``RATE_LIMITED``) come from writing to the store or
commanding the daemon, which the tools that are not read-only do, and from ``get_job``: it is read-only, but
when no daemon serves an active job it starts one, and answers ``DAEMON_UNAVAILABLE`` when it cannot.

The texts also tell a calling agent what decides whether real use goes well, each checked against the code
it describes:

- every invented or unusual name goes in as a hint, the term alone if it needs no respelling (section 9.1:
  QA collapses a hinted term to one token in ``wer_adj``, and an unhinted one counts as word errors);
- a job is kept to a scene, and ``get_results`` is asked without word times unless they are needed, so a
  client can read the result whole;
- each segment's ``suggested_take_id`` is the take to use: ``takes[]`` also lists failed and replaced ones;
- the voice is sent exactly as kept with its clip, since the transcript is part of ``voice_hash``;
- the options by their real names: ``dry_run``, ``strict_text``, ``takes``, ``max_retakes``, ``priority``.

A tool this build lists but cannot run yet (``NOT_IN_THIS_BUILD``) is marked so in its description, and in
the instructions and prompts that name it. None of these texts enters a key: ``narration.keys`` hashes
requests, text and pins, never what the surface says.
"""

from __future__ import annotations

from collections.abc import Collection
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

CLIENT_TEXT_LIMIT: Final = 2048
"""The longest server instructions, and tool description, a client is held to show in full. KNOW (2026-09-27):
Claude Code 2.1.258 cuts server instructions longer than 2048 characters ("Server instructions truncated from
N to 2048 chars"). BELIEVE: it holds tool descriptions to the same length. A test keeps every text within it."""

NAMES_AS_HINTS: Final = (
    "Send every invented or unusual name in the text as a hint, in every request that speaks it; the term alone "
    "is enough, a respelling is optional. QA then scores the name as one word (a close spelling, or one of the "
    "hint's asr_aliases, costs nothing). A name without a hint is scored as misheard words (one heard as three "
    "words is three errors), which can fail the take (WER_HIGH) and use up its retakes."
)
"""Section 9.1 ("An invented name that is not sent as a hint counts against the word error rate") and 11.1
step 4 (``wer_adj`` collapses each hinted term); section 21, asks of the flow, item 3."""
VOICE_AS_KEPT: Final = (
    "Send the voice as kept with its clip: path, sha256 (lower-case hex) and the transcript copied character for "
    "character, never retyped; a transcript that differs by one character is another voice, with no "
    "measurement yet."
)
"""``voice_hash`` hashes the clip's sha256 and the transcript as sent (NFC applied), section 10.2."""
JOB_SIZE: Final = (
    "Keep a job to a scene, about 8 to 10 segments: a chapter-sized job's get_results can exceed what a client "
    "accepts from one tool call."
)
READ_RESULTS: Final = (
    "Call get_results with include_words false unless you need word times, and use each segment's "
    "suggested_take_id: takes[] also lists failed and replaced attempts, so takes[0] may be a failed one."
)
REPORT_RESOURCE: Final = (
    "narration://jobs/{job_id}/report summarises a job in Markdown (per segment: the suggested take, every take's "
    "verdict and flags)."
)
REPORT_HOLDS: Final = (
    "narration://jobs/{job_id}/report holds each segment's suggested take and every take's verdict and flags, "
    "but no file paths and no full cue timings (only the spans it lists to listen to first): use it to review a "
    "large job, then take paths and cue timings from get_results or narration://takes/{take_id}."
)
"""What the report resource holds (``qa.report.report_md``), for the texts that offer it as the smaller read."""
TIER_4: Final = (
    "A suggestion of tier 4 means every take failed QA (the job's outcome is then needs_attention): resolve its "
    "flags (the resolve_flags prompt) or redo the segment before keeping it."
)
"""Section 8's tier 4 (``qa.suggest``): the suggested take is the best of the failed ones."""
OPTIONS: Final = (
    "submit_job's options: dry_run (the plan only), strict_text (refuse digits, symbols and unit-like tokens "
    "instead of speaking them as the engine reads them), takes, max_retakes, and priority 'interactive' (a short "
    "redo runs ahead of a long batch)."
)

NOT_IN_THIS_BUILD: Final[frozenset[str]] = frozenset({"design_voice", "profile_voice", "audition_pronunciation"})
"""Tools this build lists but cannot run yet: each answers ``BACKEND_NOT_INSTALLED`` at once, because the
daemon has no handler for its job kind (``narration.backend.service.RUNNABLE_KINDS``). Take a tool out in the
change that lands its handler (WP34: ``design_voice``, ``profile_voice``; WP35: ``audition_pronunciation``). A
test holds this set equal to the tools whose kind is not runnable, so a handler merged without its text, or
a text without its handler, fails. The set is kept here rather than read from the backend: these texts are
built without a backend (the front-end serves any ``Backend``), and importing the concrete backend here
would pull the job engine into the front-end."""


def not_in_this_build(tool: str) -> str:
    """The mark on a tool this build lists but cannot run yet (``NOT_IN_THIS_BUILD``)."""
    return (
        f"Not in this build yet: {tool} answers BACKEND_NOT_INSTALLED at once. The narration tools "
        "(measure_voice, check_text, submit_job) work meanwhile."
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
        "threads, limits and versions (version is the package's; spec_revision is the MCP specification "
        "revision the server is built on); the engine profiles; the capabilities (the pace, context and "
        "instruct controls are all unsupported); the measured alignment error; and admission: whether a "
        "submission would be accepted now and when to come back. Check it before heavy jobs (measure_voice, "
        "design_voice, a large submit_job).",
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
        "making). wait_s (up to 55) holds the answer until the job's status changes or wait_s has passed: call "
        "again, with wait_s, until status is completed, failed or cancelled, then read get_results. If the "
        "request carries a progressToken, progress notifications are sent while it waits. Cancelling this "
        "request ends only the wait, never the job (use cancel_job). A failed job is still a successful call: "
        "the job's error is in error. If the daemon died, get_job starts it again, so the job resumes.",
        retention=True,
    ),
    "get_results": ToolText(
        "Get a job's results",
        "A finished job's result. For a narration job, per segment: suggested_take_id and suggestion (advice; "
        "you choose), the text echo, and takes[]: every take rendered or found for this request, failed and "
        "replaced attempts included, so use suggested_take_id, not takes[0]. "
        + TIER_4
        + " Each take has delivery (path, sha256, samples, sample_rate, duration_s; audio is never inlined), "
        "cues[] (start_s and end_s, null for a cue that could not be placed; words[] with include_words), qa "
        "(verdict, flags, wer_adj, terms: what the recogniser heard for each hinted term) and flags (such as "
        "RETAKEN, apart from qa.flags). Then consistency, listen_first and report_md. Send include_words false "
        "unless you need word times: they make the result much larger, and a job of more than about 10 "
        "segments can exceed what a client accepts from one tool call. "
        + REPORT_HOLDS
        + " For a design, measurement, profile or audition job, its result. A failed or cancelled job returns "
        "what it finished, and job.error.",
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
        "the service records no choice. Every candidate's clip goes on the service's provenance list, so "
        "measure_voice and submit_job accept it as synthetic with no allowlist edit. Copy the chosen clip out of "
        "the store into your own folder (the store's copy is removed after the retention period), and keep its "
        "sha256 and transcript exactly as the candidate gives them. A candidate flagged WER_HIGH "
        "does not say its transcript as the recogniser heard it, and measuring it fails (REF_TEXT_MISMATCH).",
        retention=True,
    ),
    "profile_voice": ToolText(
        "Profile a voice",
        "Measure any WAV you can read, a clip or a take (pitch, pauses, loudness, brightness, HNR, CPPS), and "
        "draw its spectrogram and pitch contour, on the CPU in seconds. It is sent no transcript, so its "
        "speaking rate is null; a design candidate's profile has one. Returns a job; get_results gives the "
        "measurements and the pictures' paths.",
        retention=True,
    ),
    "measure_voice": ToolText(
        "Measure a voice",
        "Measure a voice once per clip and engine: a transcript check, a calibration set and a length ladder, "
        "giving max_segment_chars and max_segment_seconds, the pace curve and the similarity baseline. A heavy "
        "GPU job (20 to 50 minutes): check get_server_status first. If the voice is already measured under "
        "this engine, the measurement comes back at once; otherwise it returns a job: call get_job with wait_s "
        "until the job has completed before you run submit_job with this voice. " + VOICE_AS_KEPT,
        retention=True,
    ),
    "check_text": ToolText(
        "Check text",
        "Renders nothing. Per cue: the received, spoken and engine text, the hints applied (hints_applied), "
        "text warnings (digits, symbols, unit-like tokens) and exact spans. With a measured voice, each "
        "segment's spoken length against the voice's reliable length and an estimated duration. Resolve text "
        "warnings by editing the text; the service speaks text as sent and never rewords it. "
        + NAMES_AS_HINTS
        + " Send the hints you will send to submit_job: hints_applied shows where each term matched (as whole "
        "words, case-sensitively; the term followed by 's matches too).",
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
        "of failing takes, cue alignment and QA. Returns a job: follow it with get_job, then get_results. Text "
        "is spoken as sent: write numbers and symbols as words. "
        + NAMES_AS_HINTS
        + " "
        + JOB_SIZE
        + " Options (each described in the schema): dry_run, strict_text, takes, max_retakes, and priority "
        "'interactive' for a short redo while a long batch runs. The same request gives the same job or the "
        "cached result; to hear new deliveries, name new attempts. Refusals are tool errors with a hint "
        "(VOICE_NOT_MEASURED: run measure_voice first).",
        retention=True,
    ),
}
assert tuple(TOOL_TEXTS) == TOOL_NAMES, "one text per tool, in the section 7.1 order"


def tool_description(name: str, retention: RetentionConfig, *, unbuilt: Collection[str] = NOT_IN_THIS_BUILD) -> str:
    """The published description of a tool: the mark of a tool not in this build, what it does, then the rules
    every caller must know."""
    text = TOOL_TEXTS[name]
    parts = [not_in_this_build(name)] if name in unbuilt else []
    parts += [text.does, NO_CALLER_STATE, LENGTH_IS_YOURS, RESPELLING_IS_A_HINT]
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


def _listed(tools: list[str]) -> str:
    return tools[0] if len(tools) == 1 else ", ".join(tools[:-1]) + " and " + tools[-1]


def server_instructions(unbuilt: Collection[str] = NOT_IN_THIS_BUILD) -> str:
    """The server's ``instructions``: the flow, then what decides whether real use goes well, within
    ``CLIENT_TEXT_LIMIT``. The rules every tool description states (section 7.1) are left to them, except that
    the service keeps no caller state."""
    if "design_voice" in unbuilt:
        voice = "a synthetic clip whose sha256 the operator allowlisted"
    else:
        voice = "design_voice, then a person listens and chooses a clip (or a synthetic clip the operator allowlisted)"
    parts = [
        "narration-mcp designs voices, measures them, and narrates paragraphs of cues in a chosen voice, "
        "returning QA'd, cue-aligned takes as local WAV files (path and sha256).",
        f"The flow: {voice}; measure_voice once per clip; check_text; submit_job; get_job with wait_s until the "
        "job has ended; get_results.",
        NAMES_AS_HINTS,
        VOICE_AS_KEPT,
        JOB_SIZE,
        READ_RESULTS,
        REPORT_RESOURCE,
        OPTIONS,
    ]
    missing = [name for name in TOOL_NAMES if name in unbuilt]
    if missing:
        verb = "answers" if len(missing) == 1 else "answer"
        parts.append(f"Not in this build yet: {_listed(missing)} {verb} BACKEND_NOT_INSTALLED at once.")
    parts += [
        NO_CALLER_STATE,
        "Heavy jobs share a GPU: read get_server_status (admission) first.",
    ]
    return " ".join(parts)


SERVER_TITLE: Final = "Narration"
SERVER_INSTRUCTIONS: Final = server_instructions()

RESOURCE_DESCRIPTIONS: Final[dict[str, str]] = {
    "status": "The server's status, as get_server_status returns it.",
    "design": "A design's candidates: clips, exact transcripts, seeds and profiles.",
    "measurement": "A voice's measurements, one per engine profile.",
    "job": "A job, as get_job returns it. Subscribable (2026-07-28 clients, through subscriptions/listen).",
    "job report": "A job's report, in Markdown: per segment, the suggested take, every take's verdict and its "
    "flags, and the spans to listen to first. Much smaller than get_results for a large job, but with no file "
    "paths and no full cue timings: take those from get_results or narration://takes/{take_id}.",
    "take": "A take: its delivery file, its render and its analyses.",
}


@dataclass(frozen=True, slots=True)
class PromptText:
    """A prompt's title, description, argument descriptions and message template (section 7.8)."""

    title: str
    description: str
    arguments: dict[str, str]
    template: str


def prompt_texts(unbuilt: Collection[str] = NOT_IN_THIS_BUILD) -> dict[str, PromptText]:
    """The four prompts of section 7.8, with the tools not in this build marked where a prompt names them."""
    if "audition_pronunciation" in unbuilt:
        try_variants = (
            "audition_pronunciation is not in this build yet, so to hear a respelling, submit a short segment "
            "that contains the term with that respelling as its hint, one respelling per request, and read "
            "what the recogniser heard in qa.terms;"
        )
        audition_steps = (
            "2. audition_pronunciation is not in this build yet. To hear a respelling, run submit_job with a "
            "short segment that contains the term (a sentence from the text, if there is one) and that "
            "respelling as the term's hint: one respelling per request, since a request takes each term once. "
            "The voice must be measured.\n"
            "3. Report what the recogniser heard for each (qa.terms in get_results) and give the person each "
            "take's file. "
        )
        audition_note = " audition_pronunciation is not in this build yet; the prompt hears variants with submit_job."
    else:
        try_variants = "try variants with audition_pronunciation;"
        audition_steps = (
            "2. Run audition_pronunciation with them and the voice, in a carrier sentence from the text if there "
            "is one.\n"
            "3. Report what the ASR heard for each variant. "
        )
        audition_note = ""
    if "design_voice" in unbuilt:
        design_first = (
            "design_voice is not in this build yet and answers BACKEND_NOT_INSTALLED at once; until it is, the "
            "person supplies a synthetic clip the operator allowlisted, with its exact transcript. Once it is:\n"
        )
        design_note = " design_voice is not in this build yet."
    else:
        design_first = ""
        design_note = ""
    return {
        "narrate_script": PromptText(
            "Narrate a script",
            "Narrate a script's paragraphs with a voice: measure, hint the names, check the text, submit a scene "
            "at a time, wait, collect.",
            {"voice_path": "the absolute path of the voice's WAV clip"},
            "Narrate a script with the voice whose clip is at {voice_path}. The service keeps no caller state, "
            "so send the voice (path, sha256 and transcript, copied exactly as kept with the clip; never retype "
            "the transcript) and your pronunciation hints with every call.\n"
            "1. If the clip has no measurement under the current engine, run measure_voice. It is a heavy GPU "
            "job: call get_server_status first and read admission.\n"
            "2. List every invented or unusual name in the script as a hint; the term alone is enough, and a "
            "respelling is optional. A name sent without a hint is scored as misheard words and can fail its "
            "takes (WER_HIGH).\n"
            "3. Run check_text on every paragraph's cues, with the hints. The cues must already be the words to "
            "be spoken, with numbers and symbols written as words.\n"
            "4. Resolve every text warning by editing the text; the service never rewords it. A paragraph's "
            "length is your decision: an over-long segment is warned about, not refused.\n"
            "5. Run submit_job one scene at a time (about 8 to 10 segments per job), with the takes you want and "
            "your hints; options.strict_text true refuses any text warning left instead of speaking it. A "
            "respelling is a hint to the engine, never a guarantee.\n"
            "6. Call get_job with wait_s until the job has ended, then get_results with include_words false "
            "unless you need word times. On a retryable error, wait at least retry_after_s, add your own "
            "jitter, and send the identical request again.\n"
            "7. Per segment, use suggested_take_id: takes[] also lists failed and replaced attempts. "
            + TIER_4
            + " Copy the takes you keep and check their sha256; record the take and analysis ids; report "
            "listen_first to the person, who listens and decides.",
        ),
        "resolve_flags": PromptText(
            "Resolve flags",
            "Handle each flagged take or cue of a job with a hint, new attempts, or a text edit.",
            {"job_id": "the job whose flags to resolve"},
            "Resolve the flags of job {job_id}. Read it with get_results (include_words false), or its summary "
            "at narration://jobs/{job_id}/report. For each flagged take or cue, choose one remedy:\n"
            "- WER_HIGH on a name: send the name as a hint. The term alone is enough, and asr_aliases can list "
            "how the recogniser spelled it (qa.terms, or include_transcripts true). A hint without a respelling "
            "leaves the engine text as it was, so the cached takes are scored again without rendering anew;\n"
            "- a pronunciation hint: a respelling is a hint to the engine, never a guarantee; " + try_variants + "\n"
            "- new attempts for the segment: name attempts not rendered yet, such as [3, 4];\n"
            "- an edit of the text: the service speaks text as sent.\n"
            "Then send submit_job again with the changes; cached work is reused, and the service keeps no "
            "caller state, so send the whole request. Tell the person what you changed and what to listen to "
            "first (listen_first).",
        ),
        "design_narrator_voice": PromptText(
            "Design a narrator voice",
            "Write positive-only voice descriptions, design candidates, shortlist them, and ask a person to choose."
            + design_note,
            {"brief": "what the narrator should sound like, and for what"},
            "Design a narrator voice for this brief: {brief}\n"
            + design_first
            + "1. Write two or three voice descriptions that name only the qualities wanted. A negation such as "
            "'not gravelly' tends to come out as that quality.\n"
            "2. Run design_voice with takes 3 for each description. It uses the shared GPU: call "
            "get_server_status first.\n"
            "3. Read each candidate's profile (pitch, speaking rate, brightness, the pictures) to shortlist.\n"
            "4. Ask the person to listen to the shortlisted clips and choose; you cannot hear them, and the "
            "service records no choice. Copy the chosen clip out of the store into your own folder (the store's "
            "copy is removed after the retention period), and keep that path, its sha256 and its exact "
            "transcript: every later call needs them, the transcript copied, never retyped.",
        ),
        "add_pronunciation": PromptText(
            "Add a pronunciation",
            "Draft respelling hints for a term and hear them." + audition_note,
            {"term": "the word or words as they appear in the text"},
            "Help pronounce '{term}'.\n"
            "1. Draft up to four respellings (for example 'Oss-a-veen' for 'Ossavine').\n"
            + audition_steps
            + "A respelling is a hint to the engine, never a guarantee, and whoever owns the text decides by ear; "
            "the service records no choice.\n"
            "4. Use the chosen respelling as a hint in check_text and submit_job. If the term as written already "
            "sounds right, send the term alone: QA still needs it as a hint.",
        ),
    }


PROMPT_TEXTS: Final[dict[str, PromptText]] = prompt_texts()
