# Design log

This file is a log, so it records history. `design.md` states the design as it is now, and a log holds how
it got there. It has three parts:

1. **Revision history**: what each revision of `design.md` changed.
2. **Design changes**: how a change to the design is proposed, approved and applied, and the table of every
   change (DC-n) so far.
3. **Owner decisions**: each decision the owner made that the design states as a rule, with its date and
   the section that states it now.

Add to this file when the design changes. `design.md` itself is rewritten to the present and never gets a
"changed in revision n" note.

---

## Revision history

The design's current revision number is in the header of `design.md`.

- *Revision 2 (the same day) applied the owner's Qwen-only decision and the consuming flow's
  requirements (`evolution-simulator/…/logbook/specs/story-narration.md`, R1–R13).*
- *Revision 3 applied an independent review of revision 2 and evidence from the names probe.*
- *Revision 4 (2026-09-26) applied the owner's separation principle (the service knows nothing of how it
  is used; see Owner decisions below) and the requirements as revised under it (`story-narration.md` at
  commit `bf19d0a`): text is sent in spoken words, the normaliser and the flow's vocabulary left the
  service, exact spans replaced the number check.*
- *Revision 5 (2026-09-26) applies the owner's decisions after a blind review of revision 4: **the
  service is stateless**. It keeps no projects, scripts, cuts, choices of take, pronunciation lists or
  voices. A caller sends everything with each request, and gives its voice as a clip by location. It
  also applies the review's accepted findings. Section 21 maps the requirements again.*
- *Revision 5.1 (the same day) records the owner's answers to the last open questions: negated voice
  descriptions are warned about, not refused (Q6); the design assumes no install location, and reads a
  clip from any absolute local path (Q9); no stronger speaker model for now (Q11); the wav2vec2 aligner
  is the default, with Qwen3-ForcedAligner tested against it in Phase 0 (Q18); a written-text
  normaliser waits for a later phase (Q20); and no listening model in v1, so a voice profile is
  measurements and pictures only (Q22). No owner question is left open; Phase 0 is next.*
- *Revision 5.2 (2026-09-26) applies three design changes the owner approved (see Design changes below) for
  a public project. **DC-1**: no GPL code, so the voice profile's pitch comes from `librosa.pyin`, its
  harmonics-to-noise ratio from Boersma's autocorrelation method, and CPPS replaces jitter and shimmer
  (sections 3.6, 4, 18). **DC-2**: a backoff contract for consumers: `retry_after_s` on every retryable
  error, the retryable codes `QUEUE_FULL` and `RATE_LIMITED`, `poll_after_s` from `submit_job` and
  `get_job`, and `admission` in `get_server_status` (sections 7.2–7.4, 7.6, 14). **DC-3**: the canary is
  designed on the installing machine from shipped text, not shipped as audio (sections 6, 10.1, 15).*
- *Revision 5.3 (2026-09-26) applies five changes the owner approved (see Design changes below), found by the
  first build wave's evidence and reviews. **DC-5**: the NaN/DC signal check gets its flag code,
  `SIGNAL_INVALID` (sections 11.1, 14). **DC-6**: an `idempotency_key` reused for a different request is
  refused (sections 7.3, 14). **DC-7**: the number reader's version 2 reads each side in phrases split
  at punctuation and folds curly apostrophes (section 11.3, App. B). **DC-8**: the default loudness target
  is −20 LUFS, since −16 was never reached under the true-peak ceiling on real output (sections 13, 16,
  Q3). **DC-9**: a take with less silence than `pad_s` at an end keeps what it has, and nothing is added
  (section 13). Two clarifications come with them: the delivery key names the post-processing rules'
  version, and a take with no measurable loudness reports null (sections 10.2, 13, App. B).*
- *Revision 5.4 (the same day) applies two more owner decisions from the review of the delivery code.
  **DC-8 is amended** to −23 LUFS, EBU R128's pair with the −1 dBTP ceiling: on the same 48
  paragraphs, −22 is reached by all of them and −20 by 40. **DC-10**: the trim measures frames with
  the take's mean removed and never lets the speech threshold fall below −70 dBFS, so a DC offset or a
  near-silent take no longer defeats it; QA warns on a DC offset above 0.001 (sections 11.1, 13, 16).*
- *Revision 5.5 (the same day) applies the owner's **DC-4**: every `synthesize` and `design` call passes
  its own `max_new_tokens`, min(8192, max(128, ceil(2.5 × characters))), so a runaway render costs about
  a minute of GPU, not twenty, and `TOKEN_CAP_HIT` can fire. The cap only truncates (measured), so no
  render that ends under it changes (sections 6, 10.1, 11.1, 16, App. A). With it, from the alignment
  work: **DC-12**, a cue whose text has no word the aligner can place is not a retake trigger, since no
  retake can place it; the aligner's confidence thresholds become configuration, and its method id
  covers them (section 11.2). A take under the loudness gate keeps its measured true peak (section 13).*
- *Revision 5.6 (the same day) describes number reader `@2` as built: clock times, a leading minus, the
  degree sign, money, and the comma trade-off (section 11.3; DC-7).*
- *Revision 5.7 (the same day) records spike (d): the clone path is `bit_exact` on the pinned stack, the
  one exemption from deterministic algorithms (the voice prompt's encode), a worker's own WAV writer, and
  the tier decided by `engine pin` on the installing machine (section 10.1; ADR 0002).*
- *Revision 5.8 (the same day) applies the owner's **DC-13**: the service ships its own default design
  text, written for it in place of the bake-off's reference text, since the bake-off's texts are private
  (section 16; it is also the canary's text, DC-3). Section 3.1 and the examples in section 7.3 and
  Appendices A and B follow.*
- *Revision 5.9 (the same day) describes the cue aligner as built (section 11.2).
  - DC-11: a word that cannot be spelled, such as a number in digits, becomes a wildcard token that
    absorbs its speech.
  - A cue next to an unplaceable cue's speech snaps only to a pause within reach of its own words.
  - Confidence counts letters and apostrophes.
  - The method id also covers the reaches, the wildcard and the versions of the rules.*
- *Revision 5.10 (the same day) names the four reasons a `CUE_UNALIGNED` can carry
  (section 11.2 step 7), as contracts 1.6.3 define them.*
- *Revision 5.11 (2026-09-27) describes the daemon as built (sections 4, 4.1 and 17).
  - The daemon runs as `pythonw.exe` on Windows. The daemon and the workers start with `-P` and without
    `PYTHONPATH`, `PYTHONHOME` or `PYTHONSTARTUP`; a worker starts in its own project folder.
  - A daemon honours every stop posted after it was launched, and only those.
  - A daemon about to exit checks once more for work, so no job is stranded.*
- *Revision 5.12 (the same day) states when a job's `outcome` is `needs_attention` (sections 7.4 and 8):
  only when some segment's suggestion is a verdict fail, or some segment has no take; warnings alone
  leave `all_passed`.*
- *Revision 5.13 (the same day) applies DC-14 and DC-15 (section 11.1 steps 2 and 8) and the QA group's
  measured memory (section 4): Whisper decodes with five beams, not conditioned on the previous window;
  audio over 60 s is embedded in windows of at most 60 s; the QA group needs about 11.5 GB.*
- *Revision 5.14 (the same day) applies DC-16 (section 6, EngineProfile): `vram_need_mb` is recorded in
  the engine profile but not hashed; it changes no audio. The owner approved it.*
- *Revision 5.15 (2026-09-28) applies two changes the owner approved, and describes the daemon's
  detachment as built.
  - **DC-17**: `narration-admin voices allow <clip.wav>` adds a clip designed elsewhere to `[voices]
    allow_sha256` once the operator confirms that it is synthetic, and `voices list` shows the list. It is
    an operator command, never an MCP tool (sections 7.1, 16, and 17 items 4 and 10).
  - **DC-19**: `PACE_FAST` and `PACE_SLOW` warn only (section 11.1 already had no slow fail); pace never
    fails and never triggers a retake. The QA profile becomes `default.v4` (sections 11.1, 14, 16 and
    Appendix B).
  - **The daemon runs only once it is in no Job Object at all** (the lead's rule, PR #37; spike k). It is
    created suspended and resumed only when Windows says it is in no job; otherwise the front-end
    returns `DAEMON_UNAVAILABLE`. On a host whose jobs forbid breakaway, the route is `[daemon] autostart
    = false` and a daemon started by hand with `daemon start --foreground` (sections 4.1, 7.1, 14 and
    16). Section 7.1 also lists `narration-admin render`, as built.*
- *Revision 5.16 (2026-09-28) describes what was built since revision 5.15.
  - **A job whose daemon has gone** (PR #41). `get_job`, and `cancel_job` for a job it leaves
    `cancelling`, ask for a daemon when none serves an active job. A launch is recorded in
    `run\launch.json`. For 90 s after it (BELIEVE), while `run\daemon.json` shows no start since, no
    other is asked for as long as the launched process runs; once that process has gone, `get_job` answers
    `DAEMON_UNAVAILABLE` with the daemon's log until the window ends, also after a daemon that started and
    then failed. A stop answered `stopped: true`, by `daemon stop` or by `install` after a repair (the
    lead's decision), holds for the jobs queued before it (sections 4, 4.1, 7.4, 7.6, 14 and 15).
  - **A transcript that differs from the measured one** only in whitespace or punctuation is named in
    `VOICE_NOT_MEASURED`, with the rewrites that give the measured one (PR #41; sections 3.2, 7.3 and
    14).
  - **`narration-admin failures`** (PR #45) lists every take that failed QA or that a retake replaced,
    across jobs, with its reasons; it only reads the store, and is never an MCP tool. `report.md` gains a
    Failures section by the same rule, and `gc` lists the failed takes it would remove (sections 7.1,
    11.1, 15 and 17 item 10).
  - **`[qa] profile` is informational** (PR #46): every take is scored with the QA profile the build
    pins, and `doctor` warns when the line differs (section 16).
  - **Corrections to stale text.** `run\daemon.json` stays behind with state `stopped` (sections 4.1 and
    15). On a designed candidate, `TOKEN_CAP_HIT` and `WER_HIGH` fail and never trigger a retake (section
    14). Appendix B's QA metrics show `clipping_fraction`. Sections 0 and 20 point to section 7.1's list
    of operator commands.
  - **`CLIP_TOO_LONG`** (WP34, PR #39) is in the flag table: a designed candidate longer than `[limits]
    max_clip_seconds` fails, and never triggers a retake (section 14).
  - **DC-18** (the owner's decision D2; built by WP47, PR #44): pace is spoken characters per second of
    speaking time, the voiced span less every pause of 0.25 s or more inside it (ASSUME). The measured
    pace model and QA's pace check use this one rule. The measurement key names the pace method, and the
    measurement record and its key move to `/v2`, so a voice measured before is measured again from its
    cached renders. The QA profile becomes `default.v5`; `PACE_FAST` and `PACE_SLOW` still only warn
    (DC-19). Sections 0, 3.2, 6, 7.5, 7.6, 10.2, 11.1, 12, 14, 16 and Appendix B.
  - **DC-20** (the lead's gap-fill, approved 2026-09-28; the owner may overrule): pace is judged against
    the voice's flat level, and no trend slope is extended or extrapolated. Every ladder rung is judged
    against the band level (the median of the band rungs' medians) × (1 + tol); QA's expected pace
    follows the curve inside its range and holds its end values flat outside it (sections 3.2, 11.1 step
    9 and 21).*
- *Revision 5.17 (2026-09-29) records why the daemon stopped (WP51; a contract change the lead approved,
  contracts 1.6.11). A `stopped` `run\daemon.json` keeps its daemon's start time and says why it stopped
  (`stop_reason`: `operator`, `idle`, `interrupted` or `error`). A queued job waits after a stop only when
  the reason is `operator`; the 30 s timing rule stays only for a status an older daemon wrote. A launch
  counts as started once any daemon records a start at or after it, running or stopped, so a daemon that
  started, served and then failed inside the 90 s window is no longer reported as a failed start
  (sections 4.1 and 15).*
- *Revision 5.18 (2026-09-29) states what `audition_pronunciation` does (WP35), which revision 5 left at
  "takes plus what the ASR heard" (section 7.6; contracts 1.6.12). Each variant is the carrier (or the term
  alone) with the variant's respelling as its hint, rendered, keyed, seeded and cached as a `submit_job`
  take of that text and hint, and retaken as one. Its verdict has QA's checks except the speaker and pace
  checks, since the voice need not be measured, and the respelling counts as the term. Each take's speaker
  similarity to the clip is reported beside the verdict, never in it.*

---

## Design changes

A change to `docs/design.md` is proposed here first, as a new row with status `proposed`. Build to it only
once it is `approved`. The owner approves a change; the lead may also approve a gap-fill that the owner
can overrule, and the row says so. The lead then applies an approved change to the design as a new
revision, with a line in the revision history above, before or together with the WP that builds it.
Status values: `proposed` · `approved` · `applied` (in the design) · `rejected`.

| # | Change | Design sections | Status | Built in |
|---|---|---|---|---|
| DC-1 | **No GPL in the voice profile.** f0 comes from `librosa.pyin` (ISC). HNR uses Boersma's (1993) autocorrelation method, the one Praat implements. **CPPS** (smoothed cepstral peak prominence, Hillenbrand 1994) replaces jitter and shimmer: those are defined on sustained vowels and are noisy on running narration, while CPPS holds up on connected speech. Each measure is documented as what it is, not as Praat's. Validation uses synthetic signals with known values (pulse trains at a known f0, noise at a known SNR). No GPL code anywhere in the repo, tests included. Note: librosa depends on `soxr` (LGPL-2.1+), which is acceptable as a separately installed, replaceable package; it goes in the licence audit. | §3.6, §4 (QA worker), §18 | **applied** (rev 5.2) | WP22, WP34 |
| DC-2 | **A backoff contract** (Q8). Retries are already safe (content-hash ids, `idempotency_key`, identical request = same job); this tells a consumer *when* to come back. (1) The Error fragment gains an optional **`retry_after_s`**, set on every retryable error: the server's minimum wait, like HTTP `Retry-After`. `details` carries the facts behind it, e.g. `GPU_UNAVAILABLE {free_mb, need_mb, waited_s}`. (2) `LIMIT_EXCEEDED` stays for request-size limits (not retryable), and two retryable codes are added: **`QUEUE_FULL`** (`retry_after_s` from the queue's drain estimate) and **`RATE_LIMITED`** (`retry_after_s` until the submit window frees). (3) `submit_job` and `get_job` return **`poll_after_s`**, the earliest poll worth making, longer while `waiting_for_gpu`, beside the existing `eta_s`, `queue_position` and `phase`. (4) `get_server_status` gains **`admission`**: {accepting, queue {length, max, est_drain_s}, rate {remaining, resets_in_s}, gpu {in_use, holder, free_mb, need_mb by group, waiting_since}}, so a consumer can decide before submitting. (5) Tool descriptions state the rule: wait at least `retry_after_s`, add your own jitter, then resend the identical request, which is deduplicated. The service suggests; the consumer decides. | §7.2 (Error), §7.3, §7.4, §7.6, §14 | **applied** (rev 5.2) | WP01, WP31, WP36 |
| DC-3 | **The canary is designed on the installing machine**, not shipped as audio. §10.1 ships a canary clip, but the design promises nothing across a GPU, driver or CUDA change, so a shipped raw hash would not match on anyone else's machine, and shipping audio breaks P6. `narration-admin engine pin` designs the canary from a fixed description, text and seed (shipped as text) and stores its hash and embedding per engine profile. It still checks what §10.1 says: this engine, on this machine, over time. | §10.1, §15, §6 (EngineProfile) | **applied** (rev 5.2) | WP32, WP37 |
| DC-4 | **The effective `max_new_tokens`** (§1.3 item 1). `load` pins 8192, the value all the evidence used, as the ceiling, and every `synthesize` and `design` call passes its own cap, `min(8192, max(128, ceil(2.5 × len(text))))`, computed by the daemon from the text the call speaks; the factor and floor are `[engines.*]` keys hashed into the engine profile. WP20 measured that the cap only truncates, so no render that ends under its cap changes, and speech ran 0.70–0.82 frames per character (ADR 0003). A runaway then costs about a minute of GPU instead of about 22, and `TOKEN_CAP_HIT` can fire. The owner chose this over keeping 8192 or a fixed 2048 (2026-09-26). | §6, §10.1, §11.1, §16, App. A | **applied** (rev 5.5) | WP16 (fake), WP20, WP31, WP32 |
| DC-5 | **A flag code for the NaN/DC signal check.** §11.1 step 1 requires the check, but §14 has no code for it. `SIGNAL_INVALID`: fail for non-finite samples (retake at fail, as a fresh seed may cure it), warn for a DC offset. A lead gap-fill from WP14's review; it changes no stated behaviour. The owner approved it. | §11.1, §14 | **applied** (rev 5.3) | WP14 |
| DC-6 | **`idempotency_key` reused for a different request is refused.** The design says the key deduplicates retries, and a retry is the same request, but it leaves the other case open. Returning the old job would tell the caller that a different request had been queued. So an active job of the same kind with the same key and a different `request_sha256` is `INVALID_ARGUMENT` on `idempotency_key`. A lead gap-fill from WP12's review. The owner approved it. | §7.3, §6 (Job) | **applied** (rev 5.3) | WP12, WP36 |
| DC-7 | **The number reader, version 2** (`whisper-english-normalizer+nought@2`). The WP14 review found two ways the vendored reader fails a perfect take. (1) Whisper's normaliser strips commas before it reads number words, so "two thousand, forty" becomes 2040 in the transcript while the span "forty" reads 40. (2) It handles only the straight apostrophe, so a possessive written with ’ and the same word written with ' differ. Version 2 reads each side in phrases split at punctuation, and folds ’ ‘ ʼ to '. The version is in the analysis key, so no cached verdict of version 1 is reused. Proposed by the lead; the owner approved it. | §11.3, App. B | **applied** (rev 5.3) | WP14 |
| DC-8 | **The default loudness target is −23 LUFS**, not −16 (first −20 in rev 5.3, then −23 in rev 5.4). WP13 measured the bake-off's 48 real clone paragraphs: with the −1.0 dBTP ceiling and no limiter, every take stayed under −16 (−21.3 / −18.8 / −16.7 LUFS, min / median / max; peak-to-loudness ratio 15.7–20.3 dB), so takes of one script differed by up to 4.6 LU. At −20, 40 of the 48 reach the target exactly, the waveform is never altered, and a caller adds gain in its mix. `target_lufs` stays configurable. The owner chose this over keeping −16 or adding a limiter (2026-09-26); after the WP13 review measured that −22 is reached by all 48 and −20 by only 40, the owner moved it to −23, EBU R128's pair with −1 dBTP. | §13, §16, Q3 | **applied** (rev 5.3, 5.4) | WP13 |
| DC-9 | **Less silence than `pad_s` is kept as found.** 23 of 48 real takes had less than 0.08 s of silence at the tail (one had none), and 5 at the head. The delivery keeps what exists and adds no digital silence, so the length is raw − max(head − pad, 0) − max(tail − pad, 0), and `head_s`/`tail_s` report the silence found. §13's formula assumed at least `pad_s`. Proposed by the lead (WP13); the owner approved it. With it, two clarifications: the delivery key names the post-processing rules' version (`narration.post/1`), and a take with no measurable loudness reports null. | §13, §10.2, App. B | **applied** (rev 5.3) | WP13 |
| DC-10 | **A trim rule that a DC offset or a near-silent take cannot defeat.** The WP13 review showed two synthetic failures of §13's rule: a DC offset of 0.001 made every frame speech (nothing trimmed, and a 3 s internal gap read as 0 s, hiding `SILENCE_LONG`), and a take under 5 % speech (a near-silent hung tail) was never trimmed. The trim now measures frame RMS with the take's mean removed (for measurement only) and floors the speech threshold at −70 dBFS (`trim_floor_dbfs`). QA's DC-offset warning drops from 0.02 to 0.001. Real output had |DC| ≤ 0.00002, so normal takes are unchanged. The owner approved it (2026-09-26). | §11.1, §13, §16 | **applied** (rev 5.4) | WP13, WP14 |
| DC-11 | **A wildcard for a token the aligner cannot spell.** §11.2 step 1 says the words around a left-out token (a number in digits) "still align". WP15 measured otherwise on real takes: in all six bake-off takes a paragraph that ends in two numbers written in digits puts the next cue 1.3–3.4 s early, and three of the six raise no flag. Spike (b) put a wildcard token in the gap on that paragraph, and the worst word error fell from 89 frames to 9. Proposed: such a token becomes the aligner's wildcard instead of being left out. Approved by the lead as a gap-fill (the owner may overrule), on condition that WP15 measures it fixes that boundary on all six takes and moves no other; the design text is applied when WP15 merges with that evidence. | §11.2 | **applied** (rev 5.9); met: its two conditions held on the evidence (spike (b), `spikes/b-forced-align-cpu/`) | WP15 |
| DC-12 | **A cue no retake can place is not a retake trigger.** A cue whose text has no word the aligner can place gets `CUE_UNALIGNED`, a retake trigger, and fails the same way on every retake. It now carries `details.reason` = `no_alignable_words` and does not trigger a retake; it stays in listen-first. A lead gap-fill from WP15 (the owner may overrule); `codes.is_retake_trigger` takes the flag's details. | §11.1, §11.2 | **applied** (rev 5.5) | WP15, WP31 |
| DC-13 | **The service's own default design text.** The §16 default `design_text`, which is also the canary's design text (DC-3), was the bake-off's reference text. The bake-off's texts are private (AGENTS.md §1 rule 1), and this is a public project, so the service now ships a text written for it: "Good bread asks for patience: the dough is mixed, folded and left to rise through the morning. When the loaves come out golden and crisp, a gentle warmth fills the whole kitchen." It has two sentences and 32 words, and no digits or names. A clip designed earlier keeps the transcript `design_voice` returned for it. The owner decided (2026-09-26). | §3.1, §7.3 (example), §16, App. A, App. B | **applied** (rev 5.8) | WP18 |
| DC-14 | **Whisper decodes with five beams, not conditioned on the previous window.** §11.1 step 2 says greedy decoding conditioned on the previous text. WP22 measured that it loops on 2 of the 6 bake-off takes (WER 0.40); five beams without conditioning reproduce the bake-off's WER exactly on all six (ADR 0004, `spikes/acceptance-wp22/decoding.json`). A lead gap-fill (the owner may overrule), since the design's own acceptance is to match the bake-off. | §11.1 | **applied** (rev 5.13; approved by the lead, 2026-09-27) | WP22 |
| DC-15 | **A long clip is embedded in windows.** WavLM's memory grows with the square of the length (0.6 GB at 30 s, 8.4 GB at 119 s; WP22, spike h), and a segment longer than the voice's limit is rendered and warned about, never refused. Audio longer than **60 s** is embedded as equal windows of at most 60 s; the embedding is the mean of the L2-normalised window embeddings, normalised again. Audio of 60 s or less is embedded in one pass, as before. First approved at 30 s; WP22 then measured that 30 s windows move three of the bake-off's voicelock rows (clips of 30.6–37.0 s) out of tolerance, while 60 s windows reproduce all 52 rows, and their peak (2.2 GB above the models) stays under transcription's. A lead gap-fill (the owner may overrule). | §11.1, §4 (QA worker) | **applied** (rev 5.13; approved by the lead, 2026-09-27, and amended to 60 s the same day) | WP22 |
| DC-16 | **`vram_need_mb` is recorded in the engine profile but not hashed.** §6 lists it among the profile's fields, so it entered the profile hash, which names every render, take and measurement. It changes no audio: it only tells the scheduler how much free memory to wait for. Hashing it would make a refined memory estimate invalidate every cached take and every voice measurement (20–50 GPU minutes each). It joins the canary and `snapshot_dir` among the fields that are kept but not hashed. Decided before anything is pinned. A lead gap-fill (the owner may overrule). | §6 (EngineProfile), §10.1 | **applied** (rev 5.14; the owner approved, 2026-09-27) | WP32 |
| DC-17 | **`narration-admin voices allow <clip.wav>` and `voices list`.** The owner asked (2026-09-27) for a command that adds a clip designed elsewhere to `[voices] allow_sha256`, instead of hashing it and editing `narration.toml` by hand. It records the owner's own configuration of the machine and approves no caller's work, so §7.1's "none of them approves anything" still holds. **Operator-only, never an MCP tool:** the allowlist is the synthetic-voices gate (§17.4), and a tool would let any caller allowlist a recording of a real person. | §7.1 (operator CLI), §16, §17.4, §17 item 10 | **applied** (rev 5.15; the owner approved, 2026-09-27) | WP45 |
| DC-18 | **Pace is judged in spoken characters per second of speaking time**, with pauses of 0.25 s or more left out (decision D2). The words-per-minute curve followed the corpus's word lengths, and a span that includes the pauses between sentences makes a one-sentence segment look fast. The measured pace model (trend, curve, `tol`) and QA's pace check both use the one rate function; the measurement key gains a pace-method version, so a voice measured before is measured again from its cached renders. | §3.2, §11.1, §16 | **approved** (owner, 2026-09-27); applied (design 5.16, PR #47; code PR #44) | WP47 |
| DC-19 | **`PACE_FAST` warns only; it never fails and never triggers a retake** (QA profile `default.v4`). In the first real narration session, 24 takes failed `PACE_FAST` alone, with WER 0 and high speaker similarity, and the owner listened and found none too fast. §11.1's pace row loses its fail column (`PACE_SLOW` already had none), §14's `PACE_FAST` row becomes warn only, and §16's `[qa] profile` example becomes `default.v4`. | §11.1, §14, §16, App. B | **applied** (rev 5.15; the owner approved, 2026-09-27) | PR #42 (WP47 hotfix) |
| DC-20 | **Pace is judged against the voice's flat level; no trend slope is extended or extrapolated.** In characters per second of speaking time (DC-18) a voice's rate is flat, so a slope fitted to four band rungs is noise (±0.3 per 100 characters at 3% per-rung spread). Extending it wrongly stopped the ladder or let rushing pass, and extrapolating it skewed QA at both ends of the length range (PR #44's rate-model review, reproduced with the branch's own functions). Every rung is judged against the band level (the median of the band rungs' medians) × (1 + tol); QA's expected rate follows the measured curve inside its range and holds its end values flat outside it. A lead gap-fill (the owner may overrule). | §3.2, §11.1 step 9 | **approved** (lead, 2026-09-28); applied (design 5.16, PR #47; code PR #44) | WP47 |

---

## Owner decisions

Each decision the owner made that `design.md` states as a rule. The design states the rule in the present
tense, and this table says who decided it, when, and where. Decisions that went through the Design changes
table above (DC-n) are listed there with their dates and are not repeated here, except where the design once
named the owner beside the rule. The question numbers (Q1–Q22) are those of the design's section 19.

| Date | Decision | Stated in |
|---|---|---|
| 2026-09-25 | Qwen3-TTS is the only speaking engine. Only the Qwen Base clone engine (and VoiceDesign, for the design phase) is built. The Kokoro adapter, Higgs and Fish, and the licence gate for them are out of scope. The adapter boundary may stay, but nothing else is built behind it. | Header; section 4 (adapter seam) |
| 2026-09-26 | **The service knows nothing of how it is used.** It narrates text in a voice and reports what it did, as it would for any caller. What the text says, how its numbers and names are read, when a job may run, how long a piece must be, which voice to use, and who approves the result are all the caller's. (Revision 4.) | Header; sections 9.2, 19 (Q20) and 21 |
| 2026-09-26 | **The service is stateless.** The caller keeps its scripts, the takes it chose, its approvals, its pronunciation list and its voice. Each request carries what the service needs, and the answer is a function of the request. The service keeps only its own things: a content-addressed cache of work done, the job queue, its engine pins and measurements of voices. (Revision 5.) | Header; sections 0 (item 2), 2 and 21 |
| 2026-09-26 | **Voices by location.** The design tool returns each candidate's clip path, fingerprint (sha256) and exact transcript, and a generation call gives the same three. Agents pass locations, never file contents. There is no lock and no voice registry. (Revision 5.) | Header; sections 3, 7.3 and 17 |
| 2026-09-26 | **Synthetic voices only.** A clip is accepted only if the service designed it or its fingerprint is in the operator's configuration (`allow_sha256`). The rule is a safety rule of the service and does not depend on any use. | Header; section 17 item 4 |
| 2026-09-26 | **Exact spans confirm the value of a number, not its wording.** "Thirty-two hundred" and "three thousand two hundred" both pass as 3200. (R14.) | Sections 11.3 and 21 |
| 2026-09-26 | **An over-long paragraph is warned about, never refused** (Q17; `SEGMENT_TOO_LONG`). Length is the caller's decision. (R8.) | Sections 0 (item 7), 3.2, 19 and 21 |
| 2026-09-26 | **A voice profile** (measurements and pictures) comes with every designed candidate and from its own tool. A description in words from a listening model is not in v1 (Q22): the profile's numbers and pictures serve the shortlist, and the voice is chosen by ear. | Sections 3.6, 19 (Q22) and 20 |
| 2026-09-26 | **A negated voice description is warned about, not refused** (Q6), using a plain word list and no language model. | Section 3.5 |
| 2026-09-26 | **No install location is assumed** (Q9). The operator sets up the service's folder, and every path in section 16 is a placeholder. A voice clip or audio file is read from any absolute path on a local drive, with no list of allowed folders. | Sections 16 and 17 item 3 |
| 2026-09-26 | **Cue alignment** (Q18): `facebook/wav2vec2-large-960h-lv60-self` on the CPU is the default, chosen for fit (licence, no new toolchain, no GPU, no pronunciation dictionary) rather than proven accuracy. Qwen3-ForcedAligner-0.6B replaces it only if it is clearly better on this service's audio at boundaries with no pause. | Sections 4 (QA worker), 11.2 and 19 (Q18) |
| 2026-09-26 | **No stronger speaker-similarity model for now** (Q11). Thresholds come from each voice's measurement. | Sections 19 (Q11) and 20 |
| 2026-09-26 | **A written-text normaliser is a later phase** (Q20). v1 speaks text as sent and refuses `text_mode: "written"`. | Sections 9.2 and 19 (Q20) |
| 2026-09-26 | **Loudness** (Q3, DC-8): the default target is −23 LUFS with a −1 dBTP ceiling, not −16; `target_lufs` stays configurable. | Sections 13 and 16 (the Design changes table has the evidence) |
| 2026-09-26 | **A per-call `max_new_tokens`** (DC-4; ADR 0003). | Sections 10.1 and 16 (the Design changes table has the rule) |
| 2026-09-26 (by revision 5.1) | **Measuring before generating** (Q21) is its own step, and a request with an unmeasured clip is refused with a hint. It was settled as the design recommended; the owner may reverse it. | Section 19 (Q21) |
