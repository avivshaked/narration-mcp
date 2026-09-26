# Narration MCP server: design

*Status: revision 5.7 (2026-09-26); being implemented (see `plan.md`). Written 2026-09-25.*

*This is the repository copy of the design, and the source of truth. Revision 5.1 differed from the
bake-off's original only in two example paths (section 7.3 and Appendix A) and in this note. The evidence
files it cites (`refs/`, `eval/`, `outputs/`, `scripts/`) belong to the private bake-off that preceded this
project, and are not published. Changes are proposed and approved in `plan.md` section 1.5 and then
applied here, each listed in the revision history below.*

*Revision history:*
- *Revision 2 (the same day) applied the owner's Qwen-only decision and the consuming flow's
  requirements (`evolution-simulator/…/logbook/specs/story-narration.md`, R1–R13).*
- *Revision 3 applied an independent review of revision 2 and evidence from the names probe.*
- *Revision 4 (2026-09-26) applied the owner's separation principle (below) and the requirements as
  revised under it (`story-narration.md` at commit `bf19d0a`): text is sent in spoken words, the
  normaliser and the flow's vocabulary left the service, exact spans replaced the number check.*
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
- *Revision 5.2 (2026-09-26) applies three design changes the owner approved (`plan.md` section 1.5) for
  a public project. **DC-1**: no GPL code, so the voice profile's pitch comes from `librosa.pyin`, its
  harmonics-to-noise ratio from Boersma's autocorrelation method, and CPPS replaces jitter and shimmer
  (sections 3.6, 4, 18). **DC-2**: a backoff contract for consumers: `retry_after_s` on every retryable
  error, the retryable codes `QUEUE_FULL` and `RATE_LIMITED`, `poll_after_s` from `submit_job` and
  `get_job`, and `admission` in `get_server_status` (sections 7.2–7.4, 7.6, 14). **DC-3**: the canary is
  designed on the installing machine from shipped text, not shipped as audio (sections 6, 10.1, 15).*
- *Revision 5.3 (2026-09-26) applies five changes the owner approved (`plan.md` section 1.5), found by the
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

*Section numbers are stable, because `story-narration.md` cites them. Section 21 maps each requirement
to what changed.*

*Protocol basis: MCP specification revision **2026-07-28**, the revision marked "current" on
modelcontextprotocol.io on this date. The official Tasks extension (`io.modelcontextprotocol/tasks`,
draft) is optional and not required (section 5).*

This is a local MCP server that lets another AI agent (the **caller**, e.g. the agent that runs the
evolution-simulator story-film flow) get narration audio in **one fixed voice**, and the time of every
cue within that audio. It is built on **Qwen3-TTS 1.7B**: voices are designed with VoiceDesign, and
spoken by the Base model cloning a reference clip that the caller keeps and sends.

> **Owner principle, 2026-09-26: the service knows nothing of how it is used.** It narrates text in a
> voice and reports what it did, as it would for any caller. What the text says, how its numbers and
> names are read, when a job may run, how long a piece must be, which voice to use, and who approves the
> result all belong to the caller.

> **Owner decisions, 2026-09-26 (revision 5).**
> - **Stateless.** The caller keeps its own state: its scripts, the takes it chose, its approvals, its
>   pronunciation list and its voice. Each request carries what the service needs, and the answer is a
>   function of the request. The service keeps only its own things: a cache of work already done
>   (content-addressed, for speed), the job queue, its engine pins, and measurements of voices.
> - **Voices by location.** The design tool returns each candidate's clip path, fingerprint (sha256)
>   and exact transcript. A generation call gives the same three. Agents pass locations, never file
>   contents. There is no lock and no voice registry.
> - **Synthetic voices only.** A clip is accepted only if the service designed it, or the owner has
>   listed its fingerprint in the configuration.
> - **Exact spans confirm the value of a number**, not its wording (section 11.3).
> - **An over-long paragraph is warned about, never refused** (section 3.2).
> - **A voice profile** (measurements and pictures) comes with every designed candidate and from its own
>   tool. (A description in words, by a listening model, was dropped from v1 in revision 5.1, Q22.)

> **Owner decisions, 2026-09-26 (revision 5.1).**
> - **A negated voice description is warned about, not refused.** A plain word list finds it, with no
>   language model (section 3.5).
> - **No install location is assumed.** The owner sets up the service's folder; every path in section
>   16 is a placeholder. A voice clip or audio file is read from **any absolute path on a local drive**
>   (section 17).
> - **Cue alignment:** `facebook/wav2vec2-large-960h-lv60-self` on the CPU is the default, chosen for
>   fit (licence, no new toolchain, no GPU, no pronunciation dictionary) rather than proven accuracy.
>   Phase 0 also tests Qwen3-ForcedAligner-0.6B, which replaces it only if clearly better on this
>   service's audio at boundaries with no pause (Q18, section 11.2).
> - **Not now:** a stronger speaker-similarity model (Q11), and a listening model that describes a voice
>   in words (Q22). **A later phase:** a written-text normaliser (Q20).

> **Owner decision, 2026-09-25:** Qwen3-TTS is the only speaking engine. Build the Qwen Base clone
> engine (and VoiceDesign for the design phase) only; the Kokoro adapter, Higgs/Fish and the licence
> gate for them are out of scope. The adapter boundary can stay, but nothing else is built behind it.

Labels used throughout: **KNOW** means measured in this repository, with the script or output saved
here. **BELIEVE** means a strong expectation to verify in Phase 0. **ASSUME** marks a placeholder number
or decision.

**v1 scope in one sentence.** Design a voice; measure the clip the caller chooses; speak paragraphs of
cues, sent in spoken words, in takes that are QA'd and cue-aligned, with exact lengths, hashes and an
echo of what was spoken. Fit is only *reported*, when a budget is given. Stitching, fit remedies and a
written-text normaliser are later phases (sections 9.2, 13.1 and 20).

---

## 0. Decisions at a glance

1. **Three steps: DESIGN → MEASURE → GENERATE** (section 3). Design returns candidate clips. The caller
   chooses one by ear, keeps it, and asks the service to measure it once. Generation then takes the
   clip by location with every request.
2. **Stateless.** No request depends on an earlier one. The same request gives the same result; work
   already done comes from the cache. The caller keeps which take it chose and why.
3. **Generation takes typed inputs, never free text.** Qwen Base takes no instruction and no unspoken
   context. The only inputs that change audio are the text (as sent, plus the request's pronunciation
   hints), the voice clip and its transcript, and the attempt number.
   - `pace`, `context_before` / `context_after` are refused explicitly.
   - Unknown fields such as `instruct` are refused as tool errors with a hint (section 14).
4. **Voice-design descriptions should be positive-only.** A negation found by a plain word list is
   warned about, with a suggested rephrasing, and the design still runs. The exact description is
   stored with the candidate.
5. **A segment is a paragraph of cues, and each cue is its own unit.** A hint or a check never crosses a
   cue boundary, and the cues are joined as sent. Every cue is **aligned to the delivery file**: CTC
   forced alignment of the known text on the CPU, with Whisper as the cross-check.
6. **The text is spoken as sent** (`text_mode: "spoken"`). The service applies only the hints in the
   request, and warns about any digit, symbol or unit-like token left in a cue. It echoes, per cue, the
   text received and the text given to the engine.
7. **Each voice has a measured reliable length.** Measuring a voice runs a length ladder (at least 3
   seeds per rung). A longer paragraph is **warned about, never refused**: length is the caller's
   decision.
8. **Three cache layers** (section 10.2), all content-addressed:
   - **Render**: request key → raw audio;
   - **Delivery**: raw + delivery profile → `delivery.wav`; the `take_id` names this;
   - **Analysis**: delivery + spoken text + cue spans + exact spans + QA inputs → verdict + cue times.

   The cache is for speed. The caller's copies of the takes it uses are the record.
9. **The service suggests a take; the caller chooses.** Per segment, the suggestion prefers a passing
   take, then a warned take with every cue placed. Take ids are stable for the same clip, text, attempt
   and engine (section 8 says how long), so a caller's choice stays valid across resubmissions.
10. **No hidden timing.** There is no fitting of any kind and no fit flag without `scene_seconds`, and
    even with it, v1 only *reports*. There is no stitching in v1. Takes come back with `samples`,
    `sample_rate`, trim record, loudness record, path and sha256. The caller owns all padding.
11. **Process model.** A thin stdio front-end per client, and a detached singleton daemon (SQLite + files,
    no sockets). One Qwen worker and one QA worker (Whisper and WavLM on the GPU, the aligner and
    profiling on the CPU). CPU thread caps, documented process
    identity, `narration-admin daemon start|stop|status`. **The service does not detect other
    workloads**; the caller decides using `get_server_status`, and can free the GPU with `release_gpu`.
12. **Takes and retakes.**
    - `takes` (1–3) asks for that many **distinct deliveries** per segment, rendered in one model load;
      or a segment names its `attempts` explicitly.
    - `max_retakes` (default 2) is the number of **automatic retakes per failing take**.
    - Seeds derive from (voice, engine text, attempt), so every take is reproducible from the request.
13. **QA before hand-back.**
    - Spans the caller marks **exact** fail the take if the transcript shows a different value or
      different words there.
    - Speaker similarity thresholds come from the **voice's measurement**.
    - Pace is judged on spoken words against the voice's length curve.
    - There are also head and end insertion checks, a token-cap check, and signal checks.
    - Every flag is reported; nothing is truncated. **What to accept is the caller's decision.**
14. **Local only.** stdio; writes confined to the store root; a caller's file is read only by the
    absolute local path a request gives, and hash-checked; offline rendering; synthetic voices only.

---

## 1. Context and evidence

**Voice holding.** All figures are WavLM-base-plus-sv cosine similarities from
`refs/auditions/voicelock.csv` (KNOW). Kokoro rows are bake-off evidence only.

| Condition | Similarity | Bit-identical? |
|---|---|---|
| Qwen VoiceDesign, same prompt + same text, **unseeded** (A) | 0.898–0.932 | no |
| Qwen VoiceDesign, unseeded fresh-process rerun of the `design` step vs `refs/narrator.wav` (A0, regenerated) | 0.936 | no |
| Qwen VoiceDesign, same seed + same text, run twice **in one process** (B) | 1.000 | **yes** |
| Qwen VoiceDesign, same seed, **different texts** (B) | 0.942–0.967 | no |
| Qwen Base **cloning one fixed reference**, different texts (C) | 0.990–0.992 (0.982–0.990 vs the ref) | no |
| Same seed, description with vs without one negated clause (d5 vs d5n) | 0.863–0.928 | no |
| *Baseline:* different design descriptions | 0.72–0.93 | no |
| *Baseline:* different Kokoro voices (bake-off) | 0.60–0.95 (`bm_daniel` vs `bm_george` 0.949) | no |

What follows from this:

- **A description is not a voice.** Only a fixed reference clip holds one. Hence a clip is designed
  once, kept, and sent with every generation (DESIGN → MEASURE → GENERATE).
- **Bit-exact reproduction** is KNOW only **within one process, and only for VoiceDesign**: B called
  `generate_voice_design` twice in a row. The clone path the service uses (`generate_voice_clone`, row
  C) was rendered once per text, so its bit-exactness is BELIEVE, within one process and across a fresh
  process alike. Phase 0 (d) tests it (section 10.1).
- All of this evidence was rendered without the determinism switches that section 10.1 pins, so a
  voice's measurement is always made under the pinned engine profile, never taken from these rows.
- **WavLM-SV is a drift alarm, not an identity proof**: two different voices can score 0.949.
- A0 was mislabelled in the original run (0.774 compared two different descriptions). It has been fixed,
  re-rendered and regenerated in `voicelock.csv`, where it now reads 0.9362.
- The earlier "Body 2001" misreading is what Whisper *heard*. Nobody has confirmed by ear what Qwen said.

**Names probe** (`scripts/r48_names_probe.json`: 8 name-dense paragraphs; 6 Qwen clone takes,
`d2-late-night_take1` and `d4-radio-drama_take2` × seeds 1–3; `eval/results.csv`,
`eval/r48_names_probe.names.md`, `outputs/qwen3-tts-1.7b-clone-*`) (KNOW):

- **Numbers: zero mismatches** across all 48 paragraph-takes.
  - Whisper writes the same kind of number sometimes as words ("twenty-four bodies", "four thousand
    forty-nine seconds") and sometimes as digits ("45 bodies", "5903.5 seconds").
  - So the exact-span check must read both. Whisper's own English normaliser maps both forms to one
    (`eval/normaliser_check.py` and `.txt`, KNOW), except "nought" (section 11.3).
- **Names.**
  - "Thallus legatribens" was heard exactly in 12/12 occurrences.
  - "Gastrella cidutis" is consistently heard as "gastrella **s**idutis", a soft c. n03 has other
    variants.
  - "Natogastrum plocrele" varies in every take (plocrella, plocrole, plocrillo, …). It needs a respelling
    hint.
  - "Marufubens" is heard split ("marou foubens"). This is BELIEVED not to be clipping: an earlier
    session checked it with 0.5 s of silence prepended, but saved neither script nor output. Phase 0
    repeats the check and saves it.
  - "Vorafrons" is often heard as "vorifrons" or "vorafron is".
- **Raw WER is 5.5–8.6 % per take** (worst segment 16–21 %), and all of it comes from name spelling.
  Raw WER is unusable on name-dense text; the gate is `wer_adj`, which collapses each term the request
  lists as a hint (section 11.1).
- **Speaker.**
  - `spk_to_ref`: d2 0.978–0.984, **d4 0.966–0.969**.
  - `spk_consist`: 0.981–0.992.
  - A fixed clone threshold of warn < 0.975 would flag *every* d4 take. The thresholds are therefore
    per-voice calibrated (section 11.1).
- **Length.** The same paragraph's duration varies 2–17 % between seeds.
- **Pace vs length.** This is written-word wpm, median of the 6 takes, by characters:

  | Characters | wpm | |
  |---|---|---|
  | 81 | 117 | |
  | 108 | 91 | number-dense |
  | 114 | 131 | |
  | 167 | 108 | |
  | 174 | 112 | |
  | 232 | 144 | |
  | 262 | 155 | |
  | 280 | 152 | |

  - There is **no rushing up to 280 characters** in the probe, nor at 323–351 characters in the
    voicelock clones (captions s03 and s02: 138 and 149 wpm, a different voice, same model). Captions
    s07 (**507 characters**, 105 words) ran at about 195 wpm cloned. So rushing starts somewhere
    between about **350 and 507 characters**, and the flow's 450-character paragraph ceiling sits
    inside that window: the length ladder (section 3.2) decides it per voice.
  - Written-word wpm is confounded by digits, which expand into many spoken words. The curve is
    therefore measured on **spoken** words (section 3.2).
  - Seed spread per paragraph, pooled over both voices: n01 146–158 wpm (8 %), n06 111–130 (17 %),
    n07 115–141 (23 %). **Within one voice it is at most 17 %** (d2 on n08) and 13 % (d4 on n07); the
    pooled figures mix two voices' paces (computed from `outputs/*/r48_names_probe.eval.json`).
- **Speed.** Generation took 2.2–3.6× the audio length per run, with the evolution farm sharing the GPU.

**Aligner.** torchaudio 2.11.0+cu128's `forced_align` is BELIEVED to run on this Windows build's CPU:
an earlier session ran a smoke test with no deprecation warning, but saved neither script nor output.
Phase 0 repeats it and saves it. The GPU path is untested. Licences, checked on the Hugging Face
API:
- `facebook/wav2vec2-large-960h-lv60-self`: apache-2.0;
- `facebook/wav2vec2-base-960h`: apache-2.0;
- `facebook/mms-300m`: cc-by-nc-4.0.

**Aligners compared elsewhere** (BELIEVE: third-party figures on recordings of people, not on this
service's audio; read 2026-09-26). FA-Bench (github.com/olewave/fa-bench), TIMIT read English,
word boundaries against human marks, clean audio, mean absolute error and a 0–1 score for boundaries
within 20 ms:

| Method | Mean error | Score within 20 ms |
|---|---|---|
| TorchAudio CTC forced alignment | 39.3 ms | 0.19 |
| WhisperX (wav2vec2 CTC) | 34.5 ms | 0.18 |
| MMS-FA (non-commercial, excluded) | 30.5 ms | 0.27 |
| Qwen3-ForcedAligner | 31.7 ms | 0.40 |
| Montreal Forced Aligner 3.4 | 21.9 ms | 0.65 |
| Whisper attention timings (stable-ts) | 98.7 ms | 0.17 |

The benchmark does not name the wav2vec2 model behind its CTC entries. `Qwen/Qwen3-ForcedAligner-0.6B`
(January 2026) is Apache-2.0 per its model card, covers English and ten other languages, takes up to
5 minutes of audio, runs through the `qwen-asr` package, and documents no CPU-only use. At 24 frames per
second one film frame lasts 42 ms, and every licensable method above averages within about 17 ms of the
others.

**Qwen settings that change audio.** Read from the installed `qwen-tts` 0.1.1 source (KNOW):

- `generate_voice_clone` defaults to `x_vector_only_mode=False` (ICL: reference audio + reference text)
  and `non_streaming_mode=False`.
- `generate_voice_design` defaults to `non_streaming_mode=True`.
- The sampling parameters are merged from the snapshot's `generate_config.json`, falling back to
  hard-coded defaults (`do_sample`, `top_k`, `top_p`, `temperature`, `repetition_penalty`, four
  `subtalker_*` parameters, `max_new_tokens` = 2048).
- All clone evidence so far used these defaults, because `generate.py` never sets them.

Other facts:

- Audition clips are 24 kHz, 16-bit PCM mono. Qwen outputs 24 kHz.
- **Scale** (the one known caller, for sizing only). The r48 film has 20 scenes and 518 s of screen
  time, with paragraphs of 1–8 cues (about 50–450 written characters). At 2.2–3.6×, one take of every
  paragraph is roughly 15–25 min of GPU time (ASSUME: about 400 s of speech), and `takes: 2` doubles
  that.

The probe's paragraphs, like the voicelock captions, came from that caller's script. They are evidence
for the thresholds above. They are not the service's test material: its calibration corpus, length
ladder, alignment benchmark and tests are its own text (sections 3.2, 9.3 and 11.2).

---

## 2. Goals and non-goals

**Goals**

- G1. One voice held exactly across batches weeks apart, as long as the caller sends the same clip.
- G2. Shaped for agents, and for any caller: paragraphs of cues in spoken words in; takes with exact
  lengths, hashes and cue times out; what was spoken echoed per cue; a clear listen-first list.
- G3. Qwen3-TTS-12Hz-1.7B only; it is the largest open-weight Qwen TTS. There is an adapter seam and
  nothing else behind it.
- G4. Auditable: the clip, the text received and given to the engine, model and seed, cue times and QA
  are all recorded with each take.
- G5. Safe on a shared machine: local, confined writes, hash-checked reads, CPU thread caps, polite
  about the GPU.
- G6. Stateless: an answer depends only on the request, so any caller can reason about it and retry it.

**Non-goals (v1)**

- **Keeping a caller's state**: its scripts and their versions, the takes it chose, its approvals, its
  pronunciation list, its voice. The caller keeps them and sends what a request needs.
- **Choosing or approving a voice.** The service designs candidates and measures a clip; which clip to
  use is the caller's choice.
- Free-text style/emotion, moods (Q5), unspoken context (R10).
- **Wording the text.** Turning digits, units, symbols and invented names into words is the caller's
  (R6). A written-text normaliser is a later phase (section 9.2, Q20).
- **Padding, stitching, scene timing, caption display timing, fit remedies.** These are the caller's
  (R4). Stitching and fit remedies are a later phase (section 13.1).
- **Deciding a paragraph's length.** The service publishes how long a paragraph a voice reads reliably
  and warns beyond it; the caller decides.
- Deciding when narration may run; the caller does that with `get_server_status`.
- Remote/multi-user serving, cloud, streaming playback, training, replacing human listening.

---

## 3. The core model: DESIGN → MEASURE → GENERATE

```
 DESIGN (stochastic)                    MEASURE (once per clip)              GENERATE (deterministic)
 ──────────────────────                 ───────────────────────              ────────────────────────
 design_voice: description × seeds      measure_voice(clip):                 submit_job:
   → candidates: clip path, sha256,       calibration set + length ladder      clip + transcript + hints
     exact transcript, profile            through Qwen Base (ICL clone)        + paragraphs of cues
            │                             → reliable length, pace curve,       + takes / attempts
            ▼                               similarity thresholds, anchor      → render → postprocess
 caller listens, chooses one clip,        (cached by the clip's fingerprint;   → QA + cue alignment
 copies it into its own folder             a copy returned to the caller)      → takes + a suggestion
```

The service keeps nothing of the caller's between these steps. The caller holds the clip, its
transcript and its fingerprint, and sends their locations with each request.

### 3.1 Design returns candidates

`design_voice(description, takes, design_text?)` renders the fixed neutral design text (the bake-off's
`REF_TEXT`, or the caller's `design_text`) with *n* derived seeds through Qwen VoiceDesign. Each
**candidate** comes back with:

- the clip: path + sha256 (a WAV in the service's output area, kept for the retention period);
- the **exact transcript** of the clip, NFC, checked by ASR;
- the description (verbatim), `description_sha256`, the seed, and the engine profile (VoiceDesign,
  `non_streaming_mode=True`, pinned);
- the positive-only lint result;
- the **voice profile**: measurements and pictures (section 3.6).

Every clip the service designs has its fingerprint added to the service's **provenance list**, which is
never pruned. That list is how the service knows a clip is synthetic (section 17).

The caller listens, chooses, and copies the clip into its own folder. From then on the clip is the
caller's: the service does not keep it as a voice.

### 3.2 Measuring a voice (once per clip)

A voice, for generation, is exactly three things the caller sends: a clip path, the clip's sha256, and
its exact transcript. The model is Qwen Base in **ICL mode** (`x_vector_only_mode=false`, pinned). The
service fingerprints the voice as `voice_hash` (section 10.2) on every request.

Before a clip can be used for generation it must be **measured**, once, with
`measure_voice({path, sha256, transcript})`. That is a job of roughly **20–50 minutes of GPU time**
(ASSUME: about 13 minutes of speech at the probe's 2.2–3.6× real time; less when the ladder stops
early). The measurement is keyed by (`voice_hash`, engine profile), cached by the service, and returned
to the caller in full, as data and as a JSON file. A generation request with an unmeasured clip is
refused with `VOICE_NOT_MEASURED` and a hint to measure it first: measuring is heavy GPU work, and when
heavy work runs is the caller's decision, so it never starts inside another job.

Measuring first checks the transcript: Whisper transcribes the clip, and a transcript that does not
match is refused (`REF_TEXT_MISMATCH`), because a wrong transcript in ICL mode causes the reference to
bleed into takes. Then it renders two sets through Qwen Base, all from the service's own corpus
(`calibration/narration-en.v1`: narration prose in spoken form, with number words, measures in words and
an invented name; no caller's text):

**1. Calibration set.** The design text plus three corpus paragraphs, **3 seeds each**. From these:

- the **anchor**: the centroid embedding over the clip and the calibration takes;
- the **similarity baseline**: the distribution of each take's similarity to the anchor and to the set
  centroid. The QA thresholds come from its p5 (section 11.1).

**2. Length ladder (R8).** Corpus paragraphs at about **80, 150, 250, 300, 350, 400, 450, 500 and 560
spoken characters** (counted as in section 7.2), **at least 3 seeds per rung**, rendered from the
shortest up. The corpus marks its number words as exact spans (section 11.3).

- **Measured per take**: pace as *spoken* words per minute and spoken characters per second over the
  voiced span; `wer_adj`; the exact-span check; similarity to the anchor.
- **The pace trend**: a straight line of pace against length, fitted over the rungs ≤ 300 spoken
  characters, where the probe showed no rushing. Pace naturally rises with length (section 1), so a rung
  is judged against the **trend extended to its length**, not against a flat reference.
- **Tolerance**: `tol = max(0.10, the largest per-rung seed spread observed at ≤ 300 characters)`,
  never below the measured seed-to-seed spread (up to 17 % within one voice in the probe).
- **A rung passes** when:
  - the **median over its seeds** of pace ≤ trend × (1 + tol);
  - the median `wer_adj` passes;
  - the median similarity passes the warn threshold;
  - no seed has an exact-span mismatch, a head insertion, or a token-cap hit.
- **`max_segment_chars`** is the upper bound of the **longest run of passing rungs counting up from the
  shortest**. The first failing rung ends the run, and the rungs above it are not rendered.
  **`max_segment_seconds`** is the median duration at that rung.
- The **pace curve** (median pace by length over the passing run) is published for QA.

**What the limit means.** `max_segment_chars` is a fact about this voice on this engine: the longest
paragraph it read reliably in the ladder. It is **advice, never a refusal**. A longer segment is
rendered and flagged `SEGMENT_TOO_LONG` (warn), with its spoken length and the limit, and its takes are
QA'd as usual. How long a paragraph is remains the caller's decision. The tool descriptions say this in
so many words.

### 3.3 Generation takes a voice and typed inputs

Generation inputs are: the voice (clip path, sha256, transcript) + segments (paragraphs as cues, in
spoken words) + the request's pronunciation hints + `takes` / `attempts` / `max_retakes` + an optional
`scene_seconds` for fit *reporting*. There is no free-text field.

- Qwen's model table marks Base as having **no instruction control**.
- With a cloned clip, delivery comes from the clip.

| Input | v1 behaviour | Later phase |
|---|---|---|
| text (cues) | spoken as sent; per cue, hints → engine text (section 9) | same |
| `text_mode` | `"spoken"` only; `"written"` is refused (`INVALID_ARGUMENT`, hint → section 9.2) | a normaliser (Q20) |
| pronunciation | the request's hints: term → respelling, a *hint* to the engine, never a guarantee | same |
| `exact` spans | QA only: a different value or different words inside one fails the take (section 11.3) | same |
| `takes`, `attempts`, `max_retakes` | distinct deliveries and retakes with derived seeds | same |
| `scene_seconds` (+ `fit.lead_in_s`, `fit.tail_s`, default 0) | **fit reported only** (section 12) | fit remedies |
| `controls.pace` | **refused** (`CONTROL_UNSUPPORTED`): Qwen has no native speed, and time-stretch is a later-phase fit remedy | `time_stretch` 0.94–1.06, only with `scene_seconds`, flagged |
| `controls.context_before/after` (R10) | **refused** (`CONTROL_UNSUPPORTED`): Base cannot condition on unspoken text | only if a future backend can; it would enter the render key |
| `instruct`, style, emotion, any unknown field | **refused** as a tool error (`INVALID_ARGUMENT`, naming the field, hint → section 3.3) | same |

The service's capabilities, including `controls` (`{"pace": false, "context": false, "instruct":
false}`), are in `get_server_status`.

### 3.4 Mood and delivery variants

A different mood of the same voice is **not straightforward**: a new design gives a new voice, and Base
takes no instruction. A different delivery is simply a different clip, which is the caller's choice.
`takes` give a choice of reading within one clip (R9, Q5).

### 3.5 Positive-only voice descriptions

This applies to the `design_voice` description and the `design_narrator_voice` prompt.

A negated quality tends to be heard by the design model as that quality, so a description should name
what is wanted ("smooth") rather than what is not ("not rough").

**The check**

- **Triggers.** A plain word list, whole-word and case-insensitive, with no language model: `not`,
  `no`, `never`, `without`, `avoid`, `don't` / `dont`, the other `n't` forms, `nor`, `neither`, `none`,
  and the prefix `non-`.
- **Allowed.** Morphological negatives that name a positive quality: "unhurried", "understated",
  "effortless".
- **What it misses.** A negation without those words: "less theatrical", "anything but theatrical",
  "free of rasp". **False alarms** are possible ("a no-nonsense tone"). Both are why it warns and never
  refuses.
- **Findings.** Each finding reports the phrase, its offset, and a suggestion:

| Negated phrase | Suggested positive rephrasing |
|---|---|
| not theatrical / never theatrical | natural, understated delivery |
| not gravelly / not rough | smooth, clean tone |
| not whispery | full, clearly voiced |
| not dramatic / no movie-trailer delivery | even, conversational documentary delivery |
| not rushed / without hurry | unhurried, measured pace |
| not monotone | gently varied intonation |
| *(no entry)* | "Describe the quality you want instead (e.g. 'smooth' rather than 'not rough')." |

**Policy: warn** (owner decision, 2026-09-26, Q6). The design runs as asked, and its result's `lint`
lists the findings (section 7.6). The tool's description tells callers the rule and why. Clips designed
before the service (d1–d4, allowlisted by the owner, section 17) were made from negated prompts; that
does not matter, because the clip is what is used and judged.

**Provenance.** The description is stored verbatim with the candidate, with `description_sha256`, the
lint result, the seed, the design text and the engine profile.

### 3.6 The voice profile

An agent cannot hear a voice, so the service describes one in terms an agent can read.
`profile_voice` takes an audio file by path and sha256. It works on any clip or take the service may
read; every designed candidate carries its profile already.

- **Measurements** (CPU, seconds): pitch median and 10th–90th percentile range (Hz and semitones), which
  is how deep and how varied the voice is; speaking rate in spoken words per minute (when the audio's
  transcript is known); pause ratio; integrated loudness; spectral centroid (brightness);
  harmonics-to-noise ratio (breathiness), by Boersma's (1993) autocorrelation method; and smoothed
  cepstral peak prominence, CPPS (Hillenbrand 1994; voice quality and roughness), which holds up on
  connected speech where jitter and shimmer, defined on sustained vowels, do not. Pitch comes from
  `librosa.pyin` (ISC). Each measure is documented as what it is, not as another tool's, and validated
  on synthetic signals with known values (revision 5.2, DC-1: no GPL code, so no parselmouth).
- **Pictures**: a spectrogram and a pitch contour as PNG files, by path, which an agent can look at.

The profile helps an agent shortlist candidates and notice drift between batches. It does not choose a
voice; the caller does, by ear.

**Not in v1: a description in words** (owner decision, 2026-09-26, Q22). A listening model could add
what the numbers miss (accent, apparent age, warmth, mood), but its reliability is unproven, it would be
the heaviest model in the service, and the voice is chosen by ear anyway. Section 20 keeps it as a later
item, to add only if agents shortlist badly from the numbers.

---

## 4. Architecture

```
 ┌──────── caller host (e.g. a Claude Code session) ─────────┐
 │  agent ──MCP, JSON-RPC over stdio, rev 2026-07-28──┐       │
 └────────────────────────────────────────────────────┼───────┘
                                                      │ spawns
             ┌────────────────────────────────────────▼──────────────────┐
             │ narration-mcp  (front-end; one per client, stateless)      │
             │  tools / resources / prompts · validation (→ tool errors)  │
             │  voice clip check · text checks + hints · planning (cache) │
             │  job submit / status / results · progress notifications    │
             └──────────┬────────────────────────────────────────────────┘
                        │ SQLite (WAL) rows + files in store_root; no sockets
             ┌──────────▼────────────────────────────────────────────────┐
             │ store_root\  narration.sqlite · cache · measurements · jobs│
             └──────────▲────────────────────────────────────────────────┘
                        │ claims work, writes renders/takes/analyses
             ┌──────────┴────────────────────────────────────────────────┐
             │ narrationd  (detached singleton daemon, named mutex)        │
             │  queue · GPU scheduler (1 resident group) · retake policy   │
             │  post-processing (CPU) · canary gate · worker supervisor    │
             └───┬──────────────────────────────┬────────────────────────┘
     JSON lines  │                              │
     over stdio  ▼                              ▼
        ┌─────────────────────────┐   ┌──────────────────────────────────┐
        │ worker qwen3            │   │ worker QA                        │
        │ Base (ICL clone) +      │   │ GPU: Whisper-large-v3, WavLM-SV  │
        │ VoiceDesign (design)    │   │ CPU: CTC aligner (wav2vec2-large │
        │ own venv (tf 4.57)      │   │      960h-lv60-self), canary sim,│
        │                         │   │      voice profile               │
        └─────────────────────────┘   └──────────────────────────────────┘
                RTX 4090 24 GB, shared with non-narration GPU jobs
```

**Front-end (`narration-mcp`).**

- Stateless, one per client. It never touches the GPU and never writes audio.
- It validates input, reads and fingerprints the voice clip (section 17), checks the text and applies
  the request's hints, plans the three cache layers, writes job rows, and reads results. It keeps no
  caller state.

**Daemon (`narrationd`).**

- A singleton, enforced by a named mutex keyed on the store path. It is started detached (section 4.1)
  by the first submission, or by `narration-admin daemon start`.
- It unloads models after 120 s idle and exits after 15 min idle.
- Why a separate process: a stdio server dies with its client, and two sessions must never load two
  models.

**Workers.**

- One per role (`qwen3`, `qa`, and the optional `listen`), each in its own uv project
  (`workers/<role>/`), synced once at install (`uv sync --frozen`).
- At runtime the daemon starts **the worker venv's `python.exe` directly**: `-m narration_worker --role
  … --store …`, with `HF_HUB_OFFLINE=1` and `TRANSFORMERS_OFFLINE=1`. uv is not in the loop, so there are
  fewer processes and no network path.
- Models load from a local snapshot directory whose path contains the commit SHA.
- Batch size is 1. Audio is exchanged as files in `store_root\scratch\` (Appendix A).
- The workers are assigned to the daemon's Windows Job Object (kill-on-close), so they die with the
  daemon.

**QA worker.**

- On the GPU: Whisper-large-v3 and WavLM-base-plus-sv, the same models as `eval/`.
- On the CPU: the **CTC aligner**, `facebook/wav2vec2-large-960h-lv60-self` (apache-2.0; the default,
  owner decision Q18). It is an English character model, so it spells invented names letter by letter.
  It runs through `torchaudio.functional.forced_align`, BELIEVED to run on this build's CPU (section 1).
- Only if Phase 0 chooses it: `Qwen/Qwen3-ForcedAligner-0.6B` instead, on the GPU in the QA group,
  through the `qwen-asr` package (in the QA worker's venv if its library versions agree, otherwise in a
  worker venv of its own). The silence snap and the Whisper cross-check apply to it unchanged.
- Also on the CPU: WavLM for the **canary** check (section 10.1), so that the canary never forces a
  model swap.
- `librosa.pyin` for f0, and the voice profile's measurements and pictures (section 3.6; no GPL code,
  revision 5.2).
- Audio I/O uses soundfile, not torchaudio's torchcodec-based loader.
- If `forced_align` ever becomes unavailable, a numpy CTC Viterbi pass (about 50 lines) is the fallback.

**Adapter seam.** The worker protocol reports capabilities, so a future backend would be a new worker
role with its own engine profile. Nothing else is built (owner decision).

**GPU scheduler**

1. One resident group at a time: **Qwen** (Base, or VoiceDesign for design jobs) or **QA** (Whisper +
   WavLM). The aligner and profiling are on the CPU. VRAM to measure in Phase 0 (ASSUME Qwen ~6–8 GB,
   QA ~5 GB).
2. Before loading, it checks through NVML that free VRAM ≥ need + 1 GB. Otherwise the phase is
   `waiting_for_gpu`, rechecked every 15 s. After 30 min: `GPU_UNAVAILABLE` (retryable).
   - It **never** kills, throttles or inspects other processes.
   - Whether narration may run beside other work is the caller's decision.
3. Work is grouped by engine profile across queued jobs, ordered by priority (`interactive` > `batch`)
   then FIFO, with affinity to the resident group.
4. **Pipeline per round**:
   - render (Qwen load) the round's attempts for every segment;
   - post-process them (CPU);
   - analyse them (QA load: ASR, similarity; aligner on CPU).

   Round 0 renders the requested attempts. Round *r* ≥ 1 renders one retake for every take slot that is
   still failing. There are at most `1 + max_retakes` rounds, so ≤ 6 model loads with the default.
5. Preemption only between segments. On an out-of-memory error: unload, wait, retry once, then fail the
   segment with `GPU_OOM`.
6. **Check again at claim time.** Before it renders, post-processes or analyses an item, the daemon
   looks up the item's key again. If the result exists, it is used. If another job is producing the same
   key right now, this job waits for it instead of repeating the work. Two sessions with overlapping
   scripts therefore never render the same paragraph twice.

### 4.1 Machine load, detachment, process identity, stopping

The caller owns the machine-load rules. It reads `get_server_status` before it submits. The service
provides:

- **CPU thread cap.** `[workers] cpu_threads` (default 8) is applied to every worker and to the daemon's
  post-processing. It works through `OMP_NUM_THREADS`, `MKL_NUM_THREADS`, `OPENBLAS_NUM_THREADS`,
  `torch.set_num_threads(n)`, `torch.set_num_interop_threads(min(n, 4))` and the resampler's threads.
  Workers also run at below-normal priority.
- **Status.** `get_server_status` reports:
  - daemon state (`stopped` | `idle` | `busy` | `stopping`);
  - the current job (id, kind, label, phase, start time);
  - the GPU: `in_use` (true while a model is loaded), `holder` (`qwen` | `qa` | none), and
    `unload_in_s` (seconds until an idle model is unloaded);
  - the workers (role, pid);
  - the queue;
  - `cpu_threads`.
- **Freeing the GPU.** `release_gpu` unloads an idle model at once, so a caller about to start other GPU
  work need not wait for the idle timeout. If a job is running it changes nothing and says which job
  holds the GPU.
- **Detachment (Windows).** The front-end starts the daemon with
  `CREATE_BREAKAWAY_FROM_JOB | DETACHED_PROCESS | CREATE_NEW_PROCESS_GROUP`, with stdin, stdout and
  stderr on `NUL` and all inherited handles closed (`close_fds=True`). The working directory is
  `store_root`, and the daemon writes its own log.
  - If the client's Job Object forbids breakaway, `CreateProcess` fails with access denied. The
    front-end then does **not** start a non-detached daemon, because it would die with the client
    mid-job. It returns `DAEMON_UNAVAILABLE` with the hint *"run `narration-admin daemon start` in a
    terminal"*.
  - Phase 0 checks that a daemon started from an MCP session survives the client exiting.
- **Process identity** (for an orphan sweep):

| Role | Image | Command-line marker |
|---|---|---|
| front-end | `python.exe` (child of the `narration-mcp.exe` uv launcher; BELIEVE) | `narration-mcp --config <path>` |
| daemon | `python.exe` (server venv) | `-m narration.daemon --store <store_root>` |
| Qwen worker | `python.exe` (worker venv) | `-m narration_worker --role qwen3 --store <store_root>` |
| QA worker | `python.exe` (worker venv) | `-m narration_worker --role qa --store <store_root>` |

  `store_root\run\daemon.json` (pid, start time, workers) exists while the daemon runs.
- **Stop.**
  - `narration-admin daemon stop`: stop claiming work, finish the in-flight segment, unload, exit.
    Queued jobs resume on the next start.
  - `daemon stop --now`: terminate the Job Object and re-queue the in-flight segment. All files are
    written to a temp name and renamed, so nothing partial is ever published.

---

## 5. MCP protocol basis (revision 2026-07-28)

- **Stateless protocol, stateless service.** No `initialize`; per-request `_meta`; `server/discover`.
  - BELIEVE: the Python SDK v2 line (`MCPServer`) implements this and also answers legacy `initialize`.
    Verify in Phase 0.
- **Handles.** Server-minted handles are passed as tool arguments (the spec's "Stateful Tools"
  guidance). They name work and results, never caller state:
  - `job_id` and `design_id` are ULIDs;
  - `take_id`, `render_id`, `analysis_id` and `voice_hash` are content hashes, so the same inputs always
    give the same id.
  - Retention is stated in the tool descriptions (section 15).
- **Progress.** `notifications/progress` is sent only during an in-flight `get_job` with `wait_s > 0`
  and a `progressToken`. `progress` is monotonic, and `total` may grow with retakes.
- **Cancellation.** `notifications/cancelled` cancels only the in-flight `get_job` wait. The job is
  cancelled with `cancel_job`.
- **Tool results.**
  - `structuredContent` + `outputSchema`, plus a text copy for older clients.
  - Audio and pictures are never inlined: results carry paths + sha256 and `file:///` `resource_link`
    items. Agents pass locations, not file contents.
  - Annotations are hints only.
- **Errors (section 14).** As 2026-07-28 intends, **input validation failures are tool execution
  errors** (`isError: true` + structured error + hint), so the model can correct itself. JSON-RPC errors
  are kept for an unknown tool, a malformed request, and a missing resource (−32602).
- **Schemas.** Published tool schemas are **fully dereferenced**: they contain no `$ref`, because some
  clients do not resolve it. A build-time test asserts that no `"$ref"` appears in `tools/list`. Every
  `outputSchema` has `"type": "object"` at its root (section 7.2).
- **Resources.** The `narration://` scheme, with `ttlMs` and `cacheScope: "private"`, and job resources
  subscribable through `subscriptions/listen`.
- **Not used.** Sampling and Roots (deprecated), MCP Logging (deprecated; logs go to stderr and files),
  elicitation/MRTR (the service asks nobody for approval).
- **Tasks extension**: optional, later phase (draft; not in the official client matrix).

---

## 6. Data model

⊘ marks an immutable record. Everything here is either the service's own (engine pins, provenance,
measurements, the benchmark) or a cache of work done (renders, takes, analyses, profiles). Nothing
records a caller's script, choices or approvals.

| Entity | Key fields |
|---|---|
| **EngineProfile** ⊘ | `engine_profile_id` (`qwen3-base-1.7b.p1`, `qwen3-design-1.7b.p1`), `hash`, `model_repo`, `model_revision` (40-hex), `snapshot_dir`, `weights` {file: sha256}, `worker_project`, `uv_lock_sha256`, package versions, dtype, `attn_implementation`, **determinism switches** (section 10.1), **audio-changing settings** (`non_streaming_mode`: False for Base, True for VoiceDesign; the effective sampling parameters incl. `max_new_tokens`, which is the ceiling, and the per-call cap rule `max_new_tokens_per_char` and `max_new_tokens_floor`, DC-4), capabilities, licence, `vram_need_mb`. Observed but not hashed: GPU, driver, CUDA, cuDNN. Also not hashed: `snapshot_dir` (a local path) and the **canary** {material, seed, raw hash, embedding, calibrated threshold}, which `engine pin` makes on the installing machine (section 10.1; revision 5.2, DC-3). |
| **Candidate** | `design_id`, index, clip {path, sha256}, exact transcript, verbatim description, seed, engine profile, lint, profile. Kept for the retention period; the caller copies the clip it chooses. |
| **Provenance entry** ⊘ | clip sha256, `design_id`, date. One per clip the service designed; never pruned (section 17). |
| **Voice** *(not stored)* | What a request sends: clip path + sha256 + transcript. `voice_hash` is computed from them on every request (section 10.2). |
| **Measurement** ⊘ | (`voice_hash`, `engine_profile_id`); verified transcript; **anchor**; **similarity baseline** {anchor p5/p50, consistency p5}; **pace curve** and trend; **`max_segment_chars`**, **`max_segment_seconds`**; ladder table; calibration takes; corpus version. Returned in full to the caller as well. |
| **Render** ⊘ | `render_id` = `rn_` + 16 hex of `render_key`; seed, attempt; raw audio {path, sha256, samples, sample_rate}; `hit_token_cap`; gen timings. |
| **Take** ⊘ *(the delivery layer)* | `take_id` = `tk_` + 16 hex of `delivery_key`; `render_id`; delivery {path, sha256, samples, sample_rate, duration_s}; **trim** {head_s, tail_s, pad_s}; **loudness** {measured_lufs, gain_db, true_peak_dbtp, ceiling_applied}; `post_stretched` (always false in v1). |
| **Analysis** ⊘ | `analysis_id` = `an_` + 16 hex of `analysis_key`; `take_id`; QA verdict + flags + metrics + exact-span results; **cue alignment** {cues, words, method, model rev, cross-check, flags}. A take can have several analyses (e.g. with different hints' aliases or exact spans); a request uses the one matching its inputs. |
| **Profile** ⊘ | audio sha256 + profile version; measurements, picture paths. |
| **AlignmentBenchmark** ⊘ | `method_id` (aligner model + revision + snap parameters), benchmark id + sha256, measured error {p50_s, p95_s, n, by boundary kind}, date (R1, section 11.2). |
| **Job** | `job_id`, kind generate\|analyse\|design\|measure\|profile\|pronunciation, the request (by value), optional caller `label` (opaque), priority, status, phase, progress, items[], outcome, result, error, `idempotency_key`. Kept for the retention period. |

**Fit** is not stored in any layer. It depends on the request's `scene_seconds` and `fit` values, so it
is computed per request from the take's delivery duration (section 12).

---

## 7. MCP surface

Server name: `narration`. Tools use snake_case, and `tools/list` returns them in a fixed order.

### 7.1 Tools by step

| Step | Tool | Does | Read-only | Returns |
|---|---|---|---|---|
| any | `get_server_status` | daemon state + current job, GPU in use and by what, queue, threads, limits, versions, capabilities, measured alignment error | ✓ | sync |
| any | `release_gpu` | unload an idle model now | | sync |
| any | `get_job` | status/progress; long-poll `wait_s` ≤ 55 | ✓ | sync |
| any | `get_results` | a finished job's result | ✓ | sync |
| any | `cancel_job` | cooperative cancel; finished work kept | | sync |
| DESIGN | `design_voice` | Qwen VoiceDesign: description (positive-only) × n seeds → candidates with clip, transcript, profile | | job |
| DESIGN | `profile_voice` | measurements and pictures of a clip or take | | job |
| MEASURE | `measure_voice` | transcript check + calibration set + length ladder for one clip | | job |
| GENERATE | `check_text` | per cue: received → spoken → engine, hints applied, text warnings, exact spans; length against the voice's limit | ✓ | sync |
| GENERATE | `audition_pronunciation` | render a term with up to 4 respelling variants | | job |
| GENERATE | `submit_job` | render and analyse what a request needs (or a `dry_run` plan) | idempotent | job |
| *later* | `stitch` | sequential or timeline stitch (section 13.1) | | *not in v1* |

Every tool description states that the service keeps no caller state, that a paragraph's length is the
caller's decision, and that a respelling is a hint. Every tool that can return a retryable error also
states the **backoff rule** (revision 5.2, DC-2): wait at least `retry_after_s`, add your own jitter, then
send the identical request again, which is deduplicated. The service suggests; the caller decides.

**Operator CLI (`narration-admin`)**, for the machine, not for any use of it: `install`, `engine
pin|repin|bridge` (section 10.1), `gc` (dry-run default), `verify`, `bench alignment` (section 11.2),
`daemon start|stop [--now]|status`, `doctor`. None of them approves anything; the service has no
approvals.

### 7.2 Shared definitions

These fragments are **inlined wherever they are used**. The published schemas contain no `$ref`, and the
comments below mark where a fragment goes.

```jsonc
// Id
{"type": "string", "pattern": "^[a-z0-9][a-z0-9._-]{0,63}$"}

// Flag (codes: section 14)
{"type": "object", "required": ["code", "severity", "message"], "properties": {
  "code": {"type": "string"}, "severity": {"enum": ["info", "warn", "fail", "error"]},
  "message": {"type": "string"}, "segment_id": {"type": "string"}, "cue": {"type": "integer"},
  "retake_trigger": {"type": "boolean"}, "details": {"type": "object"}}}

// Error (retryable: the same call may succeed later unchanged; false means the arguments must change)
// retry_after_s (revision 5.2, DC-2): set on every retryable error; the server's minimum wait before the
// identical request is sent again, like HTTP Retry-After. `details` carries the facts behind it.
{"type": "object", "required": ["code", "message", "retryable"], "properties": {
  "code": {"type": "string"}, "message": {"type": "string"}, "retryable": {"type": "boolean"},
  "hint": {"type": "string"}, "field": {"type": "string"}, "details": {"type": "object"},
  "retry_after_s": {"type": "number", "minimum": 0}}}

// Voice: a clip the caller keeps, by location
{"type": "object", "additionalProperties": false, "required": ["path", "sha256", "transcript"],
 "properties": {
   "path": {"type": "string", "description": "absolute path of a WAV on a local drive"},
   "sha256": {"type": "string", "pattern": "^[0-9a-f]{64}$"},
   "transcript": {"type": "string", "minLength": 1, "maxLength": 600,
     "description": "the exact words spoken in the clip, as design_voice returned them"}}}

// Hint: a pronunciation hint for one term
{"type": "object", "additionalProperties": false, "required": ["term"], "properties": {
  "term": {"type": "string", "minLength": 1, "maxLength": 80,
    "description": "the word or words as they appear in the spoken text"},
  "respell": {"type": "string", "maxLength": 120, "description": "optional: what the engine is given instead"},
  "align_as": {"type": "string", "maxLength": 120, "description": "optional: letters for the aligner"},
  "asr_aliases": {"type": "array", "maxItems": 10, "items": {"type": "string", "maxLength": 120}}}}

// Controls: every property is refused by every current engine (section 3.3)
{"type": "object", "additionalProperties": false, "properties": {
  "pace": {"type": "object", "properties": {
    "factor": {"type": "number", "minimum": 0.94, "maximum": 1.06},
    "mode": {"enum": ["native", "time_stretch"]}}},
  "context_before": {"type": "string", "maxLength": 1200},
  "context_after":  {"type": "string", "maxLength": 1200}}}

// Segment
{"type": "object", "additionalProperties": false, "required": ["segment_id"],
 "anyOf": [{"required": ["cues"]}, {"required": ["text"]}],
 "properties": {
   "segment_id": /* Id; the caller's own name, echoed back */,
   "cues": {"type": "array", "minItems": 1, "maxItems": 40, "items": {
     "type": "object", "additionalProperties": false, "required": ["text"], "properties": {
       "text": {"type": "string", "minLength": 1, "maxLength": 600,
         "description": "the words to be spoken, as the caller wants them heard"},
       "at_s": {"type": "number", "minimum": 0, "description": "optional, informational"},
       "exact": {"type": "array", "maxItems": 20, "description": "R14: spans that must be heard exactly",
         "items": {"type": "object", "additionalProperties": false, "required": ["start", "end"],
           "properties": {
             "start": {"type": "integer", "minimum": 0},
             "end": {"type": "integer", "minimum": 1,
               "description": "Unicode code points into this cue's text as sent, end exclusive; whole words only"}}}}}}},
   "text": {"type": "string", "minLength": 1, "maxLength": 1200,
     "description": "optional with cues; if both, must equal the join of the cues"},
   "attempts": {"type": "array", "minItems": 1, "maxItems": 3, "uniqueItems": true,
     "items": {"type": "integer", "minimum": 0, "maximum": 99},
     "description": "optional: exactly which attempts to render; default 0..takes-1"},
   "scene_seconds": {"type": "number", "exclusiveMinimum": 0, "maximum": 900,
     "description": "optional budget; absent = no fit reporting of any kind (R4)"},
   "fit": {"type": "object", "additionalProperties": false, "properties": {
     "lead_in_s": {"type": "number", "minimum": 0, "maximum": 5, "default": 0},
     "tail_s": {"type": "number", "minimum": 0, "maximum": 5, "default": 0}}},
   "controls": /* Controls */}}
```

- A text-only segment is one cue. A caller that wants exact spans sends cues.
- **Canonical form** of a cue: NFC, every run of whitespace made one space, trimmed. Nothing else is
  changed: no punctuation or capital is added, so a sentence may run across two cues.
- The **join** is the cues' canonical texts joined by one space. A `text` whose canonical form differs
  from it is refused (R2).
- **Spoken length** (`spoken_chars`) is the join's length in Unicode code points. A caller can compute
  it exactly, and it is what `max_segment_chars` is compared with (R8). Hints do not count.
- **Exact spans** (R14) must lie inside the cue, must not overlap, and must start and end on word
  edges: a word is a whitespace-separated token with its leading and trailing punctuation set aside
  (so `forty-eight` is one word, and `seven` in `seven.` is a whole word). A span may end before or
  after trailing punctuation; punctuation is ignored in the comparison. A span that cuts a word is
  refused (`INVALID_ARGUMENT`, naming the field). The service keeps a span as a range of word indices,
  which canonicalisation does not change.
- **Output schemas.** Every `outputSchema` has `"type": "object"` at its root. Its properties are the
  success fields plus an optional `error` (Error); a result carries one or the other.

### 7.3 `submit_job`

**Input**

```jsonc
{"type": "object", "additionalProperties": false, "required": ["voice", "segments"],
 "properties": {
   "voice": /* Voice */,
   "expect_engine_profile": {"type": "string",
     "description": "optional engine profile hash; refuse (ENGINE_CHANGED) if the service's differs"},
   "text_mode": {"enum": ["spoken"], "default": "spoken",
     "description": "v1 speaks the text as sent; 'written' is a later phase (section 9.2)"},
   "hints": {"type": "array", "maxItems": 500, "items": /* Hint */},
   "segments": {"type": "array", "minItems": 1, "maxItems": 200, "items": /* Segment */},
   "label": {"type": "string", "maxLength": 100, "description": "opaque; shown in status, never used"},
   "options": {"type": "object", "additionalProperties": false, "properties": {
     "dry_run": {"type": "boolean", "default": false},
     "takes": {"type": "integer", "minimum": 1, "maximum": 3, "default": 1,
       "description": "distinct deliveries per segment (attempts 0..takes-1) unless a segment names its attempts (R9)"},
     "max_retakes": {"type": "integer", "minimum": 0, "maximum": 3, "default": 2,
       "description": "automatic retakes per failing take, on the next attempt numbers"},
     "strict_text": {"type": "boolean", "default": false,
       "description": "refuse (TEXT_REFUSED) while any text warning remains; default: warn and render"},
     "priority": {"enum": ["batch", "interactive"], "default": "batch"},
     "idempotency_key": {"type": "string", "maxLength": 64}}}}}
```

**Output (success)**

```jsonc
{"type": "object", "properties": {
   "job_id": {"type": ["string", "null"], "description": "null only for dry_run"},
   "status": {"enum": ["queued", "running", "completed", "planned"]},
   "voice_hash": {"type": "string"},
   "engine_profile": {"type": "object", "properties": {"id": {"type": "string"}, "hash": {"type": "string"}}},
   "plan": {"type": "object", "properties": {
     "segments_total": {"type": "integer"}, "segments_cached": {"type": "integer"},
     "renders_needed": {"type": "integer"}, "deliveries_needed": {"type": "integer"},
     "analyses_needed": {"type": "integer"},
     "est_audio_s": {"type": "number"}, "est_wall_s": {"type": "number"},
     "queue_position": {"type": "integer"}}},
   "poll_after_s": {"type": "number", "minimum": 0,
     "description": "revision 5.2 (DC-2): the earliest get_job poll worth making; longer while waiting_for_gpu"},
   "text": {"type": "array", "description": "dry_run only: per segment, the same text echo and length check as check_text (R7, R8)"},
   "warnings": {"type": "array", "items": /* Flag: text warnings and SEGMENT_TOO_LONG */},
   "error": /* Error */}}
```

- **The same request gives the same result.** When every layer is already in the cache, the job is
  created already `completed`, and `get_results` returns at once. An identical request while a job is
  running returns that job (`idempotency_key` deduplicates retries). A retry is the same request: a key
  reused for a different request while its job is active is refused (`INVALID_ARGUMENT` on
  `idempotency_key`; revision 5.3, DC-6), never answered with the other request's job.
- **New attempts are asked for, never remembered.** A resubmission renders nothing that is cached, and
  cached takes that failed are not retaken again: their retakes are cached too. To hear new deliveries,
  a segment names new `attempts` (e.g. `[3, 4]`).
- **Refusals, as tool errors:** `VOICE_NOT_MEASURED` (hint: `measure_voice`), `VOICE_FILE_MISMATCH`,
  `VOICE_NOT_SYNTHETIC`, `PATH_NOT_ALLOWED`, `ENGINE_CHANGED`, `TEXT_REFUSED` (markup, or `strict_text`
  with text warnings left), `INVALID_ARGUMENT` (an unknown field, `text_mode: "written"`, an exact span
  that cuts a word, duplicate segment ids), `CONTROL_UNSUPPORTED`.
- **A segment longer than the voice's reliable length is not refused.** It is flagged
  `SEGMENT_TOO_LONG` (warn) in `warnings` and in the results, with the voice's limits and the segment's
  spoken length, how far over it is, and each cue's spoken length, so a caller that splits knows where
  it stands:

  ```json
  {"code": "SEGMENT_TOO_LONG", "severity": "warn", "segment_id": "p11",
   "message": "p11 is 612 spoken characters; this voice read up to 450 reliably. It will be rendered.",
   "details": {"max_segment_chars": 450, "max_segment_seconds": 31.5, "spoken_chars": 612,
               "over_by_chars": 162, "cue_chars": [88, 131, 97, 140, 152]}}
  ```

  (The limits here are illustrative; the real ones come from the voice's measurement.)

**Example** (the full `_meta` is shown once here; later examples omit it)

```json
{"jsonrpc": "2.0", "id": 7, "method": "tools/call", "params": {
  "_meta": {"io.modelcontextprotocol/protocolVersion": "2026-07-28",
            "io.modelcontextprotocol/clientCapabilities": {},
            "io.modelcontextprotocol/clientInfo": {"name": "claude-code", "version": "…"}},
  "name": "submit_job",
  "arguments": {
    "voice": {"path": "C:\\reef-film\\voice\\narrator.wav", "sha256": "5b1e…",
              "transcript": "Far below the surface, where the light grows thin, small things live quiet lives. …"},
    "hints": [{"term": "Ossavine", "respell": "Oss-a-veen"}],
    "segments": [
      {"segment_id": "p03", "cues": [
        {"text": "Before dawn, the reef belongs to the Ossavine shrimp."},
        {"text": "By sunrise, some three thousand two hundred of them are back in the rock.",
         "exact": [{"start": 17, "end": 43}]}]}],
    "label": "reef film, draft 4",
    "options": {"takes": 3}}}}
```

The exact span is "three thousand two hundred".

```json
{"job_id": "job_01JBXQ7Z3M8V4T2R9K6N5P0W1C", "status": "queued", "voice_hash": "sha256:3f9a0c1e…",
 "engine_profile": {"id": "qwen3-base-1.7b.p1", "hash": "sha256:9e21…"},
 "plan": {"segments_total": 42, "segments_cached": 41, "renders_needed": 3, "deliveries_needed": 3,
          "analyses_needed": 3, "est_audio_s": 27.0, "est_wall_s": 170, "queue_position": 0},
 "warnings": [{"code": "WRITTEN_FORM_TOKEN", "severity": "warn", "segment_id": "p09", "cue": 2,
   "message": "'5' is a digit in spoken text; the engine will read it as it reads digits.",
   "details": {"token": "5", "kind": "digit", "offset": 31}}]}
```

### 7.4 `get_job`

- **Input**: `{job_id, wait_s: 0–55, include_segments: bool}`. It emits progress notifications while
  waiting, if the request carried a `progressToken`.
- **Output**:
  - `status`: queued\|running\|cancelling\|completed\|failed\|cancelled;
  - **`phase`**: waiting_for_gpu \| loading_model \| canary \| rendering \| postprocessing \| scoring
    (ASR, similarity, alignment) \| retaking \| suggesting;
  - `round`, `outcome` (all_passed\|needs_attention), `progress` {done_s, total_s, fraction,
    segments_done, segments_total}, `eta_s`, `queue_position`, **`poll_after_s`** (revision 5.2,
    DC-2: the earliest poll worth making, longer while `waiting_for_gpu`), `message`, optional
    `segments[]` {segment_id, state, takes_ok, retakes_used}, `error`, `updated_at`.

```json
{"jsonrpc": "2.0", "method": "notifications/progress",
 "params": {"progressToken": "p-9", "progress": 31.5, "total": 128.0,
            "message": "Round 0: rendering p03 take 2/3 on qwen3-base-1.7b.p1"}}
```

### 7.5 `get_results`

**Input**: `{job_id, include_words (default true), include_transcripts (default false)}`. Results stay
available for the retention period; after that, the same request is answered again from whatever is
still cached.

**Output** (for a generation job):

- `job` {job_id, status, outcome, label}; a `failed` or `cancelled` job returns the segments it finished,
  and the others with their state.
- `voice` {voice_hash, clip sha256}, `engine_profile` {id, hash}, `measurement` {max_segment_chars,
  max_segment_seconds, measured_at}
- `segments[]`, each with:
  - `segment_id`, `status`, **`suggested_take_id`** and `suggestion` {tier, reason} (section 8). The
    suggestion is advice; the caller chooses.
  - **`text`** (R7), once per segment, since every take of a segment speaks the same engine text:
    `cues[]` {index, **received** (exactly as sent), **spoken** (its canonical form, section 7.2),
    **engine** (as given to the engine, after hints), `hints_applied[]` {term, respell, offset},
    `warnings[]`, `exact[]` {start, end}}, and `spoken_chars`.
  - **`takes[]`**, every take rendered or found for this request. Each carries:
    - `take_id`, `render_id`, `attempt`, `seed`, **`fresh`** (true if this job rendered it, false if it
      came from the cache: R11)
    - `delivery` {path, sha256, samples, sample_rate, duration_s} (R5, R12)
    - `trim` {head_s, tail_s, **pad_s**} and `loudness` {measured_lufs, gain_db, true_peak_dbtp,
      ceiling_applied} (section 13)
    - `analysis_id`
    - **`cues[]`** {index, start_s, end_s, confidence, words[] {text, start_s, end_s}} (delivery-file
      seconds; null times for an unplaced cue)
    - `alignment` {method, model, revision, cross_check, max_disagreement_s, **measured_error**
      {p50_s, p95_s, n, benchmark}, flags} (R1)
    - `qa` {verdict, flags[], wer_raw, wer_adj, **exact_ok**, **exact[]** {cue, start, end, expected,
      heard, match}, terms[], spk_sim_anchor, pace, pace_expected}
    - `fit` (computed only if `scene_seconds` was given)
- **`consistency`**: speaker similarity across the suggested takes of this request, {min, median,
  outliers[] (take ids below the voice's consistency baseline)}. It is a report on the set, never part
  of a take's verdict (section 11.1).
- **`listen_first[]`** {segment_id, take_id, cue, reason, from_s, to_s}
- `report_md`, `licence`
- `resource_link`s for every WAV

```json
{"job": {"job_id": "job_01JBXQ7Z3M8V4T2R9K6N5P0W1C", "status": "completed", "outcome": "all_passed",
         "label": "reef film, draft 4"},
 "voice": {"voice_hash": "sha256:3f9a0c1e…", "clip_sha256": "5b1e…"},
 "engine_profile": {"id": "qwen3-base-1.7b.p1", "hash": "sha256:9e21…"},
 "measurement": {"max_segment_chars": 450, "max_segment_seconds": 31.5, "measured_at": "2026-10-02"},
 "segments": [
  {"segment_id": "p03", "status": "passed", "suggested_take_id": "tk_8c41d2e07a9b3f55",
   "suggestion": {"tier": 1, "reason": "lowest passing attempt"},
   "text": {"spoken_chars": 127, "cues": [
     {"index": 0, "received": "Before dawn, the reef belongs to the Ossavine shrimp.",
      "spoken": "Before dawn, the reef belongs to the Ossavine shrimp.",
      "engine": "Before dawn, the reef belongs to the Oss-a-veen shrimp.",
      "hints_applied": [{"term": "Ossavine", "respell": "Oss-a-veen", "offset": 37}], "warnings": [], "exact": []},
     {"index": 1, "received": "By sunrise, some three thousand two hundred of them are back in the rock.",
      "spoken": "By sunrise, some three thousand two hundred of them are back in the rock.",
      "engine": "By sunrise, some three thousand two hundred of them are back in the rock.",
      "hints_applied": [], "warnings": [], "exact": [{"start": 17, "end": 43}]}]},
   "takes": [
    {"take_id": "tk_8c41d2e07a9b3f55", "render_id": "rn_77e0c4a1b2d93f08", "attempt": 0, "seed": 1834112093, "fresh": true,
     "delivery": {"path": "<store_root>\\takes\\8c\\tk_8c41d2e07a9b3f55\\delivery.wav",
                  "sha256": "…", "samples": 420480, "sample_rate": 48000, "duration_s": 8.76},
     "trim": {"head_s": 0.27, "tail_s": 0.39, "pad_s": 0.08},
     "loudness": {"measured_lufs": -23.0, "gain_db": -1.6, "true_peak_dbtp": -9.3, "ceiling_applied": false},
     "analysis_id": "an_0f3b91c2d5e7a468",
     "cues": [
       {"index": 0, "start_s": 0.08, "end_s": 3.02, "confidence": 0.93,
        "words": [{"text": "Before", "start_s": 0.08, "end_s": 0.41}, "…"]},
       {"index": 1, "start_s": 3.58, "end_s": 8.68, "confidence": 0.95, "words": ["…"]}],
     "alignment": {"method": "ctc-forced-align+silence-snap", "model": "facebook/wav2vec2-large-960h-lv60-self",
                   "revision": "…", "cross_check": "whisper-large-v3 word timestamps", "max_disagreement_s": 0.06,
                   "measured_error": {"p50_s": "…", "p95_s": "…", "n": "…", "benchmark": "alignment-en.v1"},
                   "flags": []},
     "qa": {"verdict": "pass", "wer_raw": 0.0, "wer_adj": 0.0, "exact_ok": true,
            "exact": [{"cue": 1, "start": 17, "end": 43, "expected": "3200", "heard": "3200", "match": "same"}],
            "terms": [{"term": "Ossavine", "cue": 0, "heard": "Ossavine", "ok": true}],
            "spk_sim_anchor": 0.981, "pace": {"spoken_wpm": 158}, "pace_expected": {"spoken_wpm": 150}, "flags": []}},
    {"take_id": "tk_2f07b1d9c4e8a613", "attempt": 1, "seed": 902214557, "fresh": true,
     "delivery": {"path": "…", "sha256": "…", "samples": 446400, "sample_rate": 48000, "duration_s": 9.30},
     "trim": {"head_s": 0.31, "tail_s": 0.35, "pad_s": 0.08}, "cues": ["…"], "qa": {"verdict": "pass"}}]}],
 "consistency": {"min": 0.978, "median": 0.986, "outliers": []},
 "listen_first": [],
 "report_md": "<store_root>\\jobs\\job_01JBX…\\report.md"}
```

The `measured_error` values are Phase 0 output and unknown today.

### 7.6 The other tools (inputs → outputs)

- **`get_server_status`** `{}` → version, spec revision, `store_root`, **daemon** {state, pid,
  current_job {job_id, kind, label, phase, started_at}, workers[] {role, pid}}, **gpu** {name, total_mb,
  free_mb, **in_use**, **holder**, **unload_in_s**}, `cpu_threads`, queue, engine profiles {id, hash,
  installed, env_ok, determinism tier}, **capabilities** {controls, text_modes}, limits, `text_checks_version`, and **`alignment`** {method_id, model, revision,
  **measured_error** {p50_s, p95_s, n, by boundary kind}, benchmark {id, sha256, description},
  measured_at} (R1, section 11.2). And **`admission`** (revision 5.2, DC-2), so a consumer can decide
  before it submits: {accepting, queue {length, max, est_drain_s}, rate {remaining, resets_in_s}, gpu
  {in_use, holder, free_mb, need_mb by group, waiting_since}}.
- **`release_gpu`** `{}` → `{released: bool, holder_before, busy_job?}`. It unloads an idle model at
  once; while a job runs it changes nothing and names the job.
- **`design_voice`** `{name, description ≤ 600, takes 1–4, design_text? ≤ 400}` → `{job_id, design_id,
  lint}`. The results give each candidate's clip {path, sha256}, **exact transcript**, seed, engine
  profile and profile (section 3.1). A description with a negation is still rendered, and `lint`
  lists what the word list found (section 3.5):

  ```json
  {"job_id": "job_01JBY…", "design_id": "01JBY…",
   "lint": {"policy": "warn", "findings": [
     {"phrase": "Not gravelly", "offset": 214, "suggestion": "smooth, clean tone"},
     {"phrase": "not theatrical", "offset": 228, "suggestion": "natural, understated delivery"}],
    "note": "Negated qualities tend to come out as that quality; rephrase and design again if the candidates show it."}}
  ```

- **`profile_voice`** `{audio: {path, sha256}}` → `{job_id}`. The results give the measurements and
  picture paths (section 3.6). The job runs on the CPU and finishes in seconds.
- **`measure_voice`** `{voice}` → `{job_id}`, or at once the cached measurement if one exists for this
  voice and engine profile. The results give the verified transcript, the ladder table (per rung and
  seed: spoken wpm, `wer_adj`, similarity, verdict), `max_segment_chars`, `max_segment_seconds`, the pace
  curve and trend, the similarity baseline, and the path of the measurement JSON for the caller to keep.
- **`check_text`** `{voice?, hints?, segments ≤ 200}` (R6, R7, R8). It renders nothing. Per segment:
  - **`cues[]`** {index, received, spoken, engine, `hints_applied[]` {term, respell, offset},
    **`warnings[]`** (the text warnings of section 9.1; what `strict_text` refuses on), `exact[]`
    {start, end, words}}
  - `spoken_chars`; and, when a measured voice is given, `max_segment_chars`, `over_by_chars` and
    `est_duration_s` (from the voice's pace curve).
- **`audition_pronunciation`** `{voice, term, variants: [{label, respell}] ≤ 4, carrier?}` → `{job_id}`.
  It produces takes plus what the ASR heard. The voice need not be measured. Whoever owns the text
  decides by ear; the service records no choice.
- **`cancel_job`** `{job_id, reason}` → `{status, completed}`. Finished renders, takes and analyses are
  kept in the cache.

### 7.7 Resources (`narration://`)

| URI (template) | MIME | ttlMs |
|---|---|---|
| `narration://status` | application/json | 5 000 |
| `narration://designs/{design_id}` (candidates, clips, transcripts, profiles) | application/json | 60 000 |
| `narration://measurements/{voice_hash}` (per engine profile) | application/json | 60 000 |
| `narration://jobs/{job_id}` (subscribable), `…/report` (markdown) | json / markdown | 2 000 running; 86 400 000 terminal |
| `narration://takes/{take_id}` (delivery + render + analyses) | application/json | 86 400 000 |

All resources use `cacheScope: "private"`. Audio and pictures are reached through paths and `file:///`
links only.

### 7.8 Prompts

- **`narrate_script`** `(voice_path)`:
  1. `measure_voice` if the clip has no measurement (a heavy job: check `get_server_status` first);
  2. `check_text` on every paragraph's cues, which must already be in the words to be spoken;
  3. resolve every **text warning** by editing the text (the service does not reword it);
  4. `submit_job` with the chosen `takes` and the caller's hints;
  5. `get_job(wait_s)`, then `get_results`;
  6. copy the takes to keep, check their sha256, record take and analysis ids, and report
     `listen_first`.
- **`resolve_flags`** `(job_id)` handles each flagged take or cue with a hint, new attempts, or a text
  edit.
- **`design_narrator_voice`** `(brief)` writes positive-only descriptions, runs `design_voice` with 3
  takes each, reads the profiles to shortlist, and ends by asking the person to listen and choose.
- **`add_pronunciation`** `(term)` drafts respelling hints and runs `audition_pronunciation`. It reminds
  the agent that a respelling is a hint, and that whoever owns the text decides by ear.

---

## 8. Job lifecycle, suggestions and main flows

**Job states and phases**

```
queued ─► running ─► completed  (outcome: all_passed | needs_attention)
  │         │  per round: [canary] → rendering → postprocessing → scoring (ASR, similarity, alignment)
  │         │  rounds: 0 = requested attempts; 1..max_retakes = retakes of failing take slots; then suggesting
  │         └────────► failed     (job-level: ENGINE_DRIFT, GPU_UNAVAILABLE, BACKEND_NOT_INSTALLED…)
  └── cancel_job ──► cancelling ─► cancelled  (finished renders/takes/analyses kept in the cache)
```

A job whose every layer is cached is created `completed`. A job that needs only scoring, because a take
is cached but its analysis key differs (other hints' aliases, other exact spans, a new QA profile), runs
only that phase. If every take of a segment then fails, its slots are retaken as in any job.

**Segment states**: planned → cached | rendering → rendered → postprocessed → scoring → passed | warned
| failed_qa | error | skipped.

**Takes, retakes, suggestion (per segment)**

```
round 0: the segment's attempts (default 0..takes-1), one per take slot → render → postprocess → score
for round r in 1..max_retakes:
    for each slot whose current take is a retake trigger (any fail flag, CUE_UNALIGNED, HEAD_INSERTION):
        attempt = next attempt number above every attempt so far (in slot order)   # from the request alone
        render → postprocess → score
suggest (V2), the first tier that has a take, lowest attempt within it:
    1. verdict pass                                   (a pass has every cue placed)
    2. verdict warn, every cue placed
    3. verdict warn, some cue unplaced (CUE_UNALIGNED)
    4. verdict fail (outcome needs_attention): every cue placed first, then the fewest fail flags
list: every take of every slot, with cues, samples and QA
```

Every attempt number, and so every seed, follows from the request alone. The same request therefore
asks for the same work, and the cache answers it.

**The caller chooses.** The suggestion is advice. The caller records the take it chose (its `take_id`,
or simply its attempt number) in its own files. There is no `select_take` and nothing to inherit: a take
id depends only on the clip, the engine text, the attempt and the engine profile, so the caller's choice
stays valid for as long as those are unchanged. If rendering proves bit-exact across processes (Phase 0
(d)), even a take evicted from the cache comes back with the same id; if not, it comes back as a new take
of the same attempt, and the caller's own copy of the old file is the record.

**What changed since last time (R11)** is the caller's comparison, made from two facts the results give
for every take: `fresh` (rendered by this job, or found in the cache) and its ids. A take id that differs
from the one in the caller's files means new audio; an `analysis_id` that differs with the same take id
means the cue times or the verdict may have moved (e.g. after an aligner re-pin). There is no `changes[]`
and no cut.

### Flow A: first narration of a script

```
Agent                     front-end                 SQLite/store          narrationd              workers
 │ design_voice ────────────►│ candidates: clip, sha256, transcript, profile                         │
 │ the person listens and chooses; the agent copies the clip into its own folder                     │
 │ get_server_status ───────►│ (the caller applies its own load rules)                               │
 │ measure_voice(clip) ─────►│ transcript check, calibration + ladder (one-off, 20–50 min of GPU)    │
 │ check_text(cues) ────────►│ per cue: received → spoken → engine, hints applied, text warnings     │
 │ fixes any text warning by editing its own text                                                    │
 │ submit_job(clip, hints, takes=2) ─►│ job row ───────►│ claim                 │                     │
 │◄── job_id                 │                          │                 │ load Qwen; canary hash ─►│
 │ get_job(wait_s=50) ──────►│ progress notifications ◄─┤◄── states ──────│ round 0: render ×2      │
 │   … repeated …            │                          │◄── takes ───────│ postprocess (CPU)       │
 │                           │                          │◄── analyses ────│ load QA; score; align   │
 │                           │                          │                 │ round 1: retakes if any │
 │ get_results ─────────────►│ text echo, takes (samples, sha256, trim, loudness, cues, QA),        │
 │                           │ suggestions, consistency, listen_first                               │
 │ copies takes + checks hashes; records its choices; its own review                                 │
```

What the caller does with the takes (timing, review, approval) is its own business.

### Flow B: a batch weeks later that must match

1. The caller sends the same clip path, sha256 and transcript, and the `expect_engine_profile` it
   recorded last time.
2. The service refuses with `ENGINE_CHANGED` if its engine profile differs, so a changed engine can never
   slip in unnoticed. The worker's fingerprint must also match the pinned profile, including `uv.lock`,
   the weights and the audio-changing settings (`ENGINE_DRIFT` otherwise).
3. **Canary gate** (section 10.1), on the service's own canary.
4. The measurement for this clip and engine is reused. Paragraphs already rendered come from the cache
   with the same take ids; new ones are held to the same anchor.
5. If the environment changed: restore it, or re-pin the engine (section 10.1), which the caller then
   sees as a new engine profile.

### Flow C: text edit and resubmission (R11)

1. The caller has chosen attempt 1 of `p07` by ear, and recorded its take id. It edits one cue of `p12`
   and sends the whole script again.
2. Only `p12` renders (× `takes`). Every other segment comes back from the cache, `fresh: false`, with
   the same take ids, `p07`'s attempt 1 among them.
3. The caller compares take ids with its files: only `p12`'s differ. It re-times `p12` alone. Its choice
   for `p07` still names the same take, so nothing is lost.

### Flow D: a name sounds wrong

1. The caller changes the hint for the term in the list it keeps, and sends it with the next request.
   The spoken text, and so anything the caller shows from it, stays the same.
2. `audition_pronunciation` renders the variants first, if wanted, and the caller chooses by ear.
3. The segments that contain the term have new engine text, so they render; the rest are cached.

### Flow E: another voice

The caller sends a different clip. Its `voice_hash` differs, so every segment renders anew (after the
clip is measured). Nothing of the first voice is touched.

### Flow F: designing a voice

`design_voice` (positive-only) → candidates with clips, transcripts and profiles → the agent shortlists
from the profiles' measurements and pictures → the person listens and
chooses → the caller copies the clip into its folder → `measure_voice`.

---

## 9. Text handling and pronunciation hints

**The service speaks the text it is given, and says exactly what it did to it** (R6, R7). The words are
the caller's decision: each cue arrives already in the words to be spoken. The service changes only
whitespace and Unicode form, applies only the pronunciation hints the request carries, and reports
anything that looks unspoken as a warning. A `text_checks_version` and a rules hash go into every
sidecar.

### 9.1 Pipeline (v1, `text_mode: "spoken"`)

1. **Sanitise** each cue. Control characters are refused, and so are `[`, `]`, `<|` and `|>`, which the
   engine could take as markup (`TEXT_REFUSED`; section 17, item 5). The cue is put in canonical form
   (section 7.2).
2. **Segment.** A text-only segment is one cue. If both `text` and `cues` are given, the text must
   equal their join (R2).
3. **Per cue** (R3: whatever the service does stays inside the cue it came from):
   1. **Hints.** The request's terms are matched case-sensitively, longest first, inside the cue, where
      a term stands as whole words: bounded by the cue's start or end, a space, or punctuation, the
      apostrophe included. So `Gastrella` matches in `Gastrella's`, and only the term is replaced
      (`Gas-trella's`). The *engine* text takes the hint's respelling, if it has one; the *spoken* text is
      unchanged. Each application is recorded in `hints_applied` {term, respell, offset}.
   2. **Checks.** They change nothing. Each finding is a text warning, `WRITTEN_FORM_TOKEN` (warn),
      with {token, kind, offset}:
      - `digit`: any digit, including superscripts and other Unicode digits;
      - `symbol`: any character that is not a letter, a space, or the punctuation a reader voices as
        prosody (`. , ; : ! ? ' ’ ‘ " “ ” ( ) - – — …`). This catches `%`, `&`, `/`, `×`, `°`, `·`, `#`,
        currency signs and the like;
      - `unit_like`: a standalone token of two or more characters in the service's generic list of
        unit symbols (SI symbols with their prefixes, and common abbreviations such as `km`, `kg`, `mph`,
        `ft`), matched case-sensitively, leaving out tokens that are ordinary English words in that case
        (`in`, `am`), plus the lone lower-case unit letters `m`, `s`, `g`;
      - `letter` (info, not warn): any other lone letter except `a`, `A` and `I`. A lone capital such as
        `C` or `J` may be an initial or a unit, so it is reported for a listener, not counted as an
        error.
   3. **Exact spans** are checked (section 7.2) and kept as word ranges.
4. **Join** the cues' spoken texts with one space, and the engine texts likewise. Record each cue's
   **span** in both. Nothing is added at the join: a cue with no full stop stays that way, because a
   sentence may run across two cues.
5. **Across cues.** A term that would match only across a cue boundary is not applied, and gets
   `TERM_SPLIT_ACROSS_CUES` (warn).

**Text warnings never change the text, and they block only on request.** A digit left in a cue is
spoken however the engine reads digits. `strict_text: true` refuses instead. Each warning also puts its
cue on `listen_first`.

**Pronunciation hints.** The caller keeps its pronunciation list and sends the hints a request needs,
with the request (the Hint fragment, section 7.2). The service keeps no list:

- `term`: the word or words as they appear in the spoken text;
- `respell` (optional): what the engine is given instead. Qwen has no phoneme input, so a respelling is
  a hint, never a guarantee. `audition_pronunciation` lets the caller hear variants and choose;
- `align_as` (optional): letters for the aligner (section 11.2);
- `asr_aliases`: how the ASR tends to write the term, for QA;
- `note`.

The service keeps no approval of a term either. A term sent without a respelling still helps: QA
recognises it as a name, collapses it in `wer_adj`, and reports whether it was heard (section 11.1).
An invented name that is not sent as a hint counts against the word error rate like any misheard word.

**Consistency.**

- A different term or respelling changes the engine text of the segments that contain the term, so
  only those render again; the rest come from the cache.
- Different `asr_aliases` / `align_as` change only the analysis key, so those segments are re-scored,
  never re-rendered.
- Every segment of one request is rendered with the same hints, so one request never mixes two
  renderings of a term.

### 9.2 Later phase: a normaliser for written text

A caller that sends written text (digits, units, symbols) would need a normaliser in front of 9.1,
selected by `text_mode: "written"`. Revision 3 designed one: en-GB number words, identifiers read after
set keywords, a unit table. Much of it was one caller's conventions. That caller now sends spoken words
(R6), and under the owner's principle how numbers, units and names are read is a caller's choice. So v1
has no normaliser, and `"written"` is refused (Q20).

If one is built later, it is generic:

- its conventions (locale, number style, zero word, how an identifier is read) are request settings
  with neutral defaults, never one caller's vocabulary;
- it shows every substitution it makes;
- its output goes through 9.1 unchanged;
- its tests use the service's own material (9.3).

### 9.3 Tests

The text tests use the service's own fixtures, never a caller's script:

- canonical form, the join, and a `text` that differs from the join (R2);
- hints: whole word, case, longest match, a possessive (`Gastrella's`), never across a cue (R3), and
  the recorded offsets;
- every warning kind, the exclusions (`a`, `A`, `I`, `in`), lone capitals as `letter` (info), and
  punctuation that must pass (`—`, `…`, curly quotes);
- exact spans: code-point offsets in non-ASCII text, a span that cuts a word refused, and word ranges
  that survive canonicalisation;
- markup refused.

---

## 10. Determinism and consistency

### 10.1 Layers, strongest first

1. **Identity is pinned by the request and the service**: the voice hash (clip, transcript, ICL mode),
   which the caller controls by sending the same clip, and the engine profile hash (model, weights,
   env, audio-changing settings), which the service reports on every take and a caller can require with
   `expect_engine_profile`.
2. **Cache before work**, per layer (section 10.2), checked again at claim time (section 4).
3. **Reproducible re-render.**
   - Seeds are derived from the request (section 10.3); batch size is 1.
   - Determinism switches, pinned in the engine profile:
     - `sdpa` attention;
     - TF32 off;
     - cuDNN deterministic, benchmark off;
     - `torch.use_deterministic_algorithms(True, warn_only=True)`, except during the whole
       `create_voice_clone_prompt` call (the voice prompt's encode and speaker embedding). There the worker
       turns it off and restores the previous mode on every exit path. Torch 2.11's deterministic
       replicate padding cannot take the Mimi encoder's tensor-valued pad, so with the switch on, encoding
       a voice fails. Padding's forward pass is a gather, which is deterministic either way (KNOW; ADR 0002);
     - **`CUBLAS_WORKSPACE_CONFIG=:4096:8`**, set in the worker's environment before CUDA starts. This
       is required for deterministic cuBLAS.
   - Audio-changing Qwen settings, pinned explicitly (never left to library defaults):
     - `x_vector_only_mode=false` (ICL, in the voice hash);
     - `non_streaming_mode=false` for Base, which is what every piece of clone evidence used. Change it
       only if a Phase 0 A/B justifies `true`, and that would mean a new engine profile;
     - `non_streaming_mode=true` for VoiceDesign (its default);
     - the effective sampling values from the pinned snapshot's `generate_config.json`, or the library's
       fallbacks, passed explicitly;
     - `max_new_tokens`, in two parts (DC-4, the owner's decision of 2026-09-26; ADR 0003):
       - `load` passes the snapshot's value, 8192, as the ceiling;
       - every `synthesize` and `design` call passes its own cap, computed by the daemon from the text
         the call speaks: `min(8192, max(floor, ceil(per_char × len(text))))`, with `per_char` = 2.5 and
         `floor` = 128 from `[engines.*]`, hashed into the engine profile. Speech measured 0.70–0.82
         frames per character (KNOW), so the factor leaves about three times the largest rate, and the
         floor is about 10 s of audio.
       - The cap only truncates (KNOW: renders at 8192, 2048 and the exact step count are
         bit-identical), so a render that ends under its cap is the same under any cap. A runaway
         render stops after about a minute instead of about 22 (BELIEVE, extrapolated).
   - **Files.** Identical audio makes an identical file. A worker writes its WAV itself (format, `fact`
     chunk and data), because `soundfile.write` adds a `PEAK` chunk holding the time of writing, which
     would give two identical renders different hashes (KNOW; ADR 0002).
   - **Tier.** The Phase 0 repeat test (d) found both paths **`bit_exact`** on the pinned stack (KNOW, on
     one machine; ADR 0002). Forty clone renders in seven processes, in three orders, gave one hash per
     item, with the switches above or with torch's defaults; VoiceDesign matched too. Nothing is promised
     across machines, so `narration-admin engine pin` repeats a short version of that test on the
     installing machine and records the tier: all hashes equal gives `bit_exact`, anything else
     `similar`, which `doctor` reports. The tier is reported in `get_server_status`:
     - **`bit_exact`**: a take evicted from the cache comes back with the same id; the canary gate
       compares hashes.
     - **`similar`** (if the pin's repeat test differs): the same request renders the same attempts again but not the same
       bytes, so a take id is stable only while its take is cached. The canary gate compares similarity
       only, and `CANARY_MISMATCH` is not raised on every batch. Callers keep their own copies (they do;
       R12), and the design says plainly that an attempt reproduces a delivery, not a file.
4. **Canary gate** before every batch, on the **service's own canary**: a clip designed **on the
   installing machine** by `narration-admin engine pin` (revision 5.2, DC-3), from a description, a text
   and a seed that ship with the source as text. Its raw hash and embedding are stored per engine
   profile. No canary audio ships: nothing is promised across a GPU, driver or CUDA change, so a shipped
   hash would not match on another machine. The gate checks this engine, on this machine, over time,
   not any caller's voice.
   1. After loading Qwen, render the canary (a few seconds).
   2. **Hash equal**: proceed at once. No QA is needed and there is no model swap.
   3. **Hash different**: pause the batch. The QA worker computes the canary's similarity to its stored
      embedding with WavLM **on the CPU**, while Qwen stays resident: seconds, with no swap.
      - If it is ≥ the canary's calibrated threshold: continue, and add `CANARY_MISMATCH` (info) to
        every take in the batch (e.g. after a driver update; never in the `similar` tier).
      - Otherwise: fail the job with `ENGINE_DRIFT` before rendering anything.
5. **Perceptual check**: QA similarity of every take to its voice's anchor (section 11.1).

There is no promise across a GPU, driver, CUDA or cuDNN change. These are recorded, and the canary
detects them.

**Engine pins.** `narration-admin engine pin` records an engine profile. `engine repin` makes a new
profile the one in use: its hash differs, so every render key differs, every request renders anew, and a
caller that sends `expect_engine_profile` is refused until it accepts the new hash. Measurements are per
engine profile, so each clip is measured again under the new one. `engine bridge <old> <new>` renders the
service's canary and the calibration corpus under both profiles and reports their similarity, so the
owner can judge whether a re-pin will be heard before making it. Both are operator commands; neither
changes a cached file.

### 10.2 Keys: three cache layers

All hashes are sha256 over RFC 8785 canonical JSON. Every key is computed from the request and the
service's pins, never from earlier requests.

| Layer | Key | Produces |
|---|---|---|
| — | `voice_hash` = H({schema: "narration.voice/v2", model, clip_sha256, transcript (NFC), language, x_vector_only_mode}) | |
| — | measurement key = H({voice_hash, engine_profile_hash, corpus version, ladder settings}) | the voice's measurement |
| **Render** | `render_key` = H({schema: "narration.render/v1", engine_profile_hash, voice_hash, engine_text, seed}) | `raw.wav` (`render_id`) |
| **Delivery** | `delivery_key` = H({raw_sha256, delivery profile (trim rule, target LUFS, TP ceiling, sample rate, subtype, fades), **the resampler's and loudness meter's names and versions**, the post-processing rules' version (`narration.post/1`), stretch: null}) | `delivery.wav` (**`take_id`**) |
| **Analysis** | `analysis_key` = H({delivery_sha256, spoken_text, cue spans, exact spans, the QA inputs of the hints used (`term`, `asr_aliases`, `align_as`), QA profile version, text-checks version, number reader version, ASR model rev, SV model rev, aligner method id, the voice's measurement key}) | QA verdict, flags, exact-span results, cue and word times (`analysis_id`) |

- **Planning** walks the layers. A render hit with a delivery miss is post-processing only; a take hit
  with an analysis miss is scoring only.
- **Fit** is outside every key: it is computed per request from the take's duration and the request's
  `scene_seconds` and `fit`.
- **Consistency across a request** (section 11.1) is outside every key too: it depends on the whole
  request, so it is computed per job.

### 10.3 Seed derivation

```
seed = uint32( sha256( "narration-seed/v1" ‖ voice_hash ‖ sha256(engine_text) ‖ attempt ) [0:4] ) & 0x7FFFFFFF
```

- The worker seeds `torch`, `torch.cuda`, `numpy` and `random` with it.
- The segment id is left out on purpose: the same text in the same voice gives the same take in every
  script.
- A segment's attempts are its `attempts`, or 0..`takes`−1; retakes use the next attempt numbers above
  every attempt so far, in slot order.
- A caller may name attempts, never an arbitrary seed.

---

## 11. QA pipeline, thresholds and cue alignment

### 11.1 QA per take (always on the delivery file)

**Order:** render → post-process → QA. Every analysis runs on `delivery.wav`, so cue times are
delivery-file seconds. A take's verdict depends only on that take and the request's inputs for it, never
on other takes, because verdicts are cached (section 10.2).

1. **Signal checks**: clipping (on raw), longest internal silence, NaN/DC (`SIGNAL_INVALID`, DC-5), and
   **token cap**. A render
   whose generation reached its call's `max_new_tokens` (DC-4) is `TOKEN_CAP_HIT` (fail), because the
   model stopped mid-text or ran away, and that must never pass silently. The worker reports it exactly
   (the last token was not the end token), never from the audio's length.
2. **ASR**: Whisper-large-v3 (fp16, English, greedy, word timestamps; revision pinned). A take can be
   longer than Whisper's 30 s window, so the QA profile pins sequential long-form transcription (30 s
   windows, each conditioned on the text before it) with word timestamps.
3. **Cue alignment** (section 11.2).
4. **Text match** against `spoken_text`.
   - `wer_raw` uses Whisper's normaliser and is reported for comparison only. On name-dense text it
     reaches 5.5–8.6 % (worst segment 16–21 %) purely from name spelling.
   - **`wer_adj`**, the gate, additionally removes apostrophes, maps `asr_aliases`, and collapses each
     hinted term (fuzzy-matched) to one token on both sides.
   - A rate alone is unfair on a short segment: one slip in a three-word title is 33 %. So the gate also
     counts words: **warn** when `wer_adj` > 0.02 and there is at least 1 word error; **fail** when
     `wer_adj` > 0.06 and there are at least 2.
5. **Exact spans** (R14; section 11.3). A span where the transcript shows a different value or different
   words is `EXACT_SPAN_MISMATCH` (fail), whatever `wer_adj` says. A misread number in a paragraph of
   sixty words is under 2 % of it, which the rate alone would pass.
6. **Terms.** For each occurrence of a hinted term, with a possessive `'s` set aside, a letters-only
   fuzzy match (≥ 0.75, or an alias) is `ok`. Otherwise `TERM_UNVERIFIED` (warn, never fail).
7. **Insertions.**
   - **Head:** ASR words before the first aligned word of cue 0, or speech before the aligned onset that
     fuzzy-matches the voice's transcript. The latter is reference bleed, a known ICL failure mode.
     `HEAD_INSERTION` is a warn at ≥ 1 word and a fail at ≥ 3 words, or on any match of the voice's
     transcript.
   - **Tail:** `END_INSERTION`, a warn at ≥ 1 word and a fail at ≥ 3.
8. **Speaker**: WavLM-SV similarity to the voice's anchor, from its measurement. This is in the verdict.
9. **Pace**: **spoken** words per minute (and spoken characters per second) over the voiced span,
   compared with the voice's pace curve at this segment's spoken length.
10. **Fit** (only with `scene_seconds`): reported, never remedied, in v1.

**Consistency across the request (a report, not a verdict).** After every take is scored and each
segment's suggestion is made, the service embeds the suggested takes, and compares each with their
centroid. Takes below the voice's consistency baseline (p5 − 0.01) are listed as `outliers` in the
result's `consistency`, flagged `SPK_OUTLIER` (info), and put on `listen_first`. This never changes a
verdict or a suggestion: it depends on which other takes are in the request, so it is computed per job
and cached nowhere. (Revision 4 compared each take with a running centroid inside its cached verdict,
which made verdicts depend on render order and leak from one script to another.)

**Default thresholds** (QA profile `default.v3`)

| Check | Warn | Fail | Rationale / evidence |
|---|---|---|---|
| `wer_adj` | > 0.02 and ≥ 1 word error | > 0.06 and ≥ 2 word errors | Takes with no real misreading scored raw 0.00–0.038 on caption text; the name-dense probe scored raw 5.5–8.6 %, all from names, which `wer_adj` collapses. The word counts stop one slip failing a short segment. |
| exact spans (R14) | — | any different value or word | "Body 2001" passed at WER 0.015; a word rate cannot see one wrong number. The probe had zero number mismatches in 48 paragraph-takes. |
| terms | unverified | never | The probe heard stable, unstable and split variants of names (section 1). |
| spk_sim vs anchor | < **voice `anchor_p5` − 0.01** | < **0.90** (absolute floor, ASSUME) | Per-voice thresholds from the voice's measurement: the probe's d4 scored 0.966–0.969 vs its clip and d2 0.978–0.984, so a fixed 0.975 would flag every d4 take. The floor only catches gross failure; unseeded VoiceDesign drift sat at 0.898–0.932. |
| consistency across the request | — (report only: `SPK_OUTLIER`, info) | — | Probe `spk_consist` 0.981–0.992. |
| pace vs curve at this length | > curve × (1 + tol) or < curve × (1 − tol) | > curve × (1 + 2·tol) | `tol` from the ladder (≥ 10 % and ≥ the measured seed spread, up to 17 % within one voice in the probe). |
| head / end insertion | ≥ 1 word | ≥ 3 words, or a match of the voice's transcript at the head | Reference bleed; hallucinated tail. |
| longest internal silence | > 1.2 s | > 2.5 s | Dropout or hang. |
| clipping (raw) | > 0.01 % of samples at full scale | — | Gain problem. |
| signal (raw, `SIGNAL_INVALID`) | DC offset > 0.001 | any non-finite sample | Real output had |DC| ≤ 0.00002 (DC-5, DC-10). |
| token cap | — | reached | Truncation by the model. |
| cue alignment | `CUE_LOW_CONFIDENCE`, `CUE_ALIGNMENT_DISAGREE`, `CUE_UNALIGNED` (warn, R1) | `ALIGNMENT_ERROR` (fail) | Section 11.2. |
| fit (only with `scene_seconds`) | `FIT_TIGHT` (slack < 1.0 s) | — (`OVER_SCENE` is a warn in v1: reported, not remedied) | Section 12. |

**Retake triggers.** Any fail-severity flag (so `EXACT_SPAN_MISMATCH` too), **`CUE_UNALIGNED`**
(warn-level, but a caller cannot time the cue), or `HEAD_INSERTION`. Each failing take slot gets up to
`max_retakes` automatic retakes (section 8). The exception (DC-12): a `CUE_UNALIGNED` whose
`details.reason` is `no_alignable_words` is not a trigger. That cue's text gives the aligner no word to
place, so every retake would fail the same way; it stays in listen-first.

**Report and listen-first.**

- `report.md` lists every flag, including replaced attempts, plus each cue's received → engine text.
- `listen_first` is what QA found, in this order:
  1. fails;
  2. exact-span and term flags;
  3. cue-alignment flags;
  4. insertion, similarity and pace warnings, and consistency outliers;
  5. cues with a text warning, and segments over the voice's reliable length (the listener hears what
     the engine made of them).

  How much more to listen to is the caller's review, so there is no spot-check quota.

**No approval of results.** The service approves nothing and records no choice. Accepting a take is the
caller's decision, recorded on the caller's side.

### 11.2 Cue alignment (R1, v1)

**Forced alignment is primary, and Whisper is the cross-check** (BELIEVE; the benchmark below
decides).

- The spoken text is known exactly.
- Whisper's word times come from cross-attention alignment, which is loose.
- Whisper spells invented names unpredictably, which makes mapping its words to cues fragile at the
  names.

**Steps** (on the delivery file, resampled to 16 kHz internally; times are reported in delivery-file
seconds):

1. **Transcript.** Spoken text per cue, in the model's alphabet (letters folded to A–Z with accents
   dropped, apostrophe, `|` between words; a hyphen inside a word becomes `|` too). Spoken text should
   hold no digits (R6). A token with characters outside the alphabet, a digit or symbol left in, is
   already a text warning; it is left out of the transcript, and the words around it still align.
   Hinted terms use `align_as` letters if given, otherwise their spoken letters. A token → (cue, word)
   map is kept.
2. **Emission.** `facebook/wav2vec2-large-960h-lv60-self` (revision pinned) **on the CPU**, with the
   thread cap. A 35–45 s paragraph fits in one pass at 20 ms frames.
3. **Viterbi.** `torchaudio.functional.forced_align` + `merge_tokens` (BELIEVED to run on this build's
   CPU; Phase 0 re-checks and saves it).
   - **Guard:** `forced_align` **raises** when the frames are fewer than tokens + repeats. The worker
     pre-checks `T ≥ L + R` and also catches the exception.
   - Either way, every cue of the take becomes `CUE_UNALIGNED` and the take gets `ALIGNMENT_ERROR`
     (fail, retake trigger). An audio too short for its text is a broken take.
4. **Snap into pauses.** Each boundary between cue *k* and *k+1* is moved to the speech edges of the
   silence gap between the two aligned words (20 ms energy frames). With no gap, the CTC times are kept
   and `CUE_BOUNDARY_NO_PAUSE` (info) is added. The first onset and last offset are snapped the same
   way.
5. **Confidence.** The mean token posterior per cue. Below `low_confidence_below` (0.75, ASSUME;
   set from the benchmark): `CUE_LOW_CONFIDENCE`.
6. **Cross-check.** At each boundary, compare with Whisper's word boundary where the neighbouring words
   match (names excluded). A difference above 0.25 s (ASSUME; set from the benchmark) gives
   `CUE_ALIGNMENT_DISAGREE`. `max_disagreement_s` is reported.
7. **Unplaceable cue**: confidence below `unplaced_below` (0.50, ASSUME), or its words are missing
   from the transcript. It gets `start_s` / `end_s` = null, `CUE_UNALIGNED` (warn, listen-first, retake
   trigger). **It is never interpolated.** A cue with no word the aligner can place at all has
   `details.reason` = `no_alignable_words`, and is not a retake trigger (DC-12, section 11.1).
8. **Output**: `cues[]` with `words[]` in delivery-file seconds, and `alignment` {method
   `ctc-forced-align+silence-snap`, model, revision, cross-check, measured error, flags}.
   - The analysis key names the aligner by its **method id**, `ctc-snap/<model>@<revision>+p<12 hex>`.
     The hash covers every setting the aligner decides by (the pause and snap parameters and the
     thresholds), so a changed threshold changes the key and no cached verdict is reused.

**Accuracy (R1): measured and published, not promised.** The service sets no target for callers. How
much error a caller can bear is the caller's. The service measures its own error and publishes it with
the method:

- **Benchmark** `alignment-en.v1`, the service's own material: about 12 paragraphs of narration in
  spoken form, with common words, number words, invented names, short and long cues, and cue
  boundaries with and without a pause. The builder writes them to those rules, with no digits or
  symbols, and the owner listens to a sample once before they are frozen. No caller's script is used.
- **Marks.** The renders are hand-marked in a waveform editor, by the owner or a delegate, at each
  cue's first-word onset and last-word offset: about 60 marks per render.
  - **Phase 0** marks one voice at one seed, about 30 marks, with a prototype of the aligner that Phase
    3 keeps. That is enough to choose the method and see the size of the error.
  - **Phase 3** completes it: two voices (the service's canary voice and one other), two seeds, about
    240 marks.
- **Error** = |service time − hand mark| over every cue start and end. Reported as p50 and p95 with n,
  overall and split by boundary kind (in a pause, with no pause, segment edge).
- Computed for CTC, CTC + snap, Qwen3-ForcedAligner (+ snap), and Whisper.
- **Choosing the method** (owner decision, 2026-09-26, Q18):
  - The wav2vec2 CTC aligner with the snap is primary by default, and Whisper is the cross-check.
  - Qwen3-ForcedAligner replaces it only if it is **clearly better at boundaries with no pause**: a p95
    lower by at least 20 ms there (ASSUME: one CTC frame), and no worse overall. Within a pause the
    snap decides the time, whichever aligner found the pause.
  - Phase 0 makes the choice on its ~30 marks. Phase 3's full benchmark confirms it; a reversal then is
    a new method id, measured like any other.
  - If Qwen3-ForcedAligner is chosen, step 5's confidence comes from whatever score it exposes, or the
    Whisper agreement alone; Phase 0 settles which.
- The spread also sets the thresholds of steps 5 and 6.
- **Published** by `narration-admin bench alignment` as an AlignmentBenchmark record: in
  `get_server_status.alignment` and in every take's `alignment.measured_error`, with the method id
  (aligner model + revision + snap parameters) and the benchmark's id, sha256 and description. A new
  method id is measured again before it is used.

The GPU path for the wav2vec2 aligner is untested and not needed.

### 11.3 Exact spans (R14)

A caller may mark spans of a cue's spoken text that must be heard exactly, such as a number in words
(section 7.2 gives the form).

1. **Normalise both sides the same way.** The span's expected words and the whole transcript are put
   through Whisper's English text normaliser, after one extra rule that reads "nought" as "zero". Since
   revision 5.3 (DC-7, reader version `@2`), each side is read in phrases split at punctuation, so number
   words never merge across a comma ("two thousand, forty" is `2000 40`, not `2040`), and ’ ‘ ʼ are
   read as the straight apostrophe. Version `@1` failed perfect takes on both counts. As built
   (revision 5.6), `@2` also reads both sides alike in four more places:
   - it joins a clock time (one or two digits, a colon, two digits), so "4:30" reads like "four thirty";
   - it reads a hyphen-minus or minus sign directly before a digit, at the start of a word, as "minus";
   - it rewrites "°C", "°F" and "°" as "degrees celsius", "degrees fahrenheit" and "degrees";
   - after the normaliser, it splits money at the decimal point, so "£3.50" reads like "three pounds
     fifty".

   The phrase split has one known cost: a comma inside one spoken number splits it ("three thousand, two
   hundred" reads `3000 200`, while "3,200" reads `3200`). Other known limits are listed for the
   acceptance tests to measure (WP14's status, WP40). That
   normaliser turns number words into digits and treats spelling variants alike. Tested here (KNOW,
   `eval/normaliser_check.py` and its output `.txt`, transformers 5.17.0):
   - "fourteen per cent", "fourteen percent" and "14%" all become `14%`;
   - "two oh one", "two o one" and "two zero one" all become `201`;
   - "four thousand and forty-nine" and "four thousand forty-nine" both become `4049`;
   - "thirty-five fifty-nine" becomes `3559`; "3,200" becomes `3200`;
   - "nought point five four" is the one miss (`nought .54`), hence the extra rule.
2. **Locate.** The normalised transcript is aligned to the normalised spoken text, word by word (an edit
   alignment). What the transcript has at the span's words, and anything inserted between them, is what
   was heard there. No times are used, so this works on a cue the aligner could not place.
3. **Compare.** The two normalised forms must be equal. Otherwise the span is `EXACT_SPAN_MISMATCH`
   (fail, retake trigger), with {cue, start, end, expected, heard}. Each span's outcome is in
   `qa.exact[]` with `match`: `same` | `different` | `missing`, and `qa.exact_ok` sums them up.

**What this confirms, and what it does not** (owner decision, 2026-09-26). Because both sides become
digits, the check confirms **which number was spoken**, not how it was worded: "thirty-two hundred" and
"three thousand two hundred" both become 3200 and pass. The danger R14 names, a wrong number, is caught.
A change of wording is unlikely anyway, because the voice reads the words it is given. Words that are not
numbers are still compared word for word.

A hinted term inside a span matches only by its spelling or an `asr_alias`. Exact spans suit words the
ASR can spell; a name is better left to the term check.

---

## 12. Scene timing and fit

**v1 reports; it never remedies.** Without `scene_seconds` there is no fit computation, no fit flag,
and no fit block (R4). This is the usual case, where the caller sets its lengths *from* the narration.

With `scene_seconds`:

- budget = `scene_seconds − fit.lead_in_s − fit.tail_s`. Both default to **0**, so the service adds no
  padding assumptions of its own (padding is the caller's).
- The result is computed per request from each take's delivery duration:
  - `slack_s`, always;
  - `FIT_TIGHT` (warn) when slack < 1.0 s;
  - `OVER_SCENE` (warn), with `overrun_s`, when over budget. How to shorten the text is the caller's.
- Prediction before render comes from the voice's pace curve (`check_text`, `dry_run`).
- Nothing is ever truncated or stretched.

**Later phase (section 13.1):** fit remedies, i.e. retake-for-length (seeds vary length by 2–17 %) and
time-stretch ≤ 1.06, flagged `post_stretched`. They exist for a future picture-first case. The one known
caller wants none (Q7).

---

## 13. Audio output and post-processing

- **Raw** (render layer): the model output, WAV float32 mono at 24 kHz, untrimmed. Its sha256 is the
  reproducibility check.
- **Delivery** (take layer): deterministic, on the CPU, thread-capped, in this order:
  1. **Trim.** The threshold is **relative to the take**: its speech level (the 95th percentile of 20 ms
     frame RMS) − 40 dB, and never below −70 dBFS. Frame RMS is measured on the take with its mean
     removed; the audio itself is not changed (revision 5.4, DC-10: a DC offset of 0.001 made every frame
     speech, and a take under 5 % speech was never trimmed). It is gain independent for every take whose
     speech level is above −30 dBFS. `head_s` and `tail_s` are the silence found below the
     threshold at each end of the raw audio; up to `pad_s` (0.08 s) of it is kept at each end. A take
     with less silence than `pad_s` at an end keeps what it has, and no silence is added (revision 5.3,
     DC-9; 23 of 48 real takes had less at the tail). So the delivery's length is the raw length −
     max(`head_s` − `pad_s`, 0) − max(`tail_s` − `pad_s`, 0), before resampling rounds it to whole
     samples, and `head_s`/`tail_s` report the silence found.
  2. **Resample** to 48 kHz with a pinned resampler (its name and version are in the delivery key).
  3. **Gain.** Static gain to **−23 LUFS** integrated by default (`target_lufs`; BS.1770-4, measured on
     the mono signal as a single channel with weight 1.0); no limiter. The meter's name and version are in
     the delivery key. *Revision 5.3 (DC-8):* the default was −16 LUFS, but on the bake-off's 48 real
     clone paragraphs the ceiling below held every take under it (−21.3 / −18.8 / −16.7 LUFS, min /
     median / max; peak-to-loudness ratio 15.7–20.3 dB), so takes of one script differed by up to
     4.6 LU. At −23, EBU R128's pair with this ceiling, all 48 reach the target (−22 does too; −20
     only 40; revision 5.4).
  4. **Fades**: 0.01 s at the edges.
  5. **Quantise** to WAV PCM_24 mono.
  6. **True peak**, measured on **this final 48 kHz file** (4× oversampled). If it is above −1.0 dBTP,
     lower the gain until it is ≤ −1.0, re-quantise, and flag `LOUDNESS_UNDER_TARGET` (info) with the
     shortfall. A take with no measurable loudness (every block under the −70 LUFS gate) reports
     `measured_lufs` as null. `true_peak_dbtp` is null only for an all-zero file: a quiet take still has
     a measurable peak (revision 5.5).

  **The true-peak ceiling wins over the loudness target**, so takes of one script can sit at slightly
  different loudness. Each take's **loudness record** {measured_lufs, gain_db, true_peak_dbtp,
  ceiling_applied} is in the results, so a caller mixing under music can level them without measuring
  again. A gain above +12 dB is flagged `GAIN_HIGH` (info).
- **Exact lengths (R5, R12).** `samples`, `sample_rate`, `duration_s`, `sha256`, `trim` {head_s,
  tail_s, pad_s} and `loudness` for every take, in `get_results`. Cue times use the same coordinates.
- **Handover.** Absolute path + sha256. A file stays at least for the retention period after it was last
  used (section 15). There is no export alias; the caller copies what it keeps and checks the hash. The
  service's cache is for speed; the caller's copies are the record.
- **Idempotency.** An identical `submit_job` returns the active job or completes from the cache.
  `idempotency_key` deduplicates retries.

### 13.1 Later phase: stitching and fit remedies (not in v1)

Placement and padding are the caller's (R4, Q7), so these wait until a caller needs them:

- `stitch`, with modes `sequential` (with `pause_after_s`) and `timeline` (with `timeline_offset_s` and
  per-segment `start_s`), taking the takes to stitch by id in the request;
- the fit remedies of section 12.

None of these fields are in the v1 schemas.

---

## 14. Error model

**JSON-RPC protocol errors** are only for:

- an unknown tool (−32602);
- a malformed request that is not a valid `CallToolRequest` (−32600 / −32602);
- a missing resource (−32602);
- an internal failure outside a tool call (−32603). A failure inside a tool call is a tool error with
  `INTERNAL`, so the model sees it (revision 5.2; ADR 0001).

**Tool execution errors** are for everything about the *arguments*, including JSON-Schema validation
failures. Following 2026-07-28's intent, the model gets actionable feedback: `isError: true`,
`structuredContent: {"error": Error}` (with `field` and `hint`), and a text copy.

- The front-end validates arguments itself, inside the handler, so that SDK-level validation never turns
  them into JSON-RPC errors.
- A Phase 4 test: sending `instruct` returns `isError` + `INVALID_ARGUMENT`, with `field: "instruct"` and a
  hint pointing at section 3.3.
- `retryable` is true only when the same call may succeed later unchanged (the GPU was busy, the disk
  was full). When the arguments must change first, it is false.

| Code | Retryable | Meaning / hint |
|---|---|---|
| `INVALID_ARGUMENT` | no | schema or semantic failure, e.g. an unknown field, duplicate segment ids, `text` ≠ the join (R2), `text_mode: "written"`, an exact span that cuts a word, an `idempotency_key` reused for a different request (DC-6) |
| `LIMIT_EXCEEDED` | no | request-size limits (segments, cues, characters, hints); a full queue is `QUEUE_FULL` |
| `NOT_FOUND` | no | a job, design, take or measurement (e.g. expired after the retention period) |
| `PATH_NOT_ALLOWED` | no | a clip or audio path that is not absolute, resolves to a network share or a device path, or is not a regular file (section 17) |
| `UNSUPPORTED_AUDIO` | no | not a WAV the service reads, or longer than 30 s / larger than 20 MB for a voice clip |
| `VOICE_FILE_MISMATCH` | no | the file's sha256 is not the one sent; the clip changed or the path is wrong |
| `VOICE_NOT_SYNTHETIC` | no | the clip is neither one the service designed nor one the owner allowlisted (section 17) |
| `VOICE_NOT_MEASURED` | no | no measurement for this voice under the current engine; hint: `measure_voice` |
| `REF_TEXT_MISMATCH` | no | measuring found that the transcript does not match the clip |
| `ENGINE_CHANGED` | no | `expect_engine_profile` differs from the service's engine profile; hint: accept the new hash, or ask the owner to restore the old one |
| `CONTROL_UNSUPPORTED` | no | `pace`, `context_before/after` (R10) |
| `TEXT_REFUSED` | no | markup characters, or `strict_text` with text warnings left; every offender listed |
| `ENGINE_DRIFT` | no | fingerprint mismatch, or the canary similarity is below threshold |
| `BACKEND_NOT_INSTALLED` | no | weights or worker env missing |
| `DAEMON_UNAVAILABLE` | yes | cannot start detached (breakaway refused), or stopping; hint: `narration-admin daemon start` |
| `GPU_UNAVAILABLE` | yes | the VRAM wait timed out |
| `STORE_FULL` | yes | free disk below the minimum |
| `JOB_NOT_CANCELLABLE` | no | the job is already terminal |
| `INTERNAL` | maybe | a bug; the log path is included |
| `QUEUE_FULL` | yes | the job queue is full; `retry_after_s` from the queue's drain estimate (revision 5.2, DC-2) |
| `RATE_LIMITED` | yes | too many submissions in the rate window; `retry_after_s` until the window frees (revision 5.2, DC-2) |

Every retryable error carries **`retry_after_s`** (revision 5.2, DC-2), and `details` the facts behind
it, e.g. `GPU_UNAVAILABLE` {free_mb, need_mb, waited_s}.

**Flags** (inside results; "R" marks a retake trigger)

| Code | Severity | R | Meaning |
|---|---|---|---|
| `WRITTEN_FORM_TOKEN` | warn (info for a lone `letter`) | | text check: a digit, symbol or unit-like token in spoken text (section 9.1) |
| `TERM_SPLIT_ACROSS_CUES` | warn | | text check: a term would match only across a cue boundary, so no hint was applied |
| `SEGMENT_TOO_LONG` | warn | | the segment is longer than the voice's reliable length; it is still rendered (section 3.2) |
| `WER_HIGH` | warn / fail | fail | `wer_adj` above the threshold, with the word-count rule |
| `EXACT_SPAN_MISMATCH` | fail | ✓ | a different value or different words inside a span the caller marked exact (R14, section 11.3) |
| `TERM_UNVERIFIED` | warn | | ASR did not match a hinted term |
| `SPK_SIM_LOW` | warn / fail | fail | similarity to the voice's anchor below the measured warn threshold / the floor |
| `SPK_OUTLIER` | info | | a suggested take stands apart from the rest of the request (a report, never a verdict) |
| `PACE_FAST` / `PACE_SLOW` | warn / fail | fail | against the voice's pace curve at this length |
| `HEAD_INSERTION` | warn / fail | ✓ | words before cue 0, or reference bleed |
| `END_INSERTION` | warn / fail | fail | words after the last cue |
| `SILENCE_LONG` | warn / fail | fail | longest internal silence |
| `CLIPPING` | warn | | raw samples at full scale |
| `SIGNAL_INVALID` | warn / fail | fail | non-finite samples in the raw take (fail), or a DC offset (warn); revision 5.3, DC-5 |
| `TOKEN_CAP_HIT` | fail | ✓ | generation stopped at `max_new_tokens` |
| `CUE_UNALIGNED` | warn | ✓ | cue not placed; times null; never interpolated |
| `CUE_LOW_CONFIDENCE` | warn | | alignment posterior below threshold |
| `CUE_ALIGNMENT_DISAGREE` | warn | | CTC vs Whisper boundary differ > threshold |
| `CUE_BOUNDARY_NO_PAUSE` | info | | no silence gap at a cue boundary |
| `ALIGNMENT_ERROR` | fail | ✓ | the aligner raised or could not run (frames < tokens + repeats) |
| `FIT_TIGHT` / `OVER_SCENE` | warn | | only with `scene_seconds`; reported, not remedied |
| `CANARY_MISMATCH` | info | | canary hash differed but similarity passed (`bit_exact` tier only) |
| `LOUDNESS_UNDER_TARGET` / `GAIN_HIGH` | info | | the true-peak ceiling lowered the gain / gain above +12 dB |
| `RETAKEN` | info | | an earlier attempt of this slot failed (listed) |
| `RENDER_FAILED`, `WORKER_CRASHED`, `GPU_OOM`, `QA_UNAVAILABLE`, `CANCELLED` | error | | segment-level execution problems |

---

## 15. Storage layout

`store_root` is wherever the owner's configuration puts it; the design assumes no location (Q9).
Everything in it is either
the service's own or a cache of work done; nothing in it is a caller's record.

```
<store_root>\
  narration.sqlite                   jobs, queue, cache index, retention (WAL)
  run\daemon.json                    pid, workers (while running)
  engines\<engine_profile_id>.json   ⊘ (+ the canary's raw hash and embedding)
  alignment\<method_id>.json         ⊘ measured cue-boundary error on the benchmark (R1)
  provenance.jsonl                   ⊘ append-only: the fingerprint of every clip the service designed
  designs\<design_id>\<cand>\        clip.wav · candidate.json · profile\ (pictures)
  measurements\<voice_hash>\<engine_profile_id>\   ⊘ measurement.json · calibration\ · ladder\
  renders\<ab>\rn_<16hex>\           ⊘ raw.wav · render.json
  takes\<ab>\tk_<16hex>\             ⊘ delivery.wav · take.json
      analyses\an_<16hex>.json       ⊘ QA + cue alignment (one per analysis key)
  profiles\<ab>\<sha256>\            ⊘ profile.json · spectrogram.png · pitch.png
  jobs\<job_id>\  job.json · report.md · report.json
  scratch\  ·  logs\
```

- Paths are content-addressed.
- **Retention.** Designs, renders, takes, analyses, profiles and jobs are kept for `retention_days`
  (default 30) after they were last used, then `gc` may remove them. A caller copies what it keeps
  (R12); after that, a request for the same work simply renders it again. Measurements are kept longer
  (`measurement_retention_days`, default 365), since they cost 20–50 minutes of GPU time each.
  `provenance.jsonl`, `engines\` and `alignment\` are never collected.
- Immutable files are read-only and re-hashed by `verify`.
- `gc` is an operator command and a dry run by default.
- The service's own material (the canary's description, text and seed, the calibration corpus, the
  ladder texts, the alignment benchmark's text and hand marks, the text-check and QA fixtures) ships with
  its source, versioned and hashed. None of it comes from a caller. Audio is not shipped: the canary is
  designed at install time (revision 5.2, DC-3), and the benchmark's audio is rendered by whoever runs
  `bench alignment`.

---

## 16. Configuration

The design assumes no install location (owner decision, Q9). `<service_root>` below is the folder the
owner sets up; the store and models may live anywhere else. There is no list of readable folders: a
voice clip or audio file is read from any absolute path on a local drive (section 17).

```toml
[server]
store_root   = '<service_root>\store'
models_root  = '<service_root>\models'     # pinned HF snapshots (read-only at runtime)

[voices]
# Synthetic voices only (section 17). Clips the service designed are accepted by its provenance list;
# these are clips designed before the service existed, allowlisted by the owner by sha256.
allow_sha256 = [
  "8ab91fd91dee1d6d80e38f80fef5d36ee3dc9a3af39be3d3e74b7df3a5b851ac",   # refs/auditions/qwen3-tts-voicedesign_d2-late-night_take1.wav
  "e07a0199b33f34ac6d1d81b9ae54ae8fac49a13a0421148c51595462a804ba75",   # refs/auditions/qwen3-tts-voicedesign_d4-radio-drama_take2.wav
]

[retention]
retention_days = 30
measurement_retention_days = 365

[daemon]
autostart = true             # detached: BREAKAWAY_FROM_JOB | DETACHED_PROCESS | NEW_PROCESS_GROUP
idle_unload_s = 120
idle_exit_min = 15

[workers]
cpu_threads = 8
priority = "below_normal"
env = { CUBLAS_WORKSPACE_CONFIG = ":4096:8", HF_HUB_OFFLINE = "1", TRANSFORMERS_OFFLINE = "1" }

[gpu]
device = "cuda:0"
one_group_at_a_time = true   # only one of qwen | qa is loaded at once (section 4)
min_free_margin_mb = 1024
wait_timeout_min = 30

[limits]
max_segments_per_job = 200
max_cues_per_segment = 40
max_chars_per_segment = 1200   # hard cap on input; the voice's reliable length is advice, never a cap
max_chars_per_job = 60000
max_hints_per_job = 500
max_queued_jobs = 20
max_submits_per_min = 10
max_description_chars = 600
max_clip_seconds = 30
min_free_disk_gb = 5

[defaults]
takes = 1
max_retakes = 2
# No project, pronunciation list, number style or voice ships with the service: those are a caller's.

[text]
checks = "text-1.1.0"          # section 9.1: digit / symbol / unit_like / letter; the generic unit list
refuse = ["[", "]", "<|", "|>"]

[delivery]
sample_rate = 48000
subtype = "PCM_24"
target_lufs = -23.0            # revisions 5.3/5.4, DC-8 (was -16.0)
true_peak_dbtp = -1.0          # wins over target_lufs
trim_rel_db = -40.0            # relative to the take's p95 frame RMS (mean removed)
trim_floor_dbfs = -70.0        # DC-10: the speech threshold never goes below this
trim_pad_s = 0.08
fade_s = 0.01

[voice_design]
design_text = "Far below the surface, where the light grows thin, small things live quiet lives. Some of them thrive, and some of them simply disappear, and the water keeps no record of either."

[measurement]
corpus = "narration-en.v1"     # the service's own spoken-form paragraphs (section 3.2)
seeds = 3
length_ladder_spoken_chars = [80, 150, 250, 300, 350, 400, 450, 500, 560]
trend_band_max_chars = 300
pace_tol_min = 0.10            # effective tol = max(this, measured seed spread in the trend band)
sim_warn_margin = 0.01
sim_fail_floor = 0.90          # ASSUME

[alignment]
model = "facebook/wav2vec2-large-960h-lv60-self"   # apache-2.0; the default (Q18); revision pinned at install
# model = "Qwen/Qwen3-ForcedAligner-0.6B"          # only if Phase 0 chooses it (section 11.2); device = "cuda:0"
device = "cpu"
disagree_threshold_s = 0.25    # ASSUME; set from the benchmark
low_confidence_below = 0.75    # ASSUME; CUE_LOW_CONFIDENCE below this mean token posterior (section 11.2)
unplaced_below = 0.50          # ASSUME; below this the cue is not placed (CUE_UNALIGNED)
benchmark = "alignment-en.v1"  # the service's own; its measured error is published (R1, section 11.2)

[qa]
profile = "default.v3"

[engines.qwen3_base]
non_streaming_mode = false     # all clone evidence used this; change only after a Phase 0 A/B
x_vector_only_mode = false     # ICL; part of the voice hash
max_new_tokens_per_char = 2.5  # DC-4: each call's cap is min(8192, max(floor, ceil(2.5 × characters)))
max_new_tokens_floor = 128     # frames (12.5 per second of audio)
[engines.qwen3_design]
non_streaming_mode = true
max_new_tokens_per_char = 2.5
max_new_tokens_floor = 128

[workers.qwen3]
project = '<service_root>\workers\qwen3tts'
[workers.qa]
project = '<service_root>\workers\qa'
```

Client launch configuration, e.g. a `.mcp.json` for Claude Code:

```json
{"mcpServers": {"narration": {"type": "stdio", "command": "uv",
  "args": ["run", "--project", "<service_root>", "--frozen", "--offline",
           "narration-mcp", "--config", "<service_root>\\narration.toml"]}}}
```

---

## 17. Security considerations

1. **Transport.** stdio only; no process listens on a socket. Streamable HTTP is rejected for v1. If
   ever needed: 127.0.0.1, an `Origin` check, a bearer token, off by default.
2. **Write confinement.**
   - Writes go only under `store_root`, with server-built paths from validated ids.
   - Windows reserved names are rejected. `realpath` must stay under the root, and reparse points are
     refused.
3. **Reads.** The service reads a caller's file only by a path the request gives: a voice clip, or
   audio to profile. **Any absolute path on a local drive is accepted** (owner decision, 2026-09-26,
   Q9); there is no list of allowed folders.
   - Refused (`PATH_NOT_ALLOWED`): a relative path, since the daemon's working folder is not the
     caller's; a path that resolves (with `realpath`, following links) to a network share
     (`\\server\share\…`) or a device path (`\\?\…`, `\\.\…`), because opening a network path makes
     Windows offer the owner's sign-in to that server; and anything that is not a regular file.
   - A voice clip must be a WAV of ≤ 30 s / 20 MB. Its sha256 is checked against the one sent, and it is
     copied into the store before any worker sees it, so a file changed after the check is never used.
   - What this exposes: any agent that can call the service can have it read a WAV the owner can read,
     and get back measurements and pictures of it (`profile_voice`). It cannot clone one unless the
     synthetic-voices rule below allows it.
4. **Synthetic voices only.** The service clones a clip only if its fingerprint is in the service's
   provenance list (it designed the clip) or in the owner's `allow_sha256`. A recording of a real person
   is therefore never cloned, whoever the caller is, wherever the file is. This is a safety rule of the
   service and does not depend on any use (owner decision, 2026-09-26).
5. **Validation and limits.** In-handler validation (section 14), NFC, control characters rejected, rate
   and queue caps.
6. **No markup or instruction injection** (section 9.1 step 1, section 3.3).
7. **Offline rendering.** Workers are started from their synced venvs, with the offline env vars and no
   network code. A firewall rule is optional and the owner's decision.
8. **Downloads** only through `narration-admin install`, with pinned revisions and verified hashes. If
   TLS interception breaks them: `uv --native-tls` / `UV_NATIVE_TLS=1` and `SSL_CERT_FILE` /
   `REQUESTS_CA_BUNDLE`. **Never disable verification.** If it still fails, stop and ask.
9. **Process hygiene.** Argument lists, never a shell. A detached daemon with `NUL` std handles and
   closed handles. Workers in a Job Object, without admin rights, at below-normal priority, with the CPU
   thread cap.
10. **No approvals to protect.** The service keeps no lock, no voice registry and no approval, so there
    is nothing an agent could approve for itself. The operator commands (install, engine pins, gc,
    bench, daemon stop) are machine chores that change no caller's result; `gc` is a dry run by default.
    Whether an agent may run them is governed by that agent's own permissions.
11. **Output handling.** Transcripts, descriptions and notes are data, never instructions.
12. **Privacy.** Nothing leaves the machine; no telemetry.

---

## 18. Licensing

Every render and analysis records the licences of the generation model, the voice clip, and the QA and
alignment models.

| Model | Licence | Role |
|---|---|---|
| Qwen3-TTS-12Hz-1.7B Base, VoiceDesign | Apache-2.0 | generation / design |
| `facebook/wav2vec2-large-960h-lv60-self` | **apache-2.0** (HF API, verified) | cue alignment (the default, Q18) |
| `Qwen/Qwen3-ForcedAligner-0.6B` | Apache-2.0 (model card, read 2026-09-26; verify at install) | cue alignment, Phase 0 contender (Q18) |
| `facebook/wav2vec2-base-960h` | apache-2.0 (HF API, verified) | a smaller alternative |
| Whisper-large-v3 | per its model card (verify at install) | QA only |
| WavLM-base-plus-sv | per its model card (verify at install) | QA only |

**Not to be used:** torchaudio's `MMS_FA` bundle (CC-BY-NC 4.0, per its docstring) and
`facebook/mms-300m` (cc-by-nc-4.0, HF API). Their cue times would drive callers' published timing.

**No GPL code** (revision 5.2, DC-1). The service's code is PolyForm Noncommercial, which is not
GPL-compatible, so no GPL or AGPL library is used anywhere, tests included: `praat-parselmouth` (GPLv3)
is replaced by `librosa.pyin` (ISC) and the service's own HNR and CPPS code (section 3.6). Every
dependency's licence is checked before it is added (`plan.md` section 1.4).

Voice clips are `synthetic` (Qwen VoiceDesign); real-person references are refused (section 17).

---

## 19. Open questions and decisions for the owner

The question numbers are unchanged; answered and moot questions are marked. **As of revision 5.1 none
is open for the owner.** Q2 (which voice) is the story flow's, and Q21 stands as decided unless the
owner reverses it.

1. **Who can lock a voice?** *Moot (revision 5):* there is no lock and no voice registry. A voice is a
   clip the caller keeps.
2. **Which voice?** *Moved to the caller:* the story flow chooses between `d2-late-night_take1` and
   `d4-radio-drama_take2` (both allowlisted, section 16) and has the one it chooses measured. d4's lower
   similarity to its own clip (0.966–0.969 vs 0.978–0.984) is handled by its measurement.
3. **Delivery format and loudness.** *Answered (story flow):* 48 kHz / 24-bit mono, −16 LUFS per take
   (*revised by the owner to a −23 LUFS default, EBU R128's pair with the ceiling, in revisions 5.3
   and 5.4, DC-8, after real output never reached −16 under the ceiling; `target_lufs` stays
   configurable*),
   with the true-peak ceiling winning; each take's loudness record is returned (section 13).
4. **Numbers.** *Moot for the service (R6 revised):* callers send numbers already in words.
5. **Mood/delivery variants.** *Moot for the service:* a different delivery is a different clip, which
   the caller chooses (section 3.4).
6. **Negation policy.** *Answered (owner, 2026-09-26):* warn, never refuse, using a plain word list
   and no language model (section 3.5). Clips made before the service from negated prompts are
   allowlisted and never linted.
7. **Fit remedies.** *Answered (story flow):* none for films. v1 reports fit only; remedies are a later
   phase.
8. **Daemon.** *Answered:* acceptable, with detachment, idle exit, thread cap, identity, and
   `daemon start|stop`.
9. **Where things live.** *Answered (owner, 2026-09-26):* the design assumes no location; the owner
   sets up the service's folder, and section 16's paths are placeholders. Clips are read from any
   absolute local path, with no list of allowed folders (section 17). No export alias.
10. **GPU etiquette.** *Answered:* wait rather than fail fast; the caller decides whether narration may
    run, and can free the GPU with `release_gpu`.
11. **A stronger SV model.** *Answered (owner, 2026-09-26):* not now. Thresholds come from each voice's
    measurement; section 20 keeps it as a later item.
12. *Moot (Qwen only).*
13. **Captions vs prose.** *Answered (story flow):* captions, as cues. The service takes any text as
    cues, or as a one-cue segment.
14. *Moot (Qwen only).*
15. **Number reading style.** *Moot for the service (R6 revised).*
16. **Em dash.** *Moot for the service:* punctuation is passed through as sent (section 9.1).
17. **Over-long segments.** *Answered (owner, 2026-09-26):* warned about, never refused (section 3.2).
18. **Aligner model.** *Answered (owner, 2026-09-26):* `facebook/wav2vec2-large-960h-lv60-self`
    (apache-2.0, English characters, on the CPU) is the default, chosen for fit rather than proven
    accuracy. Phase 0 also tests `Qwen/Qwen3-ForcedAligner-0.6B` (Apache-2.0, GPU), which replaces it
    only if clearly better at boundaries with no pause (section 11.2). Montreal Forced Aligner is more
    accurate on published figures but needs its own toolchain and a pronunciation for every invented
    name; it is considered only if both disappoint. `MMS_FA` / `mms-300m` are excluded (NC).
19. **Intended pronunciations.** *Moved to the caller:* the story flow keeps its pronunciation list,
    approves each name by ear, and sends the hints with its requests.
20. **A normaliser for written text.** *Answered (owner, 2026-09-26):* a later phase, so v1 speaks text
    as sent and refuses `text_mode: "written"` (section 9.2). Under the owner's principle any reading convention
    is a caller's choice, and a normaliser in the service would carry one convention to every caller
    that used it.
21. **Measuring before generating** *(new; decided as recommended, the owner may reverse it).* Measuring
    is its own step, and a request with an unmeasured clip is refused with a hint. The other way, a
    generation job would measure the clip first, adding 20–50 minutes of GPU work the caller did not
    schedule; the outcome is the same either way.
22. **The listening model for descriptions.** *Answered (owner, 2026-09-26):* not in v1. The profile's
    measurements and pictures serve the shortlist, and the voice is chosen by ear. If it is ever wanted,
    a trial on the four auditioned voices (d1–d4) comes first; a candidate then was
    Qwen2-Audio-7B-Instruct (BELIEVE Apache-2.0).

---

## 20. Phased implementation plan (plan only)

| Phase | Scope | Exit criteria |
|---|---|---|
| **0: spikes and decisions** | The flow settles Q2 (which voice). The service's own material is written: a designed canary clip, the calibration corpus and ladder texts, the alignment benchmark's paragraphs, and the text and QA fixtures, all in spoken form to the rules of section 11.2; the owner listens to a sample. Measure: (a) **cue-boundary error, small**: a prototype of the aligner that Phase 3 keeps, one voice, one seed, about 30 hand marks, CTC / CTC + snap / Qwen3-ForcedAligner (+ snap) / Whisper, and the choice between the two aligners by the rule of section 11.2; (b) the aligner model revisions, **re-check and save** that `forced_align` runs on this CPU, and whether `qwen-asr` shares the QA worker's library versions or needs its own venv; (c) the **length ladder** for d2 and d4 with the trend rule (20–50 min of GPU each); (d) **repeat test on the Base clone path**, in one process and across processes, with the determinism env: it sets the tier (section 10.1); (e) optional `non_streaming_mode` A/B for Base; (f) **re-run and save** the "Marufubens" check with silence prepended; (g) **the daemon survives the MCP client exiting**; (h) VRAM + load times; (i) offline loading from a SHA-named snapshot; (j) the Python MCP SDK v2 version, the client protocol version, and in-handler validation producing tool errors. | A first error table; `max_segment_chars` for d2/d4; the determinism tier; both re-checks saved as scripts and outputs; the aligner chosen. |
| **1: store, engine, workers** | SQLite with the cache index and retention; the three cache layers; the claim-time re-check and waiting on in-flight work; engine profiles + fingerprints; worker protocol; the Qwen worker (ICL Base + VoiceDesign) with pinned settings and determinism env; post-processing (relative trim, 48 kHz, gain with the TP ceiling on the final file, loudness record); thread caps; `narration-admin render`, `daemon start\|stop\|status`. | Rendering twice (and in a fresh process) behaves as the tier says; drift detected; `daemon stop --now` leaves no partial files; two overlapping jobs render a shared paragraph once. |
| **2: text handling and hints** | Canonical form and the join; hints from the request, with the possessive rule; the text checks (digit, symbol, unit-like, letter); exact spans to word ranges; spans in the joined texts; `check_text`. | The text tests of section 9.3 pass; cue spans exact; markup refused. |
| **3: QA, alignment, measurement, suggestions** | QA worker (GPU ASR/SV; CPU aligner, canary similarity, profile); the exact-span check through the normaliser; thresholds from the voice's measurement; length-aware pace; the word-count rule for `wer_adj`; head/end insertion; token cap; `takes` / `attempts` / `max_retakes`; the suggestion tiers; the consistency report; scoring-only jobs; `measure_voice` (transcript check, calibration, ladder with the trend rule); the full alignment benchmark and `bench alignment`; reports. | On the service's QA fixtures every planted fault is caught (a different number in an exact span, a head insertion, a render cut by the token cap, an unplaceable cue), and every planted non-fault passes ("per cent" heard as "percent", "nought" as "zero", one slip in a three-word title warns but does not fail). d4 is not flagged across its own calibration takes. The same request twice gives the same result from the cache. The measured error is published. |
| **4: MCP front-end and daemon** | stdio server (2026-07-28 + legacy via the SDK); v1 tools, resources, prompts; `wait_s` + progress; cancellation; the error model with in-handler validation and dereferenced schemas; limits; path and synthetic-voice checks; detached daemon + fallback; GPU scheduler; `release_gpu`. | A Claude Code session designs, measures and narrates the service's demo script (its own text, about 20 paragraphs of cues) with `takes: 2`; the client quits mid-job and the job completes; a resubmission after one edit renders one segment and returns the rest from the cache with the same take ids; two sessions share one queue; a recording that is not allowlisted is refused. |
| **5: design and profile** | `design_voice` with the provenance list and the positive-only lint; `profile_voice` (measurements and pictures); the allowlist; operator CLI (install, engine pin/repin/bridge, gc, verify, bench). | Flows A–F end to end, including a "weeks later" batch with `expect_engine_profile` and the canary gate. |
| **6: later** | Stitching and fit remedies, when a picture-first caller exists; a written-text normaliser (Q20); the Tasks extension; a stronger SV model (Q11); a listening model for voice descriptions (Q22); unspoken context if a backend supports it; the aligner on the GPU if CPU time matters. | Each item separately justified. |

---

## 21. Response to `story-narration.md`

This maps the requirements (`story-narration.md` at commit `bf19d0a`, the version revised under the
owner's principle) against revision 5. Revision 5 makes the service stateless and takes voices by
location (owner decisions, 2026-09-26). That changes how R8, R11, R13 and V1 are met, and it adds asks of
the flow. Everything else stands as revision 4 answered it.

| Req | Verdict | What the design does (section) |
|---|---|---|
| **R1** caption times | Accepted; v1 | CTC forced alignment of the known spoken text on the CPU, snapped into pauses, with Whisper as the cross-check; delivery-file seconds for every take; `CUE_UNALIGNED` is never interpolated and triggers a retake (§11.2, §7.5). The service measures its cue-boundary error on its own benchmark and publishes p50 and p95 with the method, in `get_server_status` and each take's `alignment.measured_error`. It sets no target; the flow's 0.2 s is the flow's. Phase 0 gives a first small table; Phase 3 the full one. |
| **R2** cues without times | Accepted | Optional `at_s`; cues-only segments; a `text` that is not the join is refused (§7.2). |
| **R3** each cue its own unit | Accepted | Hints, checks and exact spans stay inside their cue; the join adds nothing (§9.1). |
| **R4** no fitting without a budget | Accepted (unchanged) | No fit of any kind without `scene_seconds`; v1 only reports fit even with it (§12). |
| **R5** exact lengths | Accepted | `samples`, `sample_rate`, `duration_s`, `sha256`, `trim` {head_s, tail_s, pad_s}, now defined exactly, and a `loudness` record per take (§7.5, §13). |
| **R6** text spoken as sent | Accepted; v1 is spoken-only | As revision 4. The hints now come with each request, from the flow's own list, as its Part 3 planned (§9.1). |
| **R7** what was spoken, echoed | Accepted | Per cue: `received`, `spoken`, `engine`, `hints_applied`, in `check_text`, the `dry_run` plan and `get_results` (§7.5, §7.6). |
| **R8** safe paragraph length | Accepted, as advice | `max_segment_chars` comes from the voice's measurement, judged against the voice's pace trend (§3.2). A longer segment is **rendered and warned about** (`SEGMENT_TOO_LONG`, with the limits, `spoken_chars`, `over_by_chars` and `cue_chars`), never refused (owner decision). |
| **R9** several takes per job | Accepted | `takes` 1–3 in one model load, or explicit `attempts`; `max_retakes` per failing take; a suggestion by V2's tiers (§7.3, §8). |
| **R10** unspoken context | Not supported | Refused with `CONTROL_UNSUPPORTED` (§3.3). |
| **R11** changes after resubmission | Met differently: **the flow compares** | The service keeps no earlier result to compare with. Every take carries `fresh` (rendered by this job) and ids that are stable for the same clip, text, attempt and engine. A segment rendered anew is one whose take id differs from the flow's manifest; a changed `analysis_id` with the same take means the cue times may have moved. The old take and its length are in the flow's own manifest (§8). |
| **R12** files by path and hash | Accepted | Absolute paths + sha256, kept at least 30 days after last use; the flow copies and checks, as it planned (§13, §15). |
| **R13** approval visible | Withdrawn by the flow; the service has no approvals at all | No lock, no voice registry, no cut, no approval. Choosing the voice is the flow's too (§2, §17). |
| **R14** exact spans | Accepted; v1 | Both the span and the transcript go through Whisper's English normaliser (tested here), then are compared. The check confirms **the value of a number**, not its wording: "thirty-two hundred" and "three thousand two hundred" both pass as 3200 (owner decision). Other words are compared word for word (§11.3). |
| *(Part 3)* machine load | Honoured by the caller | No workload detection; `get_server_status` now shows whether the GPU is in use, by which model, and for how long; `release_gpu` frees an idle model (§4.1). |

**Part 2b**

| Item | Verdict | What the design does (section) |
|---|---|---|
| **V1** keep picks made on a draft cut | Met by construction | There are no cuts to lose a pick. A pick is the flow's record (a take id, or an attempt number). The same clip, text, attempt and engine give the same take id, so a pick stays valid across resubmissions; if a take has left the cache and rendering is not bit-exact (Phase 0 decides), the flow's own copy is the record (§8, §10.1). |
| **V2** prefer a take with every cue placed | Accepted | The suggestion tiers: pass; warn with every cue placed; warn with a cue unplaced; fail (every cue placed first, then the fewest fail flags) (§8). |
| `SEGMENT_TOO_LONG` detail | Accepted, as a warning | See R8. |
| **V3**, **V4** | Withdrawn by the flow | Nothing of either is in the service. |

**Asks of the flow**

1. **Keep the voice.** Copy the chosen clip (Q2 is now the flow's) into the story repository, and record
   its path, sha256 and exact transcript. For d2 and d4, designed in the bake-off, the transcript is the
   bake-off's design text; `measure_voice` checks it. Any absolute local path works; the service keeps
   no list of allowed folders.
2. **Measure the voice once**, as its own step, when the machine is free: about 20–50 minutes of GPU
   time. Keep the measurement JSON with the clip.
3. **Send the hints with each request**, from the flow's pronunciation list. Send invented names without
   a respelling too: QA then recognises them, and name-dense paragraphs do not fail `wer_adj`.
4. **Keep, per segment, the take id and the analysis id** in the manifest, and re-time a segment when
   either changes. Record the engine profile hash, and send it as `expect_engine_profile`.
5. **A pick is the flow's record**: store the take id (and its attempt). To hear new deliveries, send
   the segment with new `attempts`.
6. **R1's example**: the per-cue text is `received` (not `written`), given once per segment; each take's
   `cues[]` carries only the times.
7. **Exact spans** confirm a number's value, not its wording.
8. **The splitter** treats `max_segment_chars` as advice: a longer paragraph is rendered and warned
   about.

**Their answers to Q3, Q7, Q9 and Q13 are adopted; Q4, Q15 and Q16 are moot; Q2 and Q19 have moved to
the flow (section 19).**

---

## Appendix A: worker protocol (daemon ↔ worker)

The protocol is JSON lines over stdin/stdout, UTF-8. Worker stdout carries protocol messages only; logs
go to stderr. Requests carry an `id`, and replies echo it. Audio travels as file paths under
`store_root\scratch\`; a caller's voice clip is copied into the store before a worker sees it. The
daemon computes the hashes. All times are in seconds.

```jsonc
→ {"id":1,"op":"hello"}
← {"id":1,"ok":true,"role":"qwen3","capabilities":{"controls":{"pace":false,"context":false,"instruct":false}},
    "fingerprint":{"python":"3.12.x","packages":{"qwen-tts":"0.1.1","transformers":"4.57.3","torch":"2.11.0+cu128"},
    "cuda":"12.8","cudnn":"…","gpu":"NVIDIA GeForce RTX 4090","driver":"…","cpu_threads":8,
    "env":{"CUBLAS_WORKSPACE_CONFIG":":4096:8"}}}
→ {"id":2,"op":"load","engine_profile":{…},"snapshot_dir":"<models_root>\\…\\snapshots\\<sha>",
    "determinism":{"tf32":false,"cudnn_deterministic":true,"deterministic_algorithms":"warn_only"},
    "settings":{"non_streaming_mode":false,"generation":{"do_sample":true,"top_k":50,"…":"explicit","max_new_tokens":8192}}}
← {"id":2,"ok":true,"load_s":18.2,"vram_mb":6120}
→ {"id":3,"op":"prepare_voice","voice_hash":"sha256:…","ref_wav":"…\\scratch\\voices\\5b1e….wav",
    "ref_text":"Far below the surface, …","x_vector_only_mode":false}
← {"id":3,"ok":true}
→ {"id":4,"op":"synthesize","voice_hash":"sha256:…","engine_text":"…","language":"English","seed":1834112093,
    "max_new_tokens":868,"out_path":"…\\scratch\\job_…\\p03_a0.wav"}   // the call's cap (DC-4)
← {"id":4,"ok":true,"sample_rate":24000,"samples":222720,"gen_s":21.3,"new_tokens":116,"max_new_tokens":868,
    "hit_token_cap":false}   // new_tokens: decoded frames (at most the cap − 1); hit_token_cap: the last
                             // token sampled was not the end token; max_new_tokens echoes the call's cap
→ {"id":5,"op":"design","description":"…","design_text":"…","language":"English","seed":2001,"max_new_tokens":400,
    "out_path":"…"}   // VoiceDesign
→ {"id":6,"op":"unload"}   → {"id":7,"op":"shutdown"}
← {"id":n,"ok":false,"error":{"code":"GPU_OOM","message":"CUDA out of memory …"}}
// QA worker: transcribe {wav, word_timestamps, long_form} (GPU) · embed {wav, device} · f0 {wav}
//            align {wav, tokens_per_cue, align_as} (CPU) → {words[], cues[], confidence[]} | {error: "ALIGNMENT_ERROR"}
//            profile {wav, out_dir} (CPU) → {measurements, pictures[]}
```

## Appendix B: sidecar examples

`renders\77\rn_77e0c4a1b2d93f08\render.json`

```json
{"schema": "narration.render/v1", "render_id": "rn_77e0c4a1b2d93f08", "render_key": "sha256:77e0c4a1…",
 "voice": {"voice_hash": "sha256:3f9a0c1e…", "clip_sha256": "5b1e…", "x_vector_only_mode": false},
 "engine": {"engine_profile_id": "qwen3-base-1.7b.p1", "engine_profile_hash": "sha256:9e21…",
            "model_repo": "Qwen/Qwen3-TTS-12Hz-1.7B-Base", "model_revision": "<40-hex>",
            "non_streaming_mode": false, "generation": {"…": "explicit values"},
            "observed": {"gpu": "NVIDIA GeForce RTX 4090", "driver": "…", "cuda": "12.8", "cudnn": "…"}},
 "engine_text": "Before dawn, the reef belongs to the Oss-a-veen shrimp. By sunrise, some three thousand two hundred of them are back in the rock.",
 "seed": 1834112093, "seed_scheme": "narration-seed/v1", "attempt": 0,
 "raw": {"path": "…\\raw.wav", "sha256": "…", "sample_rate": 24000, "samples": 222240, "format": "WAV FLOAT mono"},
 "hit_token_cap": false, "gen_s": 21.3, "rtf": 2.30, "canary": {"batch_status": "hash_match"},
 "licence": {"generation_model": "Apache-2.0", "voice_clip": "synthetic (Qwen3-TTS VoiceDesign, Apache-2.0)"}}
```

`takes\8c\tk_8c41d2e07a9b3f55\take.json`

```json
{"schema": "narration.take/v1", "take_id": "tk_8c41d2e07a9b3f55", "delivery_key": "sha256:8c41d2e0…",
 "render_id": "rn_77e0c4a1b2d93f08",
 "delivery": {"path": "…\\delivery.wav", "sha256": "…", "sample_rate": 48000, "samples": 420480,
              "duration_s": 8.76, "format": "WAV PCM_24 mono"},
 "trim": {"head_s": 0.27, "tail_s": 0.39, "pad_s": 0.08, "rule": "max(p95_frame_rms - 40 dB, -70 dBFS), mean removed; head_s/tail_s found, up to pad_s kept"},
 "loudness": {"target_lufs": -23.0, "measured_lufs": -23.0, "gain_db": -1.6, "true_peak_dbtp": -9.3,
              "ceiling_applied": false},
 "tools": {"resampler": "<name> <version>", "loudness_meter": "<name> <version>", "post": "narration.post/1"},
 "post_stretched": false}
```

`takes\8c\tk_8c41d2e07a9b3f55\analyses\an_0f3b91c2d5e7a468.json`

```json
{"schema": "narration.analysis/v1", "analysis_id": "an_0f3b91c2d5e7a468", "analysis_key": "sha256:0f3b91c2…",
 "take_id": "tk_8c41d2e07a9b3f55",
 "text": {"cues": [
   {"index": 0, "received": "Before dawn, the reef belongs to the Ossavine shrimp.",
    "spoken": "Before dawn, the reef belongs to the Ossavine shrimp.",
    "engine": "Before dawn, the reef belongs to the Oss-a-veen shrimp.",
    "spoken_span": [0, 53], "engine_span": [0, 55],
    "hints_applied": [{"term": "Ossavine", "respell": "Oss-a-veen", "offset": 37}], "warnings": [], "exact": []},
   {"index": 1, "received": "By sunrise, some three thousand two hundred of them are back in the rock.",
    "spoken": "By sunrise, some three thousand two hundred of them are back in the rock.",
    "engine": "By sunrise, some three thousand two hundred of them are back in the rock.",
    "spoken_span": [54, 127], "engine_span": [56, 129], "hints_applied": [], "warnings": [],
    "exact": [{"start": 17, "end": 43, "words": [3, 7]}]}],
  "text_checks": {"version": "text-1.1.0", "rules_sha256": "…"},
  "hints_used": [{"term": "Ossavine", "respell": "Oss-a-veen"}]},
 "versions": {"qa_profile": "default.v3", "asr": "openai/whisper-large-v3@…", "sv": "microsoft/wavlm-base-plus-sv@…",
              "aligner_method": "ctc-snap/wav2vec2-large-960h-lv60-self@…", "number_reader": "whisper-english-normalizer+nought@2",
              "measurement": "sha256:c07d…"},
 "alignment": {"method": "ctc-forced-align+silence-snap", "device": "cpu",
   "cross_check": {"model": "openai/whisper-large-v3", "max_disagreement_s": 0.06},
   "measured_error": {"p50_s": "…", "p95_s": "…", "n": "…", "benchmark": "alignment-en.v1"},
   "cues": [{"index": 0, "start_s": 0.08, "end_s": 3.02, "confidence": 0.93, "words": ["…"]},
            {"index": 1, "start_s": 3.58, "end_s": 8.68, "confidence": 0.95, "words": ["…"]}],
   "flags": []},
 "qa": {"verdict": "pass", "transcript": "…",
        "exact": [{"cue": 1, "start": 17, "end": 43, "expected": "3200", "heard": "3200", "match": "same"}],
        "terms": [{"term": "Ossavine", "cue": 0, "heard": "Ossavine", "ok": true}],
        "metrics": {"wer_raw": 0.0, "wer_adj": 0.0, "word_errors": 0, "exact_ok": true, "spk_sim_anchor": 0.981,
                    "spoken_wpm": 158, "expected_spoken_wpm": 150,
                    "head_insertion_words": 0, "end_insertion_words": 0, "longest_silence_s": 0.61},
        "thresholds": {"spk_warn": 0.972, "spk_fail": 0.90, "pace_tol": 0.17},
        "flags": []},
 "licence": {"aligner": "apache-2.0", "asr": "per model card", "sv": "per model card"}}
```

`measurements\3f9a0c1e…\qwen3-base-1.7b.p1\measurement.json` (abridged)

```json
{"schema": "narration.measurement/v1", "voice_hash": "sha256:3f9a0c1e…", "clip_sha256": "5b1e…",
 "engine_profile": {"id": "qwen3-base-1.7b.p1", "hash": "sha256:9e21…"},
 "transcript_check": {"heard": "Far below the surface, …", "wer": 0.0, "ok": true},
 "corpus": "narration-en.v1",
 "similarity": {"anchor_p5": 0.982, "anchor_p50": 0.987, "consistency_p5": 0.983},
 "pace": {"trend": {"intercept_wpm": 118, "per_100_chars": 12.5, "band_max_chars": 300}, "tol": 0.17,
          "curve": [{"chars": 80, "wpm": 121}, "…"]},
 "max_segment_chars": 450, "max_segment_seconds": 31.5,
 "ladder": [{"chars": 80, "seeds": [{"seed": "…", "wpm": 119, "wer_adj": 0.0, "sim": 0.986, "verdict": "pass"}, "…"], "passes": true}, "…"],
 "measured_at": "2026-10-02T14:03:11Z"}
```

All numbers in these examples are illustrative, except where section 1 marks them KNOW.
