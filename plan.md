# Narration MCP: implementation plan

*Plan revision 2, 2026-09-26. Status: **H0 answered and the plan pushed; WP00 next.** Nothing is built yet.*

**Source of truth.** The design is [docs/design.md](docs/design.md), **revision 5.1**. It is the
bake-off's design copied into this repository, and differs only in two example paths and a header note.
This plan cites it as "§n". Changes to it are proposed and approved in §1.5, and approved changes enter
it as revision 5.2. Where this plan and the design disagree, the design wins, and the plan gets fixed.

**No local paths in tracked files** (owner, 2026-09-26). This machine's paths and facts live in the
gitignored `AGENTS.local.md`. Tracked files say `<repo>` for the main checkout.

**Status is kept in this file**, in the work-package table (§4), and only by the lead, on `main`. Agents
report in `status/<WP>.md` on their own branch, and the lead rolls those up when merging (§2.4).
[HANDOFF.md](HANDOFF.md) says where things stand right now. [AGENTS.md](AGENTS.md) holds the rules every
agent follows.

Status values: `todo` · `ready` (dependencies met) · `active` · `review` (branch done, awaiting merge) ·
`done` (merged to `main`) · `blocked:<gate or WP>` · `dropped`.

---

## 1. Shape of the work

The design has six phases (§20). Phase 0 is spikes, Phases 1–5 are the build, and Phase 6 is later.
Building them strictly in order would be slow and mostly serial. This plan cuts the same scope into
**work packages (WPs)** along module boundaries instead. Each WP owns its directories and tests, so most
of them can run at once in separate worktrees. The Phase 0 spikes are folded into the WPs that need their
answers, and each design phase's exit criteria become acceptance WPs at the end (§6).

Three things limit how much can run at once:

1. **One GPU, shared.** The dev machine has one 24 GB GPU, which other people's jobs also use.
   Anything that loads Qwen, Whisper or WavLM on the GPU takes the **GPU lock**
   (§2.5), one holder at a time. Everything else is designed to be developed and tested **without the
   GPU**: workers are thin model runners behind a protocol, a `fake` worker stands in for them, and all
   decision logic is plain Python in the server package (§1.1).
2. **Contracts first.** Parallel work needs frozen interfaces: the schema fragments (§7.2), the codes
   (§14), the sidecars (App. B), the worker protocol (App. A) and the module interfaces. Wave 0 fixes
   these, serially, before anything fans out.
3. **Human gates.** A few steps need a person's ears or hands (§5). The plan schedules around them, and
   only a small part of the work waits on any of them.

### 1.1 Architectural decisions made by this plan

These are the lead's choices where the design leaves room. The owner can overturn any of them.

| # | Decision | Why |
|---|---|---|
| P1 | **Workers are model runners only.** A worker loads models, runs them and returns raw outputs: audio, transcripts with word times, embeddings, CTC emissions and alignments, f0, profile numbers. Every verdict, threshold, flag, suggestion and key lives in the server package (`src/narration/`). | Logic becomes plain Python, testable without a GPU or torch, and parallel WPs don't need a worker venv. |
| P2 | **Whisper's English normaliser is vendored** into `narration.qa.normaliser` (MIT, with its notice), plus the "nought → zero" rule, and pinned as `whisper-english-normalizer+nought@1`. | The exact-span check (§11.3) runs in the server package, and the "number reader version" in the analysis key (§10.2) is then a file this repo controls. The bakeoff's `eval/normaliser_check.py` results become its golden test. |
| P3 | **One server uv project at the repo root** (`narration`: front-end, daemon, admin, all logic), and **two worker uv projects** (`workers/qwen3tts`, `workers/qa`) that share the `narration_worker` package (`workers/common`, a path dependency). | Matches §4 and §16 (`<service_root>` = repo root; workers in `workers/<role>/`). Heavy torch dependencies stay out of the server venv. |
| P4 | **Python 3.12** everywhere, from the interpreter already installed (uv does not download one). | The design's worker fingerprint says 3.12. No global install is needed. |
| P5 | **All of this machine's dev state lives in the repo's gitignored `.dev\`**: `uv-cache\`, `models\` (question Q6), per-worktree stores, the GPU lock, the owner's local config. Tests write only under a `.pytest-tmp\` inside their worktree. This is a dev convention of this checkout, not something the product assumes (§1.4). | Nothing is written outside the project, and a uv cache on D: lets every worktree's venv hardlink the same torch wheels instead of copying ~5 GB each. |
| P6 | **No audio, weights or checkpoints in git, ever**, unless the owner approves a specific file. The service's own **text** material (corpus, benchmark text, canary description, fixtures) and spike results (JSON/CSV/MD) *are* committed: they are part of the product (§15; Q5). Audio the service needs is generated on the installing machine (DC-3), or tracked by a manifest (path, sha256, how it was made) and kept under `.dev\`. | The owner's standing rule; the remote is public. |
| P7 | **The server's dependencies are declared once, in Wave 0**, and only the lead edits `pyproject.toml` / `uv.lock` on `main` afterwards. A WP that needs a new package asks in its status file. | The lock file would otherwise conflict in every merge. |
| P8 | **Worker pins start from the bakeoff's working venvs**: `qwen3tts` = Python 3.12, qwen-tts 0.1.1, transformers 4.57.3, accelerate 1.12.0, torch/torchaudio 2.11.0+cu128 (qwen-tts pins transformers and accelerate exactly); `qa` = Python 3.12, transformers 5.17.0 (the version the eval evidence used), torch/torchaudio 2.11.0+cu128, jiwer 4.0.0, soundfile; **not** praat-parselmouth, which the bakeoff used but is GPL (§1.3 item 3). Both use the `pytorch-cu128` index as the bakeoff's `pyproject.toml` does. | The evidence in §1 was made with these versions. The two transformers versions cannot share one venv. |

### 1.2 What the bakeoff gives us

The bakeoff (private, not a git repo; its location is in `AGENTS.local.md` and
`NARRATION_BAKEOFF_ROOT`) has **no tests** and no
code for loudness, trimming, alignment, `wer_adj`, hints, SQLite or MCP. What it does have is working
model calls and evidence to test against. **It is read-only to us.** Code is ported, with attribution in
the commit message, and never imported from there. Its audio is read by absolute path from local tests
and never copied into git (P6).

| Bakeoff asset | Use here | WP |
|---|---|---|
| `models/qwen3-tts/generate.py`: `load()` (bf16, `sdpa`, `cuda:0`), `create_voice_clone_prompt(ref_audio, ref_text)` then `generate_voice_clone(text, language="English", voice_clone_prompt=…)`, `generate_voice_design(text, instruct, language)` | The call pattern of the `qwen3tts` worker | WP20 |
| `models/qwen3-tts/pyproject.toml` + `uv.lock` | The starting pins for `workers/qwen3tts` (P8) | WP00, WP20 |
| `eval/evaluate.py` `Scorer`: Whisper-large-v3 via the transformers ASR pipeline, `WhisperProcessor…tokenizer.normalize`, WavLM-base-plus-sv `WavLMForXVector` embeddings L2-normalised, `to16k` (`resample_poly`) | The `transcribe` and `embed` ops; the normaliser port | WP22, WP14 |
| `eval/names_report.py` `name_spans`, `heard_for` (maps reference → hypothesis through `jiwer.process_words` and absorbs insertions at the edges) | Essentially §11.3 step 2 "Locate": the exact-span and term checks | WP14 |
| `eval/audition_check.py` `median_f0` (parselmouth, 50–400 Hz), the int16 bit-identity check | The bit-identity check for spike (d). The f0 code is **not** ported: parselmouth is GPL (§1.3 item 3) | WP20 |
| `eval/normaliser_check.py` + `.txt` (21 number-form cases, transformers 5.17.0) | Golden tests for the vendored normaliser | WP14 |
| `outputs/qwen3-tts-1.7b-clone-{d2,d4}-seed{1,2,3}/r48_names_probe.{wav,json,eval.json}`: 6 real clone takes with per-segment sample boundaries and raw Whisper transcripts | Real audio for post-processing, alignment and QA tests; golden transcripts for `wer_adj`, terms and exact spans | WP13, WP14, WP15, WP22 |
| `eval/results.csv`, `refs/auditions/voicelock.csv`, `index.csv`, `eval/r48_names_probe.names.md` | Golden numbers: WER 5.5–8.6 %, `spk_to_ref` d2 0.978–0.984 / d4 0.966–0.969, `spk_consist` 0.981–0.992, the VoiceDesign bit-identity row | WP14, WP22, WP33 |
| `refs/auditions/qwen3-tts-voicedesign_d2-late-night_take1.wav` (sha256 `8ab91fd9…51ac`, seed 2001) and `…d4-radio-drama_take2.wav` (`e07a0199…ba75`, seed 4002); transcript = `REF_TEXT` in `generate.py` | The two allowlisted voices (§16), used for every GPU test and for Phase 0 (c) | WP20, WP33, WP39 |
| `scripts/r48_*.json` | **Not** service material (it is a caller's script, §1). Evidence only, e.g. to reproduce the names probe | – |

**Model weights.** Qwen3-TTS 1.7B Base and VoiceDesign, whisper-large-v3 and wavlm-base-plus-sv are
already in this machine's Hugging Face cache; the snapshots are listed in `AGENTS.local.md`.
wavlm-base-plus-sv's `main` has only `pytorch_model.bin`, while `refs/pr/8` has safetensors. **Not
cached, so they must be downloaded:** `facebook/wav2vec2-large-960h-lv60-self` (the default aligner) and
`Qwen/Qwen3-ForcedAligner-0.6B` (the Phase 0 contender). **Decided (Q6):** copy the cached snapshots,
hash-verified, into `<repo>\.dev\models\`, and download the missing two there. Nothing is written outside
the project.

### 1.3 Where the installed code and the design disagree

The survey read the installed `qwen-tts` 0.1.1 source and found two things the design states
differently, and the licence decision surfaced a third. None blocks the start. Items 1 and 2 are
settled by WP20 and WP14 with evidence and an ADR. Item 3 was decided by the owner (Q7).

1. **The effective `max_new_tokens` is 8192, not 2048.** `from_pretrained` loads the snapshot's
   `generation_config.json` (the design says `generate_config.json`), which sets `max_new_tokens: 8192`
   (and do_sample true, temperature 0.9, top_k 50, top_p 1.0, repetition penalty 1.05, the same for
   `subtalker_*`). 2048 is only the hard-coded fallback. All clone evidence therefore ran with a cap of
   8192 tokens, which is about 680 s of audio at 12 Hz. So `TOKEN_CAP_HIT` will practically never fire,
   while a runaway render could take about 40 minutes of GPU before stopping. **Recommendation:** pin
   the effective values explicitly, as §10.1 requires, with 8192 so the evidence stays valid, and raise
   with the owner whether a cap derived from text length is wanted. That would be a design change, and it
   enters the engine profile. The planted token-cap fault in WP40 needs a test-only low cap either way.
2. **The spelling map.** §11.3 says Whisper's normaliser "treats spelling variants alike". That holds
   only when it is built with the `normalizer.json` British → American map, as `evaluate.py` does.
   `normaliser_check.py` built it with `{}`, so the map was never tested. **Decision (P2):** vendor the
   normaliser *with* the map, pin both, and extend the golden tests to spelling variants.

3. **`praat-parselmouth` is GPLv3+** (KNOW: its installed `METADATA` in the bakeoff's eval venv says
   "License: GPLv3"). The design uses it for f0 and for the voice profile's HNR, jitter and shimmer (§3.6,
   §4 QA worker). Importing a GPL library into PolyForm-licensed code makes a combined work that the GPL
   requires to be GPL-compatible, which PolyForm Noncommercial is not. That is the FSF's reading of
   Python imports; it is contested, but a project that sells commercial licences cannot rely on the
   other reading. **Decided (Q7): replace it**, as design change DC-1 (§1.5).

Also noted, for WP20 and WP14: `generate_voice_clone` decodes the reference codes and the new codes
together, then cuts the reference part off **proportionally** (`cut = ref_len / total_len × samples`).
The first phoneme of a take can therefore be clipped, or a little reference can bleed in. That is
relevant to spike (f), to the head-insertion check and to the trim.

### 1.4 A public project (owner requirement, 2026-09-26)

This will be a **public library that others install and use**, so it needs the rigour of a public
project. The design was written for one owner on one Windows machine. The following apply on top of it:

**What "public rigour" means here**

- **Nobody's personal setup is in the defaults.** The shipped config has an empty `allow_sha256`, no
  paths, and no reference to the bakeoff, d2/d4, the evolution simulator or any one machine. The owner's
  own setup lives in a local, gitignored config. Docs say "operator" where the design says "owner".
- **Licensing (decided by the owner, 2026-09-26): the same terms as the evolution simulator.**
  - Code, and the service's own material under `material/`: **PolyForm Noncommercial 1.0.0**
    (`LICENSE`). The body is verbatim from the evolution simulator's file; only the preamble is adapted.
  - Prose: **CC BY-NC 4.0** (`LICENSE-DOCS`).
  - `COMMERCIAL.md` says what counts as commercial and how to ask. It is adapted from the evolution
    simulator's, and **the owner reviews it before the first push**.
  - This makes the project **source-available and non-commercial, not open source** in the OSI sense.
    Everything below still applies.
  - Consequences:
    - `THIRD_PARTY_NOTICES` lists anything vendored (Whisper's English normaliser and its spelling
      map, both MIT).
    - `CONTRIBUTING.md` carries the evolution simulator's contribution grant, including the right to
      relicense, because commercial licences need the project to be one work.
    - **Every dependency's licence is checked before it is added.** Permissive licences are fine.
      GPL/AGPL is not, and neither is anything non-commercial or "research only": it could not be
      combined with, or commercially relicensed alongside, this code. LGPL needs the lead's OK.
    - Each model's licence is checked at install and recorded (§18). `MMS_FA` and `mms-300m` stay
      excluded.
- **Anyone can run the tests.**
  - The default suite needs no GPU, no model and no bakeoff. It runs in CI on every push and PR.
  - Tests that need models (`model`), the GPU (`gpu`) or the owner's bakeoff evidence (`evidence`) are
    opt-in, skip cleanly when their resources are absent, and are documented so a contributor with an
    NVIDIA GPU can run them.
  - Fixtures are the service's own material, plus synthetic audio generated deterministically by the
    tests.
- **CI on GitHub Actions**: ruff, the type check (pyright), and the default test suite. The pure parts run
  on Windows and Linux; nothing in CI needs a GPU. Actions are pinned by commit SHA and run with minimal
  permissions.
- **Reproducible installs.** `uv.lock` is committed for every project. `narration-admin install`
  downloads pinned model revisions and verifies their hashes (§17.8). Its first step is a doctor that
  says exactly what is missing: a supported GPU, VRAM, disk space, the platform.
- **A documented public surface.** The MCP tools, their schemas, error codes and flags are the API. They
  are versioned (SemVer for the package; `schema` ids such as `narration.render/v1` for sidecars), and
  changing one is a breaking change that goes in `CHANGELOG.md`. Docs are generated from the schemas.
- **Security posture stated.** `SECURITY.md` has the threat model of §17 (a local server that reads any
  absolute local path, clones only synthetic voices, writes only under its store) and how to report a
  vulnerability.
- **Project hygiene.** README (what it is, requirements, install, the `.mcp.json`, a first narration),
  `CONTRIBUTING.md` (setup, the test tiers, how GPU tests are run, commit style), `CODE_OF_CONDUCT.md`,
  issue and PR templates, `CHANGELOG.md` (Keep a Changelog), Conventional Commits.
- **No release without the owner.** Tags, GitHub releases and PyPI uploads are publishing, so each one
  needs the owner's explicit go-ahead.
- **No local paths in tracked files** (owner, Q5). `tools/check_tracked.py` enforces it in the
  pre-commit hook and in CI, together with the no-binaries rule.

**The owner's answers (H0, 2026-09-26)**

| # | Question | Decision |
|---|---|---|
| Q1 | Licence | PolyForm NC 1.0.0 + CC BY-NC 4.0 + `COMMERCIAL.md`, as the evolution simulator (above) |
| Q2 | Platforms | **Windows first**; others later, maybe. Do only the cheap preparation now (below) |
| Q3 | Distribution | A git clone + `uv sync` + `narration-admin install` for v1, with the layout kept PyPI-ready. Releases only with the owner |
| Q4 | Merge flow | Pull requests with CI, merged by the lead (§2.4) |
| Q5 | What is public | **No local paths.** Read as yes to the rest: the design is in `docs/design.md`; the plan files, the service's own text material, fixtures and spike results (text) are committed. Audio, weights and databases never |
| Q6 | Models on this machine | Copy the cached snapshots into `<repo>\.dev\models\`, and download the missing two there |
| Q7 | parselmouth (GPL) | Replace it: DC-1 (§1.5) |
| Q8 | GPU etiquette | Bounded runs when free VRAM allows; long runs with the owner's OK. **And** the MCP surface must tell a consumer how to back off: DC-2 (§1.5) |
| Q9 | Human gates | The owner will listen and mark. Marking may use DaVinci Resolve through its MCP, set up by an agent, with a caveat on frame precision (WP38). Decided later |

**The cheap preparation for other platforms (Q2).** It costs little now and saves a rewrite later:
- every OS-specific call sits behind `narration.platform` (WP19), implemented for Windows only. On any
  other OS it raises a clear "unsupported platform" error, and `doctor` says so;
- the pure code is portable: `pathlib`, `subprocess.DEVNULL` (never `NUL`), `os.replace`, explicit
  UTF-8, no `ctypes.windll` or `msvcrt` outside `narration.platform`;
- CI also runs the default suite on Linux, which catches Windows-isms in the pure code at once;
- `uv.lock` files stay universal (not restricted to Windows), so they resolve on Linux too.

Not now: the POSIX singleton, detachment and kill-on-close; GPU tests on Linux; macOS (no CUDA).

**Other adaptations for a public project.** Items 1 and 2 are design changes, in §1.5.
1. **The canary is designed at install time, not shipped as audio**: DC-3.
2. **Backoff information for consumers**: DC-2.
3. **The alignment benchmark's audio and hand marks.** The measured error that gets published comes from
   the owner's benchmark run. Others read it as the method's published record. Re-running `bench
   alignment` needs the benchmark audio, which could later be published as a release asset with the
   owner's approval. Until then, `bench alignment` works on any benchmark folder an operator provides.
4. **Voices.** Others design their own voices with `design_voice` (the provenance list), which works out
   of the box. `allow_sha256` is for clips an operator designed elsewhere, and ships empty.

### 1.5 Design changes

A change to `docs/design.md` is proposed here first. Once the owner approves it, the lead applies it to
the design as revision 5.2 (with a revision-history line) before, or together with, the WP that builds
it. Status: `proposed` · `approved` · `applied` (in the design) · `rejected`.

| # | Change | Design sections | Status | Built in |
|---|---|---|---|---|
| DC-1 | **No GPL in the voice profile.** f0 comes from `librosa.pyin` (ISC). HNR uses Boersma's (1993) autocorrelation method, the one Praat implements. **CPPS** (smoothed cepstral peak prominence, Hillenbrand 1994) replaces jitter and shimmer: those are defined on sustained vowels and are noisy on running narration, while CPPS holds up on connected speech. Each measure is documented as what it is, not as Praat's. Validation uses synthetic signals with known values (pulse trains at a known f0, noise at a known SNR). No GPL code anywhere in the repo, tests included. Note: librosa depends on `soxr` (LGPL-2.1+), which is acceptable as a separately installed, replaceable package; it goes in the licence audit. | §3.6, §4 (QA worker), §18 | **approved** (Q7) | WP22, WP34 |
| DC-2 | **A backoff contract** (Q8). Retries are already safe (content-hash ids, `idempotency_key`, identical request = same job); this tells a consumer *when* to come back. (1) The Error fragment gains an optional **`retry_after_s`**, set on every retryable error: the server's minimum wait, like HTTP `Retry-After`. `details` carries the facts behind it, e.g. `GPU_UNAVAILABLE {free_mb, need_mb, waited_s}`. (2) `LIMIT_EXCEEDED` stays for request-size limits (not retryable), and two retryable codes are added: **`QUEUE_FULL`** (`retry_after_s` from the queue's drain estimate) and **`RATE_LIMITED`** (`retry_after_s` until the submit window frees). (3) `submit_job` and `get_job` return **`poll_after_s`**, the earliest poll worth making, longer while `waiting_for_gpu`, beside the existing `eta_s`, `queue_position` and `phase`. (4) `get_server_status` gains **`admission`**: {accepting, queue {length, max, est_drain_s}, rate {remaining, resets_in_s}, gpu {in_use, holder, free_mb, need_mb by group, waiting_since}}, so a consumer can decide before submitting. (5) Tool descriptions state the rule: wait at least `retry_after_s`, add your own jitter, then resend the identical request, which is deduplicated. The service suggests; the consumer decides. | §7.2 (Error), §7.3, §7.4, §7.6, §14 | **approved** (owner, 2026-09-26) | WP01, WP31, WP36 |
| DC-3 | **The canary is designed on the installing machine**, not shipped as audio. §10.1 ships a canary clip, but the design promises nothing across a GPU, driver or CUDA change, so a shipped raw hash would not match on anyone else's machine, and shipping audio breaks P6. `narration-admin engine pin` designs the canary from a fixed description, text and seed (shipped as text) and stores its hash and embedding per engine profile. It still checks what §10.1 says: this engine, on this machine, over time. | §10.1, §15, §6 (EngineProfile) | **approved** (owner, 2026-09-26) | WP32, WP37 |
| DC-4 | **The effective `max_new_tokens`** (§1.3 item 1). Pin 8192, the value all the evidence used; decide whether a cap derived from text length is wanted. | §1, §10.1, §11.1 | open; WP20's ADR proposes | WP20 |

---

## 2. Worktrees, branches and merging

### 2.1 Layout

```
<repo>\                                   main checkout (branch main), = <service_root> in dev
  plan.md  HANDOFF.md  AGENTS.md  CLAUDE.md
  AGENTS.local.md  (gitignored)           this machine's paths and facts; copied into each worktree
  README.md  LICENSE  LICENSE-DOCS  COMMERCIAL.md  THIRD_PARTY_NOTICES
  CONTRIBUTING.md  SECURITY.md  CODE_OF_CONDUCT.md  CHANGELOG.md
  narration.example.toml                  shipped config: no personal values (§1.4)
  .github\                                workflows (WP03), issue + PR templates (WP02)
  docs\design.md                          the design (source of truth; §1.5 for changes)
  pyproject.toml  uv.lock                 server project "narration" (lead-owned, P7)
  src\narration\                          server package (see §3 for who owns what)
  workers\common\                         narration_worker: protocol, framing, determinism, fingerprint
  workers\qwen3tts\  workers\qa\          worker uv projects (own pyproject + uv.lock each)
  material\                               the service's own text material, versioned + hashed (WP18)
  spikes\<id>\                            Phase 0 scripts + saved outputs (JSON/CSV/MD; no audio)
  docs\decisions\                         one short ADR per spike outcome or contract change
  tests\<area>\                           server tests, one folder per WP area
  status\<WP>.md                          per-WP status reports, written by the WP's agent
  tools\                                  dev tools: gpu_lock.py, check_tracked.py, new_worktree, githooks\
  .dev\            (gitignored)           uv-cache\ · models\ · stores\<worktree>\ · gpu.lock\ · fixtures\
  worktrees\       (gitignored)           one git worktree per active WP
    wp10-text\  wp12-store\  …
```

Worktrees live in `worktrees\` inside the project, **not** under `.claude\` (owner's instruction).

### 2.2 Branches

- `main`: always green (the full CPU test suite passes). Only the lead merges into it.
- `wp/<nn>-<slug>` per work package, e.g. `wp/10-text`, in `worktrees\wp<nn>-<slug>\`.
- A contract change goes on `wp/01-contracts-<n>`, merged before the WPs that need it rebase.

### 2.3 An agent's loop inside its WP

1. The lead creates the worktree with the helper, which runs
   `git worktree add worktrees\wp10-text -b wp/10-text main`, copies `AGENTS.local.md` in, enables the
   git hooks, and syncs the venv (`uv sync`, with `UV_CACHE_DIR` = `<repo>\.dev\uv-cache`).
2. The agent works only in its worktree, by absolute path, and only in the directories its WP owns
   (§3). It commits often, with small commits.
3. It keeps `status\<WP>.md` current: state, what's done, test results, decisions, contract change
   requests, dependency requests, questions.
4. Done means: its acceptance tests pass, the full CPU suite passes, `ruff` and the type check are
   clean, it has rebased onto the current `main`, and its status file says `review`.

### 2.4 Merging (the lead)

The flow is **pull requests** (decided, Q4), as a public project should have. They give a public review
trail and CI gates on every change.

1. The lead pushes the WP branch and opens a PR against `main`. The description comes from the status
   file and ends with the Claude Code attribution line.
2. A review subagent reviews the diff for anything non-trivial. The lead reads the status file and
   checks the §1.4 rules: no personal defaults, licences, tests anyone can run.
3. Once CI is green, the lead merges it (merge commit, no squash, so WP history stays readable), updates
   the §4 table and the log (§9) on `main`, and pushes.
4. Remove the worktree (`git worktree remove`) and delete the branch once merged.
5. Tell the still-active WPs that depend on it to rebase.

PR descriptions and commit messages follow the no-local-paths rule too. Before pushing, the lead runs
`tools/check_tracked.py` over the branch's commit messages as well as its files.

### 2.5 The GPU lock

`tools\gpu_lock.py acquire --holder <WP> --minutes <n>` / `release` / `status`. The lock is a directory,
`.dev\gpu.lock\` (creating a directory is atomic on Windows), with an `owner.json` inside it: holder,
start time, expected end, PID. A lock past its expected end + 15 min with a dead PID counts as stale and
is reported, never silently broken; the lead clears it.

Rules for holders: check `nvidia-smi` first and start only when free VRAM ≥ what the run needs + 1 GB.
Hold it for bounded runs (≤ 30 min unless the lead agrees to longer, e.g. a 50-min ladder). Never kill,
throttle or touch another process on the GPU. Release on every exit path.

---

## 3. Work packages

Each WP lists: design sections · what it owns · depends on · GPU · deliverables and acceptance. The
lane says whether it needs the GPU lock at all.

### Wave 0: foundation (WP00 and WP01 by the lead, serially; WP02 and WP03 by agents beside WP01)

**WP00 Bootstrap.**
- The repository layout (§2.1).
- The server `pyproject.toml` with every server dependency declared up front (P7), each one
  licence-checked; empty packages.
- `ruff` + pyright + pytest configuration (`--basetemp` inside the worktree; the markers of AGENTS.md
  §6).
- Tools:
  - `tools\gpu_lock.py`;
  - `tools\check_tracked.py`: no local paths and no binaries in tracked files or commit messages;
  - the git hooks in `tools\githooks\` (pre-commit, commit-msg);
  - the worktree helper, which copies `AGENTS.local.md` in.
- The worker project skeletons with the P8 pins, locked universally (Q2).
- The dev models: the cached snapshots copied hash-verified into `.dev\models\`, and the missing two
  downloaded there (Q6).

*Owns:* the root, `tools\`. *Accept:* `uv sync` works; `pytest` runs (empty); the check passes on the
tree and fails on a planted local path and a planted `.wav`; the first push, once the owner has
reviewed `COMMERCIAL.md`.

**WP01 Contracts v1.** One module per contract, frozen at the end of Wave 0. They include DC-2 (backoff,
approved), and WP01 applies DC-1 to DC-3 to `docs/design.md` as revision 5.2.
- `narration.contracts.codes`: every error and flag code with severity, `retryable` and retake trigger
  (§14 tables verbatim, plus DC-2's `QUEUE_FULL` and `RATE_LIMITED`).
- `narration.contracts.schemas`: the §7.2 fragments and every tool's input/output schema, as Python
  dicts assembled with **no `$ref`** (§5); a test that walks `tools/list` and finds no `"$ref"`, and that
  every `outputSchema` has `"type": "object"` at its root.
- `narration.contracts.models`: typed records (dataclasses) for Voice, Hint, Cue, Segment, Flag, Error,
  and the sidecars: render, take, analysis, measurement, candidate, profile, engine profile, alignment
  benchmark (§6, App. B).
- `narration.contracts.worker`: the worker protocol messages (App. A), including the QA ops.
- `narration.contracts.interfaces`: the seams between WPs, as `Protocol`s: `TextPlanner`
  (request → per-cue received/spoken/engine + spans + warnings), `KeyBuilder`, `Store`,
  `DeliveryProcessor`, `QaScorer` (raw worker outputs + request inputs + measurement → verdict + flags +
  metrics), `AlignerCore`, `WorkerClient`, `Backend` (what the MCP front-end calls).
- `narration.config`: the §16 TOML, loaded and validated.

*Owns:* `src\narration\contracts\`, `src\narration\config.py`. *Accept:* the schema tests pass; every
code in §14 is present; each interface has a docstring naming its design section. **After this, a
contract changes only through a contract-change request** in a status file, and the lead merges it on a
`wp/01-contracts-<n>` branch.

**WP02 Public project scaffolding.** The rest of §1.4's "project hygiene" list. `LICENSE`,
`LICENSE-DOCS`, `COMMERCIAL.md`, `THIRD_PARTY_NOTICES`, a first `README.md` and `docs/design.md` already
exist (written 2026-09-26, before the first push, at the owner's request). Still to write:
- `CONTRIBUTING.md`: the contribution grant as in the evolution simulator, setup with uv, the test tiers
  and markers, how to run GPU tests, Conventional Commits, the dependency-licence rule;
- `CODE_OF_CONDUCT.md` (Contributor Covenant);
- `SECURITY.md`: the §17 threat model, and a reporting route the owner chooses;
- `CHANGELOG.md`;
- issue and PR templates, including a "Commercial licence" issue template;
- `narration.example.toml`, with no personal values.
*Owns:* the repository root docs, `.github\` (except workflows). *Accept:* a reader who has never seen
this machine can tell what the project is, what it needs, and how to contribute.

**WP03 CI.** GitHub Actions: ruff (lint and format), pyright, and the default pytest suite on
`windows-latest` and `ubuntu-latest` with Python 3.12, using uv with the lock file. Actions are pinned
by commit SHA, run with `permissions: contents: read`, and use no secrets. A job runs `tools/check_tracked.py`
(no local paths, no audio, weight or database files) on the files and on the PR's commit messages. A job
runs the schema checks: no `$ref`, and every output schema is
an object. *Owns:* `.github\workflows\`. *Accept:* green on the empty skeleton; red on a planted lint
error and on a planted `.wav`.

### Wave 1: parallel, no GPU needed (except the GPU lane)

| WP | Title | Design | Owns | Depends | GPU |
|---|---|---|---|---|---|
| WP10 | Text pipeline + negation lint | §7.2 canonical form/join/spoken length/exact spans, §9.1, §9.3, §3.5 | `narration.text`, `narration.lint`, `tests\text` | WP01 | no |
| WP12 | Keys, seeds and store | §10.2, §10.3, §15, §17.2, §4 item 6, §6 | `narration.keys`, `narration.store`, `tests\store` | WP01 | no |
| WP13 | Delivery post-processing | §13 | `narration.post`, `tests\post` | WP01 | no |
| WP14 | QA logic (pure) | §11.1, §11.3, §8 (suggestion), consistency, listen-first, report | `narration.qa`, `tests\qa` | WP01 | no |
| WP15 | Cue alignment | §11.2 | `narration.align` (pure) + the `align` op in `workers\qa` | WP01 | no (CPU model) |
| WP16 | Worker protocol, common package, fake worker | App. A, §4 workers, §4.1 thread caps | `workers\common`, `narration.workers` (client), the `fake` role | WP01 | no |
| WP17 | MCP front-end skeleton | §5, §7, §14 | `narration.mcp`, `tests\mcp` | WP01 | no |
| WP18 | The service's own material | §3.2, §11.2, §10.1 canary, §15, Phase 4 demo | `material\` | WP01 (for the text rules) | no (listening: H1) |
| WP19 | Platform seam (Windows only) | §4, §4.1, §17.2, Q2 | `narration.platform`, `tests\platform` | WP01 | no |
| WP20 | GPU lane: Qwen worker + Phase 0 GPU spikes | §10.1, App. A, §20 (d)(e)(f)(h)(i) | `workers\qwen3tts`, `spikes\` | WP16 (protocol) | **yes** |

**WP10 Text pipeline + lint.** Sanitise (control characters, `[` `]` `<|` `|>` → `TEXT_REFUSED`),
canonical form, join, the `text` = join check, spoken length in code points, exact spans → word ranges
(whole-word rule, punctuation set aside), hints (case-sensitive, longest first, whole words, the
possessive rule, recorded offsets), never across a cue (`TERM_SPLIT_ACROSS_CUES`), the four text checks
(`digit`, `symbol`, `unit_like` with the generic unit list, `letter` as info), `text_checks_version` +
rules hash. The positive-only lint (§3.5): the word list, the allowed morphological negatives, the
suggestion table. *Accept:* every §9.3 case as a named test, on the service's own fixtures only.

**WP12 Keys, seeds and store.** RFC 8785 canonical JSON and sha256; `voice_hash`, measurement key,
`render_key`, `delivery_key`, `analysis_key`; ids `rn_`/`tk_`/`an_` + 16 hex; job and design ids as ULIDs;
the seed formula (§10.3) with a golden test. SQLite in WAL mode: jobs, queue, cache index for the three
layers, measurements, profiles, designs, provenance, retention bookkeeping. Content-addressed paths, writes
to a temp name then renamed, immutable files made read-only. Write confinement: server-built paths only,
`realpath` under the root, Windows reserved names and reparse points refused. **Claim-time re-check and
in-flight waiting**: a lease per key, so two jobs never produce the same key twice. `gc` (dry run by
default) and `verify` (re-hash). *Accept:* key golden tests; two concurrent claimers of one key produce it
once; a crash mid-write leaves no partial file; confinement tests.

**WP13 Delivery post-processing.** Relative trim (p95 of 20 ms frame RMS − 40 dB, `pad_s` 0.08), pinned
48 kHz resampler, static gain to −16 LUFS (BS.1770-4, mono weight 1.0), 10 ms fades, PCM_24, true peak on
the final file at 4× oversampling with the −1.0 dBTP ceiling winning, `LOUDNESS_UNDER_TARGET` / `GAIN_HIGH`,
the trim and loudness records, tool names and versions for the delivery key, CPU thread cap. *Accept:*
deterministic output (same input → same bytes, twice and across processes); the length formula holds to
the sample; loudness and true peak within tolerance on real Qwen output from the bakeoff (read-only,
referenced by path, never copied into git).

**WP14 QA logic (pure).** Given raw worker outputs (transcript + word times, embeddings, alignment,
signal stats) plus the request's inputs and the voice's measurement: `wer_raw`, `wer_adj` (apostrophes,
aliases, hinted terms collapsed) with the word-count rule; exact spans (vendored normaliser + "nought",
edit alignment, `same`/`different`/`missing`); terms (letters-only fuzzy ≥ 0.75 or alias); head and end
insertion, including reference bleed; pace against the curve at this length; speaker thresholds from the
measurement; signal checks (clipping on raw, longest silence, NaN/DC, token cap); the verdict and retake
triggers; the suggestion tiers (§8, V2); the per-job consistency report; the `listen_first` order; the
report (markdown + JSON). *Accept:* the bakeoff's evidence reproduced as golden tests (marker
`evidence`):
- the normaliser cases of §11.3, plus spelling variants through the vendored map (§1.3 item 2);
- the names probe's numbers (zero mismatches) and its raw WER range;
- d4's similarities not flagged under a d4-calibrated threshold.

Also unit tests for every threshold edge in the `default.v3` table.

**WP15 Cue alignment.** Pure part: text → wav2vec2 alphabet (accents folded, apostrophe, `|` between
words and for in-word hyphens), `align_as`, token → (cue, word) map, tokens outside the alphabet dropped;
the `T ≥ L + R` guard; snapping boundaries into pauses (20 ms energy frames) with `CUE_BOUNDARY_NO_PAUSE`;
confidence and `CUE_LOW_CONFIDENCE`; unplaceable cues null and never interpolated; the Whisper
cross-check with `max_disagreement_s`. Worker part: the `align` op in `workers\qa`, emissions from
`facebook/wav2vec2-large-960h-lv60-self` on the **CPU** with the thread cap, `forced_align` +
`merge_tokens`, exceptions → `ALIGNMENT_ERROR`. Folds in **spike (b)**: re-check and save that
`forced_align` runs on this CPU (`spikes\b-forced-align-cpu\`), and pin the model revision. *Accept:*
spike (b) saved; alignment of bakeoff takes with cue times that look right (sanity, not accuracy:
accuracy is WP38); a too-short audio raises the guard, not a crash.

**WP16 Worker protocol, common package, fake worker.** JSON lines over stdio, UTF-8, stdout for protocol
only, logs to stderr; request ids; timeouts; structured errors (`GPU_OOM`, …). Shared worker code:
determinism setup (§10.1 switches, `CUBLAS_WORKSPACE_CONFIG` set before CUDA), the CPU thread cap,
below-normal priority, the `hello` fingerprint (Python, packages, CUDA, cuDNN, GPU, driver, env). The
server-side `WorkerClient`. A `fake` role that returns deterministic synthetic audio and canned QA
outputs, so the daemon and front-end can be built and tested with no model. *Accept:* round-trip tests;
a worker crash is reported, not hung on; the fake passes the same contract tests the real workers will.

**WP17 MCP front-end skeleton.** Starts with **spike (j)**: which Python MCP SDK version implements
revision 2026-07-28 (`MCPServer`, stateless, `server/discover`, still answering legacy `initialize`), what
a current Claude Code client sends, and whether validation can be done inside the handler so that
argument failures come back as tool errors rather than JSON-RPC errors. Saved as
`spikes\j-mcp-sdk\` + an ADR. Then: every v1 tool registered in the fixed order with its description
(stating no caller state kept, length is the caller's decision, a respelling is a hint), dereferenced
input/output schemas, in-handler validation → `isError` + structured Error + hint, `structuredContent`
plus a text copy, the resource templates (§7.7) and prompts (§7.8), all against a fake `Backend`.
*Accept:* sending `instruct` returns `INVALID_ARGUMENT` with `field: "instruct"` and a hint to §3.3;
`text_mode: "written"` is refused; no `$ref` in `tools/list`; an unknown tool is JSON-RPC −32602.

**WP18 The service's own material.** Written by the agent, in spoken form, to the rules of §3.2 and
§11.2, using **no caller's text** (not the evolution-simulator script, not the bakeoff captions):
- `calibration/narration-en.v1`: three calibration paragraphs, and ladder paragraphs at about 80, 150,
  250, 300, 350, 400, 450, 500 and 560 spoken characters, with number words marked as exact spans and an
  invented name;
- `alignment-en.v1`: about 12 paragraphs of cues, with common words, number words, invented names, short
  and long cues, and cue boundaries with and without a pause;
- the canary text (and the description + seed the canary clip will be designed from);
- the Phase 4 demo script: about 20 paragraphs of cues;
- text fixtures (for WP10) and planted-fault QA fixture specs (for WP40).
Each set gets a manifest with its version and sha256. *Accept:* WP10's checks raise no text warning on
any of it; spoken lengths hit the ladder targets within ±5 %; **gate H1** (the owner listens to a sample
of renders) before the corpus is frozen as v1.

**WP19 Platform seam.** One module owns every OS-specific mechanism, behind a small interface:
- the singleton lock (a named mutex keyed on the store path / an `fcntl` lock file);
- detached start (the Windows flags of §4.1 / `start_new_session`, with the std handles closed);
- a detection of "breakaway refused" that becomes `DAEMON_UNAVAILABLE`;
- kill-on-close for workers (a Job Object / process group + `PR_SET_PDEATHSIG`);
- below-normal priority;
- path rules (§17.2–3: reserved names, reparse points or symlinks, network and device paths; UNC on
  Windows, `/proc`, `/dev` and network mounts on Linux);
- the free-disk check.

**Windows only** (Q2). On any other OS every call raises a clear "unsupported platform" error, and
`doctor` says so; the POSIX implementations are later work. *Accept:* the interface's contract tests
pass on Windows; on Linux CI they check the "unsupported platform" behaviour and skip the rest cleanly.

**WP20 GPU lane: Qwen worker + Phase 0 GPU spikes.** One agent, holding the GPU lock in bounded runs.
In order:
1. **(i) + (h)**: load Base and VoiceDesign offline (`HF_HUB_OFFLINE=1`, `TRANSFORMERS_OFFLINE=1`) from a
   snapshot directory named by its commit SHA; record VRAM and load times.
2. The `qwen3tts` worker: `hello`, `load`, `prepare_voice`, `synthesize`, `design`, `unload`,
   `shutdown`, with every audio-changing setting passed explicitly (§10.1: `x_vector_only_mode=false`,
   `non_streaming_mode` false for Base / true for VoiceDesign, the sampling values from the pinned
   `generate_config.json`, `max_new_tokens`), token-cap detection, and the determinism environment.
3. **(d)** the repeat test on the Base clone path, in one process and across fresh processes, with the
   determinism switches: this **sets the tier** (`bit_exact` or `similar`, §10.1).
4. **(f)** the "Marufubens" check with 0.5 s of silence prepended, saved this time.
5. **(e)** optional: the `non_streaming_mode` A/B for Base.
6. Settle §1.3 item 1 (`max_new_tokens`) with an ADR.
Each spike: `spikes\<letter>-<slug>\` with script, JSON/CSV results, a README, and an ADR if it decides
something. *Accept:* a render through the worker protocol, with the bakeoff's settings and seed, scores
WavLM similarity ≥ 0.98 against the bakeoff take for the same text and seed. It need not be bit-identical,
because the bakeoff set no determinism switches. Spike (d) decides the tier and its ADR is merged.

### Wave 2: integration

| WP | Title | Design | Depends | GPU |
|---|---|---|---|---|
| WP22 | QA worker (real models) | §4 QA worker, §11.1 ASR/SV, §3.6 | WP16, WP15 | yes (bounded) |
| WP30 | Daemon process management | §4, §4.1, spike (g) | WP12, WP16, WP19 | no |
| WP31 | Job engine: scheduler, rounds, retakes | §4 GPU scheduler, §8 | WP12, WP13, WP14, WP16 | no (fake worker) |
| WP32 | Engine profiles, fingerprints, canary gate | §6 EngineProfile, §10.1 | WP20, WP22, WP12 | yes (bounded) |
| WP33 | `measure_voice` | §3.2 | WP31, WP22, WP14, WP18 (H1) | yes (long) |
| WP34 | `design_voice`, provenance, `profile_voice` | §3.1, §3.5, §3.6, §17.4 | WP31, WP20, WP22, WP10 | yes (bounded) |
| WP35 | `audition_pronunciation` | §7.6 | WP31 | yes (bounded) |
| WP36 | Front-end ↔ daemon wiring | §7.3–§7.7, §12, §17.3 | WP17, WP31, WP12, WP10, WP19 | no |
| WP37 | Operator CLI `narration-admin` | §7.1, §10.1 pins, §15 gc/verify | WP12, WP30, WP32 | no |
| WP38 | Alignment benchmark + `bench alignment` | §11.2, spike (a) | WP15, WP18, WP20 | yes (renders); **H2, H3** |
| WP39 | Phase 0 (c): length ladder for d2 and d4 | §3.2, §20 (c) | WP33 (or a spike harness), H1 | yes (40–100 min) |

**WP22 QA worker.** `transcribe`: Whisper-large-v3, fp16, English, greedy, word timestamps, sequential
long-form (30 s windows conditioned on the previous text), revision pinned. `embed`: WavLM-base-plus-sv on
the GPU, and on the CPU for the canary. `f0` and `profile` per **DC-1**, with no GPL code: pitch median
and 10th–90th percentile range (Hz and semitones) from `librosa.pyin`, speaking rate, pause ratio,
loudness, spectral centroid, HNR (Boersma's autocorrelation method) and CPPS; a spectrogram and a pitch
PNG; each measure validated on synthetic signals with known values. The `align` op from WP15. Audio I/O through soundfile. Offline loading from
SHA-named snapshots; VRAM measured (spike (h), QA half). Choose and pin the WavLM revision: `main` has
only `pytorch_model.bin`, while `refs/pr/8` has safetensors, and the eval evidence probably loaded the
latter by auto-conversion. Confirm which, and that the embeddings match `voicelock.csv`. Most of it is
developed on the CPU with short audio; the GPU runs are bounded. *Accept:* on bakeoff takes, similarities and transcripts match the bakeoff's `eval/` numbers
within noise.

**WP30 Daemon process management.** The singleton (a named mutex keyed on the store path); detached start
with `CREATE_BREAKAWAY_FROM_JOB | DETACHED_PROCESS | CREATE_NEW_PROCESS_GROUP`, `NUL` handles and
`close_fds`; `DAEMON_UNAVAILABLE` when breakaway is refused (never a non-detached daemon); the Windows
Job Object with kill-on-close; the worker supervisor (the worker venv's `python.exe` started directly,
with the offline env, thread caps and below-normal priority); `run\daemon.json`; idle unload after 120 s
and exit after 15 min; `stop` (finish the in-flight segment) and `stop --now` (re-queue it); the status
that `get_server_status` reports; `release_gpu`. Folds in **spike (g)**: a daemon started from an MCP
session survives the client exiting. *Accept:* spike (g) saved; `stop --now` leaves no partial files;
workers die with the daemon.

**WP31 Job engine.** Also computes DC-2's `retry_after_s`, `poll_after_s`, queue drain estimates and
the `admission` facts. The claim loop; the GPU scheduler (NVML free-VRAM check with the 1 GB margin,
`waiting_for_gpu` every 15 s, `GPU_UNAVAILABLE` after 30 min; one resident group; grouping by engine
profile; priority then FIFO; affinity); the per-round pipeline (render → post-process → score) with
retakes on the next attempt numbers; the claim-time re-check and waiting on in-flight work (WP12's
leases); scoring-only jobs; cancel; OOM handling (unload, wait, retry once, `GPU_OOM`); progress with a
total that may grow; the suggestion and consistency hooks from WP14. All tested with the fake worker.
*Accept:* two overlapping jobs render a shared paragraph once; a failing take is retaken up to
`max_retakes` and never again on resubmission; the same request twice completes from the cache.

**WP32 Engine profiles and canary.** Engine profile records (model repo + 40-hex revision, weight
hashes, worker project + `uv.lock` sha256, package versions, dtype, attention, determinism switches,
audio-changing settings, capabilities, licence, `vram_need_mb`) and their hash; the worker fingerprint
check (`ENGINE_DRIFT`); `expect_engine_profile` (`ENGINE_CHANGED`); designing the service's **canary** on the
installing machine (DC-3) and storing its raw hash, embedding and calibrated threshold; the canary gate in both tiers; `engine
pin | repin | bridge`. *Accept:* a changed `uv.lock` or weight file is caught as drift; the canary gate
behaves as §10.1 says for the tier WP20 found.

**WP33 `measure_voice`.** The transcript check (Whisper on the clip, `REF_TEXT_MISMATCH`), the
calibration set (design text + three corpus paragraphs × 3 seeds) → anchor and similarity baseline, the
length ladder from the shortest rung up (≥ 3 seeds per rung) with the trend rule, `tol`, early stop at the
first failing rung, `max_segment_chars` / `max_segment_seconds`, the pace curve, `measurement.json`
returned in full, and `VOICE_NOT_MEASURED` for generation with an unmeasured clip. *Accept:* d4 is not
flagged across its own calibration takes (a Phase 3 exit criterion).

**WP34 Design and profile.** `design_voice`: VoiceDesign with derived seeds, the candidate's exact
transcript checked by ASR, the lint result, the profile, the provenance list (append-only, never pruned);
the synthetic-voices rule (provenance or `allow_sha256`, else `VOICE_NOT_SYNTHETIC`). `profile_voice` on
any readable WAV by path + sha256.

**WP35 `audition_pronunciation`.** A term with up to 4 respelling variants in an optional carrier
sentence; takes plus what the ASR heard; no choice recorded.

**WP36 Front-end ↔ daemon.** `submit_job` planning over the three cache layers (render, delivery,
analysis), `dry_run` plans with estimates from the pace curve, idempotency and `idempotency_key`,
`SEGMENT_TOO_LONG` warnings (never a refusal); `get_job` long-poll (`wait_s` ≤ 55) with progress
notifications and cancellation of the wait; `get_results` assembled exactly as §7.5 (text echo, takes,
cues, alignment, QA, fit only with `scene_seconds`, consistency, `listen_first`, `report_md`,
`resource_link`s); `check_text` with a voice's limits; `cancel_job`; `get_server_status`; the resources;
the path rules (absolute, local drive, regular file; network and device paths refused), the sha256
check, the clip copied into the store before a worker sees it; limits and the submit rate cap; the daemon
autostart; DC-2's backoff fields and codes on every tool that can return them, stated in the tool
descriptions.

**WP37 Operator CLI.** `install` (pinned snapshot downloads with verified hashes; TLS rules of §17.8),
`engine pin | repin | bridge`, `gc` (dry run by default), `verify`, `bench alignment`, `daemon
start | stop [--now] | status`, `doctor`, `render`.

**WP38 Alignment benchmark.** Spike (a) and its completion: render `alignment-en.v1` (one voice, one
seed first); a hand-mark format and an importer. The owner may mark in **DaVinci Resolve through its
MCP** (Q9): an agent builds a timeline of the renders with the cue texts as guides, the owner places
markers at each cue's first-word onset and last-word offset, and the agent reads the markers back.
**Caveat:** Resolve markers snap to whole frames, and at 24 fps a frame is 42 ms, which is the size of
the errors being measured. The timeline must run at ≥ 60 fps (±8 ms), or Audacity's sample-accurate
label tracks are used instead. Then: error = |service − mark| at
every cue start and end, p50/p95 with n, split by boundary kind; computed for CTC, CTC + snap,
Qwen3-ForcedAligner (+ snap), and Whisper; the choice by the rule of §11.2 (Q18); the cross-check and
confidence thresholds set from the spread; the AlignmentBenchmark record published through
`get_server_status` and every take. Phase 0 size first (~30 marks, **gate H2**), full size later (~240
marks over two voices × two seeds, **gate H3**). Also answers part of spike (b): whether `qwen-asr` fits
the QA worker's library versions or needs its own venv.

**WP39 Length ladder for d2 and d4.** Phase 0 (c). Needs the frozen corpus (H1). **Recommended: run it
once, through the real `measure_voice` (WP33)**, rather than also through a throwaway spike harness, which
would cost another 40–100 min of shared GPU time. If the flow needs `max_segment_chars` before WP33 is
ready, a spike harness built on the bakeoff's `eval/` code is the fallback.

### Wave 3: acceptance (the design's exit criteria)

| WP | Exit criteria from §20 | Depends |
|---|---|---|
| WP40 | **Phase 1 + 3.** Rendering twice and in a fresh process behaves as the tier says; drift detected; every planted fault caught (a different number in an exact span, a head insertion, a render cut by the token cap, an unplaceable cue); every planted non-fault passes ("per cent" heard as "percent", "nought" as "zero", one slip in a three-word title warns but does not fail); d4 not flagged across its calibration takes; the same request twice from the cache; the measured alignment error published. | Wave 2 |
| WP41 | **Phase 4.** A Claude Code session designs, measures and narrates the demo script with `takes: 2`; the client quits mid-job and the job completes; a resubmission after one edit renders one segment and returns the rest with the same take ids; two sessions share one queue; a recording that is not allowlisted is refused. | WP40, **H4** |
| WP42 | **Phase 5.** Flows A–F end to end (§8), including a "weeks later" batch with `expect_engine_profile` and the canary gate. | WP41 |
| WP43 | Docs: README (install, configuration, the `.mcp.json`), an operator guide, the tool reference generated from the schemas. | WP36, WP37 |
| WP44 | **Release readiness** (no release is made without the owner): a clean-machine install test from a fresh clone following only the README; `narration-admin doctor` on a machine with no GPU says so clearly; a licence audit of every dependency and model; `SECURITY.md` reviewed against the final code; a security review of the path, confinement and synthetic-voice checks; the version set and the CHANGELOG written; the GitHub description brought up to date with revision 5 (the current one still mentions "locked" voices). | WP41–WP43 |

Phase 6 (stitching, fit remedies, the written-text normaliser, the Tasks extension, a stronger SV model,
a listening model) is out of scope for this plan.

---

## 4. Status

Updated by the lead on `main` only.

| WP | Title | Wave | Depends | GPU | Status | Branch / worktree | Notes |
|---|---|---|---|---|---|---|---|
| WP00 | Bootstrap | 0 | – | – | `ready` | – | H0 answered 2026-09-26 |
| WP01 | Contracts v1 | 0 | WP00 | – | `todo` | – | |
| WP02 | Public project scaffolding | 0 | WP00 | – | `todo` | – | licences, README, design already in; beside WP01 |
| WP03 | CI | 0 | WP00 | – | `todo` | – | can run beside WP01 |
| WP10 | Text pipeline + lint | 1 | WP01 | – | `todo` | – | |
| WP12 | Keys, seeds, store | 1 | WP01 | – | `todo` | – | |
| WP13 | Delivery post-processing | 1 | WP01 | – | `todo` | – | |
| WP14 | QA logic (pure) | 1 | WP01 | – | `todo` | – | |
| WP15 | Cue alignment (+ spike b) | 1 | WP01 | CPU model | `todo` | – | |
| WP16 | Worker protocol + fake worker | 1 | WP01 | – | `todo` | – | |
| WP17 | MCP front-end skeleton (+ spike j) | 1 | WP01 | – | `todo` | – | |
| WP18 | Service material | 1 | WP01 | – | `todo` | – | H1 to freeze |
| WP19 | Platform seam (Windows only) | 1 | WP01 | – | `todo` | – | |
| WP20 | GPU lane: Qwen worker + spikes d, e, f, h, i | 1 | WP16 | **yes** | `todo` | – | sets the tier |
| WP22 | QA worker | 2 | WP16, WP15 | yes | `todo` | – | DC-1 |
| WP30 | Daemon process mgmt (+ spike g) | 2 | WP12, WP16, WP19 | – | `todo` | – | |
| WP31 | Job engine | 2 | WP12–14, WP16 | – | `todo` | – | DC-2 |
| WP32 | Engine profiles + canary | 2 | WP20, WP22, WP12 | yes | `todo` | – | DC-3 |
| WP33 | `measure_voice` | 2 | WP31, WP22, WP14, WP18 | yes (long) | `todo` | – | needs H1 |
| WP34 | Design + profile | 2 | WP31, WP20, WP22, WP10 | yes | `todo` | – | |
| WP35 | `audition_pronunciation` | 2 | WP31 | yes | `todo` | – | |
| WP36 | Front-end ↔ daemon | 2 | WP17, WP31, WP12, WP10, WP19 | – | `todo` | – | DC-2 |
| WP37 | Operator CLI | 2 | WP12, WP30, WP32 | – | `todo` | – | |
| WP38 | Alignment benchmark (spike a) | 2 | WP15, WP18, WP20 | yes | `todo` | – | H2, H3 |
| WP39 | Ladder d2/d4 (spike c) | 2 | WP33 | yes (long) | `todo` | – | needs H1 |
| WP40 | Acceptance: Phases 1 + 3 | 3 | Wave 2 | yes | `todo` | – | |
| WP41 | Acceptance: Phase 4 | 3 | WP40 | yes | `todo` | – | H4 |
| WP42 | Acceptance: Phase 5 | 3 | WP41 | yes | `todo` | – | |
| WP43 | Docs | 3 | WP36, WP37 | – | `todo` | – | |
| WP44 | Release readiness | 3 | WP41–WP43 | – | `todo` | – | a release itself needs the owner |

**Phase 0 spikes → where they live:** (a) WP38 · (b) WP15 · (c) WP39 · (d) (e) (f) (h) (i) WP20, with the
QA half of (h) in WP22 · (g) WP30 · (j) WP17.

---

## 5. Human gates

| Gate | What the owner (or a delegate) does | Blocks | Does not block |
|---|---|---|---|
| **H0** | Answers this plan's questions. **Done 2026-09-26** (§1.4) | – | – |
| **H1** | Listens to about 10 short renders of the service's own texts (a few minutes), to catch text that reads oddly; approves or asks for changes | freezing the corpus as v1 → WP33, WP38, WP39 | all of Wave 1 |
| **H2** | Marks ~30 cue boundaries on one rendered benchmark (one voice, one seed): where each cue's first word starts and last word ends; about 30–45 min, in Resolve or Audacity (WP38) | the Phase 0 aligner choice and the first error table | everything else: wav2vec2 CTC is the default anyway (Q18) |
| **H3** | Hand-marks ~240 boundaries (two voices × two seeds) | publishing the full measured error (end of WP38) | the rest of Wave 2 |
| **H4** | Runs, or watches, the Phase 4 acceptance session and listens to its takes | WP41, WP42 | – |

The design also leaves **Q2 (which voice)** to the story flow. It blocks nothing here: the service is
built and measured with both d2 and d4.

---

## 6. Schedule

```
Wave 0             WP00 (lead) ─► WP01 (lead) ─────────────────┐ contracts frozen
                               └► WP02, WP03 (agents, parallel)│
                                                              ▼
Wave 1 (parallel)  WP10  WP12  WP13  WP14  WP15  WP16  WP17  WP18  WP19  (CPU; up to 6 at once, §7)
                                              │
GPU lane (serial)                             └► WP20: i+h → qwen worker → d (tier) → f → e
                                                              ▼
Wave 2             WP30 ─┐   WP22 (GPU, bounded)   WP36 (after WP31)   WP37
                   WP31 ─┴► WP32 ─► WP34, WP35     WP38 (H2 ─► H3)
                            WP33 (H1) ─► WP39
                                                              ▼
Wave 3             WP40 ─► WP41 (H4) ─► WP42        WP43 ─► WP44 (release only with the owner)
```

**Critical path:** WP00 → WP01 → WP16 → WP20 (the Qwen worker) → WP32 / WP31 → WP33 → WP40 → WP41. The
GPU lane and the owner's gates are the scarce resources, so WP16 and WP20 start first within Wave 1.

---

## 7. How many agents at once

- **Recommended: up to 6 concurrent Wave 1 agents**, plus the GPU-lane agent, plus the lead. Wave 1 in
  order of priority: WP16, WP20 (GPU lane), WP12, WP14, WP10, WP17, then WP19, WP13, WP15, WP18 as slots
  free.
- The lead (the main session) does not implement WPs once Wave 1 starts. It creates worktrees, answers
  questions, reviews, merges, pushes, and keeps this file and HANDOFF.md current.
- Each WP agent is briefed with: its WP section from this plan, the design sections it cites, AGENTS.md,
  its worktree path, the contract modules it must not change, and the status-file format.
- Model choice: the most capable model for WP01, WP12, WP14, WP15, WP31 and WP36, where subtle errors
  are expensive; lighter models are fine for WP18 drafts and WP43.

---

## 8. Risks

| Risk | Effect | Mitigation |
|---|---|---|
| The MCP SDK does not yet implement 2026-07-28 as the design believes | WP17's shape changes | Spike (j) is WP17's first task; the `Backend` seam keeps the rest unaffected |
| The clone path is not bit-exact (spike d) | the `similar` tier: take ids stable only while cached | The design covers both tiers; build both from the start, and switch by the ADR |
| `forced_align` fails on this CPU build | no aligner | The numpy CTC Viterbi fallback (§4), ~50 lines, in WP15 |
| GPU contention with the other jobs on this machine | slow Wave 2, long waits | The GPU lock, bounded runs, most work on the fake worker; long runs (ladders) scheduled with the owner |
| Worker venv size × worktrees | disk | uv cache on D: with hardlinks (P5); worker venvs only in the WPs that need them |
| A contract proves wrong mid-wave | rebases, rework | Contract-change requests merged fast by the lead; interfaces kept narrow |
| Pushing to a public repo | exposure | Code and the service's own text only (P6); every push reviewed by the lead; no absolute personal paths or owner-specific values in code or shipped config (§1.4) |
| Owner-specific assumptions leak into the product (a 4090, Windows, d2/d4, local paths) | others can't use it | The §1.4 rules in every review; CI on a machine that has none of them; WP44's clean-machine install |
| CI cannot run GPU paths | regressions in model code go unseen | The worker contract tests run against the fake worker in CI; a documented local GPU suite is run before every merge that touches `workers\`, and its result goes in the PR |
| The scope of public rigour (platform seam, docs, CI) slows v1 | later first use by the story flow | WP02, WP03 and WP19 run in parallel with the core; WP19 is Windows only (Q2) |

---

## 9. Log

- 2026-09-26: plan revision 1 written from design revision 5.1 and a survey of the bakeoff (§1.2, §1.3).
  Local repo initialised with `origin` set; nothing pushed. The remote is **public**.
- 2026-09-26: the owner made it a public project (§1.4) and chose the evolution simulator's licences.
  `LICENSE` (PolyForm NC 1.0.0, body verbatim), `LICENSE-DOCS` (CC BY-NC 4.0), `COMMERCIAL.md`,
  `THIRD_PARTY_NOTICES` and `README.md` written before any push. Found that parselmouth is GPL
  (§1.3 item 3). Waiting on H0.
- 2026-09-26: H0 answered (§1.4 table). Windows first with cheap preparation for other platforms; PRs;
  no local paths in tracked files, so machine facts moved to the gitignored `AGENTS.local.md` and a
  checker enforces it; the design copied into `docs/design.md` as the source of truth; parselmouth
  replaced (DC-1, approved). Backoff contract (DC-2) and install-time canary (DC-3) proposed. The
  unpushed first commit was rebuilt so that no local path is ever in the public history. Plan
  revision 2.
- 2026-09-26: the owner approved `COMMERCIAL.md`, the README, DC-2 (backoff) and DC-3 (install-time
  canary), and asked for the push. `main` pushed to `origin`.
