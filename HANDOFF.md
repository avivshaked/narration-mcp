# HANDOFF

*Read this first. Then read [plan.md](plan.md) (§4 is the status table), [AGENTS.md](AGENTS.md) and the
gitignored `AGENTS.local.md`, which holds this machine's paths and facts. Keep this file current by
rewriting it to the present at the end of every session; it holds no history.*

## Where things stand

- **Built and on `main`:** every package marked `done` in plan.md §4, design revision 5.18, contracts 1.6.12.
  First narration through the MCP (M1) is reached: the owner narrates from a Claude Code session.
- **Release:** `v0.1.0a1` is tagged and the owner created the GitHub pre-release (notes in
  `.dev/pr/release-notes-0.1.0a1.md`). Private vulnerability reporting, the route `SECURITY.md` names, is on.
- **The narration voice is `a-warm-s101`** (the owner's choice, 2026-09-27). The lead designed it from the
  owner's description. It is measured (`/v2`, characters per second) in the real store `.dev/stores/service`:
  reliable up to 301 spoken characters (the ladder stopped at 300, as `narration.toml` notes). Clip, sidecar
  (the transcript is the corpus's design text) and allowlist entry are local (`.dev/voices/`, `narration.toml`;
  `[qa] profile` says `default.v5`). A measured spare, d2 re-designed with seed 2006, is kept local only.
  The owner's guide is `.dev/scratch/first-narration.md`, and the narrating agent's brief is
  `.dev/narration-brief.md`.
- **The owner's `.mcp.json`** starts the server through the venv's interpreter
  (`<repo>\.venv\Scripts\python.exe -m narration.mcp --config <repo>\narration.toml`), not `uv run … narration-mcp`,
  whose launcher `.exe` is what Avast sandboxed (`AGENTS.local.md`). A merge that changes tool texts reaches the
  owner's session after a `/mcp` reconnect.
- **The daemon** starts detached under Claude Code, because a child started with `CREATE_BREAKAWAY_FROM_JOB`
  ends up in no Job Object there (KNOW, probed from the lead's session; BELIEVE the owner's session behaves the
  same). If `get_job` ever answers `DAEMON_UNAVAILABLE` (`left_in_job`), the route is `narration-admin daemon
  start --foreground`.

## In flight

`git worktree list` shows:
- `worktrees/spike-l-pitch-speed`, branch `spike/l-pitch-speed`, **not merged**: spike (l), whether the speed of
  pitch change catches a glitch Whisper cannot hear. It did not (0 of 4 faulty takes both separated and
  located). Its README and `results.json` are there; the owner's labels and pictures are in
  `.dev/spikes/l-pitch-speed/`. The follow-up is the sibling project `tts-glitch-detector` (local git, no
  remote), which reads this project's test set `.dev/fixtures/qa-glitch-validation/` (28 WAVs with
  `manifest.json`) read-only.
- `worktrees/rel-0.1.0a1`, branch `chore/release-0.1.0a1`: merged and tagged. The worktree and branch can go.

## Next steps

1. **GPU windows, which the owner chooses** (never while the owner narrates):
   - re-score the owner's first job under the new pace rule to check the 0.25 s pause floor (`MIN_PAUSE_S`,
     ASSUME; WP47 F2);
   - the rest of the voice's ladder (350–560 characters) in 30-minute stages
     (`.dev/service-setup/staged_ladder.sh`; remove its `STOP` file first);
   - WP35's one `gpu` test (`tests/audition/test_audition_gpu.py`);
   - the d4 acceptance (`tests.measure.run_acceptance`; whether it has run is not recorded);
   - then WP38 (gates H2, H3), WP39, and Wave 3 (WP40–WP42, gate H4).
2. **WP44, release readiness:** see `status/WP44.md`. Left: the clean-machine install, `doctor` with no GPU, the
   version. Open findings S3–S6 and S9, and the licence questions, are in that file.
3. **Document how to split text into segments** (the owner asked, 2026-09-29), once the 8-segment narration is in.
   Where: the `submit_job` and `check_text` descriptions and the narration workflow prompt
   (`src/narration/mcp/descriptions.py`, then regenerate `docs/tools.md`), and a paragraph in the README's First
   run. What: one idea per segment, about half to all of the voice's `max_segment_chars`, one sentence per cue,
   no one-sentence segments (intonation restarts; pace and speaker checks are least reliable on short takes),
   never past the reliable length. Evidence first: compare the same text as 11 short segments (job
   `job_01M3QA2YFP81A57SPN1FRFB2EJ`) and as 8 longer ones (`job_01M3QBP2DNKKWMHS5T0KHWPYGR`): flags, and the
   owner's ear. The tool texts have almost no headroom (the 2048-character client limit: the instructions
   2009, `get_results` 2013, the longest prompt 1833; a test enforces it).
4. **A flaky Windows test (KNOW, CI run 36470360457):** `test_a_poll_during_a_failed_start_leaves_it_backend_not_installed_s14`
   got `psutil.NoSuchProcess` from the start instead of `WorkerFailure`. Some psutil call on the start path does
   not catch a worker that has already exited (not `_windows.set_below_normal`, which does). Find it and convert
   it to the worker's failure. Also seen once under load: `tests/engine/test_pinning.py::test_bridge_finds_a_profile_whose_recorded_folder_is_gone_s10_1[partial_only]`.

## Waiting on the owner

- **Decisions from the readiness audit** (`.dev/lead/readiness_synthesis.md`, local, section 4):
  - D1: a full ladder for the first voice, or keep the stop at 300 (needs a GPU hold over 30 min, which
    `tools/gpu_lock.py` refuses without an override);
  - D3: a compact `get_results` (a segment filter; words off by default; §7.5);
  - D4: run the service from a pinned checkout, or freeze merges that touch the engine while narrating;
  - D5: whether dev GPU work pauses while the owner narrates;
  - D6: the canary on every Qwen load (12–19 s each) or once per worker process;
  - D7: QA rule changes (a probable-name warning in `check_text`; insertion runs; a length-aware speaker margin);
  - D8: loudness: deliveries are at −23 LUFS; confirm what the consumer expects.
- **The Avast Auto-Sandbox exception** for the projects folder, or Auto-Sandbox off: the durable fix for the hangs.
- **Lead gap-fills the owner may overrule:** DC-11, DC-12, DC-14, DC-15, DC-20 (`docs/design-log.md`).
- **4a, cross-job grouping** (design §4 item 3): the lead recommends leaving it out of v1.
- **A `measure_voice` with a mistyped transcript** still queues a new, costly measurement; a refusal or warning is
  a design decision. Likewise an info-only `check_text` note for a very short segment.
- **WP44's licence questions:** LGPL (soundfile, soxr, pywin32's adodbapi), MPL-2.0 (certifi, tqdm, orjson),
  NVIDIA's CUDA wheels, and reading the model cards (`status/WP44.md`).
- **WP46's plan** (`docs/frontend-plan.md` §10): nine questions, question 1 first. Nothing is built before approval.
- **The female voice:** try 1 (job `f-deep-a`) was energetic but its median pitch was 223–288 Hz, not deep. Try 2's
  prompt (contralto, almost baritone, late fifties) is in `.dev/scratch/female-voice-try1.md`, not generated
  (the owner's call). Local script: `.dev/lead/design_batch.py`.
- **The LinkedIn post** is finished and the owner posts it. Text: `.dev/scratch/linkedin-post-v2.txt` (copy it from
  the file; pasting from the chat view loses line breaks and apostrophes). Its first comment has two placeholders
  for the owner (the story video and an earlier post). Video:
  `.dev/scratch/linkedin-video/narration-mcp-linkedin.mp4` (1080×1350, 30 fps, 1:52), thumbnails `thumb-a.png`
  and `thumb-b.png` there. Clean-up waiting for the owner's OK: `frames-final/` (4.4 GB) and the older preview
  folders there. KNOW: 3370 full-size frames took 12 min (about 0.2 s each, GPU about 30%, CPU about 25%), so the
  render is bound by one process's CPU work; next time run two or three Blender instances on one folder.
- **Later:** one run of the daemon under Claude Code itself as an MCP server (spike (g) is BELIEVE until then);
  gates H2 to H4.

## While the owner narrates (and the machine-free rule)

- **The owner needs the machine free: agents run no local pytest and no basedpyright.** They run ruff,
  `check_private` and `check_tracked` only, then push; CI on GitHub is the test runner
  (`gh run watch <id> --interval 60`). GPU work waits for a window the owner chooses.
- **Hold any merge that changes the engine's identity:** `workers/qwen3tts/` (code or lock), `workers/common`'s
  version or lock, `material/canary/`, `src/narration/engine/profile.py`, or anything feeding the render, delivery
  or analysis keys. Each forces a repin or re-QA and orphans the owner's measurement or cached work. Other merges
  change what a newly started server or daemon runs, since the service runs from the main checkout; restart the
  daemon only deliberately.
- **Dev GPU work waits** while `.dev/stores/service/run/daemon.json` shows a job or a loaded worker.
- **Never edit `narration.toml`'s `[measurement]`**; after any other edit, stop the daemon and reconnect.
- After a merge that changes a worker, the main checkout's worker venv needs `uv sync --locked` again.

## Open follow-ups

By area. Each is low unless it says otherwise.

**Pace and QA rules**
- `codes.py` lists `PACE_FAST` and `PACE_SLOW` with a fail severity, while design §14 says warn only (a profile
  can still set `pace_fail_tol_factor`): reconcile the text. WP47 F4: if the re-score shows very short
  segments reading `PACE_SLOW`, set a minimum spoken length below which pace is not flagged (a new QA profile).
- The normaliser counts "voice-over" against "voiceover" (2 errors) and "twelve gigabytes" against "12GB"
  (1 error) as word errors: `WER_HIGH` warnings on correct speech.
- `get_results` does not show what the recogniser heard (`qa.transcript` is in the stored analysis only), so an
  agent cannot see why a take was flagged.
- `submit_job` called on the backend directly ignored a top-level `takes` (it belongs in `options`); check that the
  MCP layer refuses unknown top-level keys.
- **WP40's planted faults:**
  - two takes passed QA clean (WER 0) with a word the owner heard as wrong (KNOW, 2026-09-29, job
    `job_01M3QBP2DNKKWMHS5T0KHWPYGR`: segments `s1-hook` take 1 and `s7-local` take 2): gate H1's blind spot, a
    garbled word Whisper still reads as the right one (QA did catch `s8` take 1, "that is run" for "that has run");
  - §11.1 step 7's acoustic head check is not implemented (`narration.qa.textmatch` checks Whisper's words only);
  - a garbled or dropped first word is one word error (a warning on a 13-word cue), not `HEAD_INSERTION`, so it
    is not retaken; two garbled words fail and are retaken; fillers at the start are removed by the normaliser and
    never counted; plant "garbled first word" and "filler at the start", and short-segment pace cases;
  - a take cut by the token cap is not a prefix of the uncut take; do not assume it is;
  - measure the ASSUME values in `narration.qa.textmatch` (bleed ratio 0.75 with at least 2 words or 8 letters;
    `_MIN_GAIN` 0.05) and `measure_voice`'s transcript-check rule (WER over 6 % with at least 2 errors).
- WP18: the ladder's exact spans may meet ASR number formatting ("40m", "7am"); `ladder-500` has two hour
  phrases to reword if a ladder run shows a false `EXACT_SPAN_MISMATCH`. The fixture sets `text-v1` and
  `qa-faults-v1` are still `draft` (WP40 consumes the second). `material/` sits outside `src/narration`, so a
  wheel would miss it.
- WP15/WP38: the aligner's snap reaches (`snap_reach_start_s` 0.12, `snap_reach_end_s` 0.28) are ASSUME; re-set
  them from a hand-marked boundary next to an unplaced cue (basis `spikes/b-forced-align-cpu/reach.py`), and consider
  a separate reach for wildcard edges (the method id changes whenever a reach changes). A stale comment and
  `PYTHONPATH` workaround remain in `tests/align/test_bakeoff_evidence_s11_2.py` ("not an installed package yet").
  An `align_as` with characters outside the aligner's alphabet silently
  becomes a wildcard: should `check_text` warn? One rule is wanted for a worker answering another model's snapshot
  (the qwen3 worker says `INVALID_REQUEST`, the QA aligner `BACKEND_NOT_INSTALLED`).
- WP33: a rung that passes on its medians with one failing seed shows `warned`; a `generate` job of exactly a
  ladder paragraph at a ladder attempt reuses that rung's analysis, which has no pace check (own key if it must).
- WP34: show a design candidate's `similarity_pass` as `CANARY_MISMATCH` (info; the VoiceDesign gate is a weak alarm
  by nature, lead decision); no cache answers a repeated `design_voice` or `profile_voice`; no `WER_HIGH` warn
  level; a batch design that gives way loses its unpublished clips (redesigned on resume, same seeds); the
  description check does not refuse VoiceDesign's other special tokens (`<tts_pad>`, `<think>` …; WP44 S3).
- WP35: a fully cached audition still loads the QA group once to embed the clip (cache the embedding by clip
  sha256 and model); `AuditionVariantResult` has no `suggested_take_id`; no threshold turns a low `spk_sim_clip`
  into a flag (`sim_fail_floor` 0.90 is ASSUME); the front-end queues an audition at `batch` priority;
  the README says the tool was only tested on fake workers until the `gpu` test passes.

**Daemon, front-end and admin**
- WP36: §7.3's all-cached submit should complete at once (a WP31-owned function called at submit; today the job is
  created `queued`). Proposals: `design_revision` beside `spec_revision` in `get_server_status`; lower-case
  `voice.sha256` and `audio.sha256` in `mcp/validation.py`. Three contract requests from the dead-daemon work: a
  `daemon` object in `get_job`'s output, `MeasurementRecord.transcript` with a store method
  `measurements_of_clip`, and a `check_text` flag for a transcript near-miss. Measure launch-to-status time on
  Windows (four BELIEVE values: `DAEMON_START_GRACE_S` 30 s, `START_WINDOW_S` 90 s, `STOP_TO_STOPPED_S` 30 s,
  `SPAWN_TOLERANCE_S` 10 s). Known limits: a daemon that starts and then fails at once is not held back by the
  window (each `get_job` launches one more; a throttle is suggested in design §4.1); a later failed launch within
  30 s of an operator's stop makes older queued jobs read as stopped; the transcript near-miss misses other
  whitespace and typographic dashes.
- WP30: `contracts/interfaces.py` `Platform.spawn_detached` and `codes.py` `DAEMON_UNAVAILABLE` still say "breakaway
  is refused" (it is "refused or incomplete"; a contracts fix, the lead's). `CUDA_VISIBLE_DEVICES` (BELIEVE): CUDA
  numbers only the visible GPUs while NVML numbers all, so `[gpu] device` could mean two GPUs; warn in `doctor` or
  the supervisor. `WorkerSupervisor` adds a worker to the kill-on-close group after the spawn (a suspended start,
  like `start_in_job`, would close the gap). GitHub's hosted Windows runner never exercises a successful detached
  start, so a regression that refuses every detached start stays green there.
- WP37: `bench` is a stand-in until WP38 creates `narration.bench.admin` (or `COMMAND_GROUPS` changes); `doctor`
  should check the material manifests; the store needs a single-item evict for what `verify` finds damaged;
  `reset_workers` stays a proposal. A worker's `BACKEND_NOT_INSTALLED` sticks for the daemon's lifetime, so
  `narration-admin install` stops the daemon after repairing one.
- WP45: the network-path test (`tests/admin/test_voices.py`, `\\server\share`) only checks a missing file on Linux;
  README's `voices allow` step lacks the restart advice; the `add_to_file` "added since read" branch is untested;
  two simultaneous `voices allow` runs can lose a hash (a lock file beside the configuration); the temporary file
  is briefly readable under the umask on POSIX; a Windows explicit ACL on `narration.toml` is not carried over;
  a subprocess test that `narration.mcp.server` and `narration.backend` never import `narration.admin.voices`.
- WP48: `FAILURES_SCHEMA_ID` lives in `narration.admin.failures` (move to `contracts.names` if wanted); `gc` only
  reports failed takes, so retention that waits for an audit is not implemented.
- WP43: any change to the schemas, tool descriptions or codes needs `docs/tools.md` regenerated
  (`py -3.12 tools/gen_tool_reference.py`; a default-suite test fails when it is stale).
- WP32: add `CanaryPin.calibration: tuple[float, ...] = ()` (unhashed) so `engine show` can print the calibration
  similarities; the editable worker packages are not fingerprinted (a worker code edit with an unchanged version
  is invisible to the hash; the canary would still catch an audio change); `[engines.qwen3_base]
  x_vector_only_mode` must stay `false` (the canary clones with `false`); a canary-text change forces new profile ids; `licence`, `capabilities`, `worker_project` and `engine_profile_id` are
  hashed though they change no audio (a change invalidates caches).
- WP17: what Claude Code sends first (`initialize` or an enveloped request) is unverified (ADR 0001, BELIEVE:
  either); legacy-era clients get no `resources/subscribe` and long-poll `get_job`. Check that the job engine
  reaches `FrontEnd.job_updated`.

**Workers, platform, CI**
- WP22: no worker-qa job in CI; a `cuda:N` beyond the device count answers `INTERNAL` rather than
  `INVALID_REQUEST`; `spectral_centroid_hz` is null for silent audio while the contract types it `float`; the GPU
  acceptance should assert monotonic word times; windowed aligner emissions for takes beyond about 60 s (about
  22 MB/s past 1.6 GB: 2.6 GB at 120 s); lighter word times.
- WP16: the qwen3 worker keeps its own copy of the snapshot-reference check (the fake and QA worker share
  `narration_worker.snapshots`); the shared WAV writer can still raise `struct.error` past 4 GiB; whether the fake
  should refuse a bare `{"device": "cpu"}` load (it keeps the 8192 ceiling on purpose); a worker started through a
  venv launcher that stops reading is caught only by the request timeout.
- WP12: treat a zero file id (`st_ino == 0`) as unknown in the undo check; close the thread's connection if a
  ROLLBACK ever fails (FAT and exFAT reuse file ids).
- WP19: the Windows path check refuses a link to a volume with no drive letter (`\\?\Volume{…}`).
- WP03: Dependabot for the `github-actions` ecosystem (the SHA pins repeat across jobs); `[tool.uv]
  required-version = "==0.10.10"`; `uv lock --check` for the two worker locks; an `actionlint` job (`actionlint-py`
  is MIT; shellcheck is GPL); branch protection on `main` with the required checks (the setting is not recorded);
  showing CI red on GitHub needs the owner's OK, since a planted `.wav` on the public remote stays.

## Things a new session should know

- The bake-off that preceded this project is private and read-only. Its reusable code, golden numbers and models
  are mapped in plan.md §1.2, and the findings from its installed code in §1.3. The consumer's requirements
  (R1–R14) are mapped in design §21; the service never depends on that caller.
- **Private text:** the repo is public, and the bake-off's scripts and transcripts are the owner's private story.
  `tools/check_private.py` runs in the git hooks, including pre-push; the lead runs it on every branch before
  pushing, and every reviewer runs it. A cloud session cannot run it (the private lists exist only here), so
  run it before merging a cloud PR. Cloud PRs open as drafts (`gh pr ready` before merging).
- **The PR flow:** push the branch; write the PR body to `.dev\pr\<wp>.md` and check it with
  `tools/check_tracked.py --msg-file`; `gh pr create`, then `gh pr checks N --watch`; merge locally with
  `git merge --no-ff` and push `main`; remove the worktree. Every branch gets an independent read-only reviewer
  before merge; findings are fixed first, and a branch that had a BLOCK or a data-loss finding is re-verified.
- **Removing a worktree** can fail with "Filename too long" (deep `.pytest-tmp` paths). Remove the folder with
  PowerShell `Remove-Item -LiteralPath "\\?\<path>" -Recurse -Force`, then `git worktree prune`.
- **GPU:** agents take the GPU lock in bounded runs. `owner.json` stamps UTC. After a restart, check each
  worktree for uncommitted files before resuming its agent.
- **Hangs:** run every tool as `uv run python -m …` (`AGENTS.local.md` says why).
- uv needs `UV_NATIVE_TLS=1` on this machine, because something intercepts TLS (`AGENTS.local.md`).
- The Claude Code harness reports "file changed on disk" when a file is touched through `d:\` and then `D:\`.
  That is the path's case, not an outside edit.
- **Cloud sessions** (claude.ai/code) are listed by ListAgents and can be messaged but cannot message back:
  follow them on GitHub. Briefs: `.dev/scratch/cloud-task-1.md`, `.dev/scratch/cloud-briefs-2.md`. A "remote"
  agent started from here runs locally unless the owner starts the session at claude.ai/code.
- **Agents:** at most 6 at once. Opus 5.5 by default; Fable for the hardest problems. The lead's local drafts
  and scripts are in the gitignored `.dev/lead/`.
- **Settled; do not propose again:** bounds on `[limits]` and the ladder; merging `[workers] env` over the defaults;
  typed `SubmitRequest` and `JobResults`; `Store.wait_for_job`; `tomlkit` for the allowlist edit (it broke one-line
  arrays and CRLF files).
- A change to the `SignalStats` rule (the fade exclusion in `speech_frames`, the DC-10 speech rule) needs a new
  `QA_PROFILE`; any later byte-affecting change to delivery bumps `POST_RULES` (`narration.post/1`).
- `allow_sha256` is read once at start by both the front-end and the daemon. After `narration-admin voices allow`,
  stop the daemon (`narration-admin daemon stop`), then reconnect `narration-mcp`.
- Text rulings: tab, LF and CR are whitespace, and the canonical form collapses them; every other C0 and C1
  control character is refused (`TEXT_REFUSED`).

## How to resume

1. Read this file, then plan.md §4 (status) and §9 (log), and `docs/design-log.md` when a design change is on the table.
2. Read the `status/WPnn.md` of each open package (WP44, WP46), and check each active worktree for uncommitted files.
3. Record any new answer from the owner where it belongs (plan.md, the design log, this file), then continue with
   the next steps above.
4. Rewrite this file to the present at the end of every session and every wave: where the work stands, what is
   open, and what is next. Do not add a record of what changed.
