# HANDOFF

*Updated 2026-09-26. Read this first, then [plan.md](plan.md), [AGENTS.md](AGENTS.md), and the gitignored
`AGENTS.local.md` (this machine's paths and facts).*

## Where things stand

- **Stage: H0 answered; WP00 (bootstrap) is next.** No code is written yet.
- The repository is initialised locally (`main`), with `origin` = `git@github.com:avivshaked/narration-mcp.git`,
  a **public**, empty repository. **Nothing has been pushed.** The first push waits for the owner's
  read of `COMMERCIAL.md` (below).
- Tracked so far:
  - `plan.md`: the plan, the decisions (§1.4), design changes (§1.5), and the status table (§4);
  - `AGENTS.md` + `CLAUDE.md`: the agent rules; `CLAUDE.md` also imports `AGENTS.local.md`;
  - this file;
  - `README.md`;
  - `docs/design.md`: design revision 5.1, the source of truth;
  - licences: `LICENSE` (PolyForm Noncommercial 1.0.0), `LICENSE-DOCS` (CC BY-NC 4.0), `COMMERCIAL.md`,
    `THIRD_PARTY_NOTICES`;
  - `.gitignore`, `.gitattributes`.
- Not tracked: `AGENTS.local.md`, which holds every machine-specific path and fact.

## Decided (details in plan.md §1.4 and §1.5)

- **A public project** with a public project's rigour; **no local paths in tracked files**.
- **Licences:** as the evolution simulator's; the project is source-available and non-commercial.
- **Windows first.** Only the cheap preparation for other platforms now.
- **Distribution:** a git clone + uv for v1. **Merges:** pull requests with CI.
- **Models:** copy the cached snapshots into `<repo>\.dev\models\`, and download the missing two there.
- **parselmouth (GPL) replaced** (DC-1, approved): f0 from `librosa.pyin`, HNR by Boersma's method,
  CPPS instead of jitter and shimmer.
- **GPU etiquette:** bounded runs; long ones with the owner's OK.
- **Human gates:** the owner listens (H1) and marks (H2), possibly in DaVinci Resolve through its MCP.
  Resolve markers snap to frames, so its timeline must run at ≥ 60 fps.

## Waiting on the owner

1. **Read `COMMERCIAL.md` (and `README.md`) before the first push.** `COMMERCIAL.md` is adapted from the
   evolution simulator's and speaks in the owner's name. The copyright holder is written as "Aviv Shaked".
2. **DC-2, the backoff contract** (plan.md §1.5): `retry_after_s` on retryable errors; `QUEUE_FULL` and
   `RATE_LIMITED` split out of `LIMIT_EXCEEDED`; `poll_after_s`; an `admission` block in
   `get_server_status`. Needed before WP01 freezes the contracts, or it enters later as a contract
   change.
3. **DC-3, the canary designed on the installing machine** rather than shipped as audio.

Coming later: DC-4 (`max_new_tokens`, from WP20's evidence); the GitHub description, which still says
voices are "locked" (it is outward-facing, so it waits for the owner; WP44).

## Next steps

1. **WP00 bootstrap (lead):**
   - the layout;
   - the server `pyproject.toml`, every dependency licence-checked;
   - the worker skeletons with the P8 pins;
   - pytest, ruff and pyright configuration;
   - tools: `gpu_lock.py`, `check_tracked.py`, the githooks, the worktree helper;
   - the dev models copied and downloaded into `.dev\models\`.
2. **WP01 contracts (lead)**, while agents take WP02 (public scaffolding) and WP03 (CI) in worktrees.
3. **Wave 1 fan-out**, up to 6 agents at once plus the GPU lane, in plan.md §7's order.

## Things a new session should know

- The bakeoff that preceded this project is private and read-only. Its reusable code, golden numbers
  and models are mapped in plan.md §1.2. The installed-code findings (the effective `max_new_tokens`
  8192, the normaliser's spelling map, the proportional reference cut) are in §1.3.
- The consumer's requirements (R1–R14) are mapped in design §21. The service must never depend on that
  caller (the owner's principle, at the top of the design).
- The Claude Code harness reports "file changed on disk" when a file is touched through `d:\` and then
  `D:\`. That is the path's case, not an outside edit.

## How to resume

1. Read this file, then plan.md §4 (status), §1.5 (design changes) and §9 (log).
2. Record any new answers from the owner in plan.md and here, then continue with the next step above.
3. Keep this file current at the end of every session and every wave: the stage, what changed, what is
   open, what is next.
