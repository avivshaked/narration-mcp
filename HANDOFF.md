# HANDOFF

*Updated 2026-09-26. Read this first, then [plan.md](plan.md), [AGENTS.md](AGENTS.md), and the gitignored
`AGENTS.local.md` (this machine's paths and facts).*

## Where things stand

- **Stage: WP00 done (merged to local `main`, not pushed yet); WP01 (contracts, lead) in progress;
  WP02 and WP03 agents active.** Worktrees exist for WP01, WP02, WP03, WP13, WP17, WP18, WP19, WP20.
- **Incident, 2026-09-26 ~17:14 local:** the machine froze hard or lost power (Kernel-Power 41, no
  bugcheck, no dump, nothing logged in the minutes before). Our load at the time was light: two doc/CI
  agents, one had just started basedpyright; no model on the GPU. The machine had an earlier blue screen
  (bugcheck 0x3B) on 2026-09-22, before this project began. After the restart: `git fsck` clean, all 52
  model files (16.1 GB) re-hashed and matching, worker venvs import, WP02's staged files intact, WP03's
  work lost (nothing had been written). **GPU work (WP20) is on hold** until the owner says the machine is
  fine to load models on (a driver check or memory test was suggested).
- The repository is initialised locally (`main`), with `origin` = `git@github.com:avivshaked/narration-mcp.git`,
  a **public** repository. `main` is pushed.
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

- **Whether GPU work may start** after the 2026-09-26 crash (see "Where things stand"). `COMMERCIAL.md`, the README, DC-2 (backoff) and DC-3 (install-time canary) were
approved on 2026-09-26. WP01 applies DC-1 to DC-3 to `docs/design.md` as revision 5.2.

Coming later: DC-4 (`max_new_tokens`, from WP20's evidence); the GitHub description, which still says
voices are "locked" (it is outward-facing, so it waits for the owner; WP44); the gates H1 to H4 as the
work reaches them.

## Next steps

1. ~~WP00 bootstrap~~ done 2026-09-26 (plan.md §9).
2. **WP01 contracts (lead)**, in `worktrees\wp01-contracts`, while agents finish WP02 and WP03.
3. Push `main` and open PRs once WP03's CI exists (Q4).
4. **Wave 1 fan-out** after WP01 merges, in plan.md §7's order; the CPU-only WPs first, and fewer agents at
   once than §7 allows until the machine's stability is understood. WP20 waits for the owner.

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
