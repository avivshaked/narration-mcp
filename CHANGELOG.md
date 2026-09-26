# Changelog

All notable changes to this project are documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/), and this project adheres
to [Semantic Versioning](https://semver.org/spec/v2.0.0.html). Until the first `0.1.0` release, changes
are tracked here but no version is tagged; nothing described below is installable yet.

## [Unreleased]

### Added

- Repository bootstrap: the server `pyproject.toml`, worker project skeletons, `pytest`/`ruff`/
  `basedpyright` configuration, and the dev tools (`tools/gpu_lock.py`, `tools/check_tracked.py`, the git
  hooks, the worktree helper).
- Project licensing: `LICENSE` (PolyForm Noncommercial 1.0.0), `LICENSE-DOCS` (CC BY-NC 4.0),
  `COMMERCIAL.md`, `THIRD_PARTY_NOTICES`.
- `docs/design.md`: the design document, revision 5.1, as the source of truth for the implementation.
- `plan.md` and `HANDOFF.md`: the implementation plan and the current-state handoff.
- `CONTRIBUTING.md`: the contribution grant, environment setup with uv, the test tiers and markers, how
  to run GPU tests under the GPU lock, the required checks, and the dependency-licence rule.
- `CODE_OF_CONDUCT.md`: Contributor Covenant 2.1.
- `SECURITY.md`: the service's threat model in plain terms, and how to report a vulnerability.
- `.github/ISSUE_TEMPLATE/`: bug report, feature request and commercial-licence request templates, plus
  issue template configuration.
- `.github/pull_request_template.md`: a pre-submission checklist for pull requests.
- `narration.example.toml`: the shipped configuration file, with every key and default of the design's
  configuration section and no personal or machine-specific values.
- `material/`: the service's own text material, in draft until the owner has listened to sample renders
  (gate H1): the calibration corpus and length-ladder paragraphs (`narration-en.v1`), the alignment
  benchmark (`alignment-en.v1`), the canary's description, text and seeds (`canary.v1`), the Phase 4 demo
  script (`demo-en.v1`), and the text and QA fixtures (`text-v1`, `qa-faults-v1`), each set with a
  manifest of its files' sha256 hashes.
- `narration.platform`: every OS-specific mechanism behind one interface, implemented for Windows (the
  daemon's singleton, detached start, kill-on-close worker groups, below-normal priority, the free-disk
  check, and the path rules: caller paths must be absolute local regular files, network, device and
  reserved-name paths are refused, and store writes refuse links and junctions). On any other OS each
  call reports that v1 runs on Windows only.
- `narration.text`: the text pipeline (text checks `text-1.1.0`). It speaks the text as sent, changing
  only whitespace and Unicode form. It refuses markup and control characters (`TEXT_REFUSED`, listing
  every offender) and applies pronunciation hints per cue, never across a cue (`TERM_SPLIT_ACROSS_CUES`).
  It warns about digits, symbols, unit-like tokens and lone letters (`WRITTEN_FORM_TOKEN`), turns exact
  spans into word ranges, and adds `SEGMENT_TOO_LONG` as a warning only.
- `narration.lint`: the positive-only check of voice descriptions. It warns about negated qualities
  ("not rough") and suggests a positive rephrasing; it never refuses.
