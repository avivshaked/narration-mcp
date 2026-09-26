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
