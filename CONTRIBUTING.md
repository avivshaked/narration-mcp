# Contributing

**2026-09-26**  ·  what a contribution grants, how to set up, and the checks to run before sending one

Contributions are welcome: code, prose, tests, corrections to what is written here. This document is
for anyone sending a change, human or agent. Nothing in this project runs yet beyond a skeleton — see
[README.md](README.md) and [plan.md](plan.md) for where things stand.

## What you grant

By contributing, you license your contribution under the project's licences: the [PolyForm
Noncommercial License 1.0.0](LICENSE) for code and the service's own material under `material/`, and
[CC BY-NC 4.0](LICENSE-DOCS) for prose. You also grant the copyright holder the right to license your
contribution under other terms, commercial licences included. You confirm the contribution is yours to
give. The second grant lets the project be licensed as one work rather than a patchwork of pieces with
different owners — [COMMERCIAL.md](COMMERCIAL.md) explains why that matters. A contribution that arrives
without this grant cannot be merged. No signed agreement is needed; opening the pull request is the
grant.

## Setting up

The project uses [uv](https://docs.astral.sh/uv/) with Python 3.12, from an interpreter already
installed on your machine — uv is never asked to download one:

```
uv python pin 3.12          # or point uv at your own 3.12 interpreter
uv sync --python 3.12
```

Set `UV_PYTHON_DOWNLOADS=never` so a missing interpreter fails loudly instead of uv fetching one. If
your network intercepts TLS (a corporate proxy, for example), set `UV_NATIVE_TLS=1` so uv uses your
system's certificate store — never disable certificate verification to work around a TLS failure; if
`UV_NATIVE_TLS=1` does not fix it, stop and ask before going further.

The worker projects (`workers/qwen3tts/`, `workers/qa/`) are separate uv projects with their own
`pyproject.toml` and `uv.lock`; sync each from inside its own folder if you are working on a worker.
They pin heavier, GPU-facing dependencies and can never share a virtual environment with each other or
with the server project.

Enable the git hooks once per clone:

```
git config core.hooksPath tools/githooks
```

The pre-commit hook runs `tools/check_tracked.py`, which refuses local paths and binary data (audio,
weights, checkpoints, databases) in anything you stage. The commit-msg hook runs the same check on your
commit message. If a hook refuses your change, fix the content — do not bypass the hook.

## Test tiers

The default suite needs no GPU, no downloaded model and no private evidence. It is what CI runs, and it
must pass on a machine with none of those things:

```
uv run pytest
```

Everything else is opt-in, selected with pytest markers, and skips cleanly with a clear reason when its
resource is absent:

| Marker | Needs |
|---|---|
| `model` | a model loaded on the CPU (`NARRATION_MODELS_ROOT` set to a folder with the pinned snapshots) |
| `gpu` | an NVIDIA GPU, and the GPU lock (below) |
| `evidence` | the maintainers' private bake-off evidence (`NARRATION_BAKEOFF_ROOT`); most contributors will not have this and can skip these tests |
| `slow` | more than a minute to run |

Run one tier at a time, for example:

```
uv run pytest -m model
uv run pytest -m gpu
```

### Running GPU tests

The GPU is a shared resource on a development machine, and the project takes a lock before any worker
loads a model on it. If you have an NVIDIA GPU and want to run `gpu`-marked tests:

1. Check free VRAM first (`nvidia-smi`), and only proceed if it comfortably covers what the run needs.
2. Take the lock: `uv run python tools/gpu_lock.py acquire --holder <your-name-or-WP> --minutes <n>`.
3. Run the tests: `uv run pytest -m gpu`.
4. Release the lock when done, including on failure: `uv run python tools/gpu_lock.py release`.
5. Check the lock's state at any time with `uv run python tools/gpu_lock.py status`.

Keep GPU runs bounded (a lock held for at most 30 minutes at a time is the working assumption). Never
kill, throttle or inspect another process using the GPU — if the lock looks stuck, report it rather than
breaking it.

## Checks

All of these must be clean before a pull request is ready for review:

```
uv run ruff check
uv run ruff format --check
uv run basedpyright
uv run pytest
```

## Commit style

Commits follow [Conventional Commits](https://www.conventionalcommits.org/): `feat(text): apply hints
longest-first`, `fix(store): …`, `test(qa): …`, `docs: …`, `chore: …`. The scope is the area of the
codebase you touched. Keep commits small and focused; add a body when it helps a reviewer understand
why, not just what.

Update `CHANGELOG.md` (under `## [Unreleased]`) for any change a user of the package would notice.

## Dependency licences

Every dependency's licence is checked before it is added, because this project is source-available and
non-commercial (PolyForm Noncommercial 1.0.0) and needs to stay able to offer commercial licences
alongside that.

- **Permissive licences are fine**: MIT, BSD, Apache-2.0, ISC, PSF and similar.
- **GPL and AGPL are refused**, in any version, including as a test-only or dev dependency. So is
  anything marked non-commercial or "research only" — it could not be combined with, or commercially
  relicensed alongside, this code.
- **LGPL needs a maintainer's sign-off first.** Ask before adding one, and say in the request how it is
  used (a separately installed, replaceable package is easier to accept than one that is compiled in).

If you propose a new dependency, say what it is for and what its licence is, so a maintainer can check
it quickly.

## What never goes in a commit

- **No audio, model weights, checkpoints or databases.** `*.wav`, `*.flac`, `*.mp3`, `*.safetensors`,
  `*.bin`, `*.pt`, `*.pth`, `*.onnx`, `*.sqlite` and similar are gitignored; do not force-add them, even
  as a fixture. Deterministic test audio is generated by the test itself, not checked in.
- **No local paths.** No drive letters, no home directories, no user names, in code, tests, docs, commit
  messages or pull request text. Use a placeholder such as `<service_root>` or an environment variable
  instead. `tools/check_tracked.py` enforces this in the pre-commit hook and in CI, and it also scans
  commit messages.

If either check refuses your change and you believe it is a false positive, say so in the pull request
rather than working around the hook.
