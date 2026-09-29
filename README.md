# narration-mcp

A local [MCP](https://modelcontextprotocol.io) server that gives an AI agent narration in **one fixed
voice**, and the time of every line within that audio.

> **Status: alpha (0.1.0a1), working end to end on Windows with an NVIDIA GPU, and looking for testers**
> ([Help test it](#help-test-it)). An agent can design or allow a voice, measure it once, and narrate
> paragraphs of cues, getting back QA'd, cue-aligned takes. Every published tool runs; the newest,
> `audition_pronunciation`, has so far been tested only against the service's fake workers, not on a GPU.
> Schemas may still change between alpha releases ([CHANGELOG.md](CHANGELOG.md) tracks every one), and the
> accuracy of the cue times has not been measured yet. [plan.md](plan.md) §4 has the status of every part
> of the build.

## What it does

An agent works in three steps:

1. **Design a voice.** The agent describes a voice in words. The service renders a few candidate
   clips with [Qwen3-TTS](https://github.com/QwenLM/Qwen3-TTS) VoiceDesign, each with its exact
   transcript, a fingerprint, and a profile of measurements and pictures an agent can read (pitch,
   pace, brightness, breathiness). A person listens and chooses one. The caller keeps the chosen clip.
2. **Measure the voice, once.** The service renders its own calibration texts in that voice. From them
   it learns what "the same voice" sounds like, how fast the voice speaks, and the longest paragraph it
   reads reliably.
3. **Generate narration.** The caller sends paragraphs as cues (a sentence or caption each), already in
   the words to be spoken, together with the voice clip. Every take comes back with:
   - the audio file, its exact length in samples and its sha256;
   - the start and end time of every cue within the audio, found by forced alignment of the known text;
   - quality checks on what was said, who said it and how fast: speech recognition against the text,
     speaker similarity to the voice, pace against the voice's own measured pace, and exact checks on
     spans the caller marks, such as numbers;
   - an echo of the text as it was received and as it was given to the engine.

The service keeps none of the caller's state. The same request always gives the same answer, from a
cache when the work has already been done.

## Principles

- **The service knows nothing of how it is used.** What the text says, how its numbers and names are
  read, how long a paragraph may be, and which take to keep are the caller's decisions. The service
  narrates, measures and reports.
- **Local only.** stdio transport, no network listener, models loaded offline, and nothing leaves the
  machine.
- **Synthetic voices only.** The service clones a voice only if it designed the clip itself, or the
  operator has allowlisted the clip's fingerprint. A recording of a real person is never cloned.
- **Nothing hidden.** No silent trimming, stretching or truncation. Every flag is reported, and the
  caller decides what to accept.

## Help test it

So far this has run on one machine only, the author's. The most useful thing you can do is install it on
yours, following [Install](#install) exactly, and report every place where it breaks or where the
instructions left you guessing. A report that says "step 3 failed with this error" is as valuable as a
finished narration.

**You need** a Windows 10 or 11 PC with an NVIDIA GPU that has about 12 GB of VRAM free, about 25 GB of
free disk, and Python 3.12 with uv (see [Requirements](#requirements)). **It takes** a download of about
13 GB, a few minutes of setup, and then 20 to 50 minutes of GPU time to measure your first voice, which
runs by itself.

**How far to go:**
1. Install, and run `narration-admin doctor` until every check passes.
2. From Claude Code (or another MCP client), design a voice, measure it, and narrate a paragraph of your
   own ([First run](#first-run)).
3. Listen. Tell us whether the takes QA passed sound right, and whether the ones it failed deserved it.

**Where to report:** open a [tester report](https://github.com/avivshaked/narration-mcp/issues/new?template=tester_report.yml)
on GitHub, one per problem or one for your whole run, whichever is easier. Include the version, the step,
the exact command and what it printed, and `doctor`'s output. Before you paste anything, replace your own
paths and user name with something generic. Never attach a recording of a real person's voice. A
security problem goes through [SECURITY.md](SECURITY.md) instead, never a public issue.

Testing is free for personal, non-commercial use on your own machine (see [Licence](#licence)); testing
for a company counts as commercial use.

## Requirements

- **Windows 11** (or 10). It is the only platform v1 runs on; `narration-admin doctor` says so plainly on
  any other OS. The pure parts of the code, and their tests, also run on Linux, in CI, but the daemon and
  the workers do not run there yet.
- **An NVIDIA GPU with CUDA, and about 12 GB of free VRAM.** Only one model group is loaded at a time
  (`[gpu] one_group_at_a_time`): rendering needs about 6 GB ([spike h](spikes/h-i-qwen-load/README.md)),
  and checking (speech recognition and speaker similarity together) needs about 11.5 GB
  ([spike h, QA half](spikes/h-i-qa-load/README.md)), so the larger of the two, plus a margin, is what a
  render or a measurement needs free.
- **Python 3.12** and [uv](https://docs.astral.sh/uv/) (0.10 or later). uv is also how the worker
  environments are created; nothing else needs installing by hand.
- **About 25 GB of disk:** about 13 GB for the pinned models, and 5 to 10 GB for the worker environments
  (PyTorch with CUDA; the lower figure when uv's cache is on the same drive), plus room for the store (the
  cache of rendered work; see [Where things live](#where-things-live) below).

## Not yet supported

- `narration-admin bench alignment` (measuring the cue aligner against a hand-marked benchmark) is not
  built yet.
- Only Windows runs the service; see Requirements above.

## Models

The service downloads these at install time, at pinned revisions, and records their licences with
every result. None of them is distributed with this repository.

| Model | Role | Licence |
|---|---|---|
| Qwen3-TTS-12Hz-1.7B VoiceDesign and Base | designing voices; speaking by cloning a clip | Apache-2.0 |
| `facebook/wav2vec2-large-960h-lv60-self` | aligning each cue in time | Apache-2.0 |
| `openai/whisper-large-v3` | checking what was said | per its model card, checked at install |
| `microsoft/wavlm-base-plus-sv` | checking who said it | per its model card, checked at install |

## Install

`<service_root>` is the folder of your clone; `<venv_python>` is the environment's own interpreter,
`<service_root>\.venv\Scripts\python.exe`.

1. **Clone the latest release, and create the server's environment.**

   ```sh
   git clone --branch v0.1.0a1 https://github.com/avivshaked/narration-mcp.git
   cd narration-mcp
   uv sync --locked
   ```

   `--branch v0.1.0a1` checks out the release, not `main`, which changes often; the
   [releases page](https://github.com/avivshaked/narration-mcp/releases) names the latest. `uv sync
   --locked` builds `.venv` from the committed `uv.lock` exactly, so it fails loudly rather than silently
   resolving different versions if your uv or index ever disagreed with the lock.
2. **Configure.** Copy `narration.example.toml` to `narration.toml` in `<service_root>`. It works as it
   is: the store and the models go into the folders `store` and `models` next to it. To put them
   elsewhere, such as a larger drive, set `store_root` and `models_root` under `[server]` to absolute
   paths; see [Configuration](#configuration) below for what else is in it. `narration.toml` and
   `narration.local.toml` are gitignored, so your paths never end up in version control.
3. **Install the models and the two worker environments.** This downloads the pinned model revisions
   (verifying every file's hash against what Hugging Face publishes for it) into `[server] models_root`,
   and syncs `workers/qwen3tts` and `workers/qa` from their own `uv.lock` files — the two worker projects
   have conflicting dependencies (different `transformers` versions) so they never share a venv with each
   other or with the server.

   ```sh
   <venv_python> -m narration.admin --config <service_root>\narration.toml install
   ```

   `--dry-run` lists what it would fetch without changing anything; `--from-cache <hub cache>` copies
   from a local Hugging Face cache instead of downloading, still hash-checked; `--models-only` /
   `--workers-only` run half of it. If your network intercepts TLS ("invalid peer certificate"), point
   `SSL_CERT_FILE` (or `REQUESTS_CA_BUNDLE`) at its root certificate, or set `UV_NATIVE_TLS=1` for uv —
   never turn certificate verification off.
4. **Pin the engine, and check everything.** `engine pin` records the engine profiles this installation
   and machine match, and designs the service's canary clip on this machine (so later renders can be
   checked against it, design section 10.1). It loads Qwen on the GPU, so run it only while no daemon is
   using the store.

   ```sh
   <venv_python> -m narration.admin --config <service_root>\narration.toml engine pin
   <venv_python> -m narration.admin --config <service_root>\narration.toml doctor
   ```

   `doctor` checks the platform, the configuration, the store, the models (hashed again), the worker
   venvs, the GPU and the engine pin, and says exactly what to fix for anything that fails; run it again
   after fixing something. See [docs/operator-guide.md](docs/operator-guide.md) for what each check means
   and what to do about it.

## Configuration

Everything is one TOML file; `narration.example.toml` documents every key with its default. The file the
server actually reads is found by `--config`, else the environment variable `NARRATION_CONFIG`, else
`narration.toml` in the service's own folder. A few of its sections:

| Section | What it sets |
|---|---|
| `[server]` | `store_root` (the cache and job database) and `models_root` (pinned model snapshots) |
| `[voices]` | `allow_sha256`: clips designed elsewhere that are allowed to be cloned (ships empty; see `narration-admin voices allow` in the operator guide) |
| `[retention]` | how long jobs, designs, takes and measurements stay in the store before `gc` may remove them |
| `[daemon]` | `autostart`, and how long an idle daemon keeps a model loaded or stays running at all |
| `[workers]` | the CPU thread cap and priority every worker runs at, and the offline environment variables they get |
| `[gpu]` | the CUDA device, whether only one model group loads at a time, and how long a job waits for free VRAM |
| `[limits]` | request-size limits: segments per job, cues per segment, characters, hints, the submission rate, the queue depth |
| `[defaults]` | the default number of takes and automatic retakes when a request does not say |
| `[text]` | the text-check ruleset version and the characters refused outright (markup) |
| `[delivery]` | the delivered file's sample rate, bit depth, target loudness and true-peak ceiling, and the trim and fade applied |
| `[voice_design]` | the words a designed voice's candidates speak by default |
| `[measurement]` | the calibration corpus, how many seeds, and the length ladder a voice is measured against |
| `[alignment]` | the cue-aligner model and device, and its confidence thresholds |
| `[qa]` | which QA profile (thresholds and rules) this build scores with; informational, not a switch |
| `[engines.qwen3_base]` / `[engines.qwen3_design]` | settings that change rendered audio, so they are pinned into the engine profile explicitly rather than left to library defaults |
| `[workers.qwen3]` / `[workers.qa]` | each worker's uv project folder |

`narration-mcp` and the daemon read `narration.toml` only when they start; after any change, stop the
daemon (`narration-admin daemon stop`) and reconnect your MCP client, except `[measurement]`, which must
never be edited while the store holds measurements made under the old values.

## Wire it into Claude Code

Add the server to your project's `.mcp.json`, run through the environment's own interpreter:

```json
{
  "mcpServers": {
    "narration": {
      "type": "stdio",
      "command": "<venv_python>",
      "args": ["-m", "narration.mcp", "--config", "<service_root>\\narration.toml"]
    }
  }
}
```

Start it as `python -m narration.mcp`, not through the `narration-mcp` launcher executable the
environment also installs (and that `uv run … narration-mcp` would start): some antivirus programs
sandbox a new, unsigned launcher. Without `--config`, the server reads the file `NARRATION_CONFIG` names,
else `narration.toml` in the service's own folder; with neither, it exits and says what to do.

The first job starts the background daemon that does the work, detached so it outlives the client. If a
call answers `DAEMON_UNAVAILABLE`, see [docs/operator-guide.md](docs/operator-guide.md)'s daemon section
(some hosts' terminals cannot let a process detach; there is a documented way around it).

## First run

1. **Get a voice.** Either have the agent call `design_voice` with a positive-only description ("warm,
   unhurried, low-pitched"; never "not shrill") — it renders a few candidate clips with VoiceDesign, you
   listen and choose one, and keep its clip, sha256 and exact transcript; every candidate is already on
   the service's provenance list, so it needs no further allowlisting. Or, for a clip designed some other
   way, confirm it is synthetic (never a recording of a real person) and add it with
   `narration-admin voices allow <clip.wav>`.
2. **Measure it once**, per voice and per engine: `measure_voice` with the voice (path, sha256,
   transcript). It is a heavy GPU job, 20 to 50 minutes; check `get_server_status`'s `admission` first. It
   learns the voice's pace curve, its reliable paragraph length and its speaker-similarity baseline. If
   the same clip and transcript were measured before under this engine, it answers at once from the
   cache.
3. **Check the text**, then **narrate**: `check_text` on your cues (with any pronunciation hints) catches
   digits, symbols and unusual names before you render; hint every invented or unusual name so QA scores
   it as one word instead of misheard words. Then `submit_job` with the voice, the segments and your
   hints; poll `get_job` with `wait_s` until it ends; read `get_results` (`include_words: false` unless
   you need word-level times) and use each segment's `suggested_take_id`.

You can also try the pipeline from a terminal, with no MCP client, once a voice is measured:

```sh
<venv_python> -m narration.admin --config <service_root>\narration.toml render \
  --voice <clip.wav> --transcript "<the clip's exact words>" \
  --text "Good bread asks for patience." --out take.wav
```

## Where things live

- **The store** (`[server] store_root`) holds the job database (`narration.sqlite`), the cache of
  renders, deliveries and analyses, voice measurements, designs, the engine profiles and canary, and the
  alignment benchmark — everything is either the service's own or a cache of work already done, never a
  caller's script or choices (see [SECURITY.md](SECURITY.md)).
- **Logs**: `<store_root>\logs\narration-mcp.log` (one per client session) and
  `<store_root>\logs\daemon.log` (the daemon's, rotated).
- **The daemon's live status**: `<store_root>\run\daemon.json`, also readable with
  `narration-admin daemon status`.
- **An audit of failed or retaken takes**: `narration-admin failures` (add `--export <dir>` for the WAVs
  and a CSV index). `narration-admin gc` (a dry run by default) shows what retention would remove next.

See [docs/operator-guide.md](docs/operator-guide.md) for the daemon, `doctor`, the engine pins, the
voice allowlist, garbage collection and every error code an operator may see, and
[docs/tools.md](docs/tools.md) for every MCP tool's full input and output schema.

## Project status, contributing and security

The service is specified in [docs/design.md](docs/design.md), and the work is organised in
[plan.md](plan.md). Decisions with their evidence are recorded in [docs/decisions/](docs/decisions/).
Issues are welcome, including a "Commercial licence" issue if you want to use this project commercially.

- **Contributing:** see [CONTRIBUTING.md](CONTRIBUTING.md) for the environment setup, the test tiers,
  and the checks a change needs to pass. Everyone participating is expected to follow the
  [Code of Conduct](CODE_OF_CONDUCT.md).
- **Security:** see [SECURITY.md](SECURITY.md) for what this service does and does not protect against,
  and how to report a vulnerability.
- **Changes:** notable changes are tracked in [CHANGELOG.md](CHANGELOG.md).

## Licence

This project is **source-available and non-commercial**:

- **Code**, and the service's own texts under `material/`: the
  [PolyForm Noncommercial License 1.0.0](LICENSE).
- **Written work** (this README and the other prose): [CC BY-NC 4.0](LICENSE-DOCS).

You may run it, change it and share it for any non-commercial purpose. Commercial use, including use
inside a company, needs a licence from the author. [COMMERCIAL.md](COMMERCIAL.md) says what counts as
commercial and how to ask. The models have licences of their own, which apply alongside these.
