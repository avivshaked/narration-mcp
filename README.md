# narration-mcp

A local [MCP](https://modelcontextprotocol.io) server that gives an AI agent narration in **one fixed
voice**, and the time of every line within that audio.

> **Status: in development. It cannot be installed and used yet.** Many of the parts are built and tested
> on their own, but not yet joined into a working service:
> - the store and its keys;
> - the text rules;
> - the quality checks;
> - the MCP front-end;
> - the worker protocol;
> - the Qwen3-TTS worker.
>
> The cue alignment is merged. The background daemon and the job engine are in review, and the QA worker is being built. This README describes what is
> being built, and it will say plainly when a first version can be installed. [plan.md](plan.md) §4 has
> the current status of every part.

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

## Requirements (planned)

- An NVIDIA GPU with CUDA. The voice models need about 6 GB of VRAM while rendering: about 5.8 GB
  was measured for a 30-second render ([spike h](spikes/h-i-qwen-load/README.md)). The checking
  models are measured next, and the total will be stated here.
- Python 3.12 and [uv](https://docs.astral.sh/uv/).
- About 15 GB of disk for the models, plus a working store.
- Windows 11 is the first platform. The tests that need no GPU also run on Linux in CI. Supported
  platforms will be stated before the first release.

## Models

The service downloads these at install time, at pinned revisions, and records their licences with
every result. None of them is distributed with this repository.

| Model | Role | Licence |
|---|---|---|
| Qwen3-TTS-12Hz-1.7B VoiceDesign and Base | designing voices; speaking by cloning a clip | Apache-2.0 |
| `facebook/wav2vec2-large-960h-lv60-self` | aligning each cue in time | Apache-2.0 |
| `openai/whisper-large-v3` | checking what was said | per its model card, checked at install |
| `microsoft/wavlm-base-plus-sv` | checking who said it | per its model card, checked at install |

## Use it from Claude Code

The server is not released yet; this is how a development checkout is wired up. `<service_root>` is the
folder of your clone.

1. Copy `narration.example.toml` to `<service_root>/narration.toml`, and set `store_root` and
   `models_root` under `[server]`. Create the environment once with `uv sync --frozen` in
   `<service_root>`.
2. Add the server to your project's `.mcp.json`. It runs the server with the environment's own Python
   interpreter, `<venv_python>`: `<service_root>/.venv/Scripts/python.exe` on Windows, or
   `<service_root>/.venv/bin/python` elsewhere.

   ```json
   {
     "mcpServers": {
       "narration": {
         "type": "stdio",
         "command": "<venv_python>",
         "args": ["-m", "narration.mcp", "--config", "<service_root>/narration.toml"]
       }
     }
   }
   ```

   Start it as `python -m narration.mcp`, not through the `narration-mcp` launcher executable that the
   environment also installs (and that `uv run … narration-mcp` starts): some antivirus programs sandbox
   a new, unsigned launcher. Without `--config`, the server reads the file `NARRATION_CONFIG` names, else
   `<service_root>/narration.toml`. If it finds neither, it exits and says what to do.
3. The first job starts the background daemon that does the work. If a call answers
   `DAEMON_UNAVAILABLE` because the client's session cannot start it, start the daemon yourself from a
   terminal first:

   ```sh
   <venv_python> -m narration.admin --config <service_root>/narration.toml daemon start
   ```

   It detaches and keeps running after the terminal closes. If Windows refuses to detach it from the
   terminal, the command says so and offers `--foreground`, which runs the daemon in that terminal until
   it exits; only then must the terminal stay open.

   The server clones only voices this service designed, or clips whose sha256 is in `[voices]
   allow_sha256`. The same command with `voices allow <clip.wav>` adds a clip designed elsewhere to that
   list, after you confirm that it is synthetic. The daemon and the server read `narration.toml` only
   when they start. After a change, stop the daemon first (the same command with `daemon stop`; the next
   job, or `daemon start`, starts it again), then reconnect the client to the server (`/mcp` in Claude
   Code).
4. In a session, call `measure_voice` on your voice clip once. Then call `submit_job`, poll `get_job`,
   and read the takes with `get_results`. The server's instructions tell the calling agent how to use the
   tools well: among other things, to send every invented name as a pronunciation hint (the term alone
   is enough), to keep a job to a scene, and to read results with `include_words` false.

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
- **Configuration:** [narration.example.toml](narration.example.toml) shows every configuration key and
  its default.

## Licence

This project is **source-available and non-commercial**:

- **Code**, and the service's own texts under `material/`: the
  [PolyForm Noncommercial License 1.0.0](LICENSE).
- **Written work** (this README and the other prose): [CC BY-NC 4.0](LICENSE-DOCS).

You may run it, change it and share it for any non-commercial purpose. Commercial use, including use
inside a company, needs a licence from the author. [COMMERCIAL.md](COMMERCIAL.md) says what counts as
commercial and how to ask. The models have licences of their own, which apply alongside these.
