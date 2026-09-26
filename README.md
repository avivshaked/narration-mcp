# narration-mcp

A local [MCP](https://modelcontextprotocol.io) server that gives an AI agent narration in **one fixed
voice**, and the time of every line within that audio.

> **Status: design complete, implementation starting.** Nothing here runs yet. This README describes
> what is being built; it will say plainly when a first version can be installed.

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

- An NVIDIA GPU with CUDA. How much VRAM is needed will be measured and stated here.
- Python 3.12 and [uv](https://docs.astral.sh/uv/).
- About 15 GB of disk for the models, plus a working store.
- Platform support will be stated before the first release. Development happens on Windows 11.

## Models

The service downloads these at install time, at pinned revisions, and records their licences with
every result. None of them is distributed with this repository.

| Model | Role | Licence |
|---|---|---|
| Qwen3-TTS-12Hz-1.7B VoiceDesign and Base | designing voices; speaking by cloning a clip | Apache-2.0 |
| `facebook/wav2vec2-large-960h-lv60-self` | aligning each cue in time | Apache-2.0 |
| `openai/whisper-large-v3` | checking what was said | per its model card, checked at install |
| `microsoft/wavlm-base-plus-sv` | checking who said it | per its model card, checked at install |

## Project status, contributing and security

The work is organised in [plan.md](plan.md). Issues are welcome, including a "Commercial licence" issue
if you want to use this project commercially.

- **Contributing:** see [CONTRIBUTING.md](CONTRIBUTING.md) for the environment setup, the test tiers,
  and the checks a change needs to pass. Everyone participating is expected to follow the
  [Code of Conduct](CODE_OF_CONDUCT.md).
- **Security:** see [SECURITY.md](SECURITY.md) for what this service does and does not protect against,
  and how to report a vulnerability.
- **Changes:** notable changes are tracked in [CHANGELOG.md](CHANGELOG.md).
- **Configuration:** [narration.example.toml](narration.example.toml) shows every configuration key and
  its default, once there is code to read it.

## Licence

This project is **source-available and non-commercial**:

- **Code**, and the service's own texts under `material/`: the
  [PolyForm Noncommercial License 1.0.0](LICENSE).
- **Written work** (this README and the other prose): [CC BY-NC 4.0](LICENSE-DOCS).

You may run it, change it and share it for any non-commercial purpose. Commercial use, including use
inside a company, needs a licence from the author. [COMMERCIAL.md](COMMERCIAL.md) says what counts as
commercial and how to ask. The models have licences of their own, which apply alongside these.
