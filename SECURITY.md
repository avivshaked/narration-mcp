# Security policy

**2026-09-28**  ·  the threat model this service is built to, what is in and out of scope, and how to
report a vulnerability

## What this service is

`narration-mcp` is a **local MCP server**. It talks to its client over stdio only; nothing ever listens
on a network socket, and no version of it is meant to. It runs on the same machine as the agent that
calls it, with the same user's permissions. There is no login, no multi-tenant boundary, and no remote
attacker model to speak of — the person running it is the operator, and the process the operator's own
agent talks to is trusted to the same degree the operator trusts that agent.

That said, the service still reads files, writes files, runs models and executes as a subprocess, so it
is worth being precise about what it will and will not do.

## The threat model

- **It reads any absolute local path a request names.** A caller can ask the service to read a voice
  clip or an audio file to profile from any absolute path on a local drive; there is no allowlist of
  folders. In effect: **any agent that can call this service can have it read any file its operator's own
  account can read.** For a WAV it gets back measurements and pictures, and the file's sha256 when the one
  sent is wrong. For any other file it learns only whether the file exists, is a regular file, and is a WAV
  of at most 20 MB, never its content or its hash. Relative paths,
  network shares and Windows device paths are refused, because opening a network path can prompt the
  operating system for the operator's own credentials. A file is checked against its declared sha256
  hash, and copied into the service's own store before any model sees it, so a file changed after the
  check is never the one that gets used.
- **It clones only synthetic voices.** A clip is used as a voice to clone from only if the service
  itself designed it (it is on the service's own provenance list) or the operator has explicitly
  allowlisted its sha256 hash. A recording of a real person, found on disk by any means, is never
  cloned — this rule does not depend on who is asking or where the file lives. Allowing a clip
  (`narration-admin voices allow`) is an operator command that asks a person to confirm the clip is
  synthetic; it is never an MCP tool. An agent that can run shell commands as the operator can run it
  too, so what such an agent may run is the operator's decision.
- **It writes only under its own store.** Every write the MCP server and its daemon make goes to a path
  the service itself builds from a validated id, always under the store root it was configured with.
  Reserved Windows device names and filesystem reparse points are refused, and every write lands at a
  temporary name before being renamed into place, so a crash never leaves a half-written file where a
  finished one is expected. The operator's own commands write where the operator points them:
  `narration-admin install` writes the models folder and the worker environments, `voices allow` edits
  the configuration file, and `render --out` copies a take to the file named.
- **The store, the models and the configuration are trusted.** The provenance list lives in the store,
  and the allowlist in the configuration file; the models and the worker environments are code and data
  the workers load. Anyone who can write them can allow a clip for cloning or run code in a worker. Keep
  the store folder, the models folder, the worker projects and the configuration file writable by the
  operator's account only, which matters most on a machine shared with other accounts.
- **Rendering is offline.** The models it runs are loaded from local, pinned snapshots. Model downloads
  happen only through the operator's own install step, with pinned revisions and verified file hashes —
  never as a side effect of answering a request.
- **No telemetry.** Nothing about a request, a voice, a transcript or a result leaves the machine. The
  only network activity the service ever performs is the operator's install step: the model download
  above, and `uv sync` fetching the worker environments' packages from their package indexes.
- **Text and file contents are data, never instructions.** A transcript, a voice description or a note
  that comes back from a model is something the service reports, not something it acts on.
- **Untrusted input is treated as such.** Control characters and the models' chat-template markup
  (`<|`, `|>`, and by default `[` and `]` in spoken text) are rejected outright, and every argument is
  validated before it reaches any file operation or model call. Some of the speech model's other control
  tokens (such as `<tts_pad>`) are not refused yet; one in a caller's text can affect only that caller's
  own take (see [docs/security-review.md](docs/security-review.md), S3).
- **Windows only, for now.** On any other operating system every path check refuses, so the service does
  not read or write a caller's files there.

## Out of scope

- **The security of the machine itself.** The service assumes the operator's account, filesystem
  permissions and installed software are trustworthy. It does not sandbox itself against a hostile local
  user or a compromised operating system.
- **The security of the calling agent or its host.** If the agent that talks to this service is
  compromised or malicious, it can ask the service to read any file the operator's account can read, or
  clone any allowlisted voice. That capability is a deliberate design choice (see above), not a bug to
  report.
- **A network-facing deployment.** This service is not designed, tested or supported as anything other
  than a local, stdio-only process. Running it as a network-reachable service is unsupported and its own
  risk.
- **Denial of service against a shared GPU.** Before it loads a model, the service checks that enough
  GPU memory is free and waits rather than competing for it, but it does not defend against another
  process on the machine behaving badly.

## Reporting a vulnerability

Please use **GitHub's private vulnerability reporting** for this repository (the "Security" tab, "Report
a vulnerability") to report a security issue. That report goes to the maintainer privately, rather than
appearing in a public issue.

Please do not open a public issue for a vulnerability report until the maintainer has had a chance to
assess and, where appropriate, fix it.
