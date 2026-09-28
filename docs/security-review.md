# Security review (WP44)

Date: 2026-09-28. Scope: the path checks of `narration.platform` (design §17.2, §17.3) and every tool argument
that names a file; the synthetic-voices rule (§17.4, `narration-admin voices allow`), including the window
between hashing a clip and cloning it; the text refusals (`[text] refuse`, and `narration.design`'s description
check); network listeners; the daemon's control files under the store's `run/`; subprocess calls; and
[SECURITY.md](../SECURITY.md) compared with the code.

Method: the code was read, and the pure parts were exercised on Linux (the `winpaths` rules, the text pipeline,
the front-end's clip admission through the test platform). The Windows file-system half
(`narration.platform._windows`) was read but not run: this review ran on a Linux machine, and the Windows CI
job is what runs its tests. Evidence is labelled as the design labels it: **KNOW** (read or run here, with the
test named), **BELIEVE** (expected, not verified here).

The threat model is SECURITY.md's: a local, stdio-only service run by its operator. The caller is an agent the
operator trusts to the degree they trust that agent. A hostile local user, or another process that can write
the operator's files, is out of scope. Findings that need one are rated against that model and kept as
information, so the operator knows where the edge is.

## Summary

| id | severity | finding | state |
|---|---|---|---|
| S1 | Low | A wrong sha256 sent for any local file gave back that file's sha256 | **fixed**, with tests |
| S2 | Low | `narration-admin install` and `doctor` could run a `uv.exe` from the working folder (Windows) | **fixed**, with tests |
| S3 | Low | Model control tokens not shaped `<\|…\|>` (`<tts_pad>`, `<think>` …) pass the text and description checks | open: needs a design decision |
| S4 | Low (Medium on a shared machine) | Anyone who can write the store, the configuration, the models or the worker venvs can get round §17.4, or run code in a worker | SECURITY.md now says so; a `doctor` check is open |
| S5 | Info | The read check and the open are not atomic (time of check to time of use) | open, out of scope |
| S6 | Info | Only two kinds of reparse point are treated as links when a read path is resolved | open |
| S7 | Info | Case is compared with `str.lower`, not the file system's upcase table | no change |
| S8 | Info | Error codes tell whether any local path exists, is a file, and is a WAV | documented |
| S9 | Info | Another local user can hold the daemon's `Global\` mutex name first | open, out of scope |
| S10 | Info | `voices allow --yes` can be run by any agent that has a shell as the operator | documented |
| S11 | Info | `run/daemon.json` and `run/launch.json` are advice only; nothing acts on a pid from them | no change |
| S12 | — | SECURITY.md did not match the code in six places | **fixed** in SECURITY.md |

What holds (KNOW, from the code, and from the tests named):

- **The transport.** The front-end serves on stdio only (`src/narration/mcp/server.py:228-231`). The daemon and
  the front-end meet only through the store, and workers speak over pipes
  (`src/narration/workers/client.py:173-182`). No module of the service or its workers imports a socket or
  server module, and only `narration.admin.install` imports an HTTP client. A new test pins this:
  `tests/mcp/test_no_listener.py`.
- **Subprocesses.** Every call passes an argument list: no `shell=True`, no `os.system` or `os.popen`, in
  `src/`, `workers/*/src` or `tools/`. The daemon and workers start as `<python> -P -m …` with `PYTHONPATH`,
  `PYTHONHOME` and `PYTHONSTARTUP` removed (§17.9).
- **The synthetic-voices gate** runs before any file is read, on the sha256 the request sent: in the front-end
  (`src/narration/backend/service.py:507`, `:815`, `:1009`, through `clips.check_synthetic`), and again in the
  daemon before a clip is copied or cloned (`src/narration/jobs/engine.py:151`,
  `src/narration/measure/handler.py:229`). It compares with the provenance list (`store.py:999`) and
  `allow_sha256`, which the configuration requires to be 64 lower-case hex digits (`config.py:333-335`).
- **Hash to clone.** The bytes that are hashed are the bytes copied into the store: `clips._copy` and
  `voice._copy` hash each chunk as they write it, and the copy is renamed into
  `scratch/voices/<sha256>.wav` only when that hash is the one sent (`src/narration/backend/clips.py:101-114`,
  `src/narration/jobs/voice.py:98-121`). A worker is only ever given that copy. So a file changed after the
  check is never cloned. `voices allow` hashes the same bytes it checked (`src/narration/admin/voices.py:162-204`).
- **Store writes.** Every store path is built from components matching `_COMPONENT`
  (`src/narration/store/layout.py:80`, `:206-210`), then confined by `realpath` and the platform's reparse-point
  check (`layout.py:187-204`, `_windows.py:671-703`). A sha256 or an id from a request cannot name a folder.
- **Resources** (`narration://…`) are looked up by id with bound SQL parameters
  (`src/narration/backend/resources.py:36-95`); no URI names a file.
- **Text refusals hold across joins.** Cues are joined with a space, and a hint applies only to a whole term
  between spaces or punctuation, so `<|` and `|>` cannot be assembled from cues or respellings that each pass.
  Pinned by the new `tests/text/test_markup_joins_s9_1.py`.
- **Model files.** `narration-admin install` checks each file against the hash Hugging Face publishes for the
  pinned revision, and skips pickled weights where safetensors exist. The QA worker loads any `.bin` weights with
  `weights_only=True` (`workers/qa/src/narration_worker_qa/sv.py:127`, `asr.py:162`, `align.py:440`). No code
  passes `trust_remote_code`.

## Findings

### S1. A wrong sha256 gave back the sha256 of any local file (Low, fixed)

**What.** `profile_voice` reads any absolute local path (§17.3) and has no synthetic gate, by design. When the
sha256 sent was not the file's, the error `VOICE_FILE_MISMATCH` carried `details.actual`: the file's own
sha256. The WAV check came after the sha256 check, so this held for any file of up to 20 MB, audio or not. The
same held for `submit_job`, `measure_voice` and `audition_pronunciation` for a caller that knows one allowed
sha256 (any clip it designed). A sha256 is enough to confirm a guess of a small file's content (a short
token, a PIN), offline. SECURITY.md promised only measurements of audio files.

**Evidence** (before the fix). `src/narration/backend/clips.py`: `admit_clip` ran `_check_digest` before
`_check_wav`, on both paths (now lines 97-98 and 103-104); `_check_digest` puts `actual` in `details`
(line 186). `src/narration/jobs/voice.py:112`, the same at job time. The route with no gate:
`src/narration/backend/service.py:985-992`.

**Fix.** The WAV check now comes before the sha256 check, so `details.actual` is only ever the hash of a WAV
(which `profile_voice` measures anyway). At job time (`stage_clip`), the file is no longer the one admitted and
may not be audio, so `actual` is dropped there.

**Tests.** `tests/backend/test_steps.py::test_a_file_that_is_not_a_wav_never_has_its_sha256_given_back_s17_3`
and `tests/backend/test_submit.py::` the test of the same name; `tests/jobs/test_units.py::
test_a_clip_that_is_not_the_one_sent_is_voice_file_mismatch_s17` now checks `details`. All three fail on the
old code (KNOW: run with the fix stashed).

**Residual.** Whether a path exists, is a file, and is a WAV can still be learned (S8).

### S2. `uv` was looked up in the working folder first, on Windows (Low, fixed)

**What.** `narration-admin install` and `doctor` find uv as `os.environ["UV"]`, else `shutil.which("uv")`. On
Windows, Python 3.12's `shutil.which` searches the working folder before `PATH` unless
`NoDefaultCurrentDirectoryInExePath` is set (KNOW: CPython 3.12's `shutil.which` and
`_win_path_needs_curdir`). An operator who runs `narration-admin install` outside `uv run`, in a folder that
holds a `uv.exe` or `uv.bat` (a download folder, a cloned repository), would run that program. `install` runs
uv to sync the worker venvs.

**Evidence** (before the fix). `src/narration/admin/install.py:394`, `src/narration/admin/doctor.py:165`.

**Fix.** `narration.platform.find_program` searches only the absolute entries of `PATH` (with `PATHEXT` on
Windows); an empty or relative entry, which on POSIX also means the working folder, is skipped. Both call
sites use it (`src/narration/platform/__init__.py:129`).

**Tests.** `tests/platform/test_find_program.py`: a program planted in the working folder is never found, and
one only there is not found at all. Both fail with `shutil.which` in its place (KNOW).

### S3. Control tokens not shaped `<|…|>` pass the text and description checks (Low, open)

**What.** `[text] refuse` defaults to `[`, `]`, `<|` and `|>` (`src/narration/text/rules.py:37`), and the
description check refuses `<|` and `|>` (`src/narration/design/description.py:22`). The pinned Qwen3-TTS
tokenizer also has added tokens of other shapes, such as `<tts_pad>`, `<tts_text_bos>`, `<think>` and
`<tool_call>` (KNOW for the description, from WP34's review, recorded in HANDOFF.md). A cue text or a respelling
holding one is spoken as sent today (KNOW: the planner passes `<tts_text_eod>` through unchanged). A Hugging
Face tokenizer splits added tokens out of plain text, so the model would read it as its control token
(BELIEVE; not run here, since no model is available).

**Impact.** The caller spoils only its own take, which QA then judges. No other caller or file is reached. But
§17.6 ("no markup or instruction injection") is not fully met.

**Fix proposed.** Refuse every added token of the pinned tokenizers, taken from their `tokenizer_config.json`
at pin time and stored in the engine profile, or refuse `<` + name + `>` for the names the pinned tokenizer
defines. Either changes the rules' sha256 (`text.planner.rules_sha256`), and so the text rules' identity. That
is a design decision, not a small fix.

### S4. The trust anchors are files the operator must keep private (Low; Medium on a shared machine)

**What.** §17.4's gate trusts two lists: the provenance list in the store, and `[voices] allow_sha256` in the
configuration file. A line appended to `<store_root>/provenance.jsonl` is indexed at the next start
(`src/narration/store/store.py:1006-1022`, called at `:295`), after which that clip may be cloned. The models
under `models_root` and the worker venvs are code and data that workers load. The service neither sets nor
checks who can write any of them. `store_root` and `models_root` are the operator's choice. A folder made under
the root of a second drive usually inherits an access list that lets every signed-in user change files
(BELIEVE: the Windows default for a non-system drive root; not checked here).

**Impact.** Out of SECURITY.md's scope (a hostile local user), but only if the operator knows it: on a machine
shared with other accounts, any of them could allow a clip for cloning, stop or mislead the daemon (S11), or
run code in a worker.

**Fix.** SECURITY.md now says these folders and the configuration must be writable only by the operator.
Proposed: `narration-admin doctor` warns when the store root, `models_root`, the configuration file or a
worker project is writable by another account (Windows access-list code in `narration.platform`; a test needs
the Windows CI job).

### S5. The read check and the open are not atomic (Info, open)

**What.** `check_readable_path` resolves links one name at a time and refuses a network or device target
before opening anything (`src/narration/platform/_windows.py:410-441`, `:645-669`). The caller then opens the
returned path by name (`src/narration/backend/clips.py:122-125`). A process that can change a folder on that
path in between could swap in a link to `\\server\share`, and the open would then reach that server with the
operator's sign-in.

**Impact.** Needs another process that writes the operator's folders: out of scope. The synthetic-voices rule
does not depend on it, because only the bytes hashed are ever cloned.

**Fix proposed.** Open first, then check the handle: `GetFinalPathNameByHandleW` on the open handle, refused
unless it is on a local drive. Opening a network path already offers the sign-in, so the full fix opens each
name with `FILE_FLAG_OPEN_REPARSE_POINT` and checks each handle. Windows-only code with a Windows-only test.

### S6. Only symbolic links and junctions count as links when a read path is resolved (Info, open)

**What.** `_resolve_links` follows only `IO_REPARSE_TAG_SYMLINK` and `IO_REPARSE_TAG_MOUNT_POINT`
(`src/narration/platform/_windows.py:87`, `:424`). Any other reparse point is taken as an ordinary name, and
Windows follows it when the file is opened. Other name-surrogate tags exist (container and silo links, for
example). None is known to reach a network path, and making one needs rights a caller does not have.
(BELIEVE.) OneDrive placeholders are not name surrogates, but opening one downloads the file.

**Fix proposed.** Refuse any reparse point whose tag has the name-surrogate bit (`0x20000000`) other than the two
that are followed, as `check_store_path` already refuses every reparse point.

### S7. Case is compared with `str.lower` (Info, no change)

`winpaths.relative_names` compares names with `ntpath.normcase`, which is `str.lower`
(`src/narration/platform/winpaths.py:166-174`). NTFS compares with its own upcase table, and the two differ
for a few characters: for example, KELVIN SIGN (U+212A) lowers to `k` (KNOW: `relative_names` treats
`D:\store\K` with U+212A as `D:\store\k`). It matters only for store confinement, where every name below the
root is ASCII (`_COMPONENT`) and the root is the operator's own. No change.

### S8. Error codes tell whether a path exists, is a file, and is a WAV (Info, documented)

`PATH_NOT_ALLOWED` names its rule (`not_found`, `not_regular_file`, `reparse_point` …), and `UNSUPPORTED_AUDIO`
says a file is not a WAV or is over 20 MB. An agent can therefore learn which files exist anywhere the
operator can read. This follows from §17.3 (any local path may be read) and helps a caller fix a wrong path.
SECURITY.md now states it.

### S9. The daemon's mutex name can be taken by another local user (Info, open)

The singleton is `Global\narration-mcp.daemon.<sha256 of the store path>` (`_windows.py:193-203`). Another
account on the machine can create that mutex first; `acquire` then reads `ERROR_ACCESS_DENIED` as "held"
(`_windows.py:231-236`), and no daemon ever starts for that store. `Global\` is deliberate (§4: two sessions
must never load two models). A denial of service by a local user: out of scope. If it matters, a daemon that
finds the mutex held while `run/daemon.json` names no live daemon could say so in `doctor`.

### S10. `voices allow --yes` (Info, documented)

`voices allow` asks a person to type `yes` (`src/narration/admin/voices.py:151-152`, `:207-228`); `--yes`
skips the question for the operator's own scripts. It is never an MCP tool (§17.4). An agent that can run
shell commands as the operator can run it, with or without `--yes`, and the design says so (§17.10). SECURITY.md
now says it too.

### S11. The daemon's control files (Info, no change)

`run/daemon.json` (the daemon's status) and `run/launch.json` (the last launch) are written by the service with a
temporary name and a rename. The front-end and `narration-admin` read them to tell whether a daemon runs. No
reader signals, kills or inspects the pid they name beyond its existence and start time
(`src/narration/daemon/sweep.py:79-114`), so a pid reused by another program is not mistaken for the daemon.
`daemon stop` is a row in the store's database, not a file. Whoever can write `run/` can at most make a front-end
believe a daemon runs (no autostart) or that none does (a second, harmless start: the singleton). That is S4's
trust boundary.

### S12. SECURITY.md compared with the code (fixed)

| SECURITY.md said | The code | Change |
|---|---|---|
| any agent "can read any file its operator's own account can read" | it gets measurements of a WAV, and (before S1) the sha256 of any file; never another file's content | reworded, with S8's oracle |
| "It writes only under its own store" | the operator commands write elsewhere: `install` writes `models_root` and the worker venvs, `voices allow` edits the configuration file, `render --out` copies a take where it is told | the operator commands named |
| "The only network activity … is the operator-initiated model download" | `install` also runs `uv sync`, which downloads the worker packages from their indexes | added |
| "a small set of markup-like sequences … are rejected" | true for the `<\|…\|>` shape; other control tokens pass (S3) | S3 noted |
| nothing on who may write the store | the provenance list, the configuration, the models and the venvs are the gate's trust anchors (S4) | added |
| nothing on other operating systems | v1 runs on Windows; elsewhere every path check refuses (`UnsupportedOsPlatform`) | added |
