## What this changes and why

<!-- A short description. Link an issue if there is one. -->

## Checklist

- [ ] The title follows [Conventional Commits](https://www.conventionalcommits.org/) (`feat(area): …`,
      `fix(area): …`, `docs: …`, `chore: …`, …).
- [ ] `uv run pytest` passes (the default suite: no GPU, no model, no private evidence needed).
- [ ] `uv run ruff check` and `uv run ruff format --check` are clean.
- [ ] `uv run basedpyright` is clean.
- [ ] `CHANGELOG.md` has a line under `## [Unreleased]` if this change is something a user of the
      package would notice.
- [ ] No local paths (drive letters, home directories, user names) anywhere in the diff or in the commit
      messages, and no audio, model weight, checkpoint or database files were added.
- [ ] Any new dependency's licence has been checked and stated (permissive is fine; GPL/AGPL and
      anything non-commercial or research-only is refused; LGPL needs a maintainer's sign-off first).

## Notes for the reviewer

<!-- Anything that needs a closer look, a design trade-off you made, or a question you have. -->
