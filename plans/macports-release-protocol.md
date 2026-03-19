# MacPorts release protocol for Stimfit

## Purpose

This protocol captures the exact workflow for preparing a Stimfit MacPorts release so a future agent can repeat the process with minimal guidance.

## Current target

- upstream version: `0.17.1`
- MacPorts tag: `v0.17.1macports`
- source template: [`dist/macosx/macports/science/stimfit/Portfile.in`](dist/macosx/macports/science/stimfit/Portfile.in)
- checksum helper reference: [`dist/macosx/macports/insert_checksums.sh.in`](dist/macosx/macports/insert_checksums.sh.in)
- version source of truth: [`VERSION`](VERSION)

## Observed conventions

- The Stimfit MacPorts template uses placeholders `STFVERSION`, `RMD160`, `SHA256`, and `SIZE` in [`dist/macosx/macports/science/stimfit/Portfile.in`](dist/macosx/macports/science/stimfit/Portfile.in).
- The checksum regeneration flow is documented by [`dist/macosx/macports/insert_checksums.sh.in`](dist/macosx/macports/insert_checksums.sh.in:3) through [`dist/macosx/macports/insert_checksums.sh.in`](dist/macosx/macports/insert_checksums.sh.in:39).
- Historical MacPorts release tags follow the `v<version>macports` pattern, as seen in [`.git/packed-refs`](.git/packed-refs).
- Local helper [`dist/macosx/macports/mk_diff.sh`](dist/macosx/macports/mk_diff.sh) compares the generated local Portfile against a MacPorts checkout under `~/macports/dports`.

## Release workflow

### 1. Prepare the upstream tree

1. Confirm the work starts from the intended upstream branch, currently [`master`](BRANCHES.md:7).
2. Inspect [`VERSION`](VERSION) and bump it from the previous release to the target release.
3. Review whether MacPorts-related defaults need adjustment for the new release in:
   - [`dist/macosx/macports/science/stimfit/Portfile.in`](dist/macosx/macports/science/stimfit/Portfile.in)
   - [`dist/macosx/scripts/conf_macports_release.sh`](dist/macosx/scripts/conf_macports_release.sh)
4. Keep the active Stimfit MacPorts port in scope and treat legacy [`dist/macosx/macports/python/py-stfio/Portfile.in`](dist/macosx/macports/python/py-stfio/Portfile.in) as reference-only unless a compatibility reason forces an update.

### 2. Generate the dist tarball and checksums

Follow the logic embedded in [`dist/macosx/macports/insert_checksums.sh.in`](dist/macosx/macports/insert_checksums.sh.in):

1. Re-run project generation if needed.
2. Configure the release build using [`dist/macosx/scripts/conf_macports_release.sh`](dist/macosx/scripts/conf_macports_release.sh:22).
3. Build the source tarball with the equivalent of `make dist` in the release build directory.
4. Compute:
   - `rmd160`
   - `sha256`
   - archive `size`
5. Regenerate [`dist/macosx/macports/science/stimfit/Portfile`](dist/macosx/macports/science/stimfit/Portfile) from [`dist/macosx/macports/science/stimfit/Portfile.in`](dist/macosx/macports/science/stimfit/Portfile.in) by replacing the placeholders.

### 3. Validate the new Portfile

1. Confirm the generated [`dist/macosx/macports/science/stimfit/Portfile`](dist/macosx/macports/science/stimfit/Portfile) contains:
   - `version 0.17.1`
   - updated `checksums`
   - unchanged variant logic unless intentionally revised
2. If a local MacPorts checkout exists at `~/macports/dports`, run the comparison flow described by [`dist/macosx/macports/mk_diff.sh`](dist/macosx/macports/mk_diff.sh).
3. Validate that the resulting Portfile remains ready for submission under `science/stimfit` in `macports-ports`.

### 4. Prepare git history for upstream release tracking

1. Commit the version bump and any source-release changes in the Stimfit repository.
2. Commit the regenerated MacPorts files in the Stimfit repository.
3. Create an annotated tag named `v0.17.1macports` on the final packaging commit.

Recommended split if there are two commits:

- upstream/version prep commit: `Bump version to 0.17.1`
- packaging commit: `Update macports files`

If the release is prepared as a single consolidated commit, preserve the existing intent and make sure the annotated tag still lands on the packaging-ready commit.

### 5. Prepare the downstream MacPorts pull request branch

In the local clone of `https://github.com/macports/macports-ports`, create a dedicated branch for the submission.

Recommended branch naming:

- `users/cs/stimfit-0.17.1`
- or `stimfit-0.17.1`

Recommended MacPorts commit message format:

- `stimfit: update to 0.17.1`

If dependencies or build logic change materially, append a brief bullet list in the commit body describing the rationale.

### 6. Pull request readiness checklist

Before opening the downstream PR, verify:

- the generated [`dist/macosx/macports/science/stimfit/Portfile`](dist/macosx/macports/science/stimfit/Portfile) is final
- version and checksum fields match the produced tarball
- local branch for `macports-ports` is dedicated to this update only
- commit message is `stimfit: update to 0.17.1`
- upstream Stimfit tag `v0.17.1macports` exists on the final packaging commit

## Implementation-ready checklist for Code mode

- Update [`VERSION`](VERSION) from `0.17.0` to `0.17.1`.
- Review [`dist/macosx/macports/science/stimfit/Portfile.in`](dist/macosx/macports/science/stimfit/Portfile.in) for any release-specific adjustments.
- Use the checksum-generation procedure from [`dist/macosx/macports/insert_checksums.sh.in`](dist/macosx/macports/insert_checksums.sh.in) to build the tarball and regenerate [`dist/macosx/macports/science/stimfit/Portfile`](dist/macosx/macports/science/stimfit/Portfile).
- Verify the generated Portfile fields and diff against any existing MacPorts checkout if available.
- Commit upstream changes using the established local history style.
- Create annotated tag `v0.17.1macports` on the packaging commit.
- In the `macports-ports` clone, create a fresh branch and commit the new Portfile with message `stimfit: update to 0.17.1`.

## Notes for future agents

- [`dist/macosx/macports/insert_checksums.h.in`](dist/macosx/macports/insert_checksums.h.in) was requested in the task description, but the repository currently contains [`dist/macosx/macports/insert_checksums.sh.in`](dist/macosx/macports/insert_checksums.sh.in), which appears to be the actual helper script describing the checksum workflow.
- Architect mode can document and plan this workflow, but the actual file edits outside Markdown, git commands, build commands, tag creation, and downstream branch preparation require a switch to Code mode.
