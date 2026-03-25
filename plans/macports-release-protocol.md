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
- The active CMake path requires the [`src/biosig`](src/biosig) submodule to be initialized before packaging because the MacPorts port builds with [`-DSTF_BIOSIG_PROVIDER=SUBMODULE`](dist/macosx/macports/science/stimfit/Portfile.in:35).
- The generated source archive extracts to `Stimfit-<version>-Source`, so [`worksrcdir`](dist/macosx/macports/science/stimfit/Portfile.in:20) must stay aligned with that name.
- The MacPorts [`cmake` PortGroup](dist/macosx/macports/science/stimfit/Portfile.in:5) must control the build and destroot phases; custom [`build.cmd`](dist/macosx/macports/science/stimfit/Portfile.in) or [`destroot.cmd`](dist/macosx/macports/science/stimfit/Portfile.in) overrides break MacPorts’ wrapper invocation.
- For app-bundle installs, [`-DCMAKE_INSTALL_PREFIX=${applications_dir}`](dist/macosx/macports/science/stimfit/Portfile.in:33) is required so the bundle lands under `/Applications/MacPorts/stimfit.app` rather than `/usr/local` or `/Applications/MacPorts/MacPorts`.

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

1. Initialize the BioSig submodule with `git submodule update --init src/biosig`.
2. Configure the release build with explicit MacPorts wx/Python tool paths. The updated [`dist/macosx/macports/insert_checksums.sh.in`](dist/macosx/macports/insert_checksums.sh.in) now does this directly instead of relying on a legacy `make dist` flow.
3. Generate the source archive with `COPYFILE_DISABLE=1 cpack --config build-macports-release/CPackSourceConfig.cmake -G TGZ`.
4. Copy `Stimfit-<version>-Source.tar.gz` to the uploaded filename `stimfit-<version>.tar.gz`.
5. Upload the renamed tarball using [`dist/macosx/macports/upload_stimfit.in`](dist/macosx/macports/upload_stimfit.in).
6. Compute:
   - `rmd160`
   - `sha256`
   - archive `size`
7. Regenerate [`dist/macosx/macports/science/stimfit/Portfile`](dist/macosx/macports/science/stimfit/Portfile) from [`dist/macosx/macports/science/stimfit/Portfile.in`](dist/macosx/macports/science/stimfit/Portfile.in) by replacing the placeholders.

Critical packaging guardrails:

- Ensure the source archive contains [`src/biosig`](src/biosig) before uploading.
- Ensure the source archive does **not** contain `build/release`, otherwise the tarball can recursively include itself and grow to many gigabytes.
- Keep [`CPACK_SOURCE_IGNORE_FILES`](CMakeLists.txt:606) in [`CMakeLists.txt`](CMakeLists.txt) so build artifacts are excluded from the source archive.

### 3. Validate the new Portfile

1. Confirm the generated [`dist/macosx/macports/science/stimfit/Portfile`](dist/macosx/macports/science/stimfit/Portfile) contains:
   - `version 0.17.1`
   - updated `checksums`
   - [`worksrcdir          Stimfit-${version}-Source`](dist/macosx/macports/science/stimfit/Portfile.in:20)
   - [`-DCMAKE_INSTALL_PREFIX=${applications_dir}`](dist/macosx/macports/science/stimfit/Portfile.in:33)
   - unchanged variant logic unless intentionally revised
2. If a local MacPorts checkout exists at `~/macports/dports`, run the comparison flow described by [`dist/macosx/macports/mk_diff.sh`](dist/macosx/macports/mk_diff.sh).
3. Validate that the resulting Portfile remains ready for submission under `science/stimfit` in `macports-ports`.
4. Local acceptance sequence for the overlay port should include:
   - `sudo port checksum stimfit`
   - `sudo port -v build stimfit`
   - `sudo port -v install stimfit`
5. Confirm the bundle lands at `/Applications/MacPorts/stimfit.app` only.

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
