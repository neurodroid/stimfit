# MacPorts release protocol for Stimfit

## Current preparation

The target is Stimfit `0.17.4`, with the eventual release tag
`v0.17.4macports`. [`VERSION`](../VERSION) already contains `0.17.4` from
the Windows release preparation; no additional version bump is needed.
For future releases, derive the target from `VERSION` and use the tag pattern
`v<version>macports`.

Prepare and validate the source archive, Portfile, downstream patch, and draft
pull request message locally. **Do not submit the MacPorts pull request until
the user has edited its message and authorized submission.** Uploading release
files, pushing commits or tags, and submitting the pull request are separate
publication steps.

## Conventions to preserve

- The active port is [`science/stimfit`](../dist/macosx/macports/science/stimfit/Portfile.in).
  The legacy `python/py-stfio` port is outside this release's scope.
- Generate `Portfile` from `Portfile.in` by replacing `STFVERSION`, `RMD160`,
  `SHA256`, and `SIZE` with the version and the final archive's metadata.
- The uploaded filename is `stimfit-<version>.tar.gz`; its top-level directory
  is `Stimfit-<version>-Source`. Keep `worksrcdir` aligned with that directory.
- Let the `cmake` PortGroup control build and destroot. Custom `build.cmd` or
  `destroot.cmd` overrides break its wrapper invocation.
- Preserve `-DCMAKE_INSTALL_PREFIX=${applications_dir}` and app-bundle mode,
  so the destination is `/Applications/MacPorts/stimfit.app`.
- The port builds the bundled BioSig sources with `STF_BIOSIG_PROVIDER=SUBMODULE`
  and requires GNU awk. The current template already declares `port:gawk`.
- Keep the Python interpreter and wx-config from the same MacPorts Python
  branch. Accelerate is available from Darwin 22.4 onward; older systems
  default to OpenBLAS.

### Python policy checked on 2026-11-07

The [MacPorts Python PortGroup](https://github.com/macports/macports-ports/blob/master/_resources/port1.0/group/python-1.0.tcl)
defaults to Python 3.14 (`def_v 314`). Individual ports declare their supported
branches; this differs from the user's `port select` command setting.
[wxPython](https://github.com/macports/macports-ports/blob/master/python/py-wxpython-4.0/Portfile)
and [IPython](https://github.com/macports/macports-ports/blob/master/python/py-ipython/Portfile)
both provide Python 3.10–3.14 subports. Retain Stimfit's `python310` through
`python314` variants and default `+python314`. A Python 3.15 interpreter port
alone is insufficient: both dependencies need matching subports first.

Python 3.10 remains available in MacPorts, but
[CPython support ended on 2026-10-01](https://devguide.python.org/versions/).
Treat its variant as legacy availability; CPython's maintained branches are
3.11–3.15. Recheck these sources for each release.

## 1. Select and stage the source

1. Work from the intended upstream branch, currently `master`. Inspect the
   worktree and record the source commit and the recorded BioSig submodule
   commit. Initialize `src/biosig` if necessary.
2. Review `VERSION`, the active Portfile template, and the MacPorts configure
   helpers. Commit intended source changes before selecting the source commit
   so they are included in a Git archive.
3. Create a fresh source staging directory with `mktemp -d` under `/private/tmp`.
   Populate it using `git archive HEAD`, then populate its `src/biosig` directory
   using `git -C src/biosig archive <recorded-submodule-commit>`. An archive of
   the parent repository alone does not include submodule contents.
4. Package this clean staging tree, rather than the working directory. This
   excludes ignored Autotools output, caches, and local artifacts while keeping
   the exact selected sources. Keep it outside any directory named `build`:
   CPack's `/build/` exclusion can match the absolute staging path. Record the
   staging path and both commit IDs with the release files under `build/release`.

## 2. Generate the archive and Portfile

1. Configure CMake from the staged sources into a separate build directory.
   Use explicit paths to the selected MacPorts Python and matching wx-config,
   enable the BioSig submodule provider, and retain the port's app-bundle settings.
2. Run `COPYFILE_DISABLE=1 cpack --config <build-dir>/CPackSourceConfig.cmake -G TGZ`
   from the release output directory.
3. Copy `Stimfit-<version>-Source.tar.gz` to `stimfit-<version>.tar.gz`.
4. Inspect the archive before computing checksums. It must contain `VERSION`,
   the CMake files, resources, and BioSig 4.0.0 sources for this release. It must
   exclude Git metadata, build output, `_CPack_Packages`, and AppleDouble files.
   In particular, never include `build/release`: an archive can otherwise
   recursively include its own output. Preserve `CPACK_SOURCE_IGNORE_FILES`
   in [`CMakeLists.txt`](../CMakeLists.txt).
   macOS provenance attributes can make CPack's libarchive add AppleDouble
   entries despite `COPYFILE_DISABLE=1`. Remove those entries and extended
   attribute metadata from the archive before freezing it, then verify again.
5. Compute `openssl rmd160 -r`, `openssl sha256 -r`, and `wc -c` for the renamed
   archive. Regenerate the active Portfile from its template using those values.
   Verify every placeholder is replaced and the version matches `VERSION`.
6. Freeze the archive once its checksums are in the Portfile. Any source/archive
   change requires regeneration and revalidation of all checksum fields.

[`insert_checksums.sh.in`](../dist/macosx/macports/insert_checksums.sh.in) is a
historical reference, not a preparation command to execute wholesale. It also
uploads the archive, runs `sudo portindex`, and uninstalls/cleans installed
Stimfit ports. Perform local packaging and validation as separate operations.

## 3. Validate locally

1. Fetch the latest [MacPorts PR template](https://github.com/macports/macports-ports/blob/master/.github/PULL_REQUEST_TEMPLATE.md)
   and work through **every verification item** before declaring the release
   ready. Save a per-release checklist with each item's applicability, result,
   command or other evidence, and any blocker. The actions below supplement
   that live checklist; they do not replace it. Revisit this list if the
   upstream template changes during preparation.
2. Run MacPorts lint against the generated port and inspect its resolved default
   variants, dependencies, and configure arguments.
3. Point the local port's source fetching at the prepared archive with a local
   file URL or a dedicated local distfiles directory. Run `port checksum` to
   validate the actual Portfile checksums without requiring a public upload.
4. Build and run `port destroot` from the extracted release archive using the
   selected default Python variant and the applicable LAPACK provider. Test the
   source archive without Git metadata, not just the original checkout.
5. Inspect the staged bundle at `${destroot}/Applications/MacPorts/stimfit.app`.
   Confirm it contains the executable, private libraries, resources, `_stf`
   module, and generated `stf.py`. Verify no extra nested `MacPorts` directory
   appears and no build-tree paths leak into installed runtime paths.
6. Run the existing tests through `port test`, with `STF_BUILD_TESTS` and
   `STF_BUILD_NUMERIC_TESTS` enabled in an isolated validation build. The
   production Portfile disables test targets, so a bare `port test` can skip
   testing. Use explicit configure/test overrides and verify CTest actually
   executed all five suites. Record the exact command and result.
7. Attempt the template's full source/trace install (`port -vst install`) from
   a fresh validation port directory. Use `sudo` when required by the local
   MacPorts installation. Record trace failures separately from an ordinary
   source install; a successful `port -vs install` does not establish trace
   success. Retain the previous installed version for rollback.
8. Check the installed application starts and opens a representative recording;
   verify its embedded Python module and basic analysis operations. Exercise
   the default variants and important alternatives supported by the available
   dependencies, recording the exact tested combinations and gaps.
9. Complete the template's repository checks: inspect the commit message,
   minimize the downstream commit series, search current open PRs for a
   duplicate, and reference any associated Trac ticket. Remove only genuinely
   inapplicable items, as directed by the live template; do not invent tickets
   or check unfinished work.

If a required applicable check is blocked by the host or missing permissions,
record the attempted command, concrete failure, and remaining work. Report the
gap in the PR message and final handoff. A test that was merely attempted must
not be reported as passing. Store the completed checklist and logs alongside
the release artifacts.

## 4. Prepare the downstream change for review

1. Compare the generated Portfile with the current `science/stimfit/Portfile`
   in `macports/macports-ports`, preserving relevant downstream fixes.
2. If a downstream clone exists, use a dedicated branch such as
   `stimfit-<version>` and keep the change scoped to this port. This preparation
   uses `build/release/macports-ports`. If a clone is unavailable, prepare a
   patch and draft message under `build/release` for review before publishing.
3. Use the proposed title/commit subject `stimfit: update to <version>`.
4. **Fetch and inspect the current MacPorts PR template for every release**
   before drafting the message. Use
   [`.github/PULL_REQUEST_TEMPLATE.md` on the default branch](https://github.com/macports/macports-ports/blob/master/.github/PULL_REQUEST_TEMPLATE.md);
   if it moves, inspect `.github` for its replacement. Do not reuse an old
   local template without checking the current upstream version. Record the
   source URL and check date alongside the release artifacts.
5. Follow the template's current sections and checklist. Explain the changes,
   include a very brief list of major changes since the previous MacPorts
   release, record the tested macOS/Xcode versions, and cite the validation
   performed. Derive the change list from source history; prioritize changes
   affecting MacPorts users.
   Mark only completed checklist items: running CTest does not establish that
   `sudo port test` was run, and staging with `destroot` does not establish a
   full `sudo port -vst install` or an interactive GUI test. Preserve the user's
   edited wording when applying template formatting; keep a separate copy if
   needed. Leave the message available for the user to edit before submission.
6. Once the upstream packaging commit is finalized, the annotated release tag
   follows `v<version>macports`. Check that any proposed tag points at the
   packaging-ready commit and that the archive's recorded source commits are
   available in upstream history.

## 5. Publish only after review

The local preparation is complete when the final archive, matching Portfile,
validation record, downstream patch, and editable draft message are ready.
Check the current PR template again before submission if upstream has changed
since drafting. Publication can then upload the exact validated archive,
verify its public checksums, and push the intended release history. Submit the
MacPorts pull request only after the user has edited its message and authorized
submission; preparing a draft file does not satisfy that condition.

### Established SFTP upload procedure

The historical [`insert_checksums.sh.in`](../dist/macosx/macports/insert_checksums.sh.in)
invokes `sftp -b <batch-file> p8210991@ftp.schmidt-hieber.de` from the local
`build/release` directory. Its batch template,
[`upload_stimfit.in`](../dist/macosx/macports/upload_stimfit.in), changes the
remote directory to `StimfitJ` and uploads `stimfit-<version>.tar.gz`.

The tracked [`upload_release.py`](../dist/macosx/macports/upload_release.py)
helper prepares the version-specific batch from that template with an explicit
local directory. It derives paths from its repository location, checks
`VERSION` against the generated Portfile, and verifies the frozen archive's
RIPEMD160, SHA256, and size. No credentials are stored in the repository.

Prepare and inspect the batch without connecting:

```sh
python3 dist/macosx/macports/upload_release.py --dry-run
```

To publish the checked archive when authorized, run:

```sh
python3 dist/macosx/macports/upload_release.py --upload
```

The helper retains SSH host-key checks, refuses an existing remote release or
an inconclusive access check, and uploads under a temporary name before
setting public read permissions and renaming it to the public filename. It
uses legacy SFTP rename (`rename -l`) so publication refuses to replace an
existing destination, including one created after the preflight. Use the helper
rather than invoking its generated batch directly, which would skip the remote
preflight. Coordinate release uploads; failed transfers can leave a temporary
file that needs inspection.
SSH credentials remain in the user's existing configuration. A read-only batch
(`pwd`, `cd StimfitJ`, `ls`) can establish access independently. On 2026-11-07, a read-only
connection succeeded, `stimfit-0.17.2.tar.gz` was present, and
`stimfit-0.17.4.tar.gz` was absent. Generated upload batches stay under
`build/release`; the reusable helper and template belong in version control.

Upload the frozen archive without rebuilding it. The current Portfile expects
`http://www.stimfit.org/stimfit-<version>.tar.gz`; verify that a fresh public
download has the Portfile's exact RIPEMD160, SHA256, and size before submitting
the PR, since CI cannot use a local `distpath` override. If a release file
already exists, compare it with the intended archive before considering a
replacement. Do not run the full historical helper to perform an upload.
