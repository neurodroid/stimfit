# Debian release protocol for Stimfit

Prepare the Debian unstable packages from `master` and the packaging branch
`debian/sid`. Keep this procedure current during preparation: record each
failure and its resolution, exact source and dependency versions, validation
results, and the artifact location. Preparing a release ends with reviewable,
unsigned artifacts; signing, tags, pushes, and uploads are publication steps.

## Current preparation

The target is **0.17.4-1**, prepared on 2026-11-07. `VERSION` already contains
`0.17.4`. The upstream source commit is
`0ea89fe9` (`Update MacPorts package to 0.17.4`); the previous Debian release is
`0.17.2-1` at `dedb4269`. Upstream has been merged into `debian/sid`.

Artifacts are retained under [`deb/0.17.4-1`](../deb/0.17.4-1/) (Git ignored).
The running staging directory is `/tmp/stimfit-debian-release`.

| Check | Current result |
| --- | --- |
| Source package creation | Passed, including orig tarball and quilt patch |
| HDF5 1.14 Debian binary build | Passed after correcting the ABF2 fixture |
| HDF5 2.1 Debian binary build | Passed after correcting the ABF2 fixture |
| CTest | All 3 suites passed in both build environments |
| Installed stfio unit tests | 17 tests passed in each HDF5 environment, no skips |
| Installed package autopkgtest | Both tests passed in both fresh testbeds |
| Current Debian unstable | Build, upgrade from 0.17.2-1, removal and reinstallation passed |
| Source build after binary build and clean | Passed with HDF5 2.1 and current unstable HDF5 2.2 |
| Clean upstream tree | All 614 orig archive files match after clean |
| Lintian | Passed for source and both binary sets; no warnings outside sandbox |
| Git publication | Release changes and merged histories published on `origin/debian/sid` |
| Release tags | `upstream/0.17.4` pins `0ea89fe9`; `debian/0.17.4-1` includes the final import test fix |
| Signing and package upload | Signed and verified; HTTPS transfer to mentors completed, processing unconfirmed |

The HDF5 1.14 and 2.1 compatibility builds use dated Debian snapshots. A
subsequent build and installed-package checks against current unstable on
2026-11-07 use HDF5 2.2. Record those environments separately.

## 1. Review branches and Debian reports

Start with `git status --short`, `git branch -a`, `VERSION`, and
[`BRANCHES.md`](../BRANCHES.md). Preserve local changes. Merge the selected
upstream commit into `debian/sid`; use `debian/sid-0.16` only for legacy 0.16
releases. Do not copy the legacy packaging over the current CMake packaging.

Read the [source package bug list](https://bugs.debian.org/src:stimfit) and
recent archived reports. For this release the important reports are:

- [1139613](https://bugs.debian.org/1139613): the original HDF5 version/header
  bug was fixed on the 0.16 line, but the CMake migration subsequently failed
  with HDF5 2.1's unnamespaced exported targets. Verify real Debian packages
  with both HDF5 generations. Preserve the imported-target handling in
  `cmake/StimfitDependencies.cmake` and header feature detection in
  `src/libstfio/hdf5/hdf5lib.cpp`. This was closed by `0.17.2-1`.
- [1140597](https://bugs.debian.org/1140597): GTK initialization needs DISPLAY
  even for help, and the earlier help command returned 255. A test using
  `|| true` hid failures. Test installed `stimfit -h` and `stimfit --help`
  under Xvfb, require zero exit status and Usage output, and require an invalid
  option to fail with an option diagnostic. Preserve `xvfb` and `xauth` test
  dependencies. This was also closed by `0.17.2-1`.
- [1047146](https://bugs.debian.org/1047146): source package creation failed
  after a successful binary build. Run the actual build, clean, and source
  rebuild sequence; inspect for unexpected upstream changes.
- [1035273](https://bugs.debian.org/1035273) and
  [1036491](https://bugs.debian.org/1036491): requests for meaningful NumPy and
  Matplotlib autopkgtest coverage remain open. An import smoke test alone
  does not establish that those requests are resolved.

Do not repeat `Closes` entries for bugs already closed by an earlier upload.
Keep the new changelog limited to changes actually present in the package.
Fold relevant notes from earlier UNRELEASED entries into the release being
prepared and remove the superseded entries; Git history retains the development
record. Omit platform-specific changes that do not affect the Debian package.
Lead the changelog with user-facing changes: file opening, import reliability,
plotting, and Python features. Summarize test and packaging maintenance in one
sentence; keep the detailed fixes and validation results in this procedure.

## 2. Stage a clean source package

Use a directory outside the working checkout. Build from the selected upstream
commit and copy only `debian/` from the packaging checkout. This prevents local
build output and release bookkeeping in `plans/` from entering the source diff.
The procedure itself remains in the repository; the release source is pinned
to the upstream commit recorded above.

Example, from the repository root, replacing the paths and version as needed:

```sh
mkdir -p /tmp/stimfit-debian-release
git archive --format=tar.gz --prefix=stimfit-0.17.4/ \
  0ea89fe9 -o /tmp/stimfit-debian-release/stimfit_0.17.4.orig.tar.gz
tar -xzf /tmp/stimfit-debian-release/stimfit_0.17.4.orig.tar.gz \
  -C /tmp/stimfit-debian-release
cp -a debian /tmp/stimfit-debian-release/stimfit-0.17.4/
cd /tmp/stimfit-debian-release/stimfit-0.17.4
dpkg-buildpackage -S -nc -us -uc -d
```

`-nc` is suitable only for this freshly extracted, unbuilt tree. Later source
rebuild checks must omit it. Write logs outside the extracted tree: a log
created in the source root becomes an unexpected upstream modification.
`dpkg-source -b` writes artifacts relative to the current directory; run it
from the staging parent or use `dpkg-buildpackage` inside the extracted tree.

Debian uses `STF_BIOSIG_PROVIDER=SYSTEM`. A parent `git archive` omits BioSig
submodule contents; this release's Debian orig archive intentionally uses the
system provider and contains no bundled BioSig checkout. Do not use this
archive for MacPorts or another build requiring bundled BioSig.

Keep downstream source changes as DEP-3 documented quilt patches under
`debian/patches/`, listed in `series`. Do not modify the frozen orig archive to
carry a Debian-only correction. Preserve its checksum once artifacts are
selected for signing.

## 3. Build against both HDF5 generations

Use a clean Debian build environment, not the Ubuntu development host. The
available chroots for this preparation are:

| Chroot | Snapshot | HDF5 development package |
| --- | --- | --- |
| `unstable-hdf5-1-amd64-sbuild` | `20260804T051924Z` | `1.14.6+repack-2+b1` |
| `unstable-hdf5-2-amd64-sbuild` | `20260807T000000Z` | `2.1.0+repack-1` |
| `unstable-amd64-sbuild` | Current archive, 2026-11-07 | `2.2.0+repack-6` |

The host defaults sbuild to unshare mode; explicitly select schroot mode for
these existing directory chroots. They share `/build` through
`/var/lib/sbuild/build`. **Parallel builds must use different build paths**:

```sh
sbuild --chroot-mode=schroot -c unstable-hdf5-2-amd64-sbuild \
  --build-path=/build/stimfit-release-hdf5-2 -d unstable -j8 \
  --purge-build=never --purge-session=never \
  --no-run-lintian --no-run-autopkgtest stimfit_0.17.4-1.dsc
```

Run the HDF5 1 build in its own artifact directory with
`--build-path=/build/stimfit-release-hdf5-1`. Save the complete logs and exact
installed package versions. Retaining sessions is useful while investigating
failures; end only this preparation's sessions after saving their artifacts.

Use `/home/c02380/virtualenv-3.14/bin/python` for host Python commands. That
environment is unavailable inside the Debian chroots. Package builds and
installed-package tests use Debian's Python to validate the distribution ABI;
report this interpreter exception before switching.

Do not set `nocheck`, add `NDEBUG`, or suppress failed commands to obtain a
successful release build. Review all CTest results and `dh_missing` output.
Check the standalone `python3-stfio` package, the embedded `_stf.so` and
`stf.py` in `stimfit`, toolbar assets, and executable permissions on tests.
New upstream test executables should be excluded through
`debian/not-installed`, not accidentally shipped in the application package.

## 4. Test the installed packages

Use fresh matching Debian testbeds and the binaries just produced:

```sh
autopkgtest /path/to/stimfit_0.17.4-1_amd64.deb \
  /path/to/python3-stfio_0.17.4-1_amd64.deb \
  /tmp/stimfit-debian-release/stimfit-0.17.4 \
  --output-dir=/tmp/stimfit-debian-release/autopkgtest-hdf5-2 \
  -- schroot unstable-hdf5-2-amd64-sbuild
```

Repeat with the HDF5 1 binaries and testbed. Keep each output directory unique.
Do not use `-- null` on the development host: that backend tests and may modify
the host. The help test must actually execute the installed binary under Xvfb
and inspect its status and output; checking only that the executable exists
would miss the regression. Preserve the standalone stfio import test.

Where possible, additionally run `src/pystfio/unittest_stfio.py` against the
installed module, with its HDF5 fixture present, to exercise NumPy and HDF5
reading rather than just importability. Do not count a skipped fixture suite
as passing coverage.

## 5. Verify cleanup and lintian

After a successful binary build, run the following in that retained build
session and source directory:

```sh
dpkg-buildpackage -S -us -uc
```

This runs `debian/rules clean` and creates the source package. Require success
without unexpected upstream changes or ignored source deletions. Compare
upstream file contents with the orig archive after quilt patches are unapplied.
Retain this log alongside the source and binary build logs.

On the Ubuntu host explicitly use the Debian lintian profile:

```sh
lintian --profile debian /path/to/stimfit_0.17.4-1_source.changes \
  /path/to/stimfit_0.17.4-1_amd64.changes
```

Without `--profile debian`, Ubuntu lintian incorrectly rejects `unstable` as a
bad distribution for this upload. If sandboxed lintian reports only a groff
subprocess failure while direct rendering succeeds, repeat outside the sandbox
and record both results; do not add a package override for a host limitation.

## 6. Retain and publish the reviewed release

Save the orig tarball, `.dsc`, `.debian.tar.xz`, source `.changes` and
`.buildinfo`, binary packages and their metadata, validation logs, and
checksums under `deb/<version>/`. Keep HDF5 1 and 2 binary results in separate
subdirectories because their filenames are identical. Record the final source
commit, packaging changes, dependency versions, and all remaining warnings.
A successful dput transfer confirms delivery only; confirm the version on the
mentors package page or in its processing email before claiming acceptance.

Keep the release changes reviewable on `debian/sid`. Ensure the repository
matches the packaged `debian/` directory. Refresh source artifacts and repeat
affected checks after packaging changes; never sign stale artifacts. Only
after maintainer approval should the appropriate upstream and Debian tags be
created, commits pushed, and signed source changes uploaded to mentors or the
Debian archive. Do not publish both HDF5 binary variants as one upload.

## Preparation record for 0.17.4-1

- Merged upstream `0ea89fe9` into the current `debian/sid` packaging branch.
- Preserved the HDF5 target resolution and Xvfb help fixes from `0.17.2-1`.
- Added `stimfit_io_test` to `debian/not-installed` and extended the help test
  to cover both help options and rejection of an unknown option.
- Initial simultaneous sbuild runs used the shared default
  `/build/reproducible-path`; their cleanup/build operations collided. Those
  failures were environmental and are not passing build evidence. Restarted
  with distinct build paths.
- Both real Debian builds then compiled but failed `stimfit_io_test` with
  `DACSection.uBytes == sizeof(DACInfo)`. The synthetic fixture left an empty
  DAC section's record size at zero. Added
  `debian/patches/abf2-fixture-dac-size.patch`, setting it to the required 256
  bytes. Rebuilt both environments with assertions enabled: all three CTest
  suites passed, and both binary package sets were produced.
- Ran `dpkg-buildpackage -S -us -uc` after the HDF5 2 binary build. It cleaned
  successfully, built the source artifacts, and unapplied the fixture patch.
  Compared all 614 upstream archive files with the cleaned tree: zero changes.
- Both fresh autopkgtest environments passed `python3-stfio-import` and
  `stimfit-help`. Autopkgtest returned 8 because both tests carry the existing
  `superficial` restriction; this is not a failed-test status.
- Debian-profile lintian passed for the source upload and both binary sets
  with no output outside the sandbox. The sandbox-only groff warning did
  not reproduce; no override was added.
- Installed the built `python3-stfio` package in each retained build session
  and ran `src/pystfio/unittest_stfio.py` from `/tmp` using Debian Python
  3.14.6. All 17 tests passed in each environment with no skips. Expected
  diagnostics from rejected BioSig reads and unsupported writes were emitted.
- Verified every SHA256 entry in the source and both binary `.changes` files
  and confirmed that the packaged `debian/` files match the checkout.
- Initial preparation left unsigned source and packages ready for maintainer
  review, with packaging edits and this procedure in the `debian/sid` working
  tree. The upstream merge is commit `344bbc0a`.

- Saved validation logs, upload metadata, and `SHA256SUMS` under
  `deb/0.17.4-1/`, then closed both retained build sessions successfully.
- Revised the changelog after maintainer review to emphasize reader selection,
  import reliability, Python 3 plotting, and ABF metadata handling. Summarized
  test and packaging maintenance in one sentence. Refreshed the unsigned source
  upload and checksums; the retained binary validation packages contain the
  earlier changelog wording and otherwise identical code and packaging.
- The maintainer authorized staging, committing, and pushing the release
  changes. Fetching `origin` found two additional `debian/sid` commits through
  `dcafd0d1`, including an older `0.17.2-2` UNRELEASED changelog entry.
  Committed the reviewed release preparation as `adbd7d7c` and merged the
  remote history. Follow-up maintainer review consolidated the never-released
  `0.17.2-2` notes into `0.17.4-1` and removed the separate draft stanza.
  Upstream `master` at `0ea89fe9` is already an ancestor of the packaging
  branch; keep the Debian packaging on `debian/sid`.
- Removed trailing whitespace from the quilt patch context and verified that
  it applies without fuzz. Refreshed source metadata after the changelog merge;
  the applied source correction is unchanged.
- Published the release preparation commit `adbd7d7c` and remote integration
  merge `d7d6b3ee` on `origin/debian/sid` through a normal fast-forward push.
  Verified that the complete upstream `master` history is included and that
  differences from upstream are confined to `debian/` and this procedure.
  Application sources are unchanged by the remote integration. Source lintian
  and upload checksum verification passed again after regenerating metadata.
  Package signing, release tags, and the Debian or mentors upload remain pending.

- Folded the unpublished `0.17.2-2` changelog into `0.17.4-1`: retained the
  Python 2 support removal and summarized compiler-flag and test maintenance
  in one sentence. ABF metadata handling was already covered. Omitted Windows
  and macOS packaging changes that do not affect the Debian package and kept
  detailed validation results here rather than in the user-facing changelog.
  Regenerated the unsigned source upload and checksums and reran source
  lintian; application code and the previous build/test results are unchanged.

- Continued release preparation against current Debian unstable using
  `unstable-amd64-sbuild`, `--apt-update --apt-distupgrade`, and the distinct
  build path `/build/stimfit-release-current-sid`. Preserve results under
  `deb/0.17.4-1/current-sid/`; do not substitute the older snapshot results for
  this check. Debian's Python is used in the testbed because the host's default
  virtual environment is unavailable there and the package must match the
  distribution ABI.
- Checked the package tracker on 2026-11-07: unstable and testing still list
  `0.17.2-1`. The tracker reports an autoremoval date of 2026-12-06 due to
  transitive dependency bugs in libiodbc2 (#1102465) and petsc (#1152614).
  Those are external migration risks, not Stimfit build-test failures.
- The current unstable build passed all three CTest suites with HDF5
  `2.2.0+repack-6`. Source rebuilding after cleanup also passed. Explicitly
  unapplied quilt patches with `dpkg-source --after-build --unapply-patches .`
  before comparing all 614 upstream files; none differed from the orig archive.
- Current-unstable upgrade testing exposed an import-test environment issue:
  Matplotlib emitted a cache warning because the schroot user's home directory
  did not exist. The import itself succeeded, but strict stderr checking
  correctly failed the test. Updated `python3-stfio-import` to use a private
  temporary `MPLCONFIGDIR`, removed on exit; do not add `allow-stderr`.
- Recorded the orig archive in pristine-tar commit `032c7720` using tools
  extracted under `/tmp/stimfit-release-tools` without modifying host packages.
  Regenerated the archive from Git and verified byte-for-byte equality.

- Current unstable upgrade validation passed: the installed packages upgraded
  from `0.17.2-1` to `0.17.4-1`; the help test, all 17 stfio unit tests, and
  removal/reinstallation checks passed. The initial import smoke-test failure
  is retained in `current-sid/autopkgtest/`; the corrected test passes with
  strict stderr checking in `current-sid/autopkgtest-import-fixed/`. The final runner invocation is
  recorded separately below. No application code changed.
- Regenerated the final source package in the validated current Debian build
  environment with the private Matplotlib cache test fix. Source and binary
  lintian pass. The final source upload is at the artifact directory root;
  older binary compatibility builds and their metadata remain in subdirectories.
- The maintainer key `37D2370A9F7AC0CC` is available. Unattended signing with
  the GPG agent returned `No pinentry`, so local key interaction is needed.
  The maintainer resolved this by signing locally with:

  ```sh
  debsign -k37D2370A9F7AC0CC /home/c02380/stimfit/deb/0.17.4-1/stimfit_0.17.4-1_source.changes
  ```

  Never request or transmit the passphrase in chat. Once signing succeeds,
  verify the signatures and referenced checksums, refresh `SHA256SUMS`, and
  upload through the prepared HTTPS configuration:

  ```sh
  source /home/c02380/virtualenv-3.14/bin/activate
  PYTHONPATH=/usr/lib/python3/dist-packages python /usr/bin/dput \
    -c deb/0.17.4-1/dput-mentors.cf -o mentors \
    deb/0.17.4-1/stimfit_0.17.4-1_source.changes
  PYTHONPATH=/usr/lib/python3/dist-packages python /usr/bin/dput \
    -c deb/0.17.4-1/dput-mentors.cf mentors \
    deb/0.17.4-1/stimfit_0.17.4-1_source.changes
  ```

  `-o` performs local checks without uploading. The host default virtual
  environment lacks python-debian, so exposing the system dist-packages keeps
  that interpreter usable for autopkgtest and dput. Testbed Python remains the
  Debian interpreter. The draft `sponsorship-request.txt` is prepared for review;
  sending it requires an explicit instruction and has not been done.
- Published the final import-test fix and current-unstable validation record
  on `debian/sid`, together with the pristine-tar data and annotated release
  tags. `upstream/0.17.4` points to the frozen upstream source `0ea89fe9`;
  `debian/0.17.4-1` marks the matching Debian packaging and this procedure.
  These Git updates do not upload the package to Debian or mentors.

- Corrected the host runner invocation to preserve autopkgtest's shebang flag:
  use `python -u /usr/bin/autopkgtest`, not `python /usr/bin/autopkgtest`.
  Omitting `-u` left buffered output when autopkgtest closed its logging file
  descriptors, producing exit 120 during Python shutdown even after the test
  summary was written. Retained those logs and reran the corrected import
  test in `current-sid/autopkgtest-import-final/` with unbuffered output.
  No package change or release-tag replacement is required for this runner fix.

- The maintainer signed the final source upload locally. Verified good signatures
  on the `.dsc`, source `.buildinfo`, and source `.changes` using the existing
  maintainer key. Verified the signed upload's SHA256 references and refreshed
  `SHA256SUMS`; `dput -o` passed its local signature and package checks.

- The final unbuffered import check passed with exit status 8, the expected
  result for a run containing only a superficial test. Source parsing, the
  upgrade path, and strict stderr checks all succeeded. The earlier exit-120
  runner logs remain available; they are not represented as successful process
  exits. The retained build session was closed after saving its artifacts.
- Published release packaging commit `e0760007`, pristine-tar commit `032c7720`,
  and annotated tags `upstream/0.17.4` and `debian/0.17.4-1`. Later procedure-only
  updates do not move the release tags or alter the signed source upload.
- Uploaded the signed source set over HTTPS using the prepared mentors target;
  dput returned success for the `.dsc`, orig archive, Debian archive, source
  `.buildinfo`, and source `.changes`. Saved `mentors-upload.log` and the dput
  upload receipt. The package page was not yet populated immediately after
  transfer, so mentors processing/acceptance remains unconfirmed. No sponsor
  email has been sent; `sponsorship-request.txt` remains a draft for review.
