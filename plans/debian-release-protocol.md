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
| Source build after binary build and clean | Passed in the HDF5 2.1 environment |
| Clean upstream tree | All 614 orig archive files match after clean |
| Lintian | Passed for source and both binary sets; no warnings outside sandbox |
| Git publication | Release changes and merged histories published on `origin/debian/sid` |
| Signing and package upload | Pending; no upload or release tags |

The build environments use dated Debian unstable snapshots, rather than the
current archive. Record that scope when reporting results.

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
  remote history, retaining both changelog entries with `0.17.4-1` first.
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
