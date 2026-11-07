# Debian release procedure for Stimfit

Use this procedure to prepare, validate, and publish Stimfit packages for Debian
unstable. Keep it applicable to future releases. Save release-specific commit
IDs, dependency versions, failures, results, and upload receipts with the release
artifacts under `deb/<package-version>/`, rather than adding a preparation diary
here. That directory is ignored by Git.

## 1. Select the source and packaging branches

Inspect `git status`, `VERSION`, and [BRANCHES.md](../BRANCHES.md). Preserve local
changes and fetch the remote branches before integrating work.

- `master` contains the current upstream source.
- `debian/sid` contains the corresponding Debian packaging in `debian/`.
- `pristine-tar` records data needed to reproduce the exact orig archives.
- `0.16` and `debian/sid-0.16` are reserved for legacy maintenance.

Choose the upstream release commit explicitly; the latest `master` commit may
contain work for a later release. On `debian/sid`, integrate that selected source
and any remote packaging changes without rewriting published history. Review
`debian/patches/series` and remove patches already incorporated in the selected
upstream release before applying the remaining series.

For a new upstream version, ensure `VERSION` and the intended source agree. For
a Debian revision of an existing upstream version, reuse its existing orig
archive from Debian or `pristine-tar`; do not recreate it from a newer checkout.

## 2. Review reports and write the changelog

Read the [Debian package tracker](https://tracker.debian.org/pkg/stimfit), the
[source package bug list](https://bugs.debian.org/src:stimfit), and relevant
archived reports. Check current dependency transitions and migration issues.
Separate Stimfit regressions from problems in other packages.

Keep these historical failures covered:

| Report | Regression check |
| --- | --- |
| [1139613](https://bugs.debian.org/1139613) | Build against real Debian HDF5 packages across supported major versions. Check both header detection and namespaced or unnamespaced CMake targets; a successful host build alone is insufficient. |
| [1140597](https://bugs.debian.org/1140597) | Run installed help commands under Xvfb. Require successful exit status and usage output, and require invalid options to fail. Never hide failures with `\|\| true`. |
| [1047146](https://bugs.debian.org/1047146) | Build a source package after a successful binary build and cleanup, and check for unexpected upstream changes or deletions. |

Use one changelog entry for the release being prepared. Fold relevant notes from
earlier `UNRELEASED` entries into it and remove the superseded draft entries;
Git history preserves the development record. Leave previously released entries
unchanged.

Lead with user-facing changes such as file opening, import reliability,
plotting, and Python features. Omit platform-specific changes that do not affect
Debian. Summarize test and packaging maintenance in one sentence. Keep detailed
validation results with the artifacts. Add `Closes` only for bugs actually fixed
by this upload, and do not repeat closures from earlier releases.

## 3. Stage a clean source package

Build from an extracted source tree outside the working checkout. Populate it
from the selected upstream commit and copy only `debian/` from the packaging
checkout. This excludes local build output and repository release bookkeeping.

The following example creates a **new** orig archive for a new upstream version.
Replace `SELECTED_UPSTREAM_COMMIT` with the reviewed commit, and run from
`debian/sid` after updating its changelog:

```sh
repo_root=$(git rev-parse --show-toplevel)
upstream_version=$(cat VERSION)
package_version=$(dpkg-parsechangelog -S Version)
upstream_commit=SELECTED_UPSTREAM_COMMIT
release_dir="$repo_root/deb/$package_version"
stage_dir=$(mktemp -d /tmp/stimfit-debian.XXXXXX)

mkdir -p "$release_dir"
git archive --format=tar.gz --prefix="stimfit-$upstream_version/" \
  "$upstream_commit" -o "$release_dir/stimfit_${upstream_version}.orig.tar.gz"
cp "$release_dir/stimfit_${upstream_version}.orig.tar.gz" "$stage_dir/"
tar -xzf "$stage_dir/stimfit_${upstream_version}.orig.tar.gz" -C "$stage_dir"
cp -a debian "$stage_dir/stimfit-$upstream_version/"
cd "$stage_dir/stimfit-$upstream_version"
dpkg-buildpackage -S -nc -sa -us -uc -d
cp "$stage_dir/stimfit_${package_version}.dsc" \
  "$stage_dir/stimfit_${package_version}.debian.tar."* \
  "$stage_dir/stimfit_${package_version}_source.buildinfo" \
  "$stage_dir/stimfit_${package_version}_source.changes" "$release_dir/"
```

For an existing upstream version, replace archive creation with retrieval of
its exact orig archive. Check that the package version matches the selected
upstream version and Debian revision before building.

Use `-nc` only for a freshly extracted, unbuilt tree. Subsequent cleanup checks
must omit it. Write logs outside the extracted source root: a log created there
becomes an unexpected upstream modification. Run `dpkg-source -b` from the
staging parent, or `dpkg-buildpackage` inside the source tree, so outputs land
beside the orig archive.

The Debian rules use `STF_BIOSIG_PROVIDER=SYSTEM`. A parent `git archive` omits
submodule contents; an archive prepared for this Debian system-provider build
is unsuitable for builds that require bundled BioSig. Recheck this assumption
if the packaging changes provider.

Document downstream source changes as DEP-3 quilt patches under
`debian/patches/`, listed in `series`. Preserve the orig archive and its checksum
once selected. Never modify it to carry a Debian-only fix.

## 4. Build in clean Debian environments

Build the source package against **current Debian unstable**, updating package
indexes and upgrading the disposable build environment. Retain older dependency
environments for compatibility coverage, particularly across HDF5 major versions;
label snapshot results separately from current-archive results.

For existing schroot-backed environments, select the backend explicitly if the
host defaults to a different sbuild mode. For example, with `build_chroot` set
to the chosen chroot:

```sh
sbuild --chroot-mode=schroot -c "$build_chroot" -d unstable \
  --apt-update --apt-distupgrade \
  --build-path="/build/stimfit-${package_version}-current" \
  --purge-build=never --purge-deps=never --purge-session=never \
  "$release_dir/stimfit_${package_version}.dsc"
```

Run each build from a separate output directory. Chroots may share `/build`
through bind mounts: concurrent builds need distinct build paths as well as
separate artifact directories. Save exact dependency versions and complete logs.
Retain sessions while investigating failures, then end only the sessions created
for this preparation after saving their artifacts.

Follow the host's configured Python environment for host-side commands. Debian
builds and installed-package tests need the distribution interpreter to match
its ABI; explain any necessary exception before switching environments.

Review CTest results and `dh_missing` output. Do not set `nocheck`, disable
assertions, or suppress failed commands to obtain a passing release build.
Check both `stimfit` and `python3-stfio`, including the embedded `_stf` extension,
generated `stf.py`, toolbar assets, and test-script executable permissions.
Exclude upstream test executables through `debian/not-installed` rather than
shipping them in the application package.

## 5. Test the installed packages and upgrade path

Use fresh matching Debian testbeds and the binaries just built. With
`binary_dir` and `test_chroot` set for the selected build:

```sh
autopkgtest "$binary_dir/stimfit_${package_version}_amd64.deb" \
  "$binary_dir/python3-stfio_${package_version}_amd64.deb" \
  "$stage_dir/stimfit-$upstream_version" \
  --output-dir="$binary_dir/autopkgtest" \
  -- schroot "$test_chroot"
```

Replace the architecture suffix when testing another architecture. Use a fresh
output directory for each run. Do not use the null backend on a development
host: it tests and may modify the host.

Require the following checks:

- Both `stimfit -h` and `stimfit --help` run under Xvfb, return zero, and print
  usage. An invalid option returns nonzero with an option diagnostic. Preserve
  `xvfb` and `xauth` test dependencies.
- The installed `stfio` module imports. Give Matplotlib a writable private cache
  directory so tests do not depend on the test user's home directory existing;
  keep stderr checking enabled.
- `src/pystfio/unittest_stfio.py` runs against the installed module with its HDF5
  fixture present, exercising reading and NumPy conversion. Do not count a
  missing-fixture skip as passing coverage.
- Packages upgrade from the previous released Debian version, then survive
  removal and reinstallation. Perform these operations only in disposable
  testbeds. Save the installed versions before and after the upgrade.

Inspect both test summaries and process exit status. A run containing only tests
marked `superficial` can return 8 despite passing tests; record that explicitly.
Import checks alone do not establish substantive NumPy or Matplotlib coverage.

When invoking a system Python tool through the configured host interpreter,
provide its required modules and preserve its shebang options. In particular,
autopkgtest needs unbuffered output: use `python -u /usr/bin/autopkgtest`.
Omitting `-u` can cause exit 120 during buffered log cleanup. Diagnose such runner
failures and rerun affected checks; do not treat an unexplained exit as success.

## 6. Verify cleanup and lintian

After a successful binary build, run this inside its retained build session and
source directory:

```sh
dpkg-buildpackage -S -sa -us -uc
```

Require successful cleanup and source creation without unexpected upstream
changes or ignored source deletions. Unapply quilt patches before comparing the
cleaned upstream tree with the orig archive. If necessary, use:

```sh
dpkg-source --after-build --unapply-patches .
```

Run lintian on the final source and binary upload metadata:

```sh
lintian --profile debian "$release_dir/stimfit_${package_version}_source.changes" \
  "$binary_dir/stimfit_${package_version}_amd64.changes"
```

The explicit Debian profile matters on hosts whose default profile is Ubuntu.
If a sandboxed run reports a subprocess failure, investigate it in a suitable
environment. Do not add a package override for a host or sandbox limitation.

Commit the final packaging changes. Confirm that the packaged `debian/` files
match that commit and that each upload checksum matches its referenced file.
After changes, regenerate affected source artifacts and repeat affected checks;
do not sign or upload stale metadata.

## 7. Freeze archives and tags

Save the orig archive, `.dsc`, `.debian.tar.*`, source `.changes` and `.buildinfo`,
binary packages and metadata, validation logs, and checksums under
`deb/<package-version>/`. Keep different compatibility builds in separate
subdirectories because their package filenames may be identical.

Record a new orig archive with `pristine-tar commit <archive> <upstream-commit>`.
Regenerate it with `pristine-tar checkout <output-path>` and verify byte-for-byte
equality. Use the branch and tag conventions in `debian/gbp.conf`:

- `upstream/<upstream-version>` identifies the exact upstream source commit.
- `debian/<package-version>` identifies the validated Debian packaging commit.

Once publication is authorized, create missing tags and push the reviewed
branches, tags, and pristine-tar data. Verify existing tags instead of replacing
them. Preserve the orig archive, uploaded source set, and published release tags
when making subsequent documentation or development changes.

## 8. Sign and upload the source package

Use the maintainer's registered signing key. With `signing_key` set to that key:

```sh
debsign -k"$signing_key" "$release_dir/stimfit_${package_version}_source.changes"
```

If unattended signing needs local key interaction, have the maintainer sign in
their own terminal. Keep passphrases out of chat and logs. Verify signatures on
the `.dsc`, source `.buildinfo`, and source `.changes`. Recheck referenced
checksums and refresh the checksum manifest after signing, since signatures
change file contents.

For sponsored uploads, configure a mentors target using the
[mentors upload instructions](https://mentors.debian.net/intro-maintainers/),
set `dput_config` to that configuration file, then check and upload the signed
source set:

```sh
dput -c "$dput_config" -o mentors \
  "$release_dir/stimfit_${package_version}_source.changes"
dput -c "$dput_config" mentors \
  "$release_dir/stimfit_${package_version}_source.changes"
```

`-o` checks locally without uploading. Use the source-only `.changes`; local
binary compatibility builds are validation artifacts, not multiple variants to
upload together. Include the orig archive when the upstream version is new.

A successful transfer does not establish processing or acceptance. Confirm the
version on the mentors package page or in its processing email, and retain the
upload receipt. Prepare a sponsorship request with the user-facing changes,
package and Git links, and validation results. Obtain explicit authorization
before sending messages to sponsors. After the sponsored Debian upload, monitor
archive acceptance, architecture builds, autopkgtests, and migration to testing.

## 9. Carry shared fixes upstream

Keep `debian/` on `debian/sid`. Carry generic source fixes and reusable procedures
to `master` as focused commits rather than merging the entire packaging tree.
Documentation updates need not be incorporated into an already prepared release.

An upstream fix corresponding to a quilt patch can be committed to `master`
while a frozen Debian release retains that patch against its original archive.
For the next upstream release, remove patches already incorporated upstream
before importing the updated source. Avoid applying the same correction twice.
Do not regenerate an uploaded orig archive or move release tags to include later
upstream work.
