#!/usr/bin/env bash
set -euo pipefail

VERSION=0.16.11

# cowbuilder uses a base *directory* (COW base), not a .tgz
BASE_COW="${HOME}/pbuilder/sid-base.cow"

TOPDIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
DEBDIR="${TOPDIR}/../deb"
RESULTDIR="${DEBDIR}/result"

# Persist package downloads across runs
APT_CACHE_DIR="${HOME}/pbuilder/aptcache"

# Optional: set this in your environment, e.g.
#   export DEBSIGN_KEYID=0xYOURKEYID
MAINT_EMAIL="christsc@gmx.de"

DEBSIGN_KEYID="$(
  gpg --list-secret-keys --with-colons --fingerprint 2>/dev/null \
  | awk -F: -v email="$MAINT_EMAIL" '
      $1=="sec"              { fpr="" }
      $1=="fpr" && fpr==""   { fpr=$10 }
      $1=="uid" && $10 ~ email && fpr != "" { print fpr; exit }
    '
)"

make dist

mkdir -p "${DEBDIR}"
rm -rf "${DEBDIR:?}/"*
mkdir -p "${RESULTDIR}"

# Prepare orig tarball
cp -v "stimfit-${VERSION}.tar.gz" "${DEBDIR}/"
cp -v "stimfit-${VERSION}.tar.gz" "${DEBDIR}/stimfit_${VERSION}.orig.tar.gz"

# Unpack + add debian/
cd "${DEBDIR}"
tar -xzf "stimfit_${VERSION}.orig.tar.gz"
cd "stimfit-${VERSION}"

cp -rv "${TOPDIR}/../../dist/debian" ./
test -d debian

# ---------------------------------------------------------------------------
# Build the source package inside the sid cowbuilder chroot (UNSIGNED)
# ---------------------------------------------------------------------------

sudo cowbuilder execute \
  --basepath "${BASE_COW}" \
  --aptcache "${APT_CACHE_DIR}" \
  --bindmounts "${DEBDIR}" \
  -- /bin/bash -lc "
    set -euo pipefail
    apt-get update
    apt-get -y build-dep '${DEBDIR}/stimfit-${VERSION}'
    cd '${DEBDIR}/stimfit-${VERSION}'
    # Build source package WITHOUT signing; we'll sign on the host.
    dpkg-buildpackage -S -sa -us -uc -d -nc
  "

# cowbuilder execute ran as root inside chroot; artifacts may be root-owned on host.
sudo chown -R "$(id -u):$(id -g)" "${DEBDIR}" "${RESULTDIR}"

# Find the newly created source .changes / .dsc (should be in ${DEBDIR})
CHANGES="$(ls -1t "${DEBDIR}"/*_source.changes | head -n1)"
DSC="$(ls -1t "${DEBDIR}"/*.dsc | head -n1)"

echo "Source package: ${DSC}"
echo "Source changes: ${CHANGES}"

# ---------------------------------------------------------------------------
# Sign the source package on the host (recommended; keeps private key out of chroot)
# ---------------------------------------------------------------------------

SIGN_USER="${SUDO_USER:-$USER}"

if [[ -z "${DEBSIGN_KEYID}" ]]; then
  echo "ERROR: Could not find a GPG secret key for ${MAINT_EMAIL}" >&2
  exit 1
fi

sudo -u "${SIGN_USER}" debsign -k"${DEBSIGN_KEYID}" "${CHANGES}"

# Verify the signature was actually applied — debsign can exit 0 even on
# certain failures, so we check explicitly.
if ! sudo -u "${SIGN_USER}" gpg --batch --status-fd 1 --verify "${CHANGES}" 2>/dev/null \
     | grep -q '\[GNUPG:\] GOODSIG'; then
  echo "ERROR: Signature verification failed for ${CHANGES}" >&2
  echo "       Run 'gpg --verify ${CHANGES}' for details." >&2
  exit 1
fi
echo "==> Source package signed and verified OK."

# Optional: copy signed source artifacts into RESULTDIR for convenient dput from there
# (The .changes references the files by basename; having them together is convenient.)
cp -v "${DEBDIR}/"stimfit_"${VERSION}"-*.dsc "${RESULTDIR}/" 2>/dev/null || true
cp -v "${DEBDIR}/"stimfit_"${VERSION}"-*_source.changes "${RESULTDIR}/" 2>/dev/null || true
cp -v "${DEBDIR}/"stimfit_"${VERSION}"-*.debian.tar.* "${RESULTDIR}/" 2>/dev/null || true
cp -v "${DEBDIR}/"stimfit_"${VERSION}"-*.orig.tar.* "${RESULTDIR}/" 2>/dev/null || true

# ---------------------------------------------------------------------------
# Now build binaries inside cowbuilder (installs Build-Depends as needed)
# ---------------------------------------------------------------------------

sudo cowbuilder build \
  --basepath "${BASE_COW}" \
  --buildresult "${RESULTDIR}" \
  "${DSC}"

sudo chown -R "$(id -u):$(id -g)" "${RESULTDIR}"

echo "Done. Artifacts in: ${RESULTDIR}"
echo "Upload with: dput mentors $(basename "${CHANGES}")  (from ${RESULTDIR} if you used the copy step)"
