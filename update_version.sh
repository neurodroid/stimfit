#! /bin/bash

set -euo pipefail

if [ "$#" -ne 1 ]; then
    echo "Usage: $0 <new-version>"
    echo "Example: $0 0.16.12"
    exit 1
fi

NEW_VERSION="$1"
VERSION_FILE="VERSION"

if [ ! -f "${VERSION_FILE}" ]; then
    echo "Error: ${VERSION_FILE} not found"
    exit 1
fi

OLD_VERSION="$(tr -d '\r\n' < "${VERSION_FILE}")"
printf '%s\n' "${NEW_VERSION}" > "${VERSION_FILE}"

echo "Updated ${VERSION_FILE}: ${OLD_VERSION} -> ${NEW_VERSION}"
echo "Next steps:"
echo "  1. Re-run ./autogen.sh before using Autoconf artifacts"
echo "  2. Re-run CMake configure so PROJECT_VERSION is refreshed"
