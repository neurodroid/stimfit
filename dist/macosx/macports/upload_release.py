#!/usr/bin/env python3
"""Prepare a checked MacPorts SFTP batch; upload only with --upload.

Run from any directory. VERSION, the generated Portfile, and the frozen archive
under build/release must agree. Uses the existing SSH configuration and keys.
"""

import argparse
import hashlib
from pathlib import Path
import re
import shlex
import subprocess
import sys
import uuid


DESTINATION = "p8210991@ftp.schmidt-hieber.de"
REPOSITORY = Path(__file__).resolve().parents[3]
SFTP = ["sftp", "-oBatchMode=yes", "-oStrictHostKeyChecking=yes", "-oConnectTimeout=15"]


def quoted(value):
    if "\n" in value or "\r" in value:
        raise ValueError("SFTP paths must not contain newlines")
    return '"' + value.replace("\\", "\\\\").replace('"', '\\"') + '"'


def field(text, pattern, label):
    matches = re.findall(pattern, text, re.MULTILINE)
    if len(matches) != 1:
        raise ValueError(f"Expected exactly one {label} in the generated Portfile")
    return matches[0]


def prepare(repository):
    version = (repository / "VERSION").read_text().strip()
    if not re.fullmatch(r"[0-9]+(?:\.[0-9]+){1,3}", version):
        raise ValueError(f"Unexpected release version: {version!r}")
    port_directory = repository / "dist/macosx/macports"
    portfile = (port_directory / "science/stimfit/Portfile").read_text()
    if field(portfile, r"^version\s+(\S+)\s*$", "version") != version:
        raise ValueError("VERSION and the generated Portfile version disagree")

    release = repository / "build/release"
    filename = f"stimfit-{version}.tar.gz"
    archive = release / filename
    expected = {
        "ripemd160": field(portfile, r"\brmd160\s+([0-9a-fA-F]{40})\b", "RIPEMD160"),
        "sha256": field(portfile, r"\bsha256\s+([0-9a-fA-F]{64})\b", "SHA256"),
    }
    size = int(field(portfile, r"\bsize\s+([0-9]+)\b", "archive size"))
    if archive.stat().st_size != size:
        raise ValueError(f"Archive size does not match the Portfile: {archive}")
    digests = {name: hashlib.new(name) for name in expected}
    with archive.open("rb") as source:
        for block in iter(lambda: source.read(1024 * 1024), b""):
            for digest in digests.values():
                digest.update(block)
    for name, value in expected.items():
        if digests[name].hexdigest() != value.lower():
            raise ValueError(f"Archive {name} does not match the Portfile")

    template = (port_directory / "upload_stimfit.in").read_text().splitlines()
    if len(template) != 2 or not re.fullmatch(r"cd [A-Za-z0-9_./-]+", template[0]):
        raise ValueError("Upload template changed; review its remote commands first")
    if template[1] != "put stimfit-STFVERSION.tar.gz":
        raise ValueError("Upload template changed; review its filename first")
    remote_directory = template[0][3:]
    temporary = f".{filename}.upload-{uuid.uuid4().hex}"
    batch = (
        f"lcd {quoted(str(release))}\n"
        f"cd {quoted(remote_directory)}\n"
        f"put {quoted(filename)} {quoted(temporary)}\n"
        f"chmod 644 {quoted(temporary)}\n"
        # Default OpenSSH rename uses POSIX replacement semantics. Legacy
        # SFTP rename must fail when the destination already exists.
        f"rename -l {quoted(temporary)} {quoted(filename)}\n"
    )
    batch_path = release / f"upload-stimfit-{version}.sftp"
    batch_path.write_text(batch)
    return batch_path, remote_directory, filename, temporary


def check_remote(directory, filename):
    """Refuse an existing release or an inconclusive directory check."""
    result = subprocess.run(
        SFTP + ["-b", "-", DESTINATION],
        input=f"cd {quoted(directory)}\nls -l {quoted(filename)}\n",
        text=True, capture_output=True,
    )
    if result.returncode == 0:
        raise ValueError(f"Remote release already exists; refusing to replace {filename}")
    missing = r'^Can\'t ls: "(?:[^"\r\n]*/)?' + re.escape(filename) + r'" not found\r?$'
    if result.returncode != 1 or not re.search(missing, result.stderr, re.MULTILINE):
        raise ValueError("Could not establish that the remote release is absent:\n" + result.stderr.strip())


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    action = parser.add_mutually_exclusive_group()
    action.add_argument("--upload", action="store_true", help="upload the checked archive")
    action.add_argument("--dry-run", action="store_true", help="prepare the batch without connecting (default)")
    args = parser.parse_args()
    try:
        batch, directory, filename, temporary = prepare(REPOSITORY)
        command = SFTP + ["-b", str(batch), DESTINATION]
        print(f"Archive version, size, RIPEMD160, and SHA256 verified. Batch: {batch}")
        if not args.upload:
            print(batch.read_text(), end="")
            print("Dry run: no network connection made.")
            print("To upload: " + shlex.join([sys.executable, str(Path(__file__).resolve()), "--upload"]))
            return 0
        check_remote(directory, filename)
        try:
            subprocess.run(command, check=True)
        except subprocess.CalledProcessError:
            print(f"Upload failed; a temporary remote file may remain: {temporary}", file=sys.stderr)
            raise
        print("Uploaded. Verify the public download against the Portfile before submitting the PR.")
        return 0
    except (OSError, ValueError, subprocess.CalledProcessError) as error:
        print(f"Error: {error}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    sys.exit(main())
