#!/usr/bin/env python3
"""Fetch, verify, and decrypt the pretrained S2SR checkpoint.

Downloads the encrypted weights object (resumable, size- and MD5-checked),
retrieves the RSA key archive (SHA-256-checked, password-protected ZIP),
decrypts with OpenSSL SMIME, and gates the plaintext on a pinned SHA-256.
Existing verified checkpoints short-circuit the process.

Usage::

    python scripts/download_weights.py [--model-dir PATH] [--force]
"""
import argparse
import hashlib
import os
from pathlib import Path
import subprocess
import tempfile
import zipfile

import requests

from upstream import (
    UPSTREAM_WEIGHTS_ID,
    WEIGHTS_BUCKET,
    KEY_ARCHIVE_URL,
    KEY_ARCHIVE_SHA256,
    KEY_ARCHIVE_PASSWORD,
    KEY_MEMBER,
)


MODEL_NAME = "S2SR-GL-20241022.1"
MODEL_URL = f"{WEIGHTS_BUCKET}/{UPSTREAM_WEIGHTS_ID}"
MODEL_SIZE = 840_950_890
MODEL_MD5 = "c5819380a26f978ff15d8385b27a7b50"
MODEL_SHA256 = "1ac3d52cac3737842538ed09f329b0023b43cd3d5f509ccce36a0951cb2dd520"


def digest(path: Path, algorithm: str) -> str:
    hasher = hashlib.new(algorithm)
    with path.open("rb") as file:
        for chunk in iter(lambda: file.read(8 * 1024 * 1024), b""):
            hasher.update(chunk)
    return hasher.hexdigest()


def download(url: str, destination: Path) -> None:
    partial = destination.with_name(destination.name + ".part")
    offset = partial.stat().st_size if partial.exists() else 0
    headers = {"Range": f"bytes={offset}-"} if offset else {}

    with requests.get(url, headers=headers, stream=True, timeout=120) as response:
        response.raise_for_status()
        append = offset > 0 and response.status_code == 206
        mode = "ab" if append else "wb"
        if not append:
            offset = 0
        total = int(response.headers.get("Content-Length", 0)) + offset
        written = offset
        next_report = written
        with partial.open(mode) as file:
            for chunk in response.iter_content(8 * 1024 * 1024):
                if not chunk:
                    continue
                file.write(chunk)
                written += len(chunk)
                if written >= next_report:
                    print(f"Downloaded {written / 1_000_000:.0f}/{total / 1_000_000:.0f} MB")
                    next_report = written + 64 * 1024 * 1024
    os.replace(partial, destination)


def decrypt(encrypted: Path, destination: Path) -> None:
    with tempfile.TemporaryDirectory(prefix="s2sr-") as temp_dir_name:
        temp_dir = Path(temp_dir_name)
        key_archive = temp_dir / "key.zip"
        download(KEY_ARCHIVE_URL, key_archive)
        if digest(key_archive, "sha256") != KEY_ARCHIVE_SHA256:
            raise RuntimeError("Downloaded key archive failed SHA-256 verification")

        with zipfile.ZipFile(key_archive) as archive:
            private_key = archive.read(KEY_MEMBER, pwd=KEY_ARCHIVE_PASSWORD)

        key_path = temp_dir / "private-key.pem"
        key_path.write_bytes(private_key)
        key_path.chmod(0o600)

        partial = destination.with_name(destination.name + ".part")
        subprocess.run(
            [
                "openssl",
                "smime",
                "-decrypt",
                "-in",
                str(encrypted),
                "-binary",
                "-inform",
                "DER",
                "-inkey",
                str(key_path),
                "-out",
                str(partial),
            ],
            check=True,
        )
        if digest(partial, "sha256") != MODEL_SHA256:
            raise RuntimeError("Decrypted checkpoint failed SHA-256 verification")
        os.replace(partial, destination)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Download and decrypt the pretrained S2SR checkpoint."
    )
    parser.add_argument(
        "--model-dir",
        type=Path,
        default=Path(__file__).resolve().parents[1] / "models",
    )
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()

    args.model_dir.mkdir(parents=True, exist_ok=True)
    encrypted = args.model_dir / f"{MODEL_NAME}.cms"
    checkpoint = args.model_dir / f"{MODEL_NAME}.pt"

    if checkpoint.exists() and not args.force:
        if digest(checkpoint, "sha256") == MODEL_SHA256:
            print(f"Checkpoint already verified: {checkpoint}")
            return
        raise RuntimeError(f"Existing checkpoint has the wrong hash: {checkpoint}")

    if not encrypted.exists() or args.force:
        download(MODEL_URL, encrypted)
    if encrypted.stat().st_size != MODEL_SIZE or digest(encrypted, "md5") != MODEL_MD5:
        raise RuntimeError("Encrypted model failed size or MD5 verification")

    decrypt(encrypted, checkpoint)
    print(f"Checkpoint ready: {checkpoint}")


if __name__ == "__main__":
    main()
