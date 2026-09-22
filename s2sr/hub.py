"""Local checkpoint resolution from the bundled ``models/`` directory.

The ``s2sr-v3.0.0.pt`` weights ship inside the repository / container image
under ``models/``. No network access, no Hugging Face account, and no
``HF_TOKEN`` is required. ``resolve_checkpoint()`` simply locates the file
on disk.
"""
from __future__ import annotations

import hashlib
from pathlib import Path

MODEL_ID = "s2sr-v3.0.0"
MODEL_FILENAME = f"{MODEL_ID}.pt"

# ``s2sr/hub.py`` lives at ``<repo>/s2sr/hub.py`` so parents[1] is the repo root.
MODELS_DIR = Path(__file__).resolve().parents[1] / "models"


def _candidate_paths(filename: str | Path) -> list[Path]:
    name = Path(filename)
    # Absolute or explicit relative path that already exists wins.
    if name.is_absolute() or (name.parent != Path(".") and name.exists()):
        return [name]
    return [
        MODELS_DIR / name.name,
        Path.cwd() / "models" / name.name,
        Path(name),
    ]


def resolve_checkpoint(filename: str | Path = MODEL_FILENAME) -> Path:
    """Return the local checkpoint path, raising if it is missing."""
    for candidate in _candidate_paths(filename):
        if candidate.is_file():
            return candidate.resolve()
    searched = ", ".join(str(p) for p in _candidate_paths(filename))
    raise FileNotFoundError(
        f"Checkpoint {Path(filename).name!r} not found. "
        f"Place it at {MODELS_DIR / Path(filename).name} (baked into the "
        f"Docker image) or pass --model /path/to/checkpoint.pt. "
        f"Searched: {searched}"
    )


def cached_checkpoint_path(
    filename: str | Path = MODEL_FILENAME,
) -> Path | None:
    """Return the checkpoint path if present, else None (no download)."""
    try:
        return resolve_checkpoint(filename)
    except FileNotFoundError:
        return None


def checksum_path(filename: str | Path = MODEL_FILENAME) -> Path:
    """Sidecar ``<name>.pt.sha256`` next to the weights (coreutils format)."""
    return MODELS_DIR / f"{Path(filename).name}.sha256"


def verify_checkpoint(path: str | Path, *, filename: str | Path | None = None) -> Path:
    """Verify the sha256 of ``path`` against its pinned sidecar.

    Bundled weights (anything under ``MODELS_DIR``) are enforced against
    ``models/<name>.pt.sha256``. Custom ``--model`` files are checked against
    a sibling ``<file>.sha256`` only when one exists, otherwise they pass
    through unverified. A present-but-mismatched sidecar raises
    ``ValueError`` so a truncated file fails fast instead of mid-inference.
    """
    resolved = Path(path)
    try:
        resolved.relative_to(MODELS_DIR)
        sidecar = checksum_path(filename if filename is not None else resolved.name)
    except ValueError:
        sidecar = Path(f"{resolved}.sha256")
    if not sidecar.is_file():
        return resolved
    expected = sidecar.read_text(encoding="utf-8").split()[0].lower()
    digest = hashlib.sha256()
    with resolved.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    if digest.hexdigest() != expected:
        raise ValueError(
            f"Checksum mismatch for {resolved}: expected {expected}, "
            f"got {digest.hexdigest()}. The weights file is corrupt or "
            f"not {MODEL_ID}; re-acquire it and retry."
        )
    return resolved
