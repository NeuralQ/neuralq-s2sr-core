"""Checkpoint resolution from the private NeuralQ S2SR Hugging Face repo.

The checkpoint is no longer shipped as a tracked file in ``models/``; it is
downloaded (and cached by ``huggingface_hub``) from a private Hub repo on
first use. Requires a Hugging Face token with read access, via ``HF_TOKEN``
(or ``HUGGING_FACE_HUB_TOKEN``).
"""
from __future__ import annotations

import os
from pathlib import Path

HF_REPO_ID = "Khlaifiabilel/neuralq-s2sr-core"
HF_FILENAME = "s2sr-v3.0.0.pt"
MODEL_ID = Path(HF_FILENAME).stem


def _token() -> str:
    token = os.environ.get("HF_TOKEN") or os.environ.get("HUGGING_FACE_HUB_TOKEN")
    if not token:
        raise RuntimeError(
            f"{HF_REPO_ID!r} is a private Hugging Face repo; set HF_TOKEN "
            "to a Hugging Face access token with read access to download the checkpoint."
        )
    return token


def resolve_checkpoint(
    filename: str = HF_FILENAME,
    *,
    repo_id: str = HF_REPO_ID,
    revision: str | None = None,
) -> Path:
    """Download (or reuse the cached copy of) the checkpoint from the private HF repo."""
    from huggingface_hub import hf_hub_download

    path = hf_hub_download(
        repo_id=repo_id,
        filename=filename,
        revision=revision,
        token=_token(),
    )
    return Path(path)


def cached_checkpoint_path(
    filename: str = HF_FILENAME, *, repo_id: str = HF_REPO_ID
) -> Path | None:
    """Return the already-cached checkpoint path without triggering a download."""
    from huggingface_hub import hf_hub_download
    from huggingface_hub.utils import LocalEntryNotFoundError

    try:
        path = hf_hub_download(
            repo_id=repo_id,
            filename=filename,
            local_files_only=True,
        )
    except (LocalEntryNotFoundError, OSError):
        return None
    return Path(path)
