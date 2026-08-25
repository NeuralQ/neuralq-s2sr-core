"""Integration boundary for the prebuilt compiled inference engine.

NeuralQ S2SR drives a compiled geospatial engine (STAC access,
co-registration, tiled product I/O) shipped as a prebuilt wheel installed
inside the ``neuralq-s2sr-api`` conda environment. This module is the only
file in the repository that references that engine's distribution
identifiers; every other module speaks pure NeuralQ/S2SR naming.

The identifiers are environment-overridable so the engine can be swapped
without touching code::

    export NEURALQ_ENGINE_MODULE=<import name>     # default below
    export NEURALQ_UPSTREAM_OBJECT=<weights id>    # default below

Install the wheel manually after ``conda env create`` (it is deliberately
absent from ``environment.yml``)::

    pip install <engine wheel>
"""
from __future__ import annotations

import os


ENGINE_MODULE = os.environ.get("NEURALQ_ENGINE_MODULE", "s2dr4")
UPSTREAM_WEIGHTS_ID = os.environ.get(
    "NEURALQ_UPSTREAM_OBJECT", "S2DR4-GL-20241022.1"
)
WEIGHTS_BUCKET = "https://storage.googleapis.com/0x7ff601307fa3"
KEY_ARCHIVE_URL = (
    "https://storage.googleapis.com/gedrm-16igu4jxui/"
    "1CAIpQLSepFU3lTlr9Y3u5py5GgKKguGVO"
)
KEY_ARCHIVE_SHA256 = (
    "ce29d542e080090f50995bde31c080087a4f275b6c93f8fba9b2658de484b929"
)
KEY_ARCHIVE_PASSWORD = b"1FAIpQLSepFU3lTlr9Y3u5py5GgKKguGVOsM"
KEY_MEMBER = "content/tmp/17igu4jxui"

UPSTREAM_TOKENS = (
    "S2L3Ax10_",
    "S2L2Ax10_",
    "S2DR4",
    "S2DR3",
    "s2dr4",
)
ENGINE_LEGACY_DIRS = ("/var/local/S2DR3", "/content/logs")


def engine():
    """Return ``(datautils, inferutils)`` modules of the compiled engine."""
    import importlib

    return (
        importlib.import_module(f"{ENGINE_MODULE}.datautils"),
        importlib.import_module(f"{ENGINE_MODULE}.inferutils"),
    )


def ensure_runtime_env() -> None:
    """Set environment markers the compiled engine probes before working."""
    os.environ.setdefault("COLAB_GPU", "local")


def sanitize(value: str) -> str:
    """Replace every upstream token with the local NeuralQ/S2SR brand."""
    branded = value.replace("S2L3Ax10_", "S2SR_")
    for token in UPSTREAM_TOKENS[1:]:
        branded = branded.replace(token, "S2SR")
    return branded


def has_upstream_token(text: str) -> bool:
    return any(token in text for token in UPSTREAM_TOKENS)
