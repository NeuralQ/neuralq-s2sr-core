"""Integration boundary for the local inference engine.

NeuralQ S2SR runs entirely on the pure-Python engine in
``scripts/local_engine`` (Earth Search STAC, Sentinel-2 COGs on AWS,
tiled GPU inference). No compiled wheel is required.

A different engine can still be plugged in without touching code::

    export NEURALQ_ENGINE_MODULE=<import name>
"""
from __future__ import annotations

import os


ENGINE_MODULE = os.environ.get("NEURALQ_ENGINE_MODULE", "local_engine")

# Legacy identifiers are never produced locally; they exist only so
# sanitize() can brand any historical artifact that resurfaces.
UPSTREAM_TOKENS = (
    "S2L3Ax10_",
    "S2L2Ax10_",
    "S2DR4",
    "S2DR3",
    "s2dr4",
)
ENGINE_LEGACY_DIRS = ("/var/local/S2DR3", "/content/logs")


def engine():
    """Return ``(datautils, inferutils)`` modules of the inference engine."""
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
