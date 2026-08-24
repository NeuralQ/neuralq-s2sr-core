#!/usr/bin/env python3
"""Verify the local S2SR checkpoint and reconstructed network.

Checks, in order:

1. the checkpoint file matches its pinned SHA-256,
2. ``s2sr.model.load_model`` loads ``params_ema`` with exactly
   105,055,800 parameters,
3. a fixed forward pass produces the pinned output shape and a golden
   statistical fingerprint (recorded on the reference machine; tolerances
   absorb BLAS/atomic nondeterminism across hardware).

Usage::

    python scripts/verify_local.py [--checkpoint PATH] [--device cpu]
"""
import sys

sys.dont_write_bytecode = True
import argparse
import hashlib
from pathlib import Path
import sys
import time

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from s2sr.model import S2SRNet, load_model


MODEL_NAME = "S2SR-GL-20241022.1"
MODEL_SHA256 = "1ac3d52cac3737842538ed09f329b0023b43cd3d5f509ccce36a0951cb2dd520"
PARAMETERS = 105_055_800
SAMPLE_SHAPE = (1, 50, 2, 2)
OUTPUT_SHAPE = (1, 10, 20, 20)
GOLDEN_MEAN = 0.008303545415401459
GOLDEN_STD = 0.004011942073702812
GOLDEN_MIN = -0.0014040799578651786
GOLDEN_MAX = 0.030925720930099487
TOLERANCE = 1e-6


def sha256(path: Path) -> str:
    hasher = hashlib.sha256()
    with path.open("rb") as file:
        for chunk in iter(lambda: file.read(8 * 1024 * 1024), b""):
            hasher.update(chunk)
    return hasher.hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Verify the local S2SR model against pinned fingerprints."
    )
    parser.add_argument(
        "--checkpoint",
        type=Path,
        default=ROOT / "models" / f"{MODEL_NAME}.pt",
    )
    parser.add_argument("--device", default="cpu")
    args = parser.parse_args()

    if sha256(args.checkpoint) != MODEL_SHA256:
        raise SystemExit("Checkpoint SHA-256 does not match the recovered model")
    print(f"checkpoint: verified ({args.checkpoint.stat().st_size:,} bytes)")

    started = time.perf_counter()
    model = load_model(args.checkpoint, args.device)
    parameter_count = sum(parameter.numel() for parameter in model.parameters())
    if parameter_count != PARAMETERS:
        raise SystemExit(f"Unexpected parameter count: {parameter_count:,}")
    print(f"model: EMA weights loaded ({parameter_count:,} parameters)")

    sample = torch.zeros(SAMPLE_SHAPE, device=args.device)
    with torch.inference_mode():
        output = model(sample)
    if tuple(output.shape) != OUTPUT_SHAPE or not torch.isfinite(output).all():
        raise SystemExit(f"Unexpected inference result: {tuple(output.shape)}")

    fingerprint = {
        "mean": output.mean().item(),
        "std": output.std().item(),
        "min": output.min().item(),
        "max": output.max().item(),
    }
    for name, observed in fingerprint.items():
        golden = {"mean": GOLDEN_MEAN, "std": GOLDEN_STD, "min": GOLDEN_MIN, "max": GOLDEN_MAX}[name]
        if abs(observed - golden) > TOLERANCE:
            raise SystemExit(
                f"Golden {name} mismatch: observed {observed:.12g}, "
                f"expected {golden:.12g}"
            )
    print(
        "inference: passed "
        f"(shape={tuple(output.shape)}, elapsed={time.perf_counter() - started:.2f}s)"
    )
    print(f"torch: {torch.__version__}; CUDA available: {torch.cuda.is_available()}")


if __name__ == "__main__":
    main()
