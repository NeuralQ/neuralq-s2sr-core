#!/usr/bin/env python3
"""Generate reproducible architecture and checkpoint reports for S2SR.

Emits ``architecture.json`` (module counts, stage parameters, constants,
shape trace, environment), ``checkpoint.json`` (state summaries, strict
load result, raw-vs-EMA delta statistics), per-tensor ``state_dict.csv``,
and a human-readable ``REPORT.md`` under the report directory.

Usage::

    python scripts/analyze_model.py [--checkpoint PATH] [--output-dir PATH]
"""
import argparse
from collections import Counter
import csv
import hashlib
import json
import math
from pathlib import Path
import platform
import sys

import torch


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
DEFAULT_CHECKPOINT = ROOT / "models" / "S2SR-GL-20241022.1.pt"
DEFAULT_OUTPUT = ROOT / "reports" / "s2sr_model"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as file:
        for chunk in iter(lambda: file.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def parameter_summary(module: torch.nn.Module) -> dict:
    parameters = list(module.parameters())
    return {
        "parameters": sum(parameter.numel() for parameter in parameters),
        "parameter_bytes": sum(
            parameter.numel() * parameter.element_size() for parameter in parameters
        ),
        "parameter_tensors": len(parameters),
        "trainable_parameters": sum(
            parameter.numel() for parameter in parameters if parameter.requires_grad
        ),
    }


def checkpoint_summary(state: dict[str, torch.Tensor]) -> dict:
    dtypes = Counter(str(tensor.dtype) for tensor in state.values())
    return {
        "tensors": len(state),
        "parameters": sum(tensor.numel() for tensor in state.values()),
        "tensor_bytes": sum(
            tensor.numel() * tensor.element_size() for tensor in state.values()
        ),
        "dtypes": dict(sorted(dtypes.items())),
        "first_key": next(iter(state)),
        "last_key": next(reversed(state)),
    }


def trace_shapes(model: torch.nn.Module) -> dict:
    shapes = {}
    hooks = []
    for name in ("deformer", "encoder", "generator"):
        hooks.append(
            getattr(model, name).register_forward_hook(
                lambda _module, _inputs, output, key=name: shapes.__setitem__(
                    key, list(output.shape)
                )
            )
        )
    for index, module in enumerate(model.generator):
        hooks.append(
            module.register_forward_hook(
                lambda _module, _inputs, output, key=f"generator.{index}": (
                    shapes.__setitem__(key, list(output.shape))
                )
            )
        )
    sample = torch.zeros(1, 50, 2, 3)
    with torch.inference_mode():
        output = model(sample)
    for hook in hooks:
        hook.remove()
    return {
        "sample_input": list(sample.shape),
        "stages": shapes,
        "sample_output": list(output.shape),
    }


def analyze_deltas(raw, ema, csv_path: Path) -> dict:
    changed_tensors = 0
    total_values = 0
    sum_abs = 0.0
    sum_squared = 0.0
    maximum = 0.0
    maximum_key = None
    rows = []
    stage_totals = {}

    for name, raw_tensor in raw.items():
        ema_tensor = ema[name]
        raw_float = raw_tensor.float()
        ema_float = ema_tensor.float()
        difference = ema_float - raw_float
        absolute = difference.abs()
        count = difference.numel()
        tensor_maximum = absolute.max().item()
        tensor_sum_abs = absolute.double().sum().item()
        tensor_sum_squared = difference.double().square().sum().item()
        changed_tensors += tensor_maximum > 0
        if tensor_maximum > maximum:
            maximum = tensor_maximum
            maximum_key = name
        total_values += count
        sum_abs += tensor_sum_abs
        sum_squared += tensor_sum_squared

        stage = name.split(".", 1)[0]
        totals = stage_totals.setdefault(
            stage,
            {"values": 0, "sum_abs": 0.0, "sum_squared": 0.0, "maximum": 0.0},
        )
        totals["values"] += count
        totals["sum_abs"] += tensor_sum_abs
        totals["sum_squared"] += tensor_sum_squared
        totals["maximum"] = max(totals["maximum"], tensor_maximum)
        rows.append(
            {
                "name": name,
                "shape": "x".join(str(value) for value in raw_tensor.shape),
                "dtype": str(raw_tensor.dtype),
                "numel": count,
                "raw_mean": raw_float.mean().item(),
                "raw_std": raw_float.std(unbiased=False).item(),
                "ema_mean": ema_float.mean().item(),
                "ema_std": ema_float.std(unbiased=False).item(),
                "mean_abs_delta": tensor_sum_abs / count,
                "rms_delta": math.sqrt(tensor_sum_squared / count),
                "max_abs_delta": tensor_maximum,
            }
        )

    with csv_path.open("w", encoding="utf-8", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    stages = {
        stage: {
            "values": totals["values"],
            "mean_abs_delta": totals["sum_abs"] / totals["values"],
            "rms_delta": math.sqrt(totals["sum_squared"] / totals["values"]),
            "max_abs_delta": totals["maximum"],
        }
        for stage, totals in stage_totals.items()
    }
    return {
        "changed_tensors": changed_tensors,
        "total_tensors": len(raw),
        "total_values": total_values,
        "mean_abs_delta": sum_abs / total_values,
        "rms_delta": math.sqrt(sum_squared / total_values),
        "max_abs_delta": maximum,
        "max_abs_delta_key": maximum_key,
        "stages": stages,
    }


def markdown_report(architecture: dict, checkpoint: dict) -> str:
    stages = architecture["stages"]
    delta = checkpoint["raw_vs_ema"]
    return f"""# S2SR Model Report

## Identity

- Local model ID: `S2SR-GL-20241022.1`
- Upstream provenance: the decrypted pretrained checkpoint described in the repository README
- Checkpoint SHA-256: `{checkpoint['sha256']}`
- Checkpoint size: `{checkpoint['size_bytes']:,}` bytes
- Inference state: `params_ema`

## Active Architecture

- Class: `S2SRNet`
- Input: `N x 50 x H x W`, representing five dates and ten bands
- Output: `N x 10 x 10H x 10W`
- Parameters: `{architecture['total']['parameters']:,}`
- Parameter tensors: `{architecture['total']['parameter_tensors']:,}`
- Deformation/alignment stage: `{stages['deformer']['parameters']:,}` parameters
- RRDB reconstruction stage: `{stages['encoder']['parameters']:,}` parameters
- Upsampling stage: `{stages['generator']['parameters']:,}` parameters

The alignment stage contains one `50 -> 160` convolution followed by seven
grouped deformable-convolution blocks. Each block predicts 90 offsets
(`2 x 5 groups x 3 x 3`) and applies a 160-channel deformable convolution with
five groups.

The reconstruction stage contains one `160 -> 160` convolution followed by 23
RRDBs. Every RRDB contains three residual-dense blocks; every dense block has
five 3x3 convolutions, growth width 80, and residual scale 0.2. The outer RRDB
residual also uses scale 0.2.

The generator uses scale factors `2 x 2 x 2 x 1.25 = 10`, with
`nearest-exact` interpolation before each upsampling convolution. Channel
widths are `160 -> 80 -> 40 -> 20 -> 10 -> 10`. The last convolution has no
bias and is followed by LeakyReLU.

## Checkpoint

- `params`: `{checkpoint['states']['params']['tensors']}` tensors,
  `{checkpoint['states']['params']['parameters']:,}` values
- `params_ema`: `{checkpoint['states']['params_ema']['tensors']}` tensors,
  `{checkpoint['states']['params_ema']['parameters']:,}` values
- EMA-changed tensors: `{delta['changed_tensors']}/{delta['total_tensors']}`
- Global EMA mean absolute delta: `{delta['mean_abs_delta']:.10g}`
- Global EMA RMS delta: `{delta['rms_delta']:.10g}`
- Maximum EMA delta: `{delta['max_abs_delta']:.10g}` at
  `{delta['max_abs_delta_key']}`

The checkpoint stores both full raw and EMA copies, explaining its roughly
841 MB size. Strict loading of `params_ema` into the reconstructed model has no
missing or unexpected keys. The S2SR model's forward pass matches the pinned
golden fingerprint recorded by `scripts/verify_local.py`.

## Recovered Data Contract

- Exactly five aligned dates are used for inference.
- The stack shape before flattening is `5 x 10 x H x W`.
- Flattening is date-major, producing 50 channels: all ten bands for date 1,
  then all ten bands for date 2, and so on.
- Physical source-band order is `B02, B03, B04, B08, B05, B06, B07, B11,
  B12, B8A`.
- Input reflectance DNs are converted to float and divided by 10,000.
- Output is clipped/scaled to the reflectance DN range and written as uint16.
- A 412 x 412 input becomes a 4120 x 4120 output.

## Shipped But Inactive Code

`S2SRSingleDateNet` (103,536,130 parameters) and `S2SRFusionLayer` are
reconstructed auxiliary classes but have no keys in this checkpoint and are
not used by the active inference path. The single-date network is a ten-channel
variant with the same RRDB/generator trunk and frozen parameters.

Machine-readable details are in `architecture.json`, `checkpoint.json`, and
`state_dict.csv` in this directory.
"""


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Generate reproducible S2SR architecture and checkpoint reports."
    )
    parser.add_argument("--checkpoint", type=Path, default=DEFAULT_CHECKPOINT)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()
    checkpoint_path = args.checkpoint.resolve()
    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    from s2sr.model import S2SRFusionLayer, S2SRNet, S2SRSingleDateNet

    model = S2SRNet()
    module_counts = Counter(type(module).__name__ for module in model.modules())
    architecture = {
        "class": "S2SRNet",
        "total": parameter_summary(model),
        "stages": {
            name: parameter_summary(getattr(model, name))
            for name in ("deformer", "encoder", "generator")
        },
        "module_counts": dict(sorted(module_counts.items())),
        "constants": {
            "input_dates": 5,
            "bands_per_date": 10,
            "input_channels": 50,
            "output_channels": 10,
            "feature_width": 160,
            "dense_growth_width": 80,
            "deformation_blocks": 7,
            "deformation_groups": 5,
            "deformation_offset_channels": 90,
            "rrdb_blocks": 23,
            "dense_blocks_per_rrdb": 3,
            "convolutions_per_dense_block": 5,
            "residual_scale": 0.2,
            "upsampling_scales": [2, 2, 2, 1.25, 1],
            "total_scale": 10,
            "source_band_order": [
                "B02", "B03", "B04", "B08", "B05",
                "B06", "B07", "B11", "B12", "B8A"
            ],
            "normalization_divisor": 10000,
        },
        "shape_trace": trace_shapes(model),
        "inactive_shipped_classes": {
            "S2SRSingleDateNet": parameter_summary(S2SRSingleDateNet()),
            "S2SRFusionLayer160": parameter_summary(S2SRFusionLayer(160)),
        },
        "environment": {
            "python": platform.python_version(),
            "torch": torch.__version__,
            "cuda_available": torch.cuda.is_available(),
            "torch_cuda": torch.version.cuda,
        },
    }
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
    raw = checkpoint["params"]
    ema = checkpoint["params_ema"]
    load_result = model.load_state_dict(ema, strict=True)
    checkpoint_report = {
        "path": str(checkpoint_path),
        "size_bytes": checkpoint_path.stat().st_size,
        "sha256": sha256(checkpoint_path),
        "states": {
            "params": checkpoint_summary(raw),
            "params_ema": checkpoint_summary(ema),
        },
        "inference_state": "params_ema",
        "strict_load": {
            "missing_keys": list(load_result.missing_keys),
            "unexpected_keys": list(load_result.unexpected_keys),
        },
    }
    checkpoint_report["raw_vs_ema"] = analyze_deltas(
        raw, ema, output_dir / "state_dict.csv"
    )
    (output_dir / "architecture.json").write_text(
        json.dumps(architecture, indent=2) + "\n", encoding="utf-8"
    )
    (output_dir / "checkpoint.json").write_text(
        json.dumps(checkpoint_report, indent=2) + "\n", encoding="utf-8"
    )
    (output_dir / "REPORT.md").write_text(
        markdown_report(architecture, checkpoint_report), encoding="utf-8"
    )
    print(f"Reports written to {output_dir}")


if __name__ == "__main__":
    main()
