"""Spectral index computation from S2SR ten-band super-resolved MS products."""
from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import rasterio

from output_layout import BAND_ORDER, DN_DIVISOR

CATEGORIES: dict[str, list[str]] = {
    "vegetation": ["ndvi", "gndvi", "ndre", "evi2", "savi", "mtci"],
    "water": ["ndwi", "mndwi", "ndmi", "awei"],
    "burn": ["nbr", "nbr2"],
    "soil_urban": ["bsi", "ndbi", "ndti"],
}

META: dict[str, dict[str, str]] = {
    "ndvi": {"formula": "(B08 - B04) / (B08 + B04)", "bands": "B08, B04", "range": "-1 .. 1"},
    "gndvi": {"formula": "(B08 - B03) / (B08 + B03)", "bands": "B08, B03", "range": "-1 .. 1"},
    "ndre": {"formula": "(B08 - B05) / (B08 + B05)", "bands": "B08, B05", "range": "-1 .. 1"},
    "evi2": {
        "formula": "2.5 * (B08 - B04) / (B08 + 2.4 * B04 + 1)",
        "bands": "B08, B04",
        "range": "approx -1 .. 1, can exceed",
    },
    "savi": {
        "formula": "1.5 * (B08 - B04) / (B08 + B04 + 0.5)",
        "bands": "B08, B04",
        "range": "-1 .. 1",
    },
    "mtci": {
        "formula": "(B06 - B05) / (B05 - B04)",
        "bands": "B06, B05, B04",
        "range": "positive for healthy vegetation",
    },
    "ndwi": {"formula": "(B03 - B08) / (B03 + B08)", "bands": "B03, B08", "range": "-1 .. 1"},
    "mndwi": {"formula": "(B03 - B11) / (B03 + B11)", "bands": "B03, B11", "range": "-1 .. 1"},
    "ndmi": {"formula": "(B08 - B11) / (B08 + B11)", "bands": "B08, B11", "range": "-1 .. 1"},
    "awei": {
        "formula": "4 * (B03 - B11) - (0.25 * B08 + 2.75 * B12)",
        "bands": "B03, B08, B11, B12",
        "range": "unbounded, water typically > 0",
    },
    "nbr": {"formula": "(B08 - B12) / (B08 + B12)", "bands": "B08, B12", "range": "-1 .. 1"},
    "nbr2": {"formula": "(B11 - B12) / (B11 + B12)", "bands": "B11, B12", "range": "-1 .. 1"},
    "bsi": {
        "formula": "((B11 + B04) - (B08 + B02)) / ((B11 + B04) + (B08 + B02))",
        "bands": "B11, B04, B08, B02",
        "range": "-1 .. 1",
    },
    "ndbi": {"formula": "(B11 - B08) / (B11 + B08)", "bands": "B11, B08", "range": "-1 .. 1"},
    "ndti": {"formula": "(B11 - B12) / (B11 + B12)", "bands": "B11, B12", "range": "-1 .. 1"},
}


def _nd(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    numerator = a - b
    denominator = a + b
    with np.errstate(divide="ignore", invalid="ignore"):
        result = np.where(denominator == 0, np.nan, numerator / denominator)
    return result.astype(np.float32)


def _formulas() -> dict[str, object]:
    def evi2(b):
        with np.errstate(divide="ignore", invalid="ignore"):
            denominator = b["B08"] + 2.4 * b["B04"] + 1.0
            result = np.where(denominator == 0, np.nan, 2.5 * (b["B08"] - b["B04"]) / denominator)
        return result.astype(np.float32)

    def savi(b):
        denominator = b["B08"] + b["B04"] + 0.5
        with np.errstate(divide="ignore", invalid="ignore"):
            result = np.where(denominator == 0, np.nan, 1.5 * (b["B08"] - b["B04"]) / denominator)
        return result.astype(np.float32)

    def mtci(b):
        with np.errstate(divide="ignore", invalid="ignore"):
            denominator = b["B05"] - b["B04"]
            result = np.where(denominator == 0, np.nan, (b["B06"] - b["B05"]) / denominator)
        return result.astype(np.float32)

    def awei(b):
        return (
            4.0 * (b["B03"] - b["B11"]) - (0.25 * b["B08"] + 2.75 * b["B12"])
        ).astype(np.float32)

    def bsi(b):
        numerator = (b["B11"] + b["B04"]) - (b["B08"] + b["B02"])
        denominator = (b["B11"] + b["B04"]) + (b["B08"] + b["B02"])
        with np.errstate(divide="ignore", invalid="ignore"):
            result = np.where(denominator == 0, np.nan, numerator / denominator)
        return result.astype(np.float32)

    return {
        "ndvi": lambda b: _nd(b["B08"], b["B04"]),
        "gndvi": lambda b: _nd(b["B08"], b["B03"]),
        "ndre": lambda b: _nd(b["B08"], b["B05"]),
        "evi2": evi2,
        "savi": savi,
        "mtci": mtci,
        "ndwi": lambda b: _nd(b["B03"], b["B08"]),
        "mndwi": lambda b: _nd(b["B03"], b["B11"]),
        "ndmi": lambda b: _nd(b["B08"], b["B11"]),
        "awei": awei,
        "nbr": lambda b: _nd(b["B08"], b["B12"]),
        "nbr2": lambda b: _nd(b["B11"], b["B12"]),
        "bsi": bsi,
        "ndbi": lambda b: _nd(b["B11"], b["B08"]),
        "ndti": lambda b: _nd(b["B11"], b["B12"]),
    }


def compute_indices(ms_path: Path, destination: Path, categories=None) -> list[dict]:
    formulas = _formulas()
    selected_categories = categories or list(CATEGORIES)
    destination = Path(destination)
    destination.mkdir(parents=True, exist_ok=True)

    with rasterio.open(ms_path) as source:
        if source.count != len(BAND_ORDER):
            raise ValueError(
                f"{ms_path} has {source.count} bands; expected {len(BAND_ORDER)}"
            )
        dn = source.read().astype(np.float32)
        profile = source.profile.copy()

    reflectance = {
        name: dn[index - 1] / float(DN_DIVISOR)
        for index, name in enumerate(BAND_ORDER, start=1)
    }

    profile.update(count=1, dtype="float32", nodata=np.float32("nan"))
    profile.pop("compress", None)
    profile.pop("compression", None)

    records = []
    for category in selected_categories:
        category_dir = destination / category
        category_dir.mkdir(parents=True, exist_ok=True)
        for name in CATEGORIES[category]:
            array = formulas[name](reflectance)
            path = category_dir / f"{name}.tiff"
            profile_out = dict(profile)
            profile_out.pop("blockxsize", None)
            profile_out.pop("blockysize", None)
            profile_out.pop("tiled", None)
            with rasterio.open(path, "w", **profile_out) as target:
                target.write(array, 1)
            records.append(
                {
                    "category": category,
                    "index": name,
                    "path": path,
                    "size_bytes": path.stat().st_size,
                    "width": profile["width"],
                    "height": profile["height"],
                }
            )
    return records


def write_indices_readme(destination: Path, records: list[dict]) -> Path:
    lines = [
        "# Spectral Indices",
        "",
        f"Generated: {datetime.now(timezone.utc).isoformat(timespec='seconds')} UTC",
        "",
        "Computed from the S2SR super-resolved `MS.tif` product (ten Sentinel-2 "
        "bands, 1 m) with reflectance DNs divided by 10,000.",
        "",
        "Every file is a single-band `float32` GeoTIFF, `COMPRESS=NONE`, "
        "`NoData = NaN`, sharing the CRS, extent, and 1 m grid of `MS.tif`. "
        "These are raw mathematical rasters; the colorized three-band "
        "preview `NDVI.tif` at the inference root is a visualization and is "
        "not part of this catalog.",
        "",
        "| Category | File | Formula | Bands | Typical range | Size (MB) |",
        "| --- | --- | --- | --- | --- | ---: |",
    ]
    ordered = sorted(records, key=lambda item: (item["category"], item["index"]))
    for record in ordered:
        meta = META[record["index"]]
        lines.append(
            f"| {record['category']} | `{record['category']}/{record['index']}.tiff` "
            f"| `{meta['formula']}` | {meta['bands']} | {meta['range']} "
            f"| {record['size_bytes'] / 1_000_000:.1f} |"
        )
    lines.append("")
    path = Path(destination) / "README.md"
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text("\n".join(lines), encoding="utf-8")
    temporary.replace(path)
    return path
