"""Spectral indices at 1 m from the super-resolved MS cube.

All 21 indices (+ thermal LST via lst.py) are pure arithmetic on
``MS.tif`` reflectance (DN/10000) at the 1 m grid — no new satellite.
Each is single-band float32, COMPRESS=NONE, NoData=NaN, grid-identical to
MS.tif. Formulas follow the literature (e.g. Tucker NDVI, McFeeters NDWI,
Hu FAI, Cloutis hydrocarbon) and are NaN-safe (0/0 → NaN). Oil suite
(osi/hi/foi/ndoi/sr/rg) is empirical and threshold-dependent — calibrate per
scene and confirm with SAR/field. Spectral — not a detector.
"""
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
    "oil": ["osi", "hi", "foi", "ndoi", "sr", "rg"],
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
    # Thermal sharpening product (written by lst.py, cataloged here so the
    # indices README lists it; never computed by compute_indices).
    "lst": {
        "formula": "a + b * NDVI_1m + resid_1m (TsHARP sharpening)",
        "bands": "Landsat ST_B10, B08, B04",
        "range": "Celsius, ~-20 .. 60 typical",
    },
    "co2": {
        "formula": "420 + 8·NDBI* + 0.8·ΔLST* + 5·AOT*  (ppm, *=robust p5–p98)",
        "bands": "B11,B08 (NDBI) + LST + B02,B04 (AOT)",
        "range": "ppm, ~420–450, p98 ~440–450",
    },
    "osi": {
        "formula": "(B11 + B12 - B08 - B04) / (B11 + B12 + B08 + B04)",
        "bands": "B04, B08, B11, B12",
        "range": "-1 .. 1, water oil > ~0.15, land oil > ~0.08 (local calibration required)",
    },
    "hi": {
        "formula": "(B11 - B12) / (B11 + B12)",
        "bands": "B11, B12",
        "range": "-1 .. 1, thick oil > ~0.03–0.06, clean water ~0",
    },
    "foi": {
        "formula": "B08 - [B04 + (B11 - B04)*0.187]",
        "bands": "B04, B08, B11",
        "range": "reflectance, water ~0, oil < -0.01, vegetation > +0.03",
    },
    "ndoi": {
        "formula": "(B03 - B08) / (B03 + B08)",
        "bands": "B03, B08",
        "range": "-1 .. 1, water oil < ~0.2 (vs 0.5–0.7 clean water)",
    },
    "sr": {
        "formula": "B12 / B11",
        "bands": "B11, B12",
        "range": "0 .. ~2, thick oil < ~0.85, water/soil ~0.9–1.0",
    },
    "rg": {
        "formula": "B04 / B03",
        "bands": "B03, B04",
        "range": "0 .. ~2, oil on water > ~1.05, clean water ~0.6–0.8",
    },
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

    def osi(b):
        """Oil Spill Index — hydrocarbon-aware SWIR vs Red+NIR contrast.

        Physics: crude oil elevates SWIR (B11 ~1610 nm, B12 ~2190 nm — C-H
        stretch overtone near 1.73 µm and 2.30 µm, Cloutis et al. 2010;
        Lammoglia & Souza Filho 2011) relative to the Red+NIR baseline
        (B04 ~665 nm, B08 ~842 nm). Clean seawater: B11≈B12≈0, so
        OSI → -(B08+B04)/(B08+B04) negative to near-zero. Oil film on
        water: SWIR brightening → OSI positive (Kolokoussis &
        Karathanassi 2018 used Red/SWIR vs NIR for Sentinel-2 oil on sea).
        Oil on bare soil/dark land: hydrocarbon SWIR excess vs NIR
        scattering darkening → likewise positive, but bare soil itself is
        weakly positive (~0.05–0.12), so land thresholds must be higher and
        locally calibrated. Empirical — mask clouds, sunglint (B08 spike),
        and bright urban before thresholding.
        """
        numerator = (b["B11"] + b["B12"]) - (b["B08"] + b["B04"])
        denominator = b["B11"] + b["B12"] + b["B08"] + b["B04"]
        with np.errstate(divide="ignore", invalid="ignore"):
            result = np.where(denominator == 0, np.nan, numerator / denominator)
        return result.astype(np.float32)

    def hi(b):
        """Hydrocarbon Index — B11/B12 absorption depth (2.30 µm)."""
        return _nd(b["B11"], b["B12"])

    def foi(b):
        """Floating Oil Index — NIR vs Red-SWIR baseline (Hu 2009 FAI adapted)."""
        return (b["B08"] - (b["B04"] + (b["B11"] - b["B04"]) * 0.187)).astype(np.float32)

    def ndoi(b):
        """Normalized Difference Oil Index — Green vs NIR (water oil darkens Green gap)."""
        return _nd(b["B03"], b["B08"])

    def sr(b):
        """SWIR Ratio — B12/B11 hydrocarbon ratio."""
        with np.errstate(divide="ignore", invalid="ignore"):
            result = np.where(b["B11"] == 0, np.nan, b["B12"] / b["B11"])
        return result.astype(np.float32)

    def rg(b):
        """Red-Green Ratio — oil reddening."""
        with np.errstate(divide="ignore", invalid="ignore"):
            result = np.where(b["B03"] == 0, np.nan, b["B04"] / b["B03"])
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
        "osi": osi,
        "hi": hi,
        "foi": foi,
        "ndoi": ndoi,
        "sr": sr,
        "rg": rg,
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
    # Dedicated oil/README.md with scientific explanations — requested for the
    # 1 m oil suite alongside the global indices/README.md.
    oil_records = [r for r in records if r["category"] == "oil"]
    if oil_records:
        _write_oil_readme(destination / "oil", oil_records)
    # Dedicated thermal/README.md for LST
    thermal_records = [r for r in records if r["category"] == "thermal"]
    if thermal_records:
        _write_thermal_readme(destination / "thermal", thermal_records)
    return path


def _write_oil_readme(oil_dir: Path, records: list[dict]) -> Path:
    lines = [
        "# Oil Spill Indices — 1 m (super-resolved)",
        "",
        f"Generated: {datetime.now(timezone.utc).isoformat(timespec='seconds')} UTC",
        "",
        "Six hydrocarbon-aware indices at 1 m, computed from the S2SR super-"
        "resolved ten-band `MS.tif` (DN/10000). All are single-band float32, "
        "`COMPRESS=NONE`, `NoData=NaN`, grid-identical to `MS.tif`. No new "
        "satellite needed — pure arithmetic on the 1 m cube. **Empirical, "
        "threshold-dependent, and sunglint/cloud-sensitive — calibrate per "
        "scene and confirm with SAR/field data. No optical index alone is a "
        "legal detector.** S2SR 1 m texture is inferred, not measured.",
        "",
        "## Physics",
        "",
        "- **C-H stretch:** crude oil elevates SWIR (B11 1610 nm, B12 2190 nm) "
        "near 1.73 µm & 2.30 µm (Cloutis et al. 2010; Lammoglia & Souza Filho "
        "2011) vs Red (B04 665 nm) + NIR (B08 842 nm, strong water absorption).",
        "- **Sea:** clean water B11≈B12≈0 → OSI negative; oil brightens SWIR → "
        "positive. FOI goes negative (Hu FAI baseline).",
        "- **Land:** oil darkens NIR on bare soil while SWIR stays bright → "
        "same sign, but bright soils/urban mimic oil — thresholds shift.",
        "",
        "## Indices",
        "",
        "| Index | File | Formula | Bands | Sea threshold | Land threshold |",
        "| --- | --- | --- | --- | --- | --- |",
        "| osi | `oil/osi.tiff` | `(B11+B12-B08-B04)/(B11+B12+B08+B04)` | B04,B08,B11,B12 | >0.15 candidate, >0.25 high-conf | >0.08–0.12 candidate, calibrate |",
        "| hi | `oil/hi.tiff` | `(B11-B12)/(B11+B12)` | B11,B12 | >0.03 thick oil | >0.04 |",
        "| foi | `oil/foi.tiff` | `B08-[B04+(B11-B04)*0.187]` | B04,B08,B11 | <-0.01 on water | not for land (veg >>0) |",
        "| ndoi | `oil/ndoi.tiff` | `(B03-B08)/(B03+B08)` | B03,B08 | <0.2 on water (clean ~0.6) | not for land |",
        "| sr | `oil/sr.tiff` | `B12/B11` | B11,B12 | <0.85 thick oil | <0.88 |",
        "| rg | `oil/rg.tiff` | `B04/B03` | B03,B04 | >1.05 on water | not alone |",
        "",
        "## Recommended triple test (sea)",
        "",
        "```",
        "OSI > 0.15 && FOI < -0.01 && HI > 0.03  →  high-confidence oil on water",
        "OSI > 0.12 && HI > 0.04                →  candidate on dark land",
        "```",
        "Mask first: `NDWI>0.2` (water), `B08<0.12` reflectance (glint), cloud mask.",
        "",
        "## Files",
        "",
        "| File | Size (MB) |",
        "| --- | ---: |",
    ]
    for rec in sorted(records, key=lambda r: r["index"]):
        lines.append(f"| `{rec['index']}.tiff` | {rec['size_bytes']/1_000_000:.1f} |")
    lines += ["", "Literature: Cloutis et al. 2010; Lammoglia & Souza Filho 2011; Kolokoussis & Karathanassi 2018; Hu 2009 (FAI); Pisano et al. 2021.", ""]
    path = oil_dir / "README.md"
    tmp = path.with_name(path.name + ".tmp")
    tmp.write_text("\n".join(lines), encoding="utf-8")
    tmp.replace(path)
    return path


def _write_thermal_readme(thermal_dir: Path, records: list[dict]) -> Path:
    rec = records[0]
    lines = [
        "# Thermal — Land Surface Temperature (1 m, sharpened)",
        "",
        f"Generated: {datetime.now(timezone.utc).isoformat(timespec='seconds')} UTC",
        "",
        "TsHARP-style sharpening: Landsat Collection 2 L2 ST_B10 (30 m, "
        "`Kelvin = DN×0.00341802+149.0`) disaggregated with S2SR NDVI + "
        "bilinear residuals. Preserves coarse means by construction.",
        "",
        f"| File | Size | Grid | Units | NoData |",
        f"| --- | --- | --- | --- | --- |",
        f"| `thermal/lst.tiff` | {rec['size_bytes']/1_000_000:.1f} MB | 1 m, {rec['width']}×{rec['height']} | Celsius (K−273.15) | NaN |",
        "",
        "Tags: `LST_MIN_C`, `LST_MAX_C`, `LST_MEAN_C`, `LST_LEGEND`, `STATISTICS_*` on band.",
        "Sidecar: `lst_legend.json`.",
        "",
    ]
    path = thermal_dir / "README.md"
    tmp = path.with_name(path.name + ".tmp")
    tmp.write_text("\n".join(lines), encoding="utf-8")
    tmp.replace(path)
    return path
