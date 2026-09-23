"""Carbon Emission Proxy at 1 m — S2SR + LST downscaling.

Direct 1 m CO₂ column from Sentinel-2 is unphysical: S2 has 20–180 nm
bands, while CO₂ needs <0.1 nm spectroscopy at 1.61/2.06 µm (OCO-2/3,
GHG Sat, TROPOMI 5.5×3.5 km). This module builds a *proxy* that *is*
1 m and mass-conserving, by fusing what S2SR *does* measure at 1 m:

  * **Activity footprint** — NDBI = (B11−B08)/(B11+B08) (industrial
    impervious, 1 m) + BSI threshold. Gives the 1 m source mask.
  * **Thermal excess** — ΔLST = LST_C − median(LST_clean) from
    ``indices/thermal/lst.tiff`` (Celsius, 1 m, TsHARP). Flare/kiln heat.
  * **Plume opacity** — AOT_proxy = (B02−B04)/(B02+B04) (blue vs red,
    smoke scatters blue). High where plume is.

Proxy (ppm, background ~420 ppm, factory plume +10–30 ppm):

  CO₂_proxy = 420 + k1·NDBI* + k2·ΔLST* + k3·AOT*_proxy

where * denotes per-tile robust scaling to 0..1 (p5–p98). k1=8, k2=0.8,
k3=5 are empirical, scene-calibrated — not a flux. For a true emission
rate Q (t/hr), downscale TROPOMI XCO₂ via the same activity map:

  XCO₂_1m = XCO₂_coarse × (activity_1m / mean(activity_coarse))

and apply Gaussian plume Q = ΔXCO₂·wind·width (ERA5 wind, 1 m plume
width from the proxy mask). That step needs TROPOMI/S5P; this file
provides the 1 m activity/plume at 1 m and the proxy.

Output ``indices/carbon/co2.tiff`` is float32 ppm, NoData=NaN,
grid-identical to ``MS.tif``. Tags carry the legend and per-pixel meaning.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import rasterio

from output_layout import BAND_ORDER, DN_DIVISOR

CARBON_CATEGORY = "carbon"
CARBON_INDEX = "co2"
CARBON_FILENAME = "co2.tiff"
CARBON_META = {
    "formula": "420 + 8·NDBI* + 0.8·ΔLST* + 5·AOT*  (ppm, * = robust 0..1)",
    "bands": "B02,B03,B04,B08,B11 + LST",
    "range": "ppm, background ~420, plume ~430–450, p98 ~445",
}
BASELINE_PPM = 420.0
K1, K2, K3 = 8.0, 0.8, 5.0  # empirical, scene-calibrated


def _robust01(arr: np.ndarray) -> np.ndarray:
    """Robust 0..1 scaling via p5–p98, NaN-safe."""
    valid = arr[np.isfinite(arr)]
    if valid.size == 0:
        return np.full_like(arr, np.nan, dtype=np.float32)
    p5, p98 = np.nanpercentile(valid, [5, 98])
    if p98 <= p5:
        return np.zeros_like(arr, dtype=np.float32)
    scaled = (arr - p5) / (p98 - p5)
    return np.clip(scaled, 0, 1).astype(np.float32)


def compute_co2(ms_path: Path | str, indices_dir: Path | str | None = None) -> dict:
    """Generate ``indices/carbon/co2.tiff`` (ppm) from MS.tif + optional LST.

    Returns a record dict for logging. Never raises for missing LST — falls
    back to NDBI+AOT only (k2 term zero).
    """
    ms_path = Path(ms_path)
    indices_dir = Path(indices_dir) if indices_dir else ms_path.parent / "indices"
    carbon_dir = indices_dir / CARBON_CATEGORY
    carbon_dir.mkdir(parents=True, exist_ok=True)

    with rasterio.open(ms_path) as src:
        if src.count != len(BAND_ORDER):
            raise ValueError(f"{ms_path.name} has {src.count} bands; expected {len(BAND_ORDER)}")
        dn = src.read().astype(np.float32)
        profile = {
            "crs": src.crs,
            "transform": src.transform,
            "width": src.width,
            "height": src.height,
        }
    bands = {name: dn[i] / float(DN_DIVISOR) for i, name in enumerate(BAND_ORDER)}

    # NDBI — activity
    with np.errstate(divide="ignore", invalid="ignore"):
        ndbi = np.where((bands["B11"] + bands["B08"]) == 0, np.nan, (bands["B11"] - bands["B08"]) / (bands["B11"] + bands["B08"])).astype(np.float32)
        aot = np.where((bands["B02"] + bands["B04"]) == 0, np.nan, (bands["B02"] - bands["B04"]) / (bands["B02"] + bands["B04"])).astype(np.float32)

    # ΔLST — thermal excess, if available
    lst_path = indices_dir / "thermal" / "lst.tiff"
    if lst_path.is_file():
        with rasterio.open(lst_path) as s:
            lst = s.read(1).astype(np.float32)
            lst[~np.isfinite(lst)] = np.nan
            clean_median = float(np.nanmedian(lst))
            dlst = (lst - clean_median).astype(np.float32)
            # Only positive excess matters for combustion; clamp negative to 0 before scaling
            dlst = np.where(dlst < 0, 0, dlst)
    else:
        dlst = np.zeros_like(ndbi, dtype=np.float32)
        clean_median = float("nan")

    # Robust scaling per tile
    ndbi_s = _robust01(ndbi)
    aot_s = _robust01(aot)
    dlst_s = _robust01(dlst) if np.any(np.isfinite(dlst) & (dlst != 0)) else np.zeros_like(dlst, dtype=np.float32)

    co2 = (BASELINE_PPM + K1 * ndbi_s + K2 * dlst_s + K3 * aot_s).astype(np.float32)
    # Where MS was nodata, propagate NaN
    co2[~np.isfinite(ndbi)] = np.nan

    # Stats for legend
    finite = np.isfinite(co2)
    if np.any(finite):
        vmin, vmax, vmean, vmedian, vstd = [float(np.nanmin(co2)), float(np.nanmax(co2)), float(np.nanmean(co2)), float(np.nanmedian(co2)), float(np.nanstd(co2))]
        p5, p98 = [float(x) for x in np.nanpercentile(co2[finite], [5, 98])]
    else:
        vmin = vmax = vmean = vmedian = vstd = p5 = p98 = float("nan")

    out_profile = {
        "driver": "GTiff",
        "width": profile["width"],
        "height": profile["height"],
        "count": 1,
        "dtype": "float32",
        "crs": profile["crs"],
        "transform": profile["transform"],
        "nodata": float("nan"),
        "compress": None,
    }
    dest = carbon_dir / CARBON_FILENAME
    with rasterio.open(dest, "w", **out_profile) as dst:
        dst.write(co2, 1)
        dst.set_band_description(1, "CO2 proxy (ppm)")
        dst.update_tags(
            1,
            STATISTICS_MINIMUM=str(vmin),
            STATISTICS_MAXIMUM=str(vmax),
            STATISTICS_MEAN=str(vmean),
            STATISTICS_STDDEV=str(vstd),
        )
        legend = (
            f"CO2 proxy 1 m (ppm, NoData=NaN) — background {BASELINE_PPM:.0f} ppm, "
            f"min {vmin:.1f} ppm, max {vmax:.1f} ppm, mean {vmean:.1f} ppm, "
            f"p5 {p5:.1f} p98 {p98:.1f}; per-pixel value = proxy ppm at that 1 m cell "
            f"(420 + {K1}·NDBI* + {K2}·ΔLST* + {K3}·AOT*, *=robust 0..1)"
        )
        dst.update_tags(
            CO2_UNITS="ppm",
            CO2_BASELINE_PPM=str(BASELINE_PPM),
            CO2_K1_NDBI=str(K1),
            CO2_K2_DLST=str(K2),
            CO2_K3_AOT=str(K3),
            CO2_MIN_PPM=str(vmin),
            CO2_MAX_PPM=str(vmax),
            CO2_MEAN_PPM=str(vmean),
            CO2_P5_PPM=str(p5),
            CO2_P98_PPM=str(p98),
            CO2_LST_MEDIAN_C=str(clean_median) if np.isfinite(clean_median) else "nan",
            CO2_LEGEND=legend,
        )
        dst.update_tags(1, CO2_LEGEND=legend)

    import json as _json

    sidecar = carbon_dir / "co2_legend.json"
    sidecar.write_text(
        _json.dumps(
            {
                "units": "ppm",
                "baseline_ppm": BASELINE_PPM,
                "k1_ndbi": K1,
                "k2_dlst": K2,
                "k3_aot": K3,
                "nodata": "NaN",
                "min_ppm": vmin,
                "max_ppm": vmax,
                "mean_ppm": vmean,
                "median_ppm": vmedian,
                "std_ppm": vstd,
                "p5_ppm": p5,
                "p98_ppm": p98,
                "lst_median_c": clean_median if np.isfinite(clean_median) else None,
                "legend": legend,
                "per_pixel": "each pixel value = CO2 proxy ppm at that 1 m cell (mass-conserving downscale needs TROPOMI + wind; this is the 1 m activity/plume)",
                "formula": CARBON_META["formula"],
            },
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )
    _write_carbon_readme(carbon_dir, dest, vmin, vmax, vmean, p5, p98, profile["width"], profile["height"])

    return {
        "category": CARBON_CATEGORY,
        "index": CARBON_INDEX,
        "path": dest,
        "size_bytes": dest.stat().st_size,
        "width": profile["width"],
        "height": profile["height"],
        "min_ppm": vmin,
        "max_ppm": vmax,
        "mean_ppm": vmean,
        "p5_ppm": p5,
        "p98_ppm": p98,
    }


def _write_carbon_readme(carbon_dir: Path, dest: Path, vmin: float, vmax: float, vmean: float, p5: float, p98: float, width: int, height: int) -> Path:
    from datetime import datetime, timezone

    lines = [
        "# Carbon — CO₂ Proxy (1 m, S2SR + LST)",
        "",
        f"Generated: {datetime.now(timezone.utc).isoformat(timespec='seconds')} UTC",
        "",
        "1 m CO₂ proxy from the S2SR super-resolved `MS.tif` (DN/10000) + `thermal/lst.tiff` Celsius. No new satellite — pure arithmetic on the 1 m cube. **Proxy, not a direct column** — background 420 ppm + activity/plume excess; for t/hr downscale TROPOMI XCO₂ via the 1 m activity map and ERA5 wind (mass-conserving).",
        "",
        "## Formula",
        "",
        "```",
        "CO₂_proxy = 420 + 8·NDBI* + 0.8·ΔLST* + 5·AOT*   (ppm, * = robust 0..1 p5–p98)",
        "NDBI = (B11−B08)/(B11+B08)  (industrial)",
        "ΔLST = LST_C − median(LST_clean)  (thermal excess, 0 if LST missing)",
        "AOT = (B02−B04)/(B02+B04)  (smoke)",
        "```",
        "",
        f"| File | Size | Grid | Units | NoData |",
        f"| --- | --- | --- | --- | --- |",
        f"| `carbon/co2.tiff` | {dest.stat().st_size/1_000_000:.1f} MB | 1 m, {width}×{height} | ppm | NaN |",
        "",
        f"Stats: min {vmin:.1f} ppm, max {vmax:.1f} ppm, mean {vmean:.1f} ppm, p5 {p5:.1f} p98 {p98:.1f} (per-tile p5–p98).",
        "",
        "Tags: `CO2_MIN_PPM`, `CO2_MAX_PPM`, `CO2_MEAN_PPM`, `CO2_LEGEND`, `STATISTICS_*` on band. Sidecar: `co2_legend.json`.",
        "",
        "Use: threshold `co2.tiff > 430` ppm for plume (factory), `>425` for peri-urban. Always mask with `NDBI>0.15` (factory) and compare to background `420` ppm. Validate with TROPOMI + wind for Q (t/hr).",
        "",
    ]
    path = carbon_dir / "README.md"
    tmp = path.with_name(path.name + ".tmp")
    tmp.write_text("\n".join(lines), encoding="utf-8")
    tmp.replace(path)
    return path
