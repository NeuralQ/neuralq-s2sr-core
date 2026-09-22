"""Land Surface Temperature by thermal sharpening (TsHARP-style).

Sentinel-2 carries no thermal band, so LST cannot be derived from ``MS.tif``.
Instead this module fetches the real measurement — Landsat 8/9 Collection 2
Level-2 surface temperature (``lwir11``/ST_B10, 30 m) from Earth Search STAC —
and disaggregates it to the 1 m S2SR grid with the super-resolved NDVI::

    LST_30m = a + b * NDVI_30m            least squares on clear coarse pixels
    LST_1m  = a + b * NDVI_1m + resid_1m  bilinear-upsampled coarse residuals

Residual redistribution preserves the coarse means by construction. Output
``indices/thermal/lst.tiff`` is float32 Kelvin with ``NaN`` nodata,
grid-identical to ``MS.tif`` and cataloged with the spectral indices.

``usgs-landsat`` is a Requester Pays bucket, so LST needs AWS credentials
(standard chain: env, ``~/.aws``, instance role). Without them — or without a
usable scene — every entry point raises :class:`LSTUnavailable` with a short
``reason`` (``aws-credentials``, ``no-scenes``, ``all-cloudy``,
``too-few-samples``); callers treat LST as optional and record the note.

Third-party imports stay function-local so this module (and its pure
helpers) import without numpy/rasterio, e.g. for the stdlib-only unit tests.
"""
from __future__ import annotations

from datetime import date, datetime, timedelta
from pathlib import Path
import time

from output_layout import BAND_ORDER

STAC_URL = "https://earth-search.aws.element84.com/v1/search"
COLLECTION = "landsat-c2-l2"
ASSET_ST = "lwir11"
ASSET_QA = "qa_pixel"
WINDOW_DAYS = 16
MAX_SCENE_CLOUDS = 60.0
COARSE_RES_M = 30.0
# USGS Collection 2 Level-2 surface temperature conversion (authoritative,
# https://www.usgs.gov/landsat-missions/landsat-collection-2-surface-temperature):
#   Kelvin = DN * 0.00341802 + 149.0   (uint16, fill 0)
# (Collection 1 used 0.1 with no offset — do not confuse the two.)
ST_MULT_USGS = 0.00341802
ST_ADD_USGS = 149.0
ST_FILL_DN = 0
# C2 QA_PIXEL bits that invalidate a thermal sample: fill, dilated cloud,
# cirrus, cloud, cloud shadow, snow. Clear (6) and water (7) stay valid.
QA_BAD_BITS = (0, 1, 2, 3, 4, 5)
MIN_SAMPLES = 25
FIT_SUBSAMPLE_MAX = 4000
# Catalog home of the product: alongside the spectral indices, not at the
# run root. Lowercase name + .tiff match the indices/<category>/<name>.tiff
# convention so the mosaic tile discovery and the indices README pick it up.
LST_CATEGORY = "thermal"
LST_INDEX = "lst"
LST_FILENAME = "lst.tiff"
LST_META = {
    "formula": "a + b * NDVI_1m + resid_1m (TsHARP sharpening)",
    "bands": "Landsat ST_B10, B08, B04",
    "range": "Celsius, ~-20 .. 60 typical (Kelvin - 273.15)",
}
# Physical plausibility band for the coarse median (Earth surface, Kelvin).
# Catches scaling surprises (wrong scale tags, wrong asset) before they can
# produce a silently unphysical product.
PLAUSIBLE_K_MIN = 230.0
PLAUSIBLE_K_MAX = 360.0


class LSTUnavailable(RuntimeError):
    """LST cannot be produced; ``reason`` is a short machine-readable code."""

    def __init__(self, reason: str, detail: str = "") -> None:
        self.reason = reason
        super().__init__(f"LST unavailable ({reason}): {detail}" if detail else f"LST unavailable ({reason})")


def qa_clear(qa_value: int) -> bool:
    """True when no bad QA_PIXEL bit is set."""
    return not any((int(qa_value) >> bit) & 1 for bit in QA_BAD_BITS)


def plausible_kelvin(median_k: float) -> bool:
    """True when a coarse-median temperature is physically plausible."""
    import math

    return math.isfinite(median_k) and PLAUSIBLE_K_MIN <= median_k <= PLAUSIBLE_K_MAX


def st_scale_offset(tag_scale: float | None, tag_offset: float | None) -> tuple[float, float]:
    """Resolve the ST DN->Kelvin conversion from GDAL tags.

    Trusts the file tags only when they match the USGS Collection 2
    constants within tolerance; otherwise falls back to the USGS values.
    Either way the result is (mult, add) with Kelvin = DN * mult + add.
    """
    try:
        scale = float(tag_scale or 0.0)
    except (TypeError, ValueError):
        scale = 0.0
    try:
        offset = float(tag_offset or 0.0)
    except (TypeError, ValueError):
        offset = 0.0
    if abs(scale - ST_MULT_USGS) / ST_MULT_USGS < 0.01 and abs(offset - ST_ADD_USGS) < 1.0:
        return scale, offset
    return ST_MULT_USGS, ST_ADD_USGS


def window_range(target_iso: str, days: int = WINDOW_DAYS) -> tuple[str, str]:
    """Inclusive (start, end) date strings around the target."""
    target = date.fromisoformat(target_iso)
    delta = timedelta(days=days)
    return (target - delta).isoformat(), (target + timedelta(days=1)).isoformat()


def coarse_shape(width: int, height: int, res: float = COARSE_RES_M) -> tuple[int, int]:
    """Coarse (width, height) in whole cells covering a fine grid."""
    import math

    return math.ceil(width / res), math.ceil(height / res)


def fit_lst_ndvi(pairs: list[tuple[float, float]]) -> tuple[float, float, float, int]:
    """Least-squares ``LST = a + b * NDVI``; returns (a, b, rmse, n)."""
    import math

    clean = [(x, y) for x, y in pairs if math.isfinite(x) and math.isfinite(y)]
    n = len(clean)
    if n < 2:
        raise LSTUnavailable("too-few-samples", f"only {n} valid coarse pixels")
    mean_x = sum(x for x, _ in clean) / n
    mean_y = sum(y for _, y in clean) / n
    var_x = sum((x - mean_x) ** 2 for x, _ in clean)
    if var_x == 0:
        raise LSTUnavailable("too-few-samples", "no NDVI variance at coarse scale")
    slope = sum((x - mean_x) * (y - mean_y) for x, y in clean) / var_x
    intercept = mean_y - slope * mean_x
    rmse = math.sqrt(sum((y - (intercept + slope * x)) ** 2 for x, y in clean) / n)
    return intercept, slope, rmse, n


def select_scene(items: list[dict], target_iso: str) -> dict:
    """Pick the scene closest in time, breaking ties by cloud cover."""
    if not items:
        raise LSTUnavailable("no-scenes", "STAC returned no Landsat scenes")
    target = date.fromisoformat(target_iso)

    def key(item: dict) -> tuple[int, float]:
        info = item["info"]
        return (
            abs((date.fromisoformat(info["date"]) - target).days),
            float(info.get("clouds", 9999)),
        )

    return min(items, key=key)


def search_scenes(bbox: list[float], target_iso: str) -> list[dict]:
    """STAC search for Landsat C2L2 scenes around a bbox and target date."""
    import requests

    start, end = window_range(target_iso)
    payload = {
        "collections": [COLLECTION],
        "datetime": f"{start}T00:00:00Z/{end}T23:59:59Z",
        "bbox": bbox,
        "limit": 50,
        "sortby": [{"field": "properties.datetime", "direction": "asc"}],
    }
    features: list[dict] = []
    url, method, body = STAC_URL, "POST", payload
    while True:
        for attempt in range(3):
            try:
                kwargs = {"json" if method == "POST" else "params": body}
                response = requests.request(method, url, timeout=120, **kwargs)
                response.raise_for_status()
                break
            except Exception as exc:
                if attempt == 2:
                    raise RuntimeError(f"STAC search failed for {url!r}: {exc}") from exc
                time.sleep(2 * (attempt + 1))
        page = response.json()
        features.extend(page.get("features", []))
        link = next((l for l in page.get("links", []) if l.get("rel") == "next"), None)
        if link is None:
            break
        url, method = link["href"], link.get("method", "POST").upper()
        body = {**payload, **(link.get("body") or {})}
    items = []
    for feature in features:
        properties = feature["properties"]
        clouds = properties.get("eo:cloud_cover", 9999)
        try:
            clouds = float(clouds)
        except (TypeError, ValueError):
            clouds = 9999.0
        if clouds > MAX_SCENE_CLOUDS:
            continue
        items.append(
            {
                "id": feature["id"],
                "geometry": feature.get("geometry"),
                "assets": feature.get("assets", {}),
                "info": {"date": properties["datetime"][:10], "clouds": clouds},
            }
        )
    return items


def _read_coarse(item: dict, profile) -> tuple:
    """Read ST (Kelvin) and QA onto a 30 m grid clipped to the MS extent.

    Returns (lst_kelvin, qa, coarse_profile). Raises LSTUnavailable on
    missing assets, unreadable data, or missing AWS credentials.
    """
    import numpy as np
    import rasterio
    from rasterio.enums import Resampling
    from rasterio.vrt import WarpedVRT

    assets = item.get("assets", {})
    if ASSET_ST not in assets or ASSET_QA not in assets:
        raise LSTUnavailable(
            "no-scenes", f"{item.get('id', '?')} lacks {ASSET_ST}/{ASSET_QA} assets"
        )
    left, bottom, right, top = profile["bounds"]
    coarse_w, coarse_h = coarse_shape(profile["width"], profile["height"])
    coarse_transform = rasterio.Affine(
        COARSE_RES_M, 0, left, 0, -COARSE_RES_M, top
    )
    try:
        with rasterio.Env(
            AWS_REQUEST_PAYER="requester",
            AWS_REGION="us-west-2",
            GDAL_DISABLE_READDIR_ON_OPEN="EMPTY_DIR",
            GDAL_HTTP_MAX_RETRY="5",
            GDAL_HTTP_RETRY_DELAY="5",
        ):
            with rasterio.open(assets[ASSET_ST]["href"]) as source:
                mult, add = st_scale_offset(
                    (source.scales or [None])[0], (source.offsets or [None])[0]
                )
                options = dict(
                    crs=profile["crs"],
                    transform=coarse_transform,
                    width=coarse_w,
                    height=coarse_h,
                    resampling=Resampling.bilinear,
                )
                with WarpedVRT(source, **options) as vrt:
                    raw = vrt.read(1).astype(np.float32)
            with rasterio.open(assets[ASSET_QA]["href"]) as source:
                options["resampling"] = Resampling.nearest
                with WarpedVRT(source, **options) as vrt:
                    qa = vrt.read(1)
    except LSTUnavailable:
        raise
    except Exception as error:
        # Prefer typed botocore errors when available; fall back to message
        try:
            from botocore.exceptions import (  # type: ignore[import-not-found]
                NoCredentialsError,
                PartialCredentialsError,
            )

            if isinstance(error, (NoCredentialsError, PartialCredentialsError)):
                raise LSTUnavailable(
                    "aws-credentials",
                    "usgs-landsat is Requester Pays; configure AWS credentials "
                    "(env, ~/.aws/credentials, or instance role)",
                ) from error
        except ImportError:
            pass
        text = str(error)
        if any(
            token in text
            for token in (
                "AccessDenied",
                "InvalidCredentials",
                "No valid AWS credentials",
                "Requester Pays",
                "Unable to locate credentials",
            )
        ):
            raise LSTUnavailable(
                "aws-credentials",
                "usgs-landsat is Requester Pays; configure AWS credentials "
                "(env, ~/.aws/credentials, or instance role)",
            ) from error
        raise LSTUnavailable("read-error", text) from error
    kelvin = raw * mult + add
    kelvin[raw == ST_FILL_DN] = float("nan")
    coarse_profile = {
        "crs": profile["crs"],
        "transform": coarse_transform,
        "width": coarse_w,
        "height": coarse_h,
    }
    return kelvin, qa, coarse_profile


def _aggregate_ndvi(ndvi_1m, profile, coarse_profile):
    """Block-average 1 m NDVI onto the coarse grid (NaN-aware)."""
    import numpy as np
    import rasterio
    from rasterio.enums import Resampling
    from rasterio.io import MemoryFile

    coarse_h, coarse_w = coarse_profile["height"], coarse_profile["width"]
    with MemoryFile() as mem:
        with mem.open(
            driver="GTiff",
            width=profile["width"],
            height=profile["height"],
            count=1,
            dtype="float32",
            crs=profile["crs"],
            transform=profile["transform"],
            nodata=float("nan"),
        ) as dataset:
            dataset.write(ndvi_1m.astype(np.float32), 1)
            return dataset.read(
                1,
                out_shape=(coarse_h, coarse_w),
                resampling=Resampling.average,
            )


def _upsample_residuals(residuals, coarse_profile, profile):
    """Bilinear-upsample coarse residuals onto the 1 m grid."""
    import numpy as np
    import rasterio
    from rasterio.enums import Resampling
    from rasterio.io import MemoryFile

    with MemoryFile() as mem:
        with mem.open(
            driver="GTiff",
            width=coarse_profile["width"],
            height=coarse_profile["height"],
            count=1,
            dtype="float32",
            crs=coarse_profile["crs"],
            transform=coarse_profile["transform"],
            nodata=float("nan"),
        ) as dataset:
            dataset.write(residuals.astype(np.float32), 1)
            return dataset.read(
                1,
                out_shape=(profile["height"], profile["width"]),
                resampling=Resampling.bilinear,
            )


def compute_lst(
    ms_path: str | Path,
    target_iso: str,
    bbox: list[float],
) -> dict:
    """Sharpen Landsat ST onto the MS.tif grid; write indices/thermal/lst.tiff.

    Returns a record dict for the run README. Raises LSTUnavailable when the
    layer cannot be produced (callers record ``note`` and continue).
    """
    import numpy as np
    import rasterio

    ms_path = Path(ms_path)
    with rasterio.open(ms_path) as source:
        if source.count != len(BAND_ORDER):
            raise LSTUnavailable(
                "ms-unreadable",
                f"{ms_path.name} has {source.count} bands; expected {len(BAND_ORDER)}",
            )
        dn = source.read().astype(np.float32)
        profile = {
            "crs": source.crs,
            "transform": source.transform,
            "width": source.width,
            "height": source.height,
            "bounds": tuple(source.bounds),
        }
    bands = {name: dn[i] / 10_000.0 for i, name in enumerate(BAND_ORDER)}
    with np.errstate(divide="ignore", invalid="ignore"):
        denom = bands["B08"] + bands["B04"]
        ndvi_1m = np.where(denom == 0, np.nan, (bands["B08"] - bands["B04"]) / denom).astype(
            np.float32
        )

    items = search_scenes(bbox, target_iso)
    scene = select_scene(items, target_iso)
    # S3 range reads can be reset by flaky networks; the coarse read is tiny
    # so a plain retry loop is cheap. Auth/config errors are not retried.
    last_error: LSTUnavailable | None = None
    for attempt in range(3):
        try:
            kelvin_30m, qa_30m, coarse_profile = _read_coarse(scene, profile)
            break
        except LSTUnavailable as error:
            if error.reason != "read-error":
                raise
            last_error = error
            time.sleep(5 * (attempt + 1))
    else:
        assert last_error is not None
        raise last_error

    valid = np.ones_like(kelvin_30m, dtype=bool)
    valid &= np.isfinite(kelvin_30m)
    flat_qa = qa_30m.ravel()
    clear = np.fromiter((qa_clear(int(v)) for v in flat_qa), dtype=bool, count=flat_qa.size)
    valid &= clear.reshape(qa_30m.shape)
    if int(valid.sum()) < MIN_SAMPLES:
        raise LSTUnavailable(
            "all-cloudy",
            f"only {int(valid.sum())} clear coarse pixels in {scene['id']}",
        )
    coarse_median = float(np.median(kelvin_30m[valid]))
    if not plausible_kelvin(coarse_median):
        raise LSTUnavailable(
            "implausible-values",
            f"coarse median {coarse_median:.1f} K outside "
            f"{PLAUSIBLE_K_MIN:.0f}..{PLAUSIBLE_K_MAX:.0f} K; refusing to sharpen "
            f"(check ST scaling/tags for {scene['id']})",
        )

    ndvi_30m = _aggregate_ndvi(ndvi_1m, profile, coarse_profile)
    pairs = [
        (float(x), float(y))
        for x, y in zip(ndvi_30m[valid].ravel(), kelvin_30m[valid].ravel())
    ]
    stride = max(1, len(pairs) // FIT_SUBSAMPLE_MAX)
    intercept, slope, rmse, n_fit = fit_lst_ndvi(pairs[::stride])

    predicted_1m = (intercept + slope * ndvi_1m).astype(np.float32)
    predicted_30m = intercept + slope * ndvi_30m
    residuals_30m = np.where(valid, kelvin_30m - predicted_30m, np.nan).astype(np.float32)
    residuals_1m = _upsample_residuals(residuals_30m, coarse_profile, profile)
    lst_1m = predicted_1m + np.where(np.isfinite(residuals_1m), residuals_1m, 0.0)
    lst_1m[~np.isfinite(ndvi_1m)] = np.nan

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
    # Convert to Celsius for output — units requested by user. Fit stays
    # in Kelvin (USGS source), then subtract 273.15 once at write time.
    lst_1m_c = lst_1m - 273.15
    lst_1m_c[~np.isfinite(lst_1m)] = np.nan
    destination = ms_path.parent / "indices" / LST_CATEGORY / LST_FILENAME
    destination.parent.mkdir(parents=True, exist_ok=True)
    # Per-pixel temperature values are the raster itself; the global legend
    # (min/max/mean) is stored as band + dataset tags so any GIS shows it
    # without a sidecar. All numbers are now Celsius.
    finite = np.isfinite(lst_1m_c)
    if np.any(finite):
        vmin = float(np.nanmin(lst_1m_c))
        vmax = float(np.nanmax(lst_1m_c))
        vmean = float(np.nanmean(lst_1m_c))
        vmedian = float(np.nanmedian(lst_1m_c))
        vstd = float(np.nanstd(lst_1m_c))
        p2, p98 = [float(x) for x in np.nanpercentile(lst_1m_c[finite], [2, 98])]
    else:
        vmin = vmax = vmean = vmedian = vstd = p2 = p98 = float("nan")
    # Kelvin equivalents for provenance (one subtraction away)
    kmin, kmax, kmean, kmedian = [v + 273.15 for v in (vmin, vmax, vmean, vmedian)]
    with rasterio.open(destination, "w", **out_profile) as target:
        target.write(lst_1m_c.astype(np.float32), 1)
        target.set_band_description(1, "LST (Celsius)")
        # GDAL STATISTICS — recognised by QGIS for auto-stretch/legend
        target.update_tags(
            1,
            STATISTICS_MINIMUM=str(vmin),
            STATISTICS_MAXIMUM=str(vmax),
            STATISTICS_MEAN=str(vmean),
            STATISTICS_STDDEV=str(vstd),
            STATISTICS_VALID_PERCENT=str(float(finite.mean() * 100)),
        )
        legend = (
            f"LST 1 m (Celsius, NoData=NaN) — min {vmin:.2f} C ({kmin:.2f} K), "
            f"max {vmax:.2f} C ({kmax:.2f} K), mean {vmean:.2f} C, "
            f"median {vmedian:.2f} C, p2 {p2:.2f} C, p98 {p98:.2f} C; "
            f"per-pixel value = temperature at that 1 m ground cell in Celsius"
        )
        target.update_tags(
            LST_UNITS="Celsius",
            LST_KELVIN_OFFSET="273.15",
            LST_MIN_C=str(vmin),
            LST_MAX_C=str(vmax),
            LST_MEAN_C=str(vmean),
            LST_MEDIAN_C=str(vmedian),
            LST_STD_C=str(vstd),
            LST_P2_C=str(p2),
            LST_P98_C=str(p98),
            LST_MIN_K=str(kmin),
            LST_MAX_K=str(kmax),
            LST_LEGEND=legend,
        )
        target.update_tags(1, LST_LEGEND=legend)
    # Sidecar legend for non-GDAL readers (same numbers, JSON)
    import json as _json

    legend_path = destination.with_name("lst_legend.json")
    legend_path.write_text(
        _json.dumps(
            {
                "units": "Celsius",
                "kelvin_offset": 273.15,
                "nodata": "NaN",
                "min_c": vmin,
                "max_c": vmax,
                "mean_c": vmean,
                "median_c": vmedian,
                "std_c": vstd,
                "p2_c": p2,
                "p98_c": p98,
                "min_k": kmin,
                "max_k": kmax,
                "legend": legend,
                "per_pixel": "each pixel value = LST at that 1 m cell in Celsius (K - 273.15)",
            },
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )

    scene_date = scene["info"]["date"]
    offset = abs((date.fromisoformat(scene_date) - date.fromisoformat(target_iso)).days)
    return {
        "category": LST_CATEGORY,
        "index": LST_INDEX,
        "path": destination,
        "size_bytes": destination.stat().st_size,
        "width": profile["width"],
        "height": profile["height"],
        "scene_id": scene["id"],
        "scene_date": scene_date,
        "date_offset_days": offset,
        "n_valid_coarse": int(valid.sum()),
        "n_fit": n_fit,
        "intercept_k": intercept,
        "slope_k_per_ndvi": slope,
        "rmse_k": rmse,
    }
