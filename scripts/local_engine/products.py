"""Visualization products from the super-resolved MS cube (1 m).

TCI (B04/B03/B02) and IRP/CIR (B08/B04/B03) are 2–98% percentile-stretched
per band to uint8; NDVI is computed as (B08−B04)/(B08+B04) then mapped
through a 5-stop brown→green colormap. All three share the MS profile
(CRS, transform, 1 m) but are uint8, COMPRESS=NONE. Purely visual — not
scientific.
"""
from __future__ import annotations

import numpy as np
import rasterio
from pathlib import Path

BAND_ORDER = ("B02", "B03", "B04", "B08", "B05", "B06", "B07", "B11", "B12", "B8A")


def _profile(ms_path, count: int, dtype: str) -> dict:
    with rasterio.open(ms_path) as source:
        profile = source.profile.copy()
    profile.update(count=count, dtype=dtype, compress=None)
    for key in ("blockxsize", "blockysize", "tiled"):
        profile.pop(key, None)
    return profile


def _stretch(band: np.ndarray, low: float = 2, high: float = 98) -> np.ndarray:
    lo, hi = np.percentile(band, (low, high))
    if hi <= lo:
        hi = lo + 1
    stretched = (band.astype(np.float32) - lo) * (255.0 / (hi - lo))
    return np.clip(stretched, 0, 255).astype(np.uint8)


def _colormap(ndvi: np.ndarray) -> np.ndarray:
    value = np.clip((ndvi + 1) / 2, 0, 1)
    stops = np.array(
        [
            [110, 60, 30],
            [200, 150, 60],
            [255, 255, 170],
            [120, 200, 90],
            [20, 120, 50],
        ],
        dtype=np.float32,
    )
    positions = np.linspace(0, 1, len(stops))
    rgb = np.empty(value.shape + (3,), dtype=np.uint8)
    for channel in range(3):
        rgb[..., channel] = np.clip(
            np.interp(value, positions, stops[:, channel]), 0, 255
        ).astype(np.uint8)
    return rgb


def write_visuals(ms_path: str) -> list[str]:
    """Write TCI (true color), NDVI (colored ramp), and IRP (CIR) next to MS."""
    with rasterio.open(ms_path) as source:
        dn = source.read()
    bands = {name: dn[index - 1] for index, name in enumerate(BAND_ORDER, start=1)}
    written = []

    tci = np.dstack([_stretch(bands["B04"]), _stretch(bands["B03"]), _stretch(bands["B02"])])
    irp = np.dstack([_stretch(bands["B08"]), _stretch(bands["B04"]), _stretch(bands["B03"])])
    b08 = bands["B08"].astype(np.float32) / 10_000
    b04 = bands["B04"].astype(np.float32) / 10_000
    denominator = b08 + b04
    ndvi = np.where(denominator == 0, 0.0, (b08 - b04) / denominator)

    profile = _profile(ms_path, 3, "uint8")
    parent = Path(ms_path).parent
    for suffix, array in (
        ("TCI", np.moveaxis(tci, -1, 0)),
        ("NDVI", np.moveaxis(_colormap(ndvi), -1, 0)),
        ("IRP", np.moveaxis(irp, -1, 0)),
    ):
        path = str(parent / f"{suffix}.tif")
        with rasterio.open(path, "w", **profile) as target:
            target.write(array)
        written.append(path)
    return written
