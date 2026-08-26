"""inferutils: end-to-end AOI inference for the local S2SR engine."""
from __future__ import annotations

import json
import os as _os
import secrets
from datetime import date as _date, datetime, timezone
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import rasterio
from pyproj import Transformer
from rasterio.enums import Resampling
from rasterio.vrt import WarpedVRT

import local_engine.datautils as datautils
import local_engine.products as products

import resource_monitor as rm

BAND_ORDER = ("B02", "B03", "B04", "B08", "B05", "B06", "B07", "B11", "B12", "B8A")
BAND_ASSETS = {
    "B02": ("blue", False),
    "B03": ("green", False),
    "B04": ("red", False),
    "B08": ("nir", False),
    "B05": ("rededge1", True),
    "B06": ("rededge2", True),
    "B07": ("rededge3", True),
    "B11": ("swir16", True),
    "B12": ("swir22", True),
    "B8A": ("nir08", True),
}
DN_DIVISOR = 10_000
GRID_SIZE_PX = 412
INPUT_RESOLUTION_M = 10


class _ModuleProxy:
    def __init__(self, target):
        self._target = target

    def __getattr__(self, name):
        return getattr(self._target, name)


np = _ModuleProxy(np)
os = _ModuleProxy(_os)


class _IoNamespace:
    @staticmethod
    def imread(*_args, **_kwargs):
        import imagecodecs

        raise imagecodecs.DelayedImportError("local engine reads rasters via rasterio")


io = _IoNamespace()

CFG = SimpleNamespace()


def configure(args) -> None:
    for key in (
        "date", "uid", "org", "aoi", "mode", "items",
        "savepath", "datapath", "logpath",
        "model_name", "model_path", "tile", "make_preview", "force",
    ):
        if hasattr(args, key):
            setattr(CFG, key, getattr(args, key))


def _utm_crs(longitude: float, latitude: float) -> str:
    zone = int((longitude + 180) // 6) + 1
    band = 326 if latitude >= 0 else 327
    return f"EPSG:{band * 100 + zone}"


def _make_grid(lon: float, lat: float) -> dict:
    crs = _utm_crs(lon, lat)
    to_utm = Transformer.from_crs("EPSG:4326", crs, always_xy=True)
    cx, cy = to_utm.transform(lon, lat)
    half = GRID_SIZE_PX * INPUT_RESOLUTION_M / 2
    step = float(INPUT_RESOLUTION_M)
    x0 = round((cx - half) / step) * step
    y1 = round((cy + half) / step) * step
    return {
        "crs": crs,
        "width": GRID_SIZE_PX,
        "height": GRID_SIZE_PX,
        "transform": rasterio.Affine(step, 0, x0, 0, -step, y1),
        "center_utm": (cx, cy),
        "to_wgs84": Transformer.from_crs(crs, "EPSG:4326", always_xy=True),
    }


def _read_band(href: str, grid: dict, smooth: bool) -> np.ndarray:
    with rasterio.open(href) as source:
        options = dict(
            crs=grid["crs"],
            transform=grid["transform"],
            width=grid["width"],
            height=grid["height"],
            resampling=Resampling.bilinear if smooth else Resampling.nearest,
        )
        with WarpedVRT(source, **options) as vrt:
            raw = vrt.read(1).astype(np.float32)
            scale = float(vrt.scales[0] or 0.0)
            offset = float(vrt.offsets[0] or 0.0)
    # Sentinel-2 L2A quantification is fixed at DN/10000; GDAL tags of 1.0
    # are unset defaults, not real scales. Only trust fractional tags, and
    # only honor negative BOA offsets (Earth Search COGs already applied).
    if not 0.0 < scale < 1.0:
        scale = 1.0 / DN_DIVISOR
    if not -2000.0 <= offset < 0.0:
        offset = 0.0
    dn = np.rint((raw * scale + offset) * DN_DIVISOR)
    return np.clip(dn, 0, 65_535).astype(np.uint16)


def _build_stack(items: list[dict], grid: dict) -> tuple[np.ndarray, list[str]]:
    channels, dates = [], []
    bar = rm.Bar(len(items) * len(BAND_ORDER), "download+align")
    for item in items:
        assets = item["assets"]
        missing = [b for b, (key, _) in BAND_ASSETS.items() if key not in assets]
        if missing:
            raise RuntimeError(f"{item['id']} lacks assets: {missing}")
        for band in BAND_ORDER:
            key, smooth = BAND_ASSETS[band]
            channels.append(_read_band(assets[key]["href"], grid, smooth))
            bar.update()
        dates.append(item["info"]["date"])
    return np.stack(channels), dates


def _tile_inference(stack_dn: np.ndarray, model_path: str, tile_size: int) -> np.ndarray:
    import sys as _sys
    import torch

    scripts_dir = Path(__file__).resolve().parents[1]
    for entry in (str(scripts_dir), str(scripts_dir.parent)):
        if entry not in _sys.path:
            _sys.path.insert(0, entry)
    from s2sr import load_model, super_resolve_dn

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = load_model(model_path, device=device)
    if stack_dn.ndim == 3:
        count = stack_dn.shape[0]
        if count % len(BAND_ORDER):
            raise ValueError(f"stack channel count {count} not divisible by {len(BAND_ORDER)}")
        stack_dn = stack_dn.reshape(count // len(BAND_ORDER), len(BAND_ORDER), *stack_dn.shape[-2:])
    _, bands, height, width = stack_dn.shape
    out_h, out_w = height * 10, width * 10
    output = np.zeros((bands, out_h, out_w), dtype=np.uint16)
    tile_size = max(tile_size, 16)
    total = ((height + tile_size - 1) // tile_size) * ((width + tile_size - 1) // tile_size)
    bar = rm.Bar(total, "gpu-tiles")
    done = 0
    for y in range(0, height, tile_size):
        for x in range(0, width, tile_size):
            tile = stack_dn[:, :, y : y + tile_size, x : x + tile_size]
            th, tw = tile.shape[-2:]
            result = super_resolve_dn(model, tile.reshape(-1, th, tw), device=device)
            output[
                :,
                y * 10 : (y + th) * 10,
                x * 10 : (x + tw) * 10,
            ] = result
            done += 1
            bar.update()
    del model
    return output


def test(*, lonlat, date, simulate=False) -> None:
    with rm.ResourceMonitor() as monitor:
        _run_inference(lonlat=lonlat, date=date, monitor=monitor)


def _run_inference(*, lonlat, date, monitor: rm.ResourceMonitor) -> None:
    lon, lat = float(lonlat[0]), float(lonlat[1])
    target_iso = str(date)
    compact = target_iso.replace("-", "")
    args = SimpleNamespace(
        date=compact,
        uid=secrets.token_hex(5)[:9],
        org="NN",
        aoi=datautils.aoi_from_xy((lon, lat), km=2),
        mode="aoi",
    )
    configure(args)
    # run_location's patched configure() may add path/model fields onto
    # ``args`` only after invoking the original configure; re-merge here.
    for key in (
        "savepath", "datapath", "logpath",
        "model_name", "model_path", "tile",
    ):
        if hasattr(args, key):
            setattr(CFG, key, getattr(args, key))
    save_dir = Path(CFG.savepath)
    log_dir = Path(CFG.logpath)
    data_dir = Path(CFG.datapath)
    for directory in (save_dir, log_dir, data_dir):
        directory.mkdir(parents=True, exist_ok=True)

    datautils.search_collection(args)
    items = datautils.sort_items_by_recency_and_clouds(args.items)
    chosen = datautils.select_stack_dates(items, target_iso)
    target_day = _date.fromisoformat(target_iso)
    anchor = min(
        chosen, key=lambda i: abs((_date.fromisoformat(i["info"]["date"]) - target_day).days)
    )
    mgrs_full = anchor.get("mgrs_tile") or "T000XXX"
    mgrs = mgrs_full.lstrip("T")
    uid = args.uid
    grid = _make_grid(lon, lat)
    bbox = [
        grid["to_wgs84"].transform(
            grid["transform"].c, grid["transform"].f - grid["height"] * 10
        ),
        grid["to_wgs84"].transform(
            grid["transform"].c + grid["width"] * 10, grid["transform"].f
        ),
    ]
    wgs = [coord for point in bbox for coord in point]
    print(f"  stack dates: {[i['info']['date'] for i in chosen]} (MGRS {mgrs_full})", flush=True)

    stack, dates = _build_stack(chosen, grid)
    print(f"  stack built: {stack.shape}, inferring...", flush=True)
    ms = _tile_inference(stack, str(CFG.model_path), int(getattr(CFG, "tile", 128)))

    ms_path = str(save_dir / "MS.tif")
    profile = {
        "driver": "GTiff",
        "width": ms.shape[-1],
        "height": ms.shape[-2],
        "count": ms.shape[0],
        "dtype": "uint16",
        "crs": grid["crs"],
        "transform": rasterio.Affine(1, 0, grid["transform"].c, 0, 1, grid["transform"].f),
        "compress": None,
    }
    with rasterio.open(ms_path, "w", **profile) as target:
        target.write(ms)
        for index, name in enumerate(BAND_ORDER, start=1):
            target.set_band_description(index, name)
    visuals = products.write_visuals(ms_path)

    job_id = f"s{secrets.token_hex(6)}-{secrets.token_hex(2)}-11f1"
    log_record = {
        "PID": f"{mgrs}-{uid}-{compact}",
        "mode": "aoi",
        "date": compact,
        "MGRS": mgrs,
        "ISO": _country_code(lat, lon),
        "dimentions": f"{float(grid['width'])},{float(grid['height'])}",
        "bbox": ",".join(f"{value:.15f}" for value in wgs),
        "bands": ",".join(BAND_ORDER),
        "stack_dates": ",".join(dates),
        "save_path_MS": ms_path,
        "save_path_TCI": visuals[0],
        "save_path_NDVI": visuals[1],
        "save_path_IRP": visuals[2],
        "aoi_overlap": "1.0",
        "job_id": job_id,
    }
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S")
    log_path = log_dir / f"S2SRlog_{mgrs}-{uid}-{compact}_{job_id}_{stamp}.json"
    log_path.write_text(json.dumps(log_record, indent=2) + "\n", encoding="utf-8")
    print(f"  products written to {save_dir}", flush=True)
    for line in monitor.summary():
        print(line, flush=True)


def _country_code(lat: float, lon: float) -> str:
    import sys as _sys

    scripts_dir = Path(__file__).resolve().parents[1]
    if str(scripts_dir) not in _sys.path:
        _sys.path.insert(0, str(scripts_dir))
    from output_layout import country_code

    try:
        return country_code(lat, lon)
    except Exception:
        return "XX"
