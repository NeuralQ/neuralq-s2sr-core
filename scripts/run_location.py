#!/usr/bin/env python3
"""Run one S2SR super-resolution inference for a point and target date.

Drives the compiled local engine end to end: STAC scene search and ranking,
download of five usable acquisitions, co-registration into a 50-channel
stack, tiled GPU inference, and GeoTIFF product generation. Afterwards the
run is normalized into NeuralQ/S2SR branding (uncompressed rasters, flat
canonical filenames, sanitized metadata and logs) and enriched with fifteen
spectral indices.

Outputs land in ``outputs/<CC>/<date>/<inference_id>/`` with a generated
README.md. All scratch state lives in a transient ``.work/`` directory
inside that folder and is removed when the run ends, successfully or not -
nothing outside ``outputs/`` is ever created or kept.

Usage::

    python scripts/run_location.py --lon 51.531 --lat 25.2886 \
        --date 2026-08-14 [--products MS TCI] [--skip-indices] \
        [--search-only] [--tile-size 128]
"""
import sys

sys.dont_write_bytecode = True
import argparse
from datetime import datetime
import json
import os
from pathlib import Path
import shutil
import sys
from types import SimpleNamespace

from output_layout import (
    BAND_ORDER,
    country_code,
    inference_directory,
    new_inference_id,
    product_inventory,
    format_inventory_record,
    sha256_file,
    utc_now,
    write_metadata_readme,
)

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

# Bundled checkpoint id; mirrors s2sr.hub.MODEL_ID but kept local so this
# module stays importable without torch/numpy for the stdlib-only unit tests.
MODEL_ID = "s2sr-v3.0.0"


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run S2SR for a longitude, latitude, and target date."
    )
    parser.add_argument("--lon", type=float, required=True)
    parser.add_argument("--lat", type=float, required=True)
    parser.add_argument("--date", required=True, help="Target date in YYYY-MM-DD form")
    parser.add_argument(
        "--model",
        type=Path,
        default=None,
        help=(
            "Local checkpoint path; omit to use the bundled models/s2sr-v3.0.0.pt "
            "baked into the repo / container image"
        ),
    )
    parser.add_argument(
        "--output",
        type=Path,
        help="Exact output directory; overrides the organized country/date/inference layout",
    )
    parser.add_argument(
        "--output-root",
        type=Path,
        help="Base directory for organized outputs; defaults to ROOT/outputs",
    )
    parser.add_argument(
        "--country-code",
        help="ISO 3166-1 alpha-2 code; reverse-geocoded from the coordinates when omitted",
    )
    parser.add_argument(
        "--inference-id",
        help="Inference directory name; generated from UTC time and coordinates when omitted",
    )
    parser.add_argument(
        "--device",
        choices=("auto", "cuda", "cpu"),
        default="auto",
        help="Hide CUDA for a CPU run or require CUDA explicitly",
    )
    parser.add_argument(
        "--tile-size",
        type=int,
        default=128,
        help="Input tile width; smaller values reduce peak GPU memory",
    )
    parser.add_argument(
        "--min-free-gb",
        type=float,
        default=40.0,
        help="Abort before starting when free disk at the output location is lower",
    )
    parser.add_argument(
        "--search-only",
        action="store_true",
        help="Query and rank STAC scenes without downloading imagery or running the model",
    )
    parser.add_argument("--no-preview", action="store_true")
    parser.add_argument(
        "--products",
        nargs="+",
        choices=("MS", "TCI", "NDVI", "IRP"),
        default=["MS", "TCI", "NDVI", "IRP"],
        help="Products to retain in the output directory",
    )
    parser.add_argument(
        "--skip-indices",
        action="store_true",
        help="Do not compute the spectral-index rasters from MS",
    )
    parser.add_argument(
        "--skip-lst",
        action="store_true",
        help="Do not compute the sharpened land-surface-temperature raster (needs AWS credentials for Landsat ST)",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> None:
    options = parse_args(argv)
    if options.tile_size < 16:
        raise SystemExit("--tile-size must be at least 16")
    if options.device == "cpu":
        os.environ["CUDA_VISIBLE_DEVICES"] = ""
    elif options.device == "cuda":
        os.environ.setdefault("CUDA_VISIBLE_DEVICES", "0")
    os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

    target_date = datetime.strptime(options.date, "%Y-%m-%d").strftime("%Y%m%d")
    if options.model is not None:
        model_path = options.model.resolve()
        if not model_path.is_file():
            raise SystemExit(f"Model not found: {model_path}")
    else:
        from s2sr.hub import resolve_checkpoint

        try:
            model_path = resolve_checkpoint()
        except Exception as error:
            raise SystemExit(str(error))

    if os.environ.get("S2SR_SKIP_CHECKSUM", "") == "":
        from s2sr.hub import verify_checkpoint

        try:
            verify_checkpoint(model_path)
        except Exception as error:
            raise SystemExit(str(error))

    if options.output is not None:
        output_dir = options.output.resolve()
    else:
        code = (
            options.country_code or country_code(options.lat, options.lon)
        ).upper()
        inference_id = options.inference_id or new_inference_id(
            options.lat, options.lon
        )
        output_root = (options.output_root or ROOT / "outputs").resolve()
        output_dir = inference_directory(output_root, code, options.date, inference_id)

    anchor = output_dir
    while not anchor.exists():
        anchor = anchor.parent
    free_gb = shutil.disk_usage(anchor).free / 1024**3
    if free_gb < options.min_free_gb:
        raise SystemExit(
            f"Insufficient disk space: {free_gb:.1f} GB free at {anchor}, "
            f"--min-free-gb {options.min_free_gb:g} required"
        )

    work_dir = output_dir / ".work"
    data_dir = work_dir / "data"
    log_dir = work_dir / "logs"
    output_dir.mkdir(parents=True, exist_ok=True)
    work_dir.mkdir(parents=True, exist_ok=True)

    try:
        _run(options, target_date, model_path, output_dir, work_dir, data_dir, log_dir)
    finally:
        shutil.rmtree(work_dir, ignore_errors=True)
        if (
            options.search_only
            and output_dir.is_dir()
            and not any(output_dir.iterdir())
        ):
            output_dir.rmdir()


def _run(
    options: argparse.Namespace,
    target_date: str,
    model_path: Path,
    output_dir: Path,
    work_dir: Path,
    data_dir: Path,
    log_dir: Path,
) -> None:
    from upstream import (
        engine,
        ensure_runtime_env,
        sanitize as sanitize_token,
        has_upstream_token,
        ENGINE_LEGACY_DIRS,
    )

    ensure_runtime_env()
    datautils, inferutils = engine()

    import numpy as np
    import imagecodecs
    import rasterio
    from osgeo import gdal

    original_configure = inferutils.configure
    original_load = inferutils.np.load
    original_makedirs = inferutils.os.makedirs
    original_imread = inferutils.io.imread
    original_save = inferutils.np.save
    original_system = inferutils.os.system
    quota_path = work_dir / "quota.log"
    scratch_dirs: set[str] = set()

    def track_scratch(path_string: str) -> None:
        if path_string.startswith("/tmp/.__"):
            scratch_dirs.add(path_string.rstrip("/"))

    def local_makedirs(path, *args, **kwargs):
        path_string = os.fspath(path)
        track_scratch(path_string)
        if path_string in ENGINE_LEGACY_DIRS:
            return None
        return original_makedirs(path, *args, **kwargs)

    original_mkdir = os.mkdir

    def local_mkdir(path, *args, **kwargs):
        path_string = os.fspath(path)
        track_scratch(path_string)
        return original_mkdir(path, *args, **kwargs)

    def local_system(command: str) -> int:
        if command == "mv /var/log/journal/quota.log.npy /var/log/journal/quota.log":
            generated = quota_path.with_name(quota_path.name + ".npy")
            if generated.exists():
                os.replace(generated, quota_path)
            return 0
        return original_system(command)

    def local_save(file, *args, **kwargs):
        if os.fspath(file) == "/var/log/journal/quota.log":
            file = quota_path
        return original_save(file, *args, **kwargs)

    def local_load(file, *args, **kwargs):
        if os.fspath(file) == "/var/log/journal/quota.log":
            file = quota_path
        return original_load(file, *args, **kwargs)

    def local_imread(file, *args, **kwargs):
        try:
            return original_imread(file, *args, **kwargs)
        except imagecodecs.DelayedImportError:
            with rasterio.open(file) as dataset:
                return np.moveaxis(dataset.read(), 0, -1)

    def local_configure(args) -> None:
        original_configure(args)
        args.savepath = str(output_dir)
        args.datapath = str(data_dir)
        args.logpath = str(log_dir)
        args.model_name = str(model_path)
        args.model_path = str(model_path)
        args.tile = options.tile_size
        args.make_preview = not options.no_preview
        args.force = True
        for directory in (output_dir, data_dir, log_dir):
            directory.mkdir(parents=True, exist_ok=True)

    inferutils.os.makedirs = local_makedirs
    inferutils.os.mkdir = local_mkdir
    inferutils.os.system = local_system
    inferutils.np.save = local_save
    inferutils.np.load = local_load
    inferutils.io.imread = local_imread
    inferutils.configure = local_configure

    import glob as stdlib_glob

    original_glob = stdlib_glob.glob

    def scoped_glob(pattern, *args, **kwargs):
        results = original_glob(pattern, *args, **kwargs)
        try:
            text = os.fspath(pattern)
        except TypeError:
            return results
        if isinstance(text, bytes):
            text = os.fsdecode(text)
        if "/tmp/.__" not in text:
            return results
        return [
            entry
            for entry in results
            if any(
                entry == directory or entry.startswith(directory.rstrip("/") + "/")
                for directory in scratch_dirs
            )
        ]

    stdlib_glob.glob = scoped_glob

    def normalize_products() -> None:
        for path in sorted(output_dir.rglob("*.tif")):
            if any(
                part.startswith(".")
                for part in path.relative_to(output_dir).parts
            ):
                continue
            branded_name = sanitize_token(path.name)
            dataset = gdal.Open(str(path), gdal.GA_Update)
            if dataset is None:
                raise RuntimeError(f"Could not open generated product: {path}")
            for key, value in list(dataset.GetMetadata().items()):
                branded_key = sanitize_token(key)
                branded_value = sanitize_token(value)
                dataset.SetMetadataItem(key, None)
                if branded_key == key and branded_value == value:
                    dataset.SetMetadataItem(branded_key, branded_value)
                    continue
                dataset.SetMetadataItem(branded_key, branded_value)
            dataset.SetMetadataItem("S2SR_MODEL", MODEL_ID)
            dataset.SetMetadataItem("TIFFTAG_SOFTWARE", "S2SR")
            dataset.SetMetadataItem("TIFFTAG_COPYRIGHT", "NeuralQ")
            if path.name.endswith("_MS.tif") and dataset.RasterCount == len(BAND_ORDER):
                for index, band_name in enumerate(BAND_ORDER, start=1):
                    dataset.GetRasterBand(index).SetDescription(band_name)
            dataset.FlushCache()
            dataset = None
            if branded_name != path.name:
                path.replace(path.with_name(branded_name))

        for path in sorted(log_dir.glob("*.json")):
            if not has_upstream_token(path.name):
                continue
            branded = json.loads(sanitize_token(path.read_text(encoding="utf-8")))
            if isinstance(branded, dict):
                branded = {
                    key: value
                    for key, value in branded.items()
                    if "_url" not in key.lower()
                    and not str(value).startswith("http")
                }
            destination = log_dir / sanitize_token(path.name)
            destination.write_text(json.dumps(branded, indent=2) + "\n", encoding="utf-8")
            path.unlink()

    def ensure_uncompressed(tree: Path) -> None:
        for path in sorted(tree.glob("*.tif")):
            with rasterio.open(path) as source:
                profile = source.profile.copy()
                data = source.read()
                descriptions = source.descriptions
                nodata = source.nodata
            for key in ("compress", "compression", "blockxsize", "blockysize", "tiled"):
                profile.pop(key, None)
            temporary = path.with_name(path.name + ".tmp")
            with rasterio.open(temporary, "w", **profile) as target:
                target.write(data)
                if nodata is not None:
                    target.nodata = nodata
                for index, description in enumerate(descriptions, start=1):
                    if description:
                        target.set_band_description(index, description)
            os.replace(temporary, path)

    def product_type(path: Path) -> str | None:
        for product in ("MS", "TCI", "NDVI", "IRP"):
            if path.name.endswith(f"_{product}.tif"):
                return product
        return None

    def flatten_products() -> list[str]:
        sources = []

        def hidden(path: Path) -> bool:
            return any(
                part.startswith(".") for part in path.relative_to(output_dir).parts
            )

        nested = [
            path
            for path in sorted(output_dir.rglob("*.tif"))
            if path.parent != output_dir and not hidden(path)
        ]
        used = set()
        for path in nested:
            product = product_type(path)
            target_name = f"{product}.tif" if product else path.name
            counter = 1
            while target_name in used:
                target_name = (
                    path.stem if counter == 1 else f"{path.stem}_{counter}"
                ) + path.suffix
                counter += 1
            used.add(target_name)
            source = path.relative_to(output_dir)
            os.replace(path, output_dir / target_name)
            sources.append(f"{target_name} <- {source}")
        for directory in sorted(
            (
                item
                for item in output_dir.rglob("*")
                if item.is_dir() and not hidden(item)
            ),
            reverse=True,
        ):
            shutil.rmtree(directory, ignore_errors=True)
        return sources

    def prune_products(selected: set[str]) -> list[str]:
        removed = []
        for path in sorted(output_dir.glob("*.tif")):
            product = product_type(path)
            if product is not None and product not in selected:
                removed.append(f"{path.name}")
                path.unlink()
        return removed

    started_at = utc_now()
    if options.search_only:
        args = SimpleNamespace(date=target_date)
        local_configure(args)
        args.uid = "12345678"
        args.org = "TT"
        args.aoi = datautils.aoi_from_xy((options.lon, options.lat), km=2)
        args.mode = "aoi"
        datautils.search_collection(args)
        args.items = datautils.sort_items_by_recency_and_clouds(args.items)
        print(f"Found and ranked {len(args.items)} candidate scenes:")
        for item in args.items[:12]:
            info = item.get("info", {})
            print(
                f"  {item.get('id', 'unknown')} "
                f"date={info.get('date', 'unknown')} "
                f"clouds={info.get('clouds', 'unknown')}"
            )
        return

    inferutils.test(
        lonlat=(options.lon, options.lat),
        date=options.date,
        simulate=False,
    )
    normalize_products()
    for path in sorted(output_dir.rglob("*.aux.xml")):
        path.unlink()
    sources = flatten_products()
    removed = prune_products(set(options.products))
    ensure_uncompressed(output_dir)

    from spectral_indices import compute_indices, write_indices_readme

    indices_dir = output_dir / "indices"
    indices_records: list = []
    indices_note = None
    if options.skip_indices:
        indices_note = "disabled by --skip-indices"
    elif not (output_dir / "MS.tif").is_file():
        indices_note = "MS product unavailable"
    else:
        indices_records = compute_indices(output_dir / "MS.tif", indices_dir)

    from lst import LSTUnavailable, compute_lst

    lst_record = None
    lst_note = None
    if options.skip_lst:
        lst_note = "disabled by --skip-lst"
    elif not (output_dir / "MS.tif").is_file():
        lst_note = "MS product unavailable"
    else:
        try:
            search_aoi = datautils.aoi_from_xy((options.lon, options.lat), km=6)
            lst_record = compute_lst(
                output_dir / "MS.tif", options.date, search_aoi["bbox"]
            )
        except LSTUnavailable as error:
            lst_note = f"{error.reason}: {error}"
        except Exception as error:  # optional layer must never fail the run
            lst_note = f"error: {error}"

    catalog_records = list(indices_records)
    if lst_record is not None:
        catalog_records.append(lst_record)
    if catalog_records:
        write_indices_readme(indices_dir, catalog_records)

    inventory = [
        format_inventory_record(record)
        for record in product_inventory(sorted(output_dir.glob("*.tif")))
    ]
    engine_log = {}
    for path in sorted(log_dir.glob("*.json")):
        try:
            engine_log[path.name] = json.loads(path.read_text(encoding="utf-8"))
        except ValueError:
            engine_log[path.name] = path.read_text(encoding="utf-8")
    write_metadata_readme(
        output_dir,
        "S2SR Inference Metadata",
        {
            "Run": {
                "inference_id": output_dir.name,
                "country_code": output_dir.parent.parent.name,
                "longitude": options.lon,
                "latitude": options.lat,
                "request_date": options.date,
                "target_date_compact": target_date,
                "started_at": started_at,
                "finished_at": utc_now(),
                "command": " ".join([sys.executable, *sys.argv]),
            },
            "Model": {
                "model_id": MODEL_ID,
                "checkpoint": str(model_path),
                "checkpoint_sha256": sha256_file(model_path),
                "device_setting": options.device,
                "tile_size_pixels": options.tile_size,
            },
            "Data contract": {
                "input_shape": "(50, H, W); five dates x ten bands, date-major",
                "source_band_order": ", ".join(BAND_ORDER),
                "normalization": "reflectance DN / 10000",
                "super_resolution_factor": "10x",
                "output_compression": "NONE",
            },
            "Products": {
                "files": inventory or ["none"],
                "selected": ", ".join(options.products),
                "sources": sources or ["flat"],
                "excluded": removed or ["none"],
            },
            "Indices": {
                "catalog": str(indices_dir / "README.md"),
                "categories": ", ".join(
                    sorted({record["category"] for record in indices_records})
                )
                or "-",
                "count": str(len(indices_records)),
                "files": [
                    f"{Path(record['path']).relative_to(output_dir)}: "
                    f"{record['size_bytes']:,} bytes"
                    for record in indices_records
                ]
                or [indices_note or "none"],
                "note": "single-band float32, compression NONE, nodata NaN",
            },
            "LST": {
                "method": "TsHARP-style sharpening: Landsat C2L2 ST_B10 (30 m) "
                "disaggregated to 1 m with S2SR NDVI + redistributed residuals",
                "source_conversion": "Kelvin = DN x 0.00341802 + 149.0 (USGS Collection 2); "
                "LST_1m = a + b * NDVI_1m + resid_1m",
                "product": (
                    f"{Path(lst_record['path']).relative_to(output_dir)}: "
                    f"{lst_record['width']}x{lst_record['height']}, "
                    f"float32 Celsius, compression=NONE, nodata=NaN, "
                    f"{lst_record['size_bytes']:,} bytes"
                    if lst_record
                    else (lst_note or "none")
                ),
                "source_scene": (
                    f"{lst_record['scene_id']} ({lst_record['scene_date']}, "
                    f"{lst_record['date_offset_days']} days from target)"
                    if lst_record
                    else "-"
                ),
                "fit": (
                    f"LST = {lst_record['intercept_k']:.2f} + "
                    f"{lst_record['slope_k_per_ndvi']:.2f} * NDVI over "
                    f"{lst_record['n_fit']} clear coarse pixels "
                    f"(RMSE {lst_record['rmse_k']:.2f} K)"
                    if lst_record
                    else "-"
                ),
                "validation": (
                    "coarse median inside 230..360 K; coarse means preserved "
                    "by residual redistribution"
                    if lst_record
                    else (lst_note or "none")
                ),
            },
            "Engine log": engine_log or {"note": "not retained"},
        },
    )


if __name__ == "__main__":
    main()
