#!/usr/bin/env python3
"""Build a resumable S2SR mosaic over the Doha municipality.

Fetches the OpenStreetMap boundary (relation 27332, offshore components
beyond ``--max-component-distance-km`` excluded), grids it into overlapping
4.12 km tiles on a 4 km step, and processes tiles center-out through
per-tile subprocesses of ``run_location.py``. Every product is strictly
validated (dimensions, bands, dtype, CRS, resolution, compression) before
being recorded in an atomically written manifest; finished tiles are skipped
on rerun. When all tiles are complete the runner assembles boundary-clipped,
uncompressed BigTIFFs per product via VRT + gdalwarp, adds a downsampled
uncompressed GeoTIFF preview, writes a metadata README, and then removes its
transient ``.work/`` directory inside the output folder.

All state lives in ``<output_dir>/.work/`` (manifest, boundary, per-tile
products, caches, locks). Interrupted runs resume by rerunning the same
command; once the mosaic completes, only final products remain. If the
final rasters are already present and validated, rerunning is a no-op.

Usage::

    python scripts/run_mosaic.py \
        [--date 2026-08-14] [--products MS TCI] [--prune-unselected] \
        [--plan-only] [--skip-mosaic]
"""
import sys

sys.dont_write_bytecode = True
import argparse
from datetime import datetime, timezone
import fcntl
import json
import math
import os
from pathlib import Path
import shutil
import subprocess
import sys
import time

from pyproj import Transformer
import rasterio
import requests
from shapely.geometry import MultiPolygon, box, mapping, shape
from shapely.ops import transform, unary_union

from output_layout import (
    country_code,
    inference_directory,
    plan_signature,
    product_inventory,
    format_inventory_record,
    utc_now,
    write_metadata_readme,
)


ROOT = Path(__file__).resolve().parents[1]
NOMINATIM_URL = (
    "https://nominatim.openstreetmap.org/search"
    "?q=Doha%2C+Qatar&format=geojson&polygon_geojson=1&limit=5"
)
DOHA_RELATION_ID = 27332
UTM_CRS = "EPSG:32639"
PRODUCTS = {"MS": 10, "TCI": 3, "NDVI": 3, "IRP": 3}
PRODUCT_DTYPES = {"MS": "uint16", "TCI": "uint8", "NDVI": "uint8", "IRP": "uint8"}
GRID_STEP_DEFAULT = 4000.0
TILE_FOOTPRINT_DEFAULT = 4120.0


def mosaic_inference_id(
    date_iso: str,
    grid_step: float,
    tile_footprint: float,
    products: list[str],
    max_component_distance_km: float,
) -> str:
    signature = plan_signature(
        date=date_iso,
        grid_step=grid_step,
        tile_footprint=tile_footprint,
        products=list(products),
        max_component_distance_km=max_component_distance_km,
    )
    return f"mosaic-{date_iso.replace('-', '')}-{signature}"


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build a resumable S2SR mosaic over the Doha municipality."
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        help="Exact final mosaic directory; overrides the organized layout",
    )
    parser.add_argument(
        "--output-root",
        type=Path,
        help="Base directory for organized outputs; defaults to ROOT/outputs",
    )
    parser.add_argument(
        "--country-code",
        help="ISO 3166-1 alpha-2 code; reverse-geocoded from the boundary centroid when omitted",
    )
    parser.add_argument(
        "--inference-id",
        help="Inference directory name; deterministic from the mosaic plan when omitted",
    )
    parser.add_argument("--date", default="2026-08-14")
    parser.add_argument("--tile-size", type=int, default=128)
    parser.add_argument("--grid-step", type=float, default=GRID_STEP_DEFAULT)
    parser.add_argument(
        "--tile-footprint", type=float, default=TILE_FOOTPRINT_DEFAULT
    )
    parser.add_argument("--retries", type=int, default=2)
    parser.add_argument(
        "--products",
        nargs="+",
        choices=tuple(PRODUCTS),
        default=list(PRODUCTS),
        help="Final products to retain and mosaic",
    )
    parser.add_argument(
        "--prune-unselected",
        action="store_true",
        help="Delete per-tile preview products not listed by --products",
    )
    parser.add_argument(
        "--max-component-distance-km",
        type=float,
        default=20,
        help="Exclude detached administrative islands farther from mainland Doha",
    )
    parser.add_argument("--plan-only", action="store_true")
    parser.add_argument("--skip-mosaic", action="store_true")
    return parser.parse_args()


def write_json(path: Path, value: dict) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2) + "\n", encoding="utf-8")
    os.replace(temporary, path)


def fetch_boundary(max_distance_km: float) -> tuple[MultiPolygon, MultiPolygon, dict]:
    response = requests.get(
        NOMINATIM_URL,
        headers={"User-Agent": "s2sr-mosaic/1.0"},
        timeout=120,
    )
    response.raise_for_status()
    feature = next(
        item
        for item in response.json()["features"]
        if item["properties"].get("osm_id") == DOHA_RELATION_ID
    )
    boundary_wgs84 = shape(feature["geometry"])
    to_utm = Transformer.from_crs("EPSG:4326", UTM_CRS, always_xy=True)
    boundary_utm = transform(to_utm.transform, boundary_wgs84)

    components = sorted(boundary_utm.geoms, key=lambda geometry: geometry.area, reverse=True)
    mainland = components[0]
    retained = [
        component
        for component in components
        if component.distance(mainland) <= max_distance_km * 1000
    ]
    excluded = [component for component in components if component not in retained]
    filtered_utm = unary_union(retained)

    to_wgs84 = Transformer.from_crs(UTM_CRS, "EPSG:4326", always_xy=True)
    filtered_wgs84 = transform(to_wgs84.transform, filtered_utm)
    metadata = {
        "source": "OpenStreetMap/Nominatim",
        "osm_relation": DOHA_RELATION_ID,
        "crs": UTM_CRS,
        "area_km2": round(filtered_utm.area / 1_000_000, 3),
        "component_count": len(retained),
        "excluded_component_count": len(excluded),
        "excluded_area_km2": round(
            sum(component.area for component in excluded) / 1_000_000, 3
        ),
        "max_component_distance_km": max_distance_km,
    }
    return filtered_wgs84, filtered_utm, metadata


def save_boundary(path: Path, boundary, metadata: dict) -> None:
    feature_collection = {
        "type": "FeatureCollection",
        "name": "Doha municipality (local components)",
        "features": [
            {
                "type": "Feature",
                "properties": metadata,
                "geometry": mapping(boundary),
            }
        ],
    }
    write_json(path, feature_collection)


def make_tiles(
    boundary_utm,
    grid_step: float,
    footprint: float,
) -> list[dict]:
    min_x, min_y, max_x, max_y = boundary_utm.bounds
    half = footprint / 2
    start_x = math.floor(min_x / grid_step) * grid_step + grid_step / 2
    start_y = math.floor(min_y / grid_step) * grid_step + grid_step / 2
    center = boundary_utm.centroid
    candidates = []

    y = start_y
    row = 0
    while y <= max_y + grid_step:
        x = start_x
        column = 0
        while x <= max_x + grid_step:
            footprint_geometry = box(x - half, y - half, x + half, y + half)
            if footprint_geometry.intersects(boundary_utm):
                candidates.append(
                    {
                        "easting": x,
                        "northing": y,
                        "row": row,
                        "column": column,
                        "priority": math.hypot(x - center.x, y - center.y),
                    }
                )
            x += grid_step
            column += 1
        y += grid_step
        row += 1

    candidates.sort(key=lambda tile: tile["priority"])
    to_wgs84 = Transformer.from_crs(UTM_CRS, "EPSG:4326", always_xy=True)
    tiles = []
    for index, tile in enumerate(candidates, start=1):
        longitude, latitude = to_wgs84.transform(tile["easting"], tile["northing"])
        tiles.append(
            {
                "id": f"tile-{index:03d}",
                "easting": tile["easting"],
                "northing": tile["northing"],
                "longitude": round(longitude, 8),
                "latitude": round(latitude, 8),
                "row": tile["row"],
                "column": tile["column"],
                "status": "pending",
                "attempts": 0,
                "duration_seconds": None,
                "started_at": None,
                "completed_at": None,
                "error": None,
                "products": {},
            }
        )
    return tiles


def expected_raster(boundary_utm, products: list[str]) -> dict:
    min_x, min_y, max_x, max_y = boundary_utm.bounds
    aligned_bounds = [
        math.floor(min_x),
        math.floor(min_y),
        math.ceil(max_x),
        math.ceil(max_y),
    ]
    width = aligned_bounds[2] - aligned_bounds[0]
    height = aligned_bounds[3] - aligned_bounds[1]
    return {
        "crs": UTM_CRS,
        "resolution_meters": 1,
        "bounds": aligned_bounds,
        "width": width,
        "height": height,
        "pixel_count": width * height,
        "products": {
            product: {
                "bands": PRODUCTS[product],
                "dtype": PRODUCT_DTYPES[product],
                "logical_uncompressed_bytes": (
                    width
                    * height
                    * PRODUCTS[product]
                    * (2 if PRODUCT_DTYPES[product] == "uint16" else 1)
                ),
            }
            for product in products
        },
    }


def load_or_create_manifest(
    workspace: Path,
    options: argparse.Namespace,
    boundary_metadata: dict,
    tiles: list[dict],
    expected: dict,
    output_dir: Path,
) -> dict:
    path = workspace / "manifest.json"
    if path.exists():
        manifest = json.loads(path.read_text(encoding="utf-8"))
        if (
            manifest["date"] != options.date
            or len(manifest["tiles"]) != len(tiles)
            or manifest.get("selected_products") != options.products
            or manifest.get("expected_raster") != expected
            or manifest.get("output_dir") != str(output_dir)
        ):
            raise RuntimeError(
                f"Existing workspace {workspace} does not match this mosaic plan; "
                f"remove {workspace} to replan"
            )
        for tile in manifest["tiles"]:
            if tile["status"] == "running":
                tile["status"] = "pending"
        return manifest

    manifest = {
        "schema_version": 1,
        "name": "Doha municipality S2SR mosaic",
        "date": options.date,
        "state": "planned",
        "created_at": utc_now(),
        "updated_at": utc_now(),
        "started_at": None,
        "completed_at": None,
        "boundary": boundary_metadata,
        "workspace": str(workspace),
        "grid": {
            "crs": UTM_CRS,
            "step_meters": options.grid_step,
            "tile_footprint_meters": options.tile_footprint,
            "tile_count": len(tiles),
        },
        "tile_size": options.tile_size,
        "retries": options.retries,
        "selected_products": options.products,
        "output_dir": str(output_dir),
        "expected_raster": expected,
        "tiles": tiles,
        "products": {},
    }
    write_json(path, manifest)
    return manifest


def save_manifest(workspace: Path, manifest: dict) -> None:
    manifest["updated_at"] = utc_now()
    write_json(workspace / "manifest.json", manifest)


def find_products(tile_output: Path, selected_products: list[str]) -> dict[str, str]:
    products = {}
    for product in selected_products:
        band_count = PRODUCTS[product]
        matches = sorted(
            list(tile_output.glob(f"{product}.tif"))
            + list(tile_output.rglob(f"*_{product}.tif"))
        )
        if not matches:
            raise RuntimeError(f"Missing {product} output")
        path = matches[-1]
        with rasterio.open(path) as dataset:
            if dataset.count != band_count:
                raise RuntimeError(
                    f"{path.name} has {dataset.count} bands; expected {band_count}"
                )
            if dataset.crs is None or dataset.crs.to_epsg() != 32639:
                raise RuntimeError(f"{path.name} has unexpected CRS {dataset.crs}")
            if dataset.width != 4120 or dataset.height != 4120:
                raise RuntimeError(
                    f"{path.name} has unexpected dimensions "
                    f"{dataset.width}x{dataset.height}"
                )
            if dataset.res != (1.0, 1.0):
                raise RuntimeError(f"{path.name} has unexpected resolution {dataset.res}")
            if dataset.dtypes != (PRODUCT_DTYPES[product],) * band_count:
                raise RuntimeError(f"{path.name} has unexpected dtypes {dataset.dtypes}")
            if dataset.compression is not None:
                raise RuntimeError(
                    f"{path.name} has unexpected compression {dataset.compression}"
                )
        products[product] = str(path)
    return products


def run_tile(
    workspace: Path,
    options: argparse.Namespace,
    tile: dict,
) -> tuple[bool, str | None]:
    tile_output = workspace / "tiles" / tile["id"]
    tile_log = workspace / "logs" / f"{tile['id']}.log"
    command = [
        sys.executable,
        str(ROOT / "scripts" / "run_location.py"),
        "--lon",
        str(tile["longitude"]),
        "--lat",
        str(tile["latitude"]),
        "--date",
        options.date,
        "--output",
        str(tile_output),
        "--tile-size",
        str(options.tile_size),
        "--products",
        *options.products,
        "--skip-indices",
    ]
    environment = os.environ.copy()
    environment["PYTHONUNBUFFERED"] = "1"
    started = time.monotonic()
    with tile_log.open("a", encoding="utf-8") as log:
        log.write(f"\n[{utc_now()}] command: {' '.join(command)}\n")
        log.flush()
        result = subprocess.run(
            command,
            cwd=ROOT,
            env=environment,
            stdout=log,
            stderr=subprocess.STDOUT,
            text=True,
        )
    tile["duration_seconds"] = round(
        (tile.get("duration_seconds") or 0) + time.monotonic() - started,
        1,
    )
    if result.returncode != 0:
        return False, f"run_location exited with status {result.returncode}"
    try:
        tile["products"] = find_products(tile_output, options.products)
    except Exception as error:
        return False, str(error)

    if options.prune_unselected:
        selected_suffixes = tuple(f"_{product}.tif" for product in options.products)
        flat_names = tuple(f"{product}.tif" for product in options.products)
        for path in tile_output.rglob("*.tif"):
            if path.name not in flat_names and not path.name.endswith(
                selected_suffixes
            ):
                path.unlink()
    return True, None


def executable(name: str) -> str:
    path = shutil.which(name)
    if path is None:
        raise RuntimeError(f"Required command not found: {name}")
    return path


def validate_final_raster(path: Path, product: str, expected: dict) -> dict:
    specification = expected["products"][product]
    with rasterio.open(path) as dataset:
        observed_bounds = [round(value) for value in dataset.bounds]
        errors = []
        if dataset.width != expected["width"] or dataset.height != expected["height"]:
            errors.append(f"dimensions={dataset.width}x{dataset.height}")
        if dataset.count != specification["bands"]:
            errors.append(f"bands={dataset.count}")
        if dataset.dtypes != (specification["dtype"],) * specification["bands"]:
            errors.append(f"dtypes={dataset.dtypes}")
        if dataset.crs is None or dataset.crs.to_epsg() != 32639:
            errors.append(f"crs={dataset.crs}")
        if dataset.res != (1.0, 1.0):
            errors.append(f"resolution={dataset.res}")
        if observed_bounds != expected["bounds"]:
            errors.append(f"bounds={observed_bounds}")
        if dataset.compression is not None:
            errors.append(f"compression={dataset.compression}")
        if errors:
            raise RuntimeError(
                f"Final {product} validation failed: " + ", ".join(errors)
            )
        return {
            "path": str(path),
            "bands": dataset.count,
            "width": dataset.width,
            "height": dataset.height,
            "pixel_count": dataset.width * dataset.height,
            "dtype": dataset.dtypes[0],
            "compression": str(dataset.compression),
            "crs": str(dataset.crs),
            "resolution": list(dataset.res),
            "bounds": list(dataset.bounds),
            "logical_uncompressed_bytes": specification[
                "logical_uncompressed_bytes"
            ],
            "size_bytes": path.stat().st_size,
            "validation": "passed",
        }


def finals_complete(output_dir: Path, products: list[str]) -> bool:
    for product in products:
        if not list(Path(output_dir).glob(f"*_{product}_1m.tif")):
            return False
    return bool(products)


def build_mosaic(
    workspace: Path,
    manifest: dict,
    boundary_path: Path,
    final_dir: Path,
) -> None:
    final_dir.mkdir(exist_ok=True)
    compact_date = manifest["date"].replace("-", "")
    generated = {}

    for product in manifest["selected_products"]:
        sources = [tile["products"][product] for tile in manifest["tiles"]]
        vrt = final_dir / f"Doha_{compact_date}_{product}.vrt"
        destination = final_dir / f"Doha_{compact_date}_S2SR_{product}_1m.tif"
        subprocess.run(
            [
                executable("gdalbuildvrt"),
                "-overwrite",
                "-resolution",
                "highest",
                "-srcnodata",
                "0",
                "-vrtnodata",
                "0",
                str(vrt),
                *sources,
            ],
            check=True,
        )

        creation_options = [
            "-co",
            "TILED=YES",
            "-co",
            "BLOCKXSIZE=512",
            "-co",
            "BLOCKYSIZE=512",
            "-co",
            "COMPRESS=NONE",
            "-co",
            "BIGTIFF=YES",
        ]

        subprocess.run(
            [
                executable("gdalwarp"),
                "-overwrite",
                "-cutline",
                str(boundary_path),
                "-crop_to_cutline",
                "-dstnodata",
                "0",
                "-tr",
                "1",
                "1",
                "-tap",
                "-r",
                "near",
                "-multi",
                "-wo",
                "NUM_THREADS=ALL_CPUS",
                *creation_options,
                str(vrt),
                str(destination),
            ],
            check=True,
        )
        generated[product] = validate_final_raster(
            destination,
            product,
            manifest["expected_raster"],
        )
        vrt.unlink()
        print(f"Mosaic {product} ready: {destination}", flush=True)

    if "TCI" in generated:
        preview_source = Path(generated["TCI"]["path"])
        preview = final_dir / f"Doha_{compact_date}_preview.tif"
        subprocess.run(
            [
                executable("gdal_translate"),
                "-q",
                "-of",
                "GTiff",
                "-co",
                "COMPRESS=NONE",
                "-outsize",
                "2048",
                "0",
                str(preview_source),
                str(preview),
            ],
            check=True,
        )
        generated["preview"] = {
            "path": str(preview),
            "size_bytes": preview.stat().st_size,
        }
    for path in sorted(final_dir.rglob("*.aux.xml")):
        path.unlink()
    manifest["products"] = generated


def write_mosaic_readme(
    output_dir: Path, manifest: dict, options: argparse.Namespace
) -> None:
    import run_location as run_location_module

    completed = sum(
        1 for tile in manifest["tiles"] if tile["status"] == "completed"
    )
    failed = [tile for tile in manifest["tiles"] if tile["status"] != "completed"]
    validations = {
        Path(details["path"]).name: details.get("validation")
        for details in manifest.get("products", {}).values()
        if isinstance(details, dict) and "path" in details
    }
    inventory = []
    for record in product_inventory(sorted(output_dir.glob("*.tif"))):
        row = format_inventory_record(record)
        validation = validations.get(record["name"])
        inventory.append(f"{row}, validation={validation or 'not validated'}")
    preview_entry = manifest.get("products", {}).get("preview")
    previews = (
        [
            f"{Path(preview_entry['path']).name}: "
            f"{preview_entry['size_bytes']:,} bytes, compression=NONE"
        ]
        if preview_entry
        else []
    )
    checkpoint = ROOT / "models" / f"{run_location_module.MODEL_ID}.pt"
    model_section = {
        "model_id": run_location_module.MODEL_ID,
        "checkpoint": str(checkpoint),
        "device": "per-tile subprocess of scripts/run_location.py",
    }
    if checkpoint.is_file():
        from output_layout import sha256_file

        model_section["checkpoint_sha256"] = sha256_file(checkpoint)
    write_metadata_readme(
        output_dir,
        "S2SR Mosaic Metadata",
        {
            "Run": {
                "inference_id": output_dir.name,
                "country_code": output_dir.parent.parent.name,
                "request_date": manifest["date"],
                "state": manifest["state"],
                "workspace": manifest.get("workspace", str(output_dir / ".work")),
                "started_at": manifest.get("started_at"),
                "completed_at": manifest.get("completed_at"),
                "tiles_completed": completed,
                "tiles_total": len(manifest["tiles"]),
                "tiles_failed": [tile["id"] for tile in failed] or "none",
            },
            "Boundary": manifest.get("boundary", {}),
            "Expected raster": {
                "width_pixels": manifest["expected_raster"]["width"],
                "height_pixels": manifest["expected_raster"]["height"],
                "resolution_meters": manifest["expected_raster"][
                    "resolution_meters"
                ],
                "products": ", ".join(manifest["selected_products"]),
            },
            "Model": model_section,
            "Products": {"rasters": inventory or ["none"]},
            "Previews": {"files": previews or ["none"]},
        },
    )


def main() -> None:
    options = parse_args()
    if options.tile_size < 16:
        raise SystemExit("--tile-size must be at least 16")
    datetime.strptime(options.date, "%Y-%m-%d")

    boundary_wgs84, boundary_utm, boundary_metadata = fetch_boundary(
        options.max_component_distance_km
    )
    if options.output_dir is not None:
        output_dir = options.output_dir.resolve()
    else:
        to_wgs84 = Transformer.from_crs(UTM_CRS, "EPSG:4326", always_xy=True)
        centroid = boundary_utm.centroid
        center_lon, center_lat = to_wgs84.transform(centroid.x, centroid.y)
        code = (
            options.country_code or country_code(center_lat, center_lon)
        ).upper()
        inference_id = options.inference_id or mosaic_inference_id(
            options.date,
            options.grid_step,
            options.tile_footprint,
            options.products,
            options.max_component_distance_km,
        )
        output_root = (options.output_root or ROOT / "outputs").resolve()
        output_dir = inference_directory(output_root, code, options.date, inference_id)
    output_dir.mkdir(parents=True, exist_ok=True)

    if not options.skip_mosaic and finals_complete(output_dir, options.products):
        print(f"Mosaic already complete: {output_dir}", flush=True)
        return

    workspace = output_dir / ".work"
    for directory in (
        workspace,
        workspace / "logs",
        workspace / "tiles",
        workspace / "work",
    ):
        directory.mkdir(parents=True, exist_ok=True)

    lock_file = (workspace / "runner.lock").open("w", encoding="utf-8")
    try:
        fcntl.flock(lock_file, fcntl.LOCK_EX | fcntl.LOCK_NB)
    except BlockingIOError:
        raise SystemExit(f"Another mosaic runner already owns {workspace}")
    lock_file.write(str(os.getpid()))
    lock_file.flush()
    (workspace / "runner.pid").write_text(f"{os.getpid()}\n", encoding="utf-8")

    boundary_path = workspace / "boundary.geojson"
    save_boundary(boundary_path, boundary_wgs84, boundary_metadata)
    tiles = make_tiles(boundary_utm, options.grid_step, options.tile_footprint)
    expected = expected_raster(boundary_utm, options.products)
    manifest = load_or_create_manifest(
        workspace, options, boundary_metadata, tiles, expected, output_dir
    )
    print(
        f"Plan: {len(tiles)} tiles over {boundary_metadata['area_km2']:.1f} km2; "
        f"final={expected['width']}x{expected['height']} at 1m; "
        f"products={','.join(options.products)}; workspace={workspace}; "
        f"output={output_dir}",
        flush=True,
    )
    if options.plan_only:
        print(f"Manifest written: {workspace / 'manifest.json'}", flush=True)
        return

    manifest["state"] = "running"
    manifest["started_at"] = manifest["started_at"] or utc_now()
    save_manifest(workspace, manifest)
    total = len(manifest["tiles"])

    for position, tile in enumerate(manifest["tiles"], start=1):
        try:
            existing_products = find_products(
                workspace / "tiles" / tile["id"], options.products
            )
        except Exception:
            existing_products = None
        if existing_products:
            tile["products"] = existing_products
            tile["status"] = "completed"
            tile["error"] = None
            save_manifest(workspace, manifest)
            print(f"[{position}/{total}] {tile['id']} already complete", flush=True)
            continue

        maximum_attempts = options.retries + 1
        run_attempts = 0
        while run_attempts < maximum_attempts and tile["status"] != "completed":
            run_attempts += 1
            tile["attempts"] += 1
            tile["status"] = "running"
            tile["started_at"] = tile["started_at"] or utc_now()
            tile["error"] = None
            save_manifest(workspace, manifest)
            print(
                f"[{position}/{total}] {tile['id']} attempt "
                f"{run_attempts}/{maximum_attempts} "
                f"(total {tile['attempts']}) at "
                f"{tile['longitude']:.5f},{tile['latitude']:.5f}",
                flush=True,
            )
            success, error = run_tile(workspace, options, tile)
            if success:
                tile["status"] = "completed"
                tile["completed_at"] = utc_now()
                tile["error"] = None
                print(
                    f"[{position}/{total}] {tile['id']} completed in "
                    f"{tile['duration_seconds']:.0f}s",
                    flush=True,
                )
            else:
                tile["status"] = "failed"
                tile["error"] = error
                print(f"[{position}/{total}] {tile['id']} failed: {error}", flush=True)
                if run_attempts < maximum_attempts:
                    time.sleep(15)
            save_manifest(workspace, manifest)

    failures = [tile for tile in manifest["tiles"] if tile["status"] != "completed"]
    if failures:
        manifest["state"] = "partial"
        save_manifest(workspace, manifest)
        raise SystemExit(
            f"{len(failures)} tiles remain incomplete; rerun the same command to resume"
        )

    if not options.skip_mosaic:
        manifest["state"] = "mosaicking"
        save_manifest(workspace, manifest)
        build_mosaic(workspace, manifest, boundary_path, output_dir)

    manifest["state"] = "completed"
    manifest["completed_at"] = utc_now()
    save_manifest(workspace, manifest)
    write_mosaic_readme(output_dir, manifest, options)
    if not options.skip_mosaic:
        lock_file.close()
        shutil.rmtree(workspace, ignore_errors=True)
    print("Doha mosaic completed successfully", flush=True)


if __name__ == "__main__":
    main()
