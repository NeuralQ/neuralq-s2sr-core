#!/usr/bin/env python3
"""Catalog and strictly validate every raster under the outputs tree.

Recognizes four canonical product shapes: flat inference products
(``MS.tif`` ...), scene-prefixed products (``S2SR_<scene>_<PRODUCT>.tif``),
mosaic finals (``*_S2SR_<PRODUCT>_1m.tif`` plus ``*_preview.tif``), and
spectral indices (single-band ``float32`` ``*.tiff``). Each record captures
dimensions, bands, dtype, CRS, resolution, bounds, center coordinates,
compression, size, and SHA-256; ``--strict`` exits nonzero when any check
fails. Writes ``catalog.json`` and ``catalog.csv`` at the output root.

Usage::

    python scripts/catalog_outputs.py [--strict] [--no-hash] [--outputs PATH]
"""
import argparse
import csv
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path

from pyproj import Transformer
import rasterio


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_OUTPUTS = ROOT / "outputs"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as file:
        for chunk in iter(lambda: file.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def product_type(path: Path) -> str:
    if path.suffix == ".tiff":
        return "INDEX"
    if path.name.endswith("_preview.tif"):
        return "PREVIEW"
    if path.name in ("MS.tif", "TCI.tif", "NDVI.tif", "IRP.tif"):
        return path.stem
    for product in ("MS", "TCI", "NDVI", "IRP"):
        if path.name.endswith(f"_S2SR_{product}_1m.tif"):
            return product
        if path.name.endswith(f"_{product}.tif"):
            return product
    return "UNKNOWN"


def inspect(path: Path, root: Path, with_hash: bool) -> dict:
    product = product_type(path)
    is_index = product == "INDEX"
    is_preview = product == "PREVIEW"
    expected_bands = 1 if is_index else 10 if product == "MS" else 3
    expected_dtype = (
        "float32" if is_index else "uint16" if product == "MS" else "uint8"
    )
    with rasterio.open(path) as dataset:
        center_x = (dataset.bounds.left + dataset.bounds.right) / 2
        center_y = (dataset.bounds.bottom + dataset.bounds.top) / 2
        center_lon = center_lat = None
        if dataset.crs:
            to_wgs84 = Transformer.from_crs(dataset.crs, "EPSG:4326", always_xy=True)
            center_lon, center_lat = to_wgs84.transform(center_x, center_y)
        checks = {
            "s2sr_name": (
                True
                if is_index or is_preview
                else path.name.startswith("S2SR_")
                or "_S2SR_" in path.name
                or path.name in ("MS.tif", "TCI.tif", "NDVI.tif", "IRP.tif")
            ),
            "known_product": is_index or is_preview or product != "UNKNOWN",
            "band_count": dataset.count == expected_bands,
            "dtype": dataset.dtypes == (expected_dtype,) * expected_bands,
            "one_meter": True
            if is_preview
            else dataset.res == (1.0, 1.0),
            "full_tile_size": True
            if is_preview
            else dataset.width >= 4000 and dataset.height >= 4000,
            "uncompressed": dataset.compression is None,
            "has_crs": dataset.crs is not None,
        }
        record = {
            "path": str(path.relative_to(root)),
            "product": product,
            "valid_sr": all(checks.values()),
            "size_bytes": path.stat().st_size,
            "width": dataset.width,
            "height": dataset.height,
            "bands": dataset.count,
            "dtype": dataset.dtypes[0],
            "crs": str(dataset.crs),
            "resolution_x": dataset.res[0],
            "resolution_y": dataset.res[1],
            "bounds_left": dataset.bounds.left,
            "bounds_bottom": dataset.bounds.bottom,
            "bounds_right": dataset.bounds.right,
            "bounds_top": dataset.bounds.top,
            "center_lon": center_lon,
            "center_lat": center_lat,
            "compression": str(dataset.compression),
            "checks": checks,
            "sha256": sha256(path) if with_hash else None,
        }
    return record


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Catalog and strictly validate generated S2SR products."
    )
    parser.add_argument("--outputs", type=Path, default=DEFAULT_OUTPUTS)
    parser.add_argument("--no-hash", action="store_true")
    parser.add_argument("--strict", action="store_true")
    args = parser.parse_args()
    output_root = args.outputs.resolve()
    output_root.mkdir(parents=True, exist_ok=True)
    records = [
        inspect(path, output_root, not args.no_hash)
        for path in sorted(
            list(output_root.rglob("*.tif")) + list(output_root.rglob("*.tiff"))
        )
    ]
    catalog = {
        "schema_version": 1,
        "generated_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "output_root": str(output_root),
        "product_count": len(records),
        "valid_sr_count": sum(record["valid_sr"] for record in records),
        "invalid_count": sum(not record["valid_sr"] for record in records),
        "products": records,
    }
    (output_root / "catalog.json").write_text(
        json.dumps(catalog, indent=2) + "\n", encoding="utf-8"
    )
    flat_fields = [field for field in records[0] if field != "checks"] if records else []
    with (output_root / "catalog.csv").open("w", encoding="utf-8", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=flat_fields)
        writer.writeheader()
        for record in records:
            writer.writerow({field: record[field] for field in flat_fields})
    print(
        f"Cataloged {len(records)} products: "
        f"{catalog['valid_sr_count']} valid SR, {catalog['invalid_count']} invalid"
    )
    if args.strict and catalog["invalid_count"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
