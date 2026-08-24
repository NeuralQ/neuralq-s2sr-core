"""Shared helpers for organized S2SR inference output directories."""
from __future__ import annotations

import hashlib
import json
import os
from datetime import datetime, timezone
from pathlib import Path


GEO_CACHE = Path(
    os.environ.get(
        "S2SR_GEO_CACHE",
        str(Path.home() / ".cache" / "s2sr" / "geo_cache.json"),
    )
)
NOMINATIM_REVERSE_URL = "https://nominatim.openstreetmap.org/reverse"


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def coordinate_tags(latitude: float, longitude: float) -> str:
    latitude_tag = f"{abs(latitude):.5f}{'N' if latitude >= 0 else 'S'}"
    longitude_tag = f"{abs(longitude):.5f}{'E' if longitude >= 0 else 'W'}"
    return f"{latitude_tag}-{longitude_tag}"


def new_inference_id(
    latitude: float | None = None, longitude: float | None = None
) -> str:
    import secrets

    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    parts = ["inf", stamp]
    if latitude is not None and longitude is not None:
        parts.append(coordinate_tags(latitude, longitude))
    parts.append(secrets.token_hex(2))
    return "-".join(parts)


def _cache_load() -> dict:
    try:
        return json.loads(GEO_CACHE.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return {}


def _cache_save(cache: dict) -> None:
    try:
        GEO_CACHE.parent.mkdir(parents=True, exist_ok=True)
        temporary = GEO_CACHE.with_suffix(".json.tmp")
        temporary.write_text(
            json.dumps(cache, indent=2, sort_keys=True) + "\n", encoding="utf-8"
        )
        os.replace(temporary, GEO_CACHE)
    except OSError:
        pass


def country_code(latitude: float, longitude: float) -> str:
    import requests

    key = f"{latitude:.2f},{longitude:.2f}"
    cache = _cache_load()
    if key in cache:
        return cache[key]
    response = requests.get(
        NOMINATIM_REVERSE_URL,
        params={
            "lat": f"{latitude:.6f}",
            "lon": f"{longitude:.6f}",
            "format": "jsonv2",
            "zoom": 3,
        },
        headers={"User-Agent": "s2sr-mosaic/1.0"},
        timeout=60,
    )
    response.raise_for_status()
    code = str(response.json().get("address", {}).get("country_code", "")).upper()
    if len(code) != 2 or not code.isalpha():
        raise RuntimeError(
            f"Could not resolve an ISO 3166-1 alpha-2 country code for "
            f"{latitude},{longitude}; pass --country-code explicitly"
        )
    cache[key] = code
    _cache_save(cache)
    return code


def inference_directory(
    root: Path, code: str, date_iso: str, inference_id: str
) -> Path:
    return Path(root) / code.upper() / date_iso / inference_id


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as file:
        for chunk in iter(lambda: file.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def plan_signature(**parameters) -> str:
    canonical = json.dumps(parameters, sort_keys=True, separators=(",", ":"))
    return hashlib.sha1(canonical.encode("utf-8")).hexdigest()[:8]


def product_inventory(paths) -> list[dict]:
    import rasterio

    records = []
    for path in paths:
        with rasterio.open(path) as dataset:
            records.append(
                {
                    "name": path.name,
                    "size_bytes": path.stat().st_size,
                    "width": dataset.width,
                    "height": dataset.height,
                    "bands": dataset.count,
                    "dtype": dataset.dtypes[0],
                    "resolution_x": dataset.res[0],
                    "resolution_y": dataset.res[1],
                    "crs": str(dataset.crs),
                    "compression": dataset.compression or "NONE",
                    "sha256": sha256_file(path),
                }
            )
    return records


def format_inventory_record(record: dict) -> str:
    return (
        f"{record['name']}: {record['width']}x{record['height']}, "
        f"{record['bands']} bands {record['dtype']}, "
        f"{record['resolution_x']:g} m, compression={record['compression']}, "
        f"{record['size_bytes']:,} bytes, sha256={record['sha256']}"
    )


def write_metadata_readme(
    directory: Path, title: str, sections: dict[str, dict]
) -> Path:
    lines = [f"# {title}", "", f"Generated: {utc_now()} UTC", ""]
    for section, entries in sections.items():
        lines.append(f"## {section}")
        lines.append("")
        for key, value in entries.items():
            if isinstance(value, (list, tuple)):
                lines.append(f"- {key}:")
                lines.extend(f"  - {item}" for item in value)
            else:
                lines.append(f"- {key}: {value}")
        lines.append("")
    path = Path(directory) / "README.md"
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text("\n".join(lines), encoding="utf-8")
    os.replace(temporary, path)
    return path
