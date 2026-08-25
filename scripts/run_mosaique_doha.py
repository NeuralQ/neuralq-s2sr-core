#!/usr/bin/env python3
"""Plan and run a Doha S2SR super-resolution time series.

Step 1 - plan the best acquisition dates:

  python scripts/run_mosaique_doha.py --plan-dates \
    --start-date 2020-01-01 --end-date 2026-08-21 --frequency weekly

Queries the Earth Search STAC catalog over the Doha municipality boundary,
groups acquisitions by week (or month), keeps the lowest-cloud-cover scene
per period, and writes outputs/doha_timeseries/dates.json.

Step 2 - run the mosaic time series over those dates:

  python scripts/run_mosaique_doha.py
  python scripts/run_mosaique_doha.py --max-dates 4
  python scripts/run_mosaique_doha.py --workers 6

Each date runs scripts/run_mosaic.py into its own output folder
outputs/QA/<date>/<inference_id> (transient state in .work/, removed on
success) plus a README.md with the run metadata. Final mosaics are written
uncompressed (COMPRESS=NONE). Completed dates are skipped on rerun,
so the series is fully resumable. --workers N runs up to N dates
concurrently; each worker needs roughly 2 GB RAM and 1 GB VRAM, and a
date is never started when another live runner already owns its
workspace or free disk falls below --min-free-gb. Boundary selection
(--boundary-query/--osm-id) and the per-tile timeout are forwarded to
run_mosaic.py.
"""
import sys

sys.dont_write_bytecode = True
import argparse
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import date, datetime, timezone
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import threading
import time

import requests
from shapely.geometry import mapping


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

from output_layout import country_code, inference_directory
import run_mosaic as run_mosaic_module
from run_location_doha import DOHA_LATITUDE, DOHA_LONGITUDE

STAC_URL = "https://earth-search.aws.element84.com/v1/search"
COLLECTION = "sentinel-2-l2a"
DEFAULT_DATES_FILE = ROOT / "outputs" / "doha_timeseries" / "dates.json"
TIMESERIES_MANIFEST = "timeseries.json"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run the Doha S2SR super-resolution time series."
    )
    parser.add_argument(
        "--plan-dates",
        action="store_true",
        help="Query STAC and write the best-per-period dates file, then exit",
    )
    parser.add_argument("--start-date", default="2020-01-01")
    parser.add_argument("--end-date", default=date.today().isoformat())
    parser.add_argument(
        "--frequency",
        choices=("weekly", "monthly"),
        default="weekly",
        help="Grouping used to pick the best date per period",
    )
    parser.add_argument("--dates-file", type=Path, default=DEFAULT_DATES_FILE)
    parser.add_argument("--max-dates", type=int, help="Limit dates processed this run")
    parser.add_argument(
        "--workers",
        type=int,
        default=1,
        help="Run up to N dates concurrently (each worker ~2 GB RAM, ~1 GB VRAM)",
    )
    parser.add_argument(
        "--min-free-gb",
        type=float,
        default=100,
        help="Stop scheduling new dates when free disk drops below this",
    )
    parser.add_argument(
        "--workspace-root",
        type=Path,
        default=ROOT / "outputs" / "doha_timeseries",
    )
    parser.add_argument(
        "--output-root",
        type=Path,
        default=ROOT / "outputs",
        help="Base directory; each date writes to <root>/<CC>/<date>/<inference_id>",
    )
    parser.add_argument(
        "--country-code",
        help="ISO 3166-1 alpha-2 code; reverse-geocoded from central Doha when omitted",
    )
    parser.add_argument("--dry-run", action="store_true")

    parser.add_argument("--products", nargs="+", choices=("MS", "TCI", "NDVI", "IRP"))
    parser.add_argument("--tile-size", type=int)
    parser.add_argument("--grid-step", type=float)
    parser.add_argument("--tile-footprint", type=float)
    parser.add_argument("--retries", type=int)
    parser.add_argument("--max-component-distance-km", type=float, default=20)
    parser.add_argument(
        "--boundary-query",
        default=run_mosaic_module.DEFAULT_BOUNDARY_QUERY,
        help="Nominatim query for the mosaic boundary (forwarded to run_mosaic)",
    )
    parser.add_argument(
        "--osm-id",
        type=int,
        default=run_mosaic_module.DEFAULT_OSM_ID,
        help="Keep only the Nominatim feature with this OSM id (forwarded)",
    )
    parser.add_argument(
        "--tile-timeout",
        type=int,
        default=run_mosaic_module.TILE_TIMEOUT_DEFAULT,
        help="Per-tile subprocess timeout in seconds (forwarded)",
    )
    parser.add_argument("--prune-unselected", action="store_true")
    parser.add_argument("--skip-mosaic", action="store_true")
    return parser.parse_args()


def stac_search(geometry: dict, start: str, end: str) -> list[dict]:
    payload = {
        "collections": [COLLECTION],
        "datetime": f"{start}T00:00:00Z/{end}T23:59:59Z",
        "intersects": geometry,
        "limit": 100,
        "sortby": [{"field": "properties.datetime", "direction": "asc"}],
    }
    features: list[dict] = []
    url = STAC_URL
    method = "POST"
    body = payload
    while True:
        response = None
        for attempt in range(4):
            try:
                if method == "POST":
                    response = requests.post(url, json=body, timeout=120)
                else:
                    response = requests.get(url, params=body, timeout=120)
                response.raise_for_status()
                break
            except requests.RequestException:
                if attempt == 3:
                    raise
                delay = 2**attempt
                print(f"  STAC request failed; retrying in {delay}s...", flush=True)
                time.sleep(delay)
        page = response.json()
        features.extend(page.get("features", []))
        print(f"  fetched {len(features)} scenes...", flush=True)
        link = next(
            (item for item in page.get("links", []) if item.get("rel") == "next"),
            None,
        )
        if link is None:
            return features
        url = link["href"]
        method = link.get("method", "POST").upper()
        body = {**payload, **(link.get("body") or {})}


def period_key(day: date, frequency: str) -> tuple[str, date]:
    if frequency == "weekly":
        iso_year, iso_week, _ = day.isocalendar()
        return f"{iso_year}-W{iso_week:02d}", date.fromisocalendar(iso_year, iso_week, 4)
    return f"{day.year}-{day.month:02d}", date(day.year, day.month, 15)


def plan_dates(args: argparse.Namespace) -> None:
    for name in ("--start-date", "--end-date"):
        datetime.strptime(args.start_date if name == "--start-date" else args.end_date, "%Y-%m-%d")

    print("Fetching Doha boundary...", flush=True)
    boundary_wgs84, _, boundary_metadata = run_mosaic_module.fetch_boundary(
        args.max_component_distance_km,
        query=args.boundary_query,
        osm_id=args.osm_id,
    )
    geometry = mapping(boundary_wgs84)

    print(
        f"Searching {COLLECTION} from {args.start_date} to {args.end_date}...",
        flush=True,
    )
    features = stac_search(geometry, args.start_date, args.end_date)

    groups: dict[str, list[dict]] = defaultdict(list)
    for feature in features:
        properties = feature["properties"]
        day = date.fromisoformat(properties["datetime"][:10])
        key, midpoint = period_key(day, args.frequency)
        groups[key].append(
            {
                "midpoint": midpoint,
                "date": day.isoformat(),
                "cloud_cover": properties.get("eo:cloud_cover", 9999),
                "item_id": feature["id"],
                "mgrs_tile": feature["id"].split("_")[1] if "_" in feature["id"] else None,
            }
        )

    entries = []
    for key in sorted(groups):
        candidates = groups[key]
        best = min(
            candidates,
            key=lambda item: (
                item["cloud_cover"],
                abs((date.fromisoformat(item["date"]) - item["midpoint"]).days),
            ),
        )
        entries.append(
            {
                "period": key,
                "date": best["date"],
                "cloud_cover": round(best["cloud_cover"], 3),
                "item_id": best["item_id"],
                "mgrs_tile": best["mgrs_tile"],
                "candidates": len(candidates),
            }
        )

    clouds = [entry["cloud_cover"] for entry in entries if entry["cloud_cover"] < 9999]
    plan = {
        "schema_version": 1,
        "generated_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "collection": COLLECTION,
        "frequency": args.frequency,
        "start_date": args.start_date,
        "end_date": args.end_date,
        "boundary": boundary_metadata,
        "scene_count": len(features),
        "period_count": len(entries),
        "cloud_cover": {
            "mean": round(sum(clouds) / len(clouds), 3) if clouds else None,
            "min": min(clouds) if clouds else None,
            "max": max(clouds) if clouds else None,
        },
        "entries": entries,
    }

    args.dates_file.parent.mkdir(parents=True, exist_ok=True)
    temporary = args.dates_file.with_suffix(".json.tmp")
    temporary.write_text(json.dumps(plan, indent=2) + "\n", encoding="utf-8")
    os.replace(temporary, args.dates_file)
    print(
        f"Planned {len(entries)} {args.frequency} dates "
        f"(mean cloud {plan['cloud_cover']['mean']}%) -> {args.dates_file}",
        flush=True,
    )


def load_dates(path: Path) -> list[dict]:
    if not path.is_file():
        raise SystemExit(
            f"Dates file not found: {path}\n"
            "Generate it first: python scripts/run_mosaique_doha.py --plan-dates"
        )
    plan = json.loads(path.read_text(encoding="utf-8"))
    return plan["entries"]


def executable(name: str) -> str:
    path = shutil.which(name)
    if path is None:
        raise RuntimeError(f"Required command not found: {name}")
    return path


def build_command(
    args: argparse.Namespace,
    entry_date: str,
    output_dir: Path,
    inference_id: str,
) -> list[str]:
    command = [
        sys.executable,
        str(ROOT / "scripts" / "run_mosaic.py"),
        "--date",
        entry_date,
        "--output-dir",
        str(output_dir),
        "--inference-id",
        inference_id,
        "--max-component-distance-km",
        str(args.max_component_distance_km),
    ]
    if args.products:
        command += ["--products", *args.products]
    if args.tile_size:
        command += ["--tile-size", str(args.tile_size)]
    if args.grid_step:
        command += ["--grid-step", str(args.grid_step)]
    if args.tile_footprint:
        command += ["--tile-footprint", str(args.tile_footprint)]
    if args.retries:
        command += ["--retries", str(args.retries)]
    if args.prune_unselected:
        command.append("--prune-unselected")
    if args.skip_mosaic:
        command.append("--skip-mosaic")
    command += [
        "--boundary-query",
        args.boundary_query,
        "--osm-id",
        str(args.osm_id),
        "--tile-timeout",
        str(args.tile_timeout),
    ]
    return command


def finals_present(output_dir: Path, products: list[str]) -> bool:
    """Lightweight presence check; run_mosaic re-validates authoritatively."""
    if not products:
        return False
    return all(
        list(output_dir.glob(f"*_{product}_1m.tif")) for product in products
    )


def load_manifest(base: Path) -> dict:
    path = base / TIMESERIES_MANIFEST
    if path.is_file():
        return json.loads(path.read_text(encoding="utf-8"))
    return {
        "schema_version": 1,
        "created_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "updated_at": None,
        "entries": {},
    }


def save_manifest(base: Path, manifest: dict) -> None:
    manifest["updated_at"] = datetime.now(timezone.utc).isoformat(timespec="seconds")
    path = base / TIMESERIES_MANIFEST
    temporary = path.with_suffix(".json.tmp")
    temporary.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    os.replace(temporary, path)


def _pid_command(pid: int) -> str | None:
    """Return the command line of a live process, or None when it does not exist."""
    result = subprocess.run(
        ["ps", "-ww", "-p", str(pid), "-o", "command="],
        capture_output=True,
        text=True,
        timeout=10,
    )
    return result.stdout.strip() or None


def workspace_busy(workspace: Path) -> bool:
    pid_path = workspace / "runner.pid"
    if not pid_path.is_file():
        return False
    try:
        pid = int(pid_path.read_text().strip())
    except ValueError:
        return False
    try:
        command = _pid_command(pid)
    except (OSError, subprocess.TimeoutExpired):
        return True
    if command is None:
        return False
    return "run_mosaic" in command


def disk_has_room(min_free_gb: float) -> bool:
    return shutil.disk_usage(ROOT).free > min_free_gb * 1024**3


def main() -> None:
    args = parse_args()
    if args.plan_dates:
        plan_dates(args)
        return

    entries = load_dates(args.dates_file)
    selected_products = args.products or ["MS", "TCI", "NDVI", "IRP"]
    args.workspace_root.mkdir(parents=True, exist_ok=True)
    args.output_root.mkdir(parents=True, exist_ok=True)
    manifest = load_manifest(args.workspace_root)
    code = (
        args.country_code or country_code(DOHA_LATITUDE, DOHA_LONGITUDE)
    ).upper()
    lock = threading.Lock()
    stop_scheduling = threading.Event()

    def date_paths(entry_date: str) -> tuple[Path, str]:
        inference_id = run_mosaic_module.mosaic_inference_id(
            entry_date,
            args.grid_step or run_mosaic_module.GRID_STEP_DEFAULT,
            args.tile_footprint or run_mosaic_module.TILE_FOOTPRINT_DEFAULT,
            selected_products,
            args.max_component_distance_km,
            args.boundary_query,
            args.osm_id,
        )
        output_dir = inference_directory(
            args.output_root.resolve(), code, entry_date, inference_id
        )
        return output_dir, inference_id

    def record(entry_date: str, payload: dict) -> None:
        with lock:
            merged = manifest["entries"].get(entry_date, {})
            merged.update(payload)
            manifest["entries"][entry_date] = merged
            save_manifest(args.workspace_root, manifest)

    def worker(entry: dict, delay: float = 0) -> None:
        entry_date = entry["date"]
        workspace = date_paths(entry_date)[0] / ".work"
        output_dir, inference_id = date_paths(entry_date)

        if workspace_busy(workspace):
            print(f"[{entry_date}] another runner owns the workspace, skipping", flush=True)
            record(entry_date, {"state": "skipped-running", "workspace": str(workspace)})
            return
        if stop_scheduling.is_set():
            return
        if not disk_has_room(args.min_free_gb):
            stop_scheduling.set()
            print(
                f"[{entry_date}] free disk below {args.min_free_gb} GB; "
                "no new dates will be started",
                flush=True,
            )
            return

        command = build_command(args, entry_date, output_dir, inference_id)
        if args.dry_run:
            print(f"[{entry_date}] dry-run: {' '.join(command)}", flush=True)
            return
        if delay:
            time.sleep(delay)
        print(f"[{entry_date}] running: {' '.join(command)}", flush=True)
        started = time.monotonic()
        result = subprocess.run(command, cwd=ROOT)
        duration = round(time.monotonic() - started, 1)
        state = "completed" if result.returncode == 0 else "failed"
        rec = {
            "state": state,
            "period": entry["period"],
            "cloud_cover": entry["cloud_cover"],
            "workspace": str(workspace),
            "output_dir": str(output_dir),
            "duration_seconds": duration,
            "returncode": result.returncode,
            "completed_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        }
        record(entry_date, rec)
        free_gb = shutil.disk_usage(ROOT).free / 1024**3
        print(
            f"[{entry_date}] {state} in {duration:.0f}s; free disk {free_gb:.0f} GB",
            flush=True,
        )

    pending = []
    for entry in entries:
        entry_date = entry["date"]
        output_dir, _ = date_paths(entry_date)
        workspace = output_dir / ".work"
        done = finals_present(output_dir, selected_products)
        if done:
            record(entry_date, {"state": "completed", "workspace": str(workspace), "output_dir": str(output_dir)})
            continue
        if not workspace_busy(workspace):
            pending.append(entry)
    if args.max_dates is not None:
        pending = pending[: args.max_dates]

    if not pending:
        print("Nothing to do: all planned dates are complete or already running", flush=True)
        return

    print(
        f"Processing {len(pending)} dates with up to {args.workers} worker(s)",
        flush=True,
    )
    try:
        if args.workers <= 1 or len(pending) == 1:
            for entry in pending:
                worker(entry)
        else:
            with ThreadPoolExecutor(max_workers=min(args.workers, len(pending))) as pool:
                futures = [
                    pool.submit(worker, entry, min(index * 20, 120))
                    for index, entry in enumerate(pending)
                ]
                for future in as_completed(futures):
                    future.result()
    except KeyboardInterrupt:
        print("Interrupted; rerun the same command to resume", flush=True)
        raise SystemExit(130)

    states = [rec.get("state") for rec in manifest["entries"].values()]
    print(
        f"Time series update: {states.count('completed')} completed, "
        f"{states.count('failed')} failed, {len(entries)} planned dates total",
        flush=True,
    )


if __name__ == "__main__":
    main()
