#!/usr/bin/env python3
"""Report live progress of a mosaic workspace.

Reads ``manifest.json`` from the transient ``.work/`` directory of a mosaic
output folder and prints state, tile counts, runner liveness (via PID),
elapsed time, per-tile averages, ETA, expected final raster specs,
currently running and failed tiles, and finished product paths. Once a
mosaic completes, ``.work`` is removed and the script reports the completed
finals found in the output folder instead.

Usage::

    python scripts/mosaic_status.py \
        [--workspace outputs/QA/2026-08-14/<mosaic_id>/.work]
"""
import argparse
from datetime import datetime, timezone
import json
import os
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def elapsed_seconds(timestamp: str | None) -> float | None:
    if not timestamp:
        return None
    started = datetime.fromisoformat(timestamp)
    return (datetime.now(timezone.utc) - started).total_seconds()


def format_duration(seconds: float | None) -> str:
    if seconds is None:
        return "unknown"
    minutes = int(seconds // 60)
    hours, minutes = divmod(minutes, 60)
    if hours:
        return f"{hours}h {minutes}m"
    return f"{minutes}m"


def main() -> None:
    parser = argparse.ArgumentParser(description="Show Doha mosaic progress.")
    parser.add_argument(
        "--workspace",
        type=Path,
        default=ROOT / "outputs" / "QA" / "doha-202608" / ".work",
    )
    args = parser.parse_args()
    workspace = args.workspace.resolve()
    manifest_path = workspace / "manifest.json"
    if not manifest_path.exists():
        parent = workspace.parent
        finals = sorted(parent.glob("*_1m.tif"))
        if finals:
            print(f"state: completed (no active workspace; {len(finals)} finals in {parent})")
            return
        raise SystemExit(f"No manifest at {manifest_path} and no finals in {parent}")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))

    counts = {status: 0 for status in ("pending", "running", "completed", "failed")}
    for tile in manifest["tiles"]:
        counts[tile["status"]] = counts.get(tile["status"], 0) + 1

    pid_path = workspace / "runner.pid"
    pid = int(pid_path.read_text().strip()) if pid_path.exists() else None
    running = False
    if pid:
        try:
            os.kill(pid, 0)
            running = True
        except ProcessLookupError:
            pass

    completed = counts["completed"]
    durations = [
        tile["duration_seconds"]
        for tile in manifest["tiles"]
        if tile["status"] == "completed" and tile["duration_seconds"]
    ]
    average = sum(durations) / len(durations) if durations else None
    remaining = len(manifest["tiles"]) - completed
    eta = average * remaining if average else None

    print(f"state: {manifest['state']}")
    print(f"process: {'running' if running else 'not running'} (PID {pid or 'unknown'})")
    print(
        f"tiles: {completed}/{len(manifest['tiles'])} complete; "
        f"{counts['running']} running; {counts['failed']} failed; "
        f"{counts['pending']} pending"
    )
    print(f"elapsed: {format_duration(elapsed_seconds(manifest.get('started_at')))}")
    print(f"average tile: {format_duration(average)}")
    print(f"estimated tile time remaining: {format_duration(eta)}")
    expected = manifest.get("expected_raster")
    if expected:
        print(
            f"expected final raster: {expected['width']}x{expected['height']} pixels, "
            f"{expected['resolution_meters']} m, "
            f"products={','.join(manifest.get('selected_products', []))}"
        )
        for product, details in expected["products"].items():
            print(
                f"expected {product}: {details['bands']} bands, "
                f"{details['dtype']}, logical uncompressed "
                f"{details['logical_uncompressed_bytes'] / 1_000_000_000:.2f} GB"
            )

    active = [tile for tile in manifest["tiles"] if tile["status"] == "running"]
    if active:
        tile = active[0]
        print(
            f"active: {tile['id']} at "
            f"{tile['longitude']:.5f},{tile['latitude']:.5f} "
            f"(attempt {tile['attempts']})"
        )
    failed = [tile for tile in manifest["tiles"] if tile["status"] == "failed"]
    for tile in failed[:5]:
        print(f"failed: {tile['id']}: {tile.get('error')}")
    for product, details in manifest.get("products", {}).items():
        print(f"product {product}: {details['path']}")


if __name__ == "__main__":
    main()
