# `scripts/` — Pipeline

Generic, location-agnostic orchestration: STAC search → co-registration → tiled GPU inference → uncompressed GeoTIFFs → indices/LST → READMEs.

## Files

| Script | Function | When to use |
|---|---|---|
| `run_location.py` | Single-point inference (any lon/lat/date). Handles STAC ranking, 5-date stack, tiled `super_resolve_dn`, product flattening, `COMPRESS=NONE` enforcement, indices + LST, SHA256 inventory, `README.md`. | `python scripts/run_location.py --lon 10.641 --lat 35.8256 --date 2026-08-14` |
| `run_mosaic.py` | Resumable city mosaic over OSM boundary. Grids 4.12 km tiles on 4 km step, center-out, per-tile `run_location` subprocesses, `manifest.json` + `flock`, `gdalbuildvrt` + `gdalwarp` BigTIFFs. | `python scripts/run_mosaic.py --boundary-query "Lyon, France" --osm-id 35238` |
| `spectral_indices.py` | 21 spectral indices at 1 m from `MS.tif` (DN/10000). Categories: vegetation×6, water×4, burn×2, soil_urban×3, oil×6. | Called by `run_location`; standalone: `python -c "import spectral_indices; spectral_indices.compute_indices(Path('MS.tif'), Path('indices'))"` |
| `lst.py` | TsHARP LST sharpening: Landsat C2L2 ST_B10 (30 m, `K=DN×0.00341802+149`) → 1 m via `a+b·NDVI` + residuals. Celsius output, 230–360 K gate, needs AWS creds (Requester Pays). | Called by `run_location`; opt-out `--skip-lst` |
| `output_layout.py` | Organized `outputs/<CC>/<date>/<id>/`, Nominatim reverse-geocode cache, SHA256, `product_inventory`, atomic README writes. | `from output_layout import inference_directory, product_inventory` |
| `resource_monitor.py` | `Bar` (TTY-throttled) + `ResourceMonitor` (CPU/RAM/VRAM background thread). | Used by `local_engine/inferutils.py` |
| `upstream.py` | Engine pluggability (`NEURALQ_ENGINE_MODULE`) + branding `sanitize()` for legacy tokens. | `NEURALQ_ENGINE_MODULE=local_engine python scripts/run_location.py ...` |

## How to run (explicit) — single location

```bash
conda activate neuralq-s2sr-core
# Dry run (no download/GPU)
python scripts/run_location.py --lon 51.531 --lat 25.2886 --date 2026-08-14 --search-only

# Full GPU run (default bundled weights)
python scripts/run_location.py --lon 51.531 --lat 25.2886 --date 2026-08-14

# CPU, custom weights, skip LST, smaller tiles (low VRAM)
python scripts/run_location.py --device cpu --model /path/weights.pt --skip-lst --tile-size 64 --lon 10.641 --lat 35.8256 --date 2026-08-14

# Exact output path
python scripts/run_location.py --lon 10.641 --lat 35.8256 --date 2026-08-14 --output /tmp/my_run
```

## How to run — mosaic

```bash
python scripts/run_mosaic.py --date 2026-08-14 --plan-only          # write manifest, don't run tiles
python scripts/run_mosaic.py --date 2026-08-14                       # Doha default, resumable
python scripts/run_mosaic.py --boundary-query "Lyon, France" --osm-id 35238 --products MS TCI LST
```

Outputs: `outputs/<CC>/<date>/<id>/{MS.tif,TCI.tif,NDVI.tif,IRP.tif,indices/<category>/<name>.tiff,indices/thermal/lst.tiff,README.md}` — all `COMPRESS=NONE`, 4120×4120 at 1 m. See repo `README.md:232` for tree.

Transient state lives in `<output>/.work/` (manifest, logs, per-tile products) and is auto-removed; `run_mosaic` keeps `.work/manifest.json` + `boundary.geojson` until mosaic completes.
