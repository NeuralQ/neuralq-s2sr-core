# `scripts/local_engine/` — Local Pure-Python Inference Engine

Replaces the former compiled wheel. No external engine needed; `NEURALQ_ENGINE_MODULE=local_engine` (default via `Dockerfile`/`upstream.py`).

## Files

| File | Purpose | Science / details |
|---|---|---|
| `datautils.py` | AOI + STAC search/ranking | AOI 2×2 km square → WGS84 bbox (lat-corrected lon delta). Earth Search `sentinel-2-l2a`, window `target±120d`, `cloud≤80%`, rank `date desc, cloud asc`, `select_stack_dates` picks anchor closest to target + 4 clearest same-MGRS dates (unique dates). Retries 3×. |
| `inferutils.py` | End-to-end AOI inference | UTM grid 412×412 at 10 m (center snapped), `WarpedVRT` per band (`bilinear` for 20 m B05/B06/B07/B11/B12/B8A, `nearest` for 10 m), scale/offset guard (only `0<scale<1`, `-2000≤offset<0` else `1/10000`), `DMIN/10000` → `_tile_inference` (tiled `super_resolve_dn`, 128 px default, sequential to limit VRAM) → `MS.tif` 4120×4120 at 1 m + `products.write_visuals` + engine `S2SRlog_*.json` with `stack_dates`, `anchor_date`, `sr_anchor_consistency_mae_dn`, `stack_item_ids`. |
| `products.py` | Visualizations | TCI `B04/B03/B02` and IRP `B08/B04/B03` via 2–98% percentile stretch → uint8; NDVI colormap brown→green 5-stop. Same profile as `MS.tif`, `COMPRESS=NONE`. |
| `__init__.py` | Package doc | Declares the `datautils`/`inferutils` contract expected by `run_location.py`. |

## How to run (explicit) — engine alone

```bash
# Search-only (no GPU)
python -c "
import sys; sys.path.insert(0, 'scripts')
from local_engine import datautils
from types import SimpleNamespace
from datetime import datetime
args=SimpleNamespace(date='20260814', aoi=datautils.aoi_from_xy((10.641,35.8256), km=2))
datautils.search_collection(args)
print(len(args.items), 'scenes')
print(datautils.sort_items_by_recency_and_clouds(args.items)[:2])
"

# Full inference via run_location (recommended)
python scripts/run_location.py --lon 10.641 --lat 35.8256 --date 2026-08-14

# Plug a different engine (e.g. a compiled one)
NEURALQ_ENGINE_MODULE=my_engine python scripts/run_location.py --lon 10.641 --lat 35.8256 --date 2026-08-14
```

STAC: `https://earth-search.aws.element84.com/v1/search`, collection `sentinel-2-l2a`, AWS S3 COGs public (no creds). Landsat thermal for LST is separate (`lst.py`, Requester Pays).
