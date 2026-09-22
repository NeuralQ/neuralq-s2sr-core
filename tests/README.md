# `tests/` — Stdlib-Only Unit Tests

No heavy deps required; `rasterio`/`pyproj`/`shapely`/`requests` are stubbed so the suite runs anywhere (CI, laptop, container).

## Files

| File | Covers |
|---|---|
| `test_pipeline_units.py` | `output_layout` (UTM zones, `plan_signature`, `coordinate_tags`, `new_inference_id`, `band_order`), `upstream.sanitize`, `run_mosaic` (boundary cache, `place_slug`, `mosaic_inference_id`), `s2sr.hub` (bundled resolve/verify/sidecar), `lst` (QA bits, `fit_lst_ndvi`, `select_scene`, window/grid, `plausible_kelvin`, `st_scale_offset`), oil suite (`CATEGORIES["oil"]` 6 indices), Docker bake (`models/` not excluded, `sha256sum -c`, no `HF_TOKEN`) |
| `fixtures/wakashio_oil.geojson` | Synthetic 500×200 m oil + adjacent clean-water polygons near Pointe d'Esny 20.4381°S 57.7446°E — regression guard for `osi >0.15` threshold (not a ground-truth) |

## How to run (explicit)

```bash
# Stdlib only — no conda needed
python3 tests/test_pipeline_units.py
# → PASS test_... (25 tests) / ALL TESTS PASSED

# Via pytest (if installed)
python -m pytest tests/test_pipeline_units.py -v

# In container
docker compose run --rm s2sr-cpu python tests/test_pipeline_units.py

# In conda env (also verifies real imports)
conda run -n neuralq-s2sr-core python tests/test_pipeline_units.py
```

To add a test: add a `test_*` function, use `eq(got, want, label)` and `ok(cond, label)` helpers, and keep it stdlib-only (import heavy modules via `importlib.util` as done for `s2sr.hub`).
