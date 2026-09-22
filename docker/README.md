# `docker/` — Container Build Context

Minimal conda env for the image (host `environment.yml` is the full 376-line lockfile for `conda env create`; this one is for `micromamba` in Docker).

## Files

| File | Purpose |
|---|---|
| `environment.yml` | `python=3.12`, `gdal=3.12.*`, `rasterio`, `pyproj`, `shapely`, `numpy`, `imagecodecs`, `requests`, `torch==2.13.0`, `torchvision==0.28.0` — no `huggingface_hub` |

## How to build & run (explicit) — see repo `README.md:110` for full docs

```bash
docker compose build
# → micromamba env from docker/environment.yml, COPY . (262 kB) + COPY models/ (841 MB) as separate layers, sha256sum -c

docker compose run --rm s2sr-cpu python scripts/run_location.py --device cpu --lon 10.641 --lat 35.8256 --date 2026-08-14 --search-only
docker compose run --rm s2sr-gpu python scripts/run_location.py --lon 10.641 --lat 35.8256 --date 2026-08-14
```

Base: `mambaorg/micromamba:1.5.10-jammy` + `# syntax=docker/dockerfile:1.19` (needs `COPY --exclude`). Fix `PermissionError` on `./outputs` (container `mambauser:57439` vs host `1000:1000`) with `chmod -R a+rwx outputs` once.
