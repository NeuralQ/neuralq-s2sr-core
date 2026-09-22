---
license: other
license_name: neuralq-restricted-1.0
license_link: LICENSE
tags:
  - remote-sensing
  - sentinel-2
  - super-resolution
  - geospatial
  - pytorch
library_name: pytorch
---

# NeuralQ S2SR

Local Sentinel-2 temporal super-resolution. A 105M-parameter network maps five
cloud-screened, co-registered acquisitions (5 × 10 bands) to a single 10×
super-resolved multispectral product at 1 m, orchestrated end to end — STAC
search, co-registration, tiled GPU inference, spectral indexing — by a local,
resumable pipeline.

**No usage rights are granted.** See [LICENSE](LICENSE).

## Model

`S2SRNet` — inference uses `params_ema`; strict state-dict load.

| Property | Value |
| --- | --- |
| Input | `N x 50 x H x W`, reflectance DN / 10000 |
| Output | `N x 10 x 10H x 10W`, uint16 (clamped) |
| Parameters | 105,055,800 |
| Checkpoint | `models/s2sr-v3.0.0.pt`, bundled in the repo and baked into the Docker image |

The checkpoint ships locally under `models/` — no download, no account, no
token. `resolve_checkpoint()` simply locates `models/s2sr-v3.0.0.pt` on disk.
Integrity is pinned by `models/s2sr-v3.0.0.pt.sha256`: enforced at Docker
build time and at every run (set `S2SR_SKIP_CHECKSUM=1` to skip the runtime
check). Replacing the weights? Update the `.sha256` sidecar to match.

Migration from Hugging Face: older revisions downloaded `s2sr-v3.0.0.pt`
from a private Hub repo via `HF_TOKEN`/`huggingface_hub`. That integration
is removed — `s2sr.hub` no longer knows `HF_REPO_ID`/`HF_FILENAME`, the
`huggingface_hub` dependency and `hf-cache` volume are gone, and
`HF_TOKEN` does nothing. Drop it from your shell.

Band order (per date): `B02 B03 B04 B08 B05 B06 B07 B11 B12 B8A`.

Architecture:

- **deformer** — 50→160 conv + 7 grouped deformable-conv blocks
  (DCNv1, groups = 5, 90 predicted offsets per block)
- **encoder** — 160→160 conv + 23 RRDBs (3 dense blocks each, growth width 80,
  residual scale 0.2)
- **generator** — nearest-exact upsampling ×2×2×2×1.25 = 10;
  widths 160→80→40→20→10; bias-free final conv + LeakyReLU

Python API:

```python
from s2sr import load_model, resolve_checkpoint, super_resolve_dn

checkpoint = resolve_checkpoint()   # bundled models/s2sr-v3.0.0.pt, no download
model = load_model(checkpoint, device="cuda")
sr = super_resolve_dn(model, stack_dn)   # (50, H, W) uint16 -> (10, 10H, 10W) uint16
```

## Repository Layout

```
neuralq-s2sr-core/
├── models/                  bundled s2sr-v3.0.0.pt weights + .sha256 pin (see --model to override)
├── s2sr/                    model package: architecture, local loader, inference helpers
├── scripts/                 generic, location-agnostic pipeline
│   ├── upstream.py          integration boundary to the compiled preprocessing engine
│   ├── output_layout.py     shared helpers: IDs, geocoding, inventories, README writer
│   ├── run_location.py      single-location inference (any coordinates)
│   ├── run_mosaic.py        resumable boundary tiling -> clipped BigTIFFs (any city)
│   ├── spectral_indices.py  15 indices in 4 categories from MS.tif
│   └── lst.py               TsHARP-style LST sharpening (Landsat ST + S2SR NDVI)
├── tests/                   stdlib-only unit tests
└── outputs/                 empty until a run writes products here; transient .work/ scratch auto-removed
```

## Requirements

- Linux x86_64, CUDA GPU (reference: RTX 4080 16 GB), ~40 GB free disk per run
- Every run shows progress bars (download+align, GPU tiles) and live/peak
  CPU, RAM, VRAM usage plus wall time from `scripts/resource_monitor.py`
- Conda environment:

```bash
conda env create -f environment.yml    # neuralq-s2sr-core, Python 3.12
conda activate neuralq-s2sr-core
```

The bundled checkpoint at `models/s2sr-v3.0.0.pt` is used by default — no
download and no token. It is also baked into the Docker image (`.dockerignore`
must not exclude `models/`).

Use `--model /path/to/checkpoint.pt` on `run_location.py` to override with a
different local file.

The pipeline runs entirely on the local, pure-Python engine in
`scripts/local_engine/` (Earth Search STAC + AWS Sentinel-2 COGs) — no
compiled wheel or external engine is required. A different engine can be
plugged in with `NEURALQ_ENGINE_MODULE=<import name>`.
- GDAL CLI tools (`gdalbuildvrt`, `gdalwarp`, `gdal_translate`) on PATH for mosaics

## Docker — build and use

The container (`Dockerfile`, `docker-compose.yml`, `docker/environment.yml`)
runs the pipeline with no local conda install, on GPU or CPU. The
`s2sr-v3.0.0.pt` weights are **baked into the image** — no download, no
account, no token at build or run time.

### Prerequisites

- Docker Engine 23+ with the Compose plugin (`docker compose version` should
  work) and BuildKit enabled (default on 23+; the `COPY --exclude` layer
  split needs dockerfile frontend ≥ 1.19, pinned via `# syntax` — any
  current Engine qualifies).
- For GPU runs: an NVIDIA GPU plus the [NVIDIA Container
  Toolkit](https://github.com/NVIDIA/nvidia-container-toolkit) on the host
  (`nvidia-smi` must work). CPU-only runs need nothing extra.
- ~25 GB free for the image (conda stack + CUDA torch + ~800 MB weights)
  plus ~40 GB free under `./outputs` per run (`--min-free-gb 40` enforced).
- Network access to Docker Hub, conda-forge, and PyPI at **build** time;
  Earth Search STAC, AWS Sentinel-2 COGs, and Nominatim at **run** time.

### Build

From the repo root (`models/s2sr-v3.0.0.pt` must be present):

```bash
docker compose build
```

What happens: micromamba creates the env from `docker/environment.yml`,
the repo is copied in two layers (code first, `models/` second — editing
code later rebuilds only the small layer), and the build **fails fast** if
the weights are missing or fail their `sha256sum -c` check against
`models/s2sr-v3.0.0.pt.sha256`. Replacing the weights? Update the `.sha256`
sidecar to match. (`models/` must stay out of `.dockerignore`.)

Plain-`docker` equivalent: `docker build -t neuralq-s2sr-core .`

### Use

Single location, GPU (reference: 6 min wall time, 1.6 GB VRAM peak at
`--tile-size 128`):

```bash
docker compose run --rm s2sr-gpu python scripts/run_location.py \
  --lon 51.531 --lat 25.2886 --date 2026-08-14
```

CPU dry run — STAC search and ranking only, no download or inference:

```bash
docker compose run --rm s2sr-cpu python scripts/run_location.py \
  --device cpu --lon 51.531 --lat 25.2886 --date 2026-08-14 --search-only
```

City mosaic (Doha default; resumable, rerunning resumes):

```bash
docker compose run --rm s2sr-gpu python scripts/run_mosaic.py --date 2026-08-14
docker compose run --rm s2sr-gpu python scripts/run_mosaic.py \
  --boundary-query "Lyon, France" --osm-id 35238
```

Useful knobs: `--tile-size 64` halves GPU memory per tile,
`--products MS TCI` keeps only selected products, `--skip-indices` skips the
15 spectral-index rasters, `--skip-lst` skips the sharpened temperature raster, `--model /path/weights.pt` uses alternate weights
(combine with a read-only bind mount, e.g. `-v ./my-weights.pt:/app/models/s2sr-v3.0.0.pt:ro`
— see the commented line in `docker-compose.yml`).

### What goes where

- Working dir is `/app`; the container entrypoint is the micromamba
  activator, default command prints `run_location.py --help`.
- `./outputs` is bind-mounted read-write: every run lands in
  `outputs/<CC>/<date>/<inference_id>/` on your host.
- Named volume `neuralq-s2sr-core-cache` persists the reverse-geocoding
  cache across runs.
- One run at a time per output tree; concurrent runs corrupt the engine's
  shared scratch. Set `S2SR_SKIP_CHECKSUM=1` to skip the startup weights
  hash check.

### Troubleshooting

- `PermissionError` writing `./outputs`: the container runs as `mambauser`
  (uid 57439), so the bind-mounted directory must be writable by it:
  `chmod -R a+rwx outputs` (once; applies to subdirectories of past runs too).
- `could not select device driver "nvidia"` / GPU not visible: install the
  NVIDIA Container Toolkit and retry with `s2sr-gpu`; `s2sr-cpu` always works.
- `CUDA out of memory`: lower `--tile-size` (try 64, minimum 16).
- `Insufficient disk space`: free ~40 GB at the output location or lower
  `--min-free-gb` (at your own risk).
- `Checksum mismatch`: the weights file is corrupt or was swapped without
  updating `models/s2sr-v3.0.0.pt.sha256` — re-acquire or rebuild.
- `COPY --exclude` / `unknown flag`: your builder resolved a dockerfile
  frontend older than 1.19 — upgrade Docker Engine (23+) and rebuild.

## Usage

Single location (one run at a time; concurrent runs corrupt the engine's shared scratch):

```bash
python scripts/run_location.py \
  --lon 51.5310 --lat 25.2886 --date 2026-08-14 \
  [--products MS TCI NDVI IRP] [--skip-indices] [--skip-lst] [--tile-size 128] \
  [--min-free-gb 40] [--device auto] [--search-only]
```

Boundary mosaic (any city; UTM zone derived from the boundary centroid;
boundary cached under `outputs/.boundaries/`, delete to refresh):

```bash
python scripts/run_mosaic.py                                   # Doha default
python scripts/run_mosaic.py \
  --boundary-query "Lyon, France" --osm-id 35238 --country-code FR
python scripts/run_mosaic.py --products MS TCI LST             # add 1 m temperature
```

Resumable; tiles validated on accept and finals revalidated on rerun.
Budget ~37 GB peak working state per date.

Verification and tests:

```bash
python3 tests/test_pipeline_units.py         # runs anywhere; no dependencies
```

## Outputs

```
outputs/<CC>/<date>/<inference_id>/
├── MS.tif            4120x4120-class, 10-band uint16, 1 m   (scientific product)
├── TCI.tif NDVI.tif IRP.tif   3-band uint8 visualizations
├── indices/<category>/<name>.tiff   single-band float32, NoData=NaN
│   └── thermal/lst.tiff             sharpened LST in Celsius (needs AWS credentials; absent otherwise)
└── README.md         run/model/product metadata, SHA-256 inventory, engine log
```

Mosaic finals (one directory per mosaic run, same organized layout):

```
outputs/<CC>/<date>/mosaic-<date>-<plan>/
├── <Place>_<date>_S2SR_MS_1m.tif    boundary-clipped BigTIFF, 10-band uint16, 1 m
├── <Place>_<date>_S2SR_TCI_1m.tif   (and NDVI/IRP when selected)
├── <Place>_<date>_S2SR_LST_1m.tif   only with --products ... LST; float32 Celsius, NoData=NaN
├── <Place>_<date>_preview.tif       downsampled uncompressed preview (from TCI)
└── README.md                        mosaic/plan metadata, SHA-256 inventory, per-product validation
```

All rasters are written uncompressed (`COMPRESS=NONE`) and enforced at three
layers: write time, tile resume validation, final validation. Filenames,
metadata keys, and logs are normalized to NeuralQ/S2SR identity before leaving
the pipeline.

## Spectral Indices

Computed from the super-resolved `MS.tif`; grid-identical, float32, NaN nodata.

| Category | Index | Formula |
| --- | --- | --- |
| vegetation | ndvi | `(B08 - B04) / (B08 + B04)` |
| vegetation | gndvi | `(B08 - B03) / (B08 + B03)` |
| vegetation | ndre | `(B08 - B05) / (B08 + B05)` |
| vegetation | evi2 | `2.5 * (B08 - B04) / (B08 + 2.4*B04 + 1)` |
| vegetation | savi | `1.5 * (B08 - B04) / (B08 + B04 + 0.5)` |
| vegetation | mtci | `(B06 - B05) / (B05 - B04)` |
| water | ndwi | `(B03 - B08) / (B03 + B08)` |
| water | mndwi | `(B03 - B11) / (B03 + B11)` |
| water | ndmi | `(B08 - B11) / (B08 + B11)` |
| water | awei | `4*(B03 - B11) - (0.25*B08 + 2.75*B12)` |
| burn | nbr | `(B08 - B12) / (B08 + B12)` |
| burn | nbr2 | `(B11 - B12) / (B11 + B12)` |
| soil_urban | bsi | `((B11+B04)-(B08+B02)) / ((B11+B04)+(B08+B02))` |
| soil_urban | ndbi | `(B11 - B08) / (B11 + B08)` |
| soil_urban | ndti | `(B11 - B12) / (B11 + B12)` |
| oil | osi | `(B11 + B12 - B08 - B04) / (B11 + B12 + B08 + B04)` |
| oil | hi | `(B11 - B12) / (B11 + B12)` |
| oil | foi | `B08 - [B04 + (B11 - B04)*0.187]` |
| oil | ndoi | `(B03 - B08) / (B03 + B08)` |
| oil | sr | `B12 / B11` |
| oil | rg | `B04 / B03` |

21 spectral indices + thermal LST (`indices/thermal/lst.tiff`) ≈ 1.5 GB per inference (22 files × ~67 MB). Oil suite has its own `indices/oil/README.md` with physics and thresholds; thermal has `indices/thermal/README.md`.

## Land Surface Temperature

Sentinel-2 has no thermal band, so LST is **not** derived from the optical
indices — it is measured by Landsat 8/9 and sharpened to the 1 m grid
(TsHARP-style thermal sharpening, `scripts/lst.py`):

```
LST_30m = a + b · NDVI_30m              least squares on clear coarse pixels
LST_1m  = a + b · NDVI_1m + resid_1m    bilinear-upsampled coarse residuals
```

Source: Landsat Collection 2 Level-2 surface temperature (`lwir11`/ST_B10,
30 m, Kelvin = DN × 0.00341802 + 149.0 per USGS) within ±16 days of the
target date, closest date then lowest `eo:cloud_cover` wins; thermal samples
masked by C2 `qa_pixel` (fill, dilated cloud, cirrus, cloud, shadow, snow
rejected) plus a 230–360 K plausibility gate on the coarse median. Residual
redistribution preserves the coarse means by construction. Output
`indices/thermal/lst.tiff`: single-band float32 Celsius (Kelvin − 273.15), `COMPRESS=NONE`,
`NoData=NaN`, grid-identical to `MS.tif`, and cataloged in the indices
`README.md` alongside the spectral set. The run README records the source
scene, date offset, fit coefficients, and RMSE.

Requirements: AWS credentials (standard chain — env, `~/.aws/credentials`,
or instance role), because `usgs-landsat` is a Requester Pays bucket; reads
are COG-windowed so traffic is KB-scale. Without credentials, a usable scene,
or ≥ 25 clear coarse pixels, the layer is skipped with a note and the run
continues (`--skip-lst` skips it outright). Skip reasons recorded in the run
README: `aws-credentials`, `no-scenes`, `all-cloudy`, `too-few-samples`,
`implausible-values` (coarse median outside 230–360 K — refuses to sharpen
rather than write an unphysical field), or `read-error`. For mosaics LST is
opt-in: `--products ... LST` (default set unchanged); mosaic finals use NaN
nodata, never 0 K. The per-run README's `LST` section always exists and shows
either the product spec + source scene + fit, or the skip note.

## Oil Spill Index (OSI) — 1 m, hydrocarbon-aware

Built at 1 m from the super-resolved `MS.tif` (`indices/oil/osi.tiff`,
float32, NoData=NaN, `COMPRESS=NONE`):

```
OSI = (B11 + B12 - B08 - B04) / (B11 + B12 + B08 + B04)
      B11 1610 nm (SWIR1) + B12 2190 nm (SWIR2) — C-H stretch (1.73 µm, 2.30 µm)
      B08 842 nm (NIR) — water absorption / vegetation scattering
      B04 665 nm (Red) — water-leaving baseline
```

Scientific base: crude oil elevates SWIR (hydrocarbon overtone,
Cloutis et al. 2010; Lammoglia & Souza Filho 2011) and Red while NIR stays
dark over water; the normalized difference isolates that SWIR excess. On sea,
clean water → OSI negative to ~0 (≈ −0.4 to 0.05), oil film → positive
(~0.15–0.45, thick slicks higher). On land, bare soil is weakly positive
(~0.05–0.15); oil on soil/dark land drives it higher but bright urban and
dry bright soils can mimic oil — thresholds must be calibrated locally.
Literature anchor: Kolokoussis & Karathanassi 2018 used Red/SWIR vs NIR
ratios for Sentinel-2 oil on sea; Pisano et al. 2021 reviewed SWIR contrast.

Use: threshold `osi.tiff` (e.g. `>0.15` on water, `>0.25` for high-confidence
on sea; `>0.12` on dark land as candidate, always masked by `NDWI>0.2` for
water, cloud mask, and glint mask `B08<0.15` reflectance), then confirm with
SAR and field data — optical OSI is an **ancillary, clear-sky, sunglint-
sensitive** indicator, not a standalone detector, and S2SR 1 m texture is
inferred, not measured, so sub-pixel slick width is not ground truth.

## Limitations

- Requires the bundled `models/s2sr-v3.0.0.pt` weights file (~800 MB);
   the Docker image already contains it, for conda runs keep `models/` in place.
- 10x SR is a learned prior, not a measurement: the 1 m texture is inferred,
   not observed. Each run logs a self-consistency diagnostic
   (`sr_anchor_consistency_mae_dn`: block-averaged product vs. the anchor
   acquisition) plus anchor band means in the engine log — a large MAE flags
   inference/geometry faults, a small one does not prove the detail is real.
    No ground-truth validation is shipped; treat MS.tif accordingly.
- LST inherits its source: a Landsat scene up to 16 days from the target,
   clear-sky only, and a regression (`LST ~ NDVI`) that is assumed stationary
   across the tile. Check `date_offset_days` and fit RMSE in the run README;
   large offsets or RMSE mean the temperature field is weakly constrained.
- OSI is empirical and threshold-dependent: no universal global cutoff; soil
   brightness, shallow water, and sun glint all shift it. Calibrate per scene
   against known clean water/land, mask as above, and always cross-validate
   with SAR/field — do not report oil on OSI alone.
- Mosaic geometry is only as current as the cached OSM boundary; delete the
  cache entry to re-fetch.

## License

Distributed under the [NeuralQ Restricted License 1.0](LICENSE). Viewing this
repository is permitted; **no operational, research, commercial, or derivative
use is licensed without prior written consent** of the copyright holder.
