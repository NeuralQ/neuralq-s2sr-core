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
| Checkpoint | `s2sr-v3.0.0.pt`, hosted on the private Hugging Face repo [`Khlaifiabilel/neuralq-s2sr-core`](https://huggingface.co/Khlaifiabilel/neuralq-s2sr-core) |

The checkpoint is downloaded (and cached by `huggingface_hub`, so only once)
on first use; the repo is private, so a Hugging Face read token must be set
as `HF_TOKEN` in the environment.

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

checkpoint = resolve_checkpoint()   # downloads from HF (HF_TOKEN required); cached after first call
model = load_model(checkpoint, device="cuda")
sr = super_resolve_dn(model, stack_dn)   # (50, H, W) uint16 -> (10, 10H, 10W) uint16
```

## Repository Layout

```
neuralq-s2sr-core/
├── models/                  optional local checkpoint override (see --model)
├── s2sr/                    model package: architecture, HF loader, inference helpers
├── scripts/                 generic, location-agnostic pipeline
│   ├── upstream.py          integration boundary to the compiled preprocessing engine
│   ├── output_layout.py     shared helpers: IDs, geocoding, inventories, README writer
│   ├── run_location.py      single-location inference (any coordinates)
│   ├── run_mosaic.py        resumable boundary tiling -> clipped BigTIFFs (any city)
│   └── spectral_indices.py  15 indices in 4 categories from MS.tif
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

The checkpoint is downloaded from the private Hugging Face repo
[`Khlaifiabilel/neuralq-s2sr-core`](https://huggingface.co/Khlaifiabilel/neuralq-s2sr-core)
on first use — set a Hugging Face read token as `HF_TOKEN` in the
environment before running anything:

```bash
export HF_TOKEN=hf_xxx
```

Use `--model /path/to/checkpoint.pt` on `run_location.py` to override with a
local file instead (e.g. from `models/`).

The pipeline runs entirely on the local, pure-Python engine in
`scripts/local_engine/` (Earth Search STAC + AWS Sentinel-2 COGs) — no
compiled wheel or external engine is required. A different engine can be
plugged in with `NEURALQ_ENGINE_MODULE=<import name>`.
- GDAL CLI tools (`gdalbuildvrt`, `gdalwarp`, `gdal_translate`) on PATH for mosaics

## Docker

A dedicated container (`Dockerfile`, `docker-compose.yml`) runs the pipeline
without a local conda install, on GPU or CPU:

```bash
docker compose build
docker compose run --rm s2sr-gpu python scripts/run_location.py \
  --lon 51.531 --lat 25.2886 --date 2026-08-14        # GPU (needs NVIDIA Container Toolkit)
docker compose run --rm s2sr-cpu python scripts/run_location.py \
  --device cpu --lon 51.531 --lat 25.2886 --date 2026-08-14 --search-only
```

`HF_TOKEN` must be set in the shell before running compose (both services
require it to download the private checkpoint). `./outputs` is bind-mounted
read-write; named volumes `neuralq-s2sr-core-cache` and
`neuralq-s2sr-core-hf-cache` persist the reverse-geocoding cache and the
downloaded checkpoint across runs. See the top of `Dockerfile` and
`docker-compose.yml` for plain `docker run` equivalents.

## Usage

Single location (one run at a time; concurrent runs corrupt the engine's shared scratch):

```bash
python scripts/run_location.py \
  --lon 51.5310 --lat 25.2886 --date 2026-08-14 \
  [--products MS TCI NDVI IRP] [--skip-indices] [--tile-size 128] \
  [--min-free-gb 40] [--device auto] [--search-only]
```

Boundary mosaic (any city; UTM zone derived from the boundary centroid;
boundary cached under `outputs/.boundaries/`, delete to refresh):

```bash
python scripts/run_mosaic.py                                   # Doha default
python scripts/run_mosaic.py \
  --boundary-query "Lyon, France" --osm-id 35238 --country-code FR
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
└── README.md         run/model/product metadata, SHA-256 inventory, engine log
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

Full set ≈ 1 GB per inference (15 files × ~67 MB).

## Limitations

- Requires the separately distributed compiled engine wheel; the repository
  alone covers model inference from pre-aligned stacks only.
- Mosaic geometry is only as current as the cached OSM boundary; delete the
  cache entry to re-fetch.
- Requires an `HF_TOKEN` with read access to the private
  `Khlaifiabilel/neuralq-s2sr-core` repo; without it the checkpoint cannot
  be downloaded.

## License

Distributed under the [NeuralQ Restricted License 1.0](LICENSE). Viewing this
repository is permitted; **no operational, research, commercial, or derivative
use is licensed without prior written consent** of the copyright holder.
