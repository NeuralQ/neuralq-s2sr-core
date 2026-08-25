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
| Checkpoint | `models/S2SR-GL-20241022.1.pt` |
| Checkpoint SHA-256 | `1ac3d52cac3737842538ed09f329b0023b43cd3d5f509ccce36a0951cb2dd520` |

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
from s2sr import load_model, super_resolve_dn

model = load_model("models/S2SR-GL-20241022.1.pt", device="cuda")
sr = super_resolve_dn(model, stack_dn)   # (50, H, W) uint16 -> (10, 10H, 10W) uint16
```

## Repository Layout

```
neuralq-s2sr-core/
├── models/                  checkpoint (SHA-256-pinned)
├── s2sr/                    model package: architecture, loader, inference helpers
├── scripts/
│   ├── upstream.py          integration boundary to the compiled preprocessing engine
│   ├── output_layout.py     shared helpers: IDs, geocoding, inventories, README writer
│   ├── run_location.py      single-location inference pipeline
│   ├── run_location_doha.py Doha defaults wrapper
│   ├── run_mosaic.py        resumable boundary tiling -> clipped BigTIFFs
│   ├── run_mosaique_doha.py time-series orchestrator (planning + workers)
│   └── spectral_indices.py  15 indices in 4 categories from MS.tif
├── tests/                   stdlib-only unit tests
└── outputs/                 products; transient .work/ scratch auto-removed
```

## Requirements

- Linux x86_64, CUDA GPU (reference: RTX 4080 16 GB), ~40 GB free disk per run
- Conda environment:

```bash
conda env create -f environment.yml    # neuralq-s2sr-core, Python 3.12
conda activate neuralq-s2sr-core
```

- The compiled preprocessing engine (STAC access, co-registration, tiled I/O)
  is installed separately as a wheel; `scripts/upstream.py` is the single
  integration point (`NEURALQ_ENGINE_MODULE`, `NEURALQ_UPSTREAM_OBJECT`)
- GDAL CLI tools (`gdalbuildvrt`, `gdalwarp`, `gdal_translate`) on PATH for mosaics

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

Time series:

```bash
python scripts/run_mosaique_doha.py --plan-dates \
  --start-date 2020-01-01 --end-date 2026-08-21 --frequency weekly
python scripts/run_mosaique_doha.py          # resumable; --max-dates N; keep --workers 1
```

Verification and tests:

```bash
sha256sum models/S2SR-GL-20241022.1.pt       # must match the pinned digest above
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
- Golden-fingerprint verification tooling was removed in cleanup; integrity is
  anchored by the pinned checkpoint SHA-256.

## License

Distributed under the [NeuralQ Restricted License 1.0](LICENSE). Viewing this
repository is permitted; **no operational, research, commercial, or derivative
use is licensed without prior written consent** of the copyright holder.
