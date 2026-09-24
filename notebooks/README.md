# `notebooks/` — Demo & Exploration (by track)

Interactive 1 m S2SR exploration, organized in five tracks. Every notebook runs
with outputs embedded — open and read inline, no external viewer needed.

```bash
conda activate neuralq-s2sr-core
jupyter lab notebooks/<track>/<notebook>.ipynb
```

Kernel: `python3` (conda env `neuralq-s2sr-core`, Python 3.12, `torch`,
`rasterio`, `matplotlib`, `numpy`).

## demo/ — General entry points

| File | Purpose |
|---|---|
| `01_demo.ipynb` | Synthetic model demo (no STAC) → load `MS.tif` → TCI visual → NDVI histogram → all indices → oil OSI threshold → LST vs NDVI → mask export. |
| `04_mosaic_viewer.ipynb` | City mosaic: BigTIFF windowed reads, `manifest.json` tiles/retries, `boundary.geojson` folium, preview `*_preview.tif`, MS chip stretch (Doha). |

## oil/ — Oil-spill suite

| File | Purpose |
|---|---|
| `02_oil_spill.ipynb` | 6 oil indices at 1 m, water/glint masking, histograms, interactive `OSI/HI/FOI` sliders, HI vs OSI scatter, mask export (Huntington Beach 2021-10-05). |

## lst/ — Thermal sharpening

| File | Purpose |
|---|---|
| `03_lst_thermal.ipynb` | LST 1 m (Celsius): AWS config, ST check, `lst.tiff` visual + histogram, LST vs NDVI scatter, on-demand `compute_lst` without re-running S2SR (Sousse; needs AWS creds). |

## carbon/ — CO₂ proxy

| File | Purpose |
|---|---|
| `05_carbon_co2.ipynb` | Carbon 1 m proxy `420+8·NDBI*+0.8·ΔLST*+5·AOT*` ppm (robust p5–p98), Sousse + Huntington, TCI vs `co2.tiff`, histogram, factory mask export. |

## agritech/ — Crop time series → intelligence (run in order)

| # | File | Purpose |
|---|---|---|
| 06 | `06_crop_health_timeseries.ipynb` | SR ROI on 8 dates (Feb→Jul 2024, Sidi Bouzid fields) → all-index timelines → health proxy + zones → rule-tree crop types → `REPORT.md`, `timeseries.csv`, GeoTIFFs. |
| 07 | `07_batch_classification_timeseries.ipynb` | 1 km² batch deep dive: vigor/health/crop-type/KMeans/change/frequency/slope classes, parcel segmentation + fact table, per-crop health status, interactive explorers → `BATCH_REPORT.md`, 6 GeoTIFF maps. |
| 08 | `08_meteo_health.ipynb` | Open-Meteo ERA5 daily + 2015–2023 baseline → GDD, P−ET0 deficit, heat days, dry spells → overlays with series, per-crop verdicts, irrigation flags. |
| 09 | `09_crop_intelligence.ipynb` | KPI cards, growth-cycle staging (NDVI+GDD), canopy-volume & biomass proxies per parcel, days-to-harvest/health/yield predictions, parcel dossier + folium map → `CROP_DOSSIER.csv`, `INTELLIGENCE.md`. |
| 10 | `10_panoptic_segmentation.ipynb` | 3-voter ensemble semantic + per-class instances + panoptic ids + field polygons (GeoJSON, m/m²/compactness) + per-field dossier → `fields.geojson`, `panoptic_dossier.csv`. |

Headless re-execution (ci / servers):

```bash
conda run -n neuralq-s2sr-core jupyter nbconvert --execute --to notebook notebooks/agritech/07_batch_classification_timeseries.ipynb --output /tmp/07_check.ipynb --allow-errors
```
