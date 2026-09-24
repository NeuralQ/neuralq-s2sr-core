# `notebooks/` — Demo & Exploration

Interactive 1 m S2SR exploration — works on the latest `outputs/` run or on a synthetic stack.

## Files

| File | Purpose |
|---|---|
| `01_demo.ipynb` | 8 sections: 1) synthetic model demo (no STAC) → 2) load `MS.tif` → 3) TCI visual → 4) NDVI histogram → 5) all indices → 6) oil OSI threshold → 7) LST vs NDVI → 8) export mask. Overview + charts. |
| `02_oil_spill.ipynb` | Dedicated oil: 6 indices at 1 m, water/glint masking, histograms per scene, interactive `OSI/HI/FOI` sliders (ipywidgets), triple test, HI vs OSI scatter, mask export. |
| `03_lst_thermal.ipynb` | LST 1 m (Celsius): AWS config cell (env vs `~/.aws/credentials` vs instance role), ST check, `lst.tiff` visual + histogram + legend tags, LST vs NDVI scatter + 1 m fit vs 30 m fit in run README, on-demand `compute_lst` on existing `MS.tif` without re-running S2SR. |
| `04_mosaic_viewer.ipynb` | City mosaic: BigTIFF windowed reads, `manifest.json` tiles/retries, `boundary.geojson` folium, preview `*_preview.tif`, MS chip stretch. |
| `05_carbon_co2.ipynb` | Carbon 1 m proxy: `420+8·NDBI*+0.8·ΔLST*+5·AOT*` ppm (robust p5–p98), Sousse + Huntington, TCI vs `co2.tiff`, histogram, factory mask export, AWS config for LST. |\n| `06_crop_health_timeseries.ipynb` | Crop time series at 1 m: SR ROI on 3–4 dates → all-index timelines (veg/water/soil) → health proxy + zones + stress change → rule-tree crop types + KMeans check → `REPORT.md`, `timeseries.csv`, `health_map.tif`, `croptype.tif`. Sidi Bouzid default; synthetic fallback offline. |\n| `07_batch_classification_timeseries.ipynb` | Small-batch (1 km²) deep dive on the real series: 15 indices × 4 dates → vigor/health/crop-type/KMeans/change/frequency/slope classes → parcel segmentation + fact table → per-class + per-parcel series → `BATCH_REPORT.md`, 6 GeoTIFF maps. |
| `08_meteo_health.ipynb` | Meteo × health: Open-Meteo ERA5 daily (T, rain, ET0, soil moisture) + 2015–2023 baseline → GDD, P−ET0 deficit, heat days, dry spells → overlays with NDVI/health series, per-crop verdicts, irrigation flags. |

## How to run (explicit)

```bash
conda activate neuralq-s2sr-core
# Optional: ensure a product exists
python scripts/run_location.py --lon 10.641 --lat 35.8256 --date 2026-08-14

# Launch
jupyter lab notebooks/01_demo.ipynb
# or
jupyter notebook notebooks/01_demo.ipynb
```

Headless (no GUI) — the notebook still runs; charts are saved if you `plt.savefig`.

```bash
conda run -n neuralq-s2sr-core jupyter nbconvert --execute --to html notebooks/01_demo.ipynb --output demo.html
```

Kernel: `python3` (conda env `neuralq-s2sr-core`, Python 3.12, `torch`, `rasterio`, `matplotlib`, `numpy`).
