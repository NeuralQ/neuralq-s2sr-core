# `notebooks/` — Demo & Exploration

Interactive 1 m S2SR exploration — works on the latest `outputs/` run or on a synthetic stack.

## Files

| File | Purpose |
|---|---|
| `01_demo.ipynb` | 8 sections: 1) synthetic model demo (no STAC) → 2) load `MS.tif` → 3) TCI visual → 4) NDVI histogram → 5) all indices → 6) oil OSI threshold → 7) LST vs NDVI → 8) export mask. Overview + charts. |
| `02_oil_spill.ipynb` | Dedicated oil: 6 indices at 1 m, water/glint masking, histograms per scene, interactive `OSI/HI/FOI` sliders (ipywidgets), triple test, HI vs OSI scatter, mask export. |
| `03_lst_thermal.ipynb` | LST 1 m (Celsius): AWS config cell (env vs `~/.aws/credentials` vs instance role), ST check, `lst.tiff` visual + histogram + legend tags, LST vs NDVI scatter + 1 m fit vs 30 m fit in run README, on-demand `compute_lst` on existing `MS.tif` without re-running S2SR. |
| `04_mosaic_viewer.ipynb` | City mosaic: BigTIFF windowed reads, `manifest.json` tiles/retries, `boundary.geojson` folium, preview `*_preview.tif`, MS chip stretch. |

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
