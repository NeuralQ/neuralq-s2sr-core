# `notebooks/` — Demo & Exploration

Interactive 1 m S2SR exploration — works on the latest `outputs/` run or on a synthetic stack.

## Files

| File | Purpose |
|---|---|
| `demo.ipynb` | 8 sections: 1) synthetic model demo (no STAC) → 2) load `MS.tif` → 3) TCI visual → 4) NDVI compute + histogram → 5) all indices list → 6) oil OSI threshold on water → 7) LST vs NDVI 2D histogram → 8) export binary oil mask. Generates matplotlib charts. |

## How to run (explicit)

```bash
conda activate neuralq-s2sr-core
# Optional: ensure a product exists
python scripts/run_location.py --lon 10.641 --lat 35.8256 --date 2026-08-14

# Launch
jupyter lab notebooks/demo.ipynb
# or
jupyter notebook notebooks/demo.ipynb
```

Headless (no GUI) — the notebook still runs; charts are saved if you `plt.savefig`.

```bash
conda run -n neuralq-s2sr-core jupyter nbconvert --execute --to html notebooks/demo.ipynb --output demo.html
```

Kernel: `python3` (conda env `neuralq-s2sr-core`, Python 3.12, `torch`, `rasterio`, `matplotlib`, `numpy`).
