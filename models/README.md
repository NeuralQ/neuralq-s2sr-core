# `models/` — Bundled Weights

## Contents

| File | Size | Purpose |
|---|---|---|
| `s2sr-v3.0.0.pt` | 802 MB (840,950,460 B) | 105M S2SR checkpoint, `params_ema` strict load, input `50×H×W` → `10×10H×10W` |
| `s2sr-v3.0.0.pt.sha256` | 81 B | Coreutils sidecar `1ac3d52c...  s2sr-v3.0.0.pt` — pinned hash |

The checkpoint is **baked into the Docker image** (`COPY models/ → /app/models/`, verified at build via `sha256sum -c`). No download, no `HF_TOKEN`, no network at runtime.

## How to use (explicit)

```bash
# Verify (host)
sha256sum -c models/s2sr-v3.0.0.pt.sha256
# → s2sr-v3.0.0.pt: OK

# Resolve from Python (searches <repo>/models/, ./models/, explicit path)
python -c "from s2sr import resolve_checkpoint, verify_checkpoint; print(verify_checkpoint(resolve_checkpoint()))"

# Override with a different checkpoint
python scripts/run_location.py --model /path/to/custom.pt --lon 10.641 --lat 35.8256 --date 2026-08-14
# If /path/to/custom.pt.sha256 exists, it will be enforced; otherwise passthrough.

# Update the pinned hash after replacing the weights
sha256sum models/s2sr-v3.0.0.pt > models/s2sr-v3.0.0.pt.sha256
```

Do not put `models/` in `.dockerignore` — the image must contain it. `models/*.pt` is tracked via Git LFS (see `.gitattributes:22`).
