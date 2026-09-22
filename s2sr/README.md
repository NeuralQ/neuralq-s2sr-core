# `s2sr/` — Model Package

Pure-Python PyTorch package for the 105M S2SR network. No I/O, no STAC, no GDAL — just the network and checkpoint helpers.

## Files

| File | Purpose | Key exports |
|---|---|---|
| `hub.py` | Local checkpoint resolver (no Hugging Face) | `MODEL_ID="s2sr-v3.0.0"`, `MODELS_DIR`, `resolve_checkpoint()`, `cached_checkpoint_path()`, `verify_checkpoint()`, `checksum_path()` |
| `model.py` | Network definition + loader | `S2SRNet` (deformer 7×DCN groups=5 → encoder 23×RRDB → generator 2×2×2×1.25=10×), `S2SRResidualDenseBlock`, `S2SRRRDB`, `S2SRDeformationBlock`, `load_model()` |
| `inference.py` | DN ↔ normalized helpers | `predict_normalized()` (0..1 tensor), `super_resolve_dn()` (uint16 DN/10000 → uint16 10×, clamp+round) |
| `__init__.py` | Public re-export | `S2SRNet`, `load_model`, `resolve_checkpoint`, `verify_checkpoint`, `super_resolve_dn`, `predict_normalized` |

## Science

- Input: `N×50×H×W` uint16 DN, 5 dates × 10 bands (`B02 B03 B04 B08 B05 B06 B07 B11 B12 B8A`) date-major, divided by 10000 → 0..1.
- Output: `N×10×10H×10W` uint16, clamped 0..1 then ×10000, nearest-exact upsampling (no checkerboard).
- Checkpoint: `models/s2sr-v3.0.0.pt`, 802 MB, `params_ema` strict load, sha256 pinned.

## How to run (explicit)

```bash
# 1. Resolve the bundled weights (no download, no token)
python -c "from s2sr import resolve_checkpoint, verify_checkpoint; print(verify_checkpoint(resolve_checkpoint()))"
# → /.../models/s2sr-v3.0.0.pt

# 2. Load model (CPU or CUDA)
python -c "from s2sr import load_model, resolve_checkpoint; m=load_model(resolve_checkpoint(), device='cuda'); print(m)"

# 3. Super-resolve a stack (50,H,W) uint16 → (10,10H,10W) uint16
python -c "
import numpy as np
from s2sr import load_model, resolve_checkpoint, super_resolve_dn
m=load_model(resolve_checkpoint(), device='cuda')
stack=np.zeros((50,64,64), dtype=np.uint16)  # 5×10 bands
out=super_resolve_dn(m, stack, device='cuda')
print(out.shape, out.dtype)
"
```

All imports except `torch`/`numpy` are lazy; `s2sr.hub` imports without either (used by stdlib tests).
