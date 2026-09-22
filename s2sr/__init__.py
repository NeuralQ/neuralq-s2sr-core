"""NeuralQ S2SR public package.

Re-exports the three surfaces a caller needs:
  * model: :class:`s2sr.model.S2SRNet` and :func:`s2sr.model.load_model`
  * checkpoint: :func:`s2sr.hub.resolve_checkpoint` (local ``models/``) and
    :func:`s2sr.hub.verify_checkpoint` (sha256 sidecar)
  * inference: :func:`s2sr.inference.super_resolve_dn` (DN→DN, 10×) and
    :func:`s2sr.inference.predict_normalized` (0..1 tensor)

All submodules are importable without side effects; heavy deps (torch,
numpy) are only required when actually running inference.
"""
from .hub import (
    MODELS_DIR,
    MODEL_FILENAME,
    MODEL_ID,
    cached_checkpoint_path,
    checksum_path,
    resolve_checkpoint,
    verify_checkpoint,
)
from .inference import predict_normalized, super_resolve_dn
from .model import S2SRNet, load_model

__all__ = [
    "MODELS_DIR",
    "MODEL_FILENAME",
    "MODEL_ID",
    "S2SRNet",
    "cached_checkpoint_path",
    "checksum_path",
    "load_model",
    "predict_normalized",
    "resolve_checkpoint",
    "super_resolve_dn",
    "verify_checkpoint",
]
