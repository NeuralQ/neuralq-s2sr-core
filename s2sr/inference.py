"""Inference helpers — DN ↔ normalized conversion and tiled prediction.

Sentinel-2 L2A DN are reflectance×10000. The network was trained on
0..1 (DN/10000, date-major 10 bands). Outputs are clamped 0..1, rescaled,
and rounded to uint16. No cloud mask is applied here; that is the STAC
selection's job. Tiling is the caller's responsibility (GPU memory).
"""
import numpy as np
import torch
from numpy.typing import NDArray
from torch import Tensor, nn


@torch.inference_mode()
def predict_normalized(
    model: nn.Module,
    stack: Tensor | NDArray[np.floating],
    *,
    device: str | torch.device | None = None,
) -> Tensor:
    """Run a normalized 0..1 five-date, ten-band stack through S2SR."""
    tensor = torch.as_tensor(stack, dtype=torch.float32)
    squeeze_batch = tensor.ndim == 3
    if squeeze_batch:
        tensor = tensor.unsqueeze(0)
    if tensor.ndim != 4 or tensor.shape[1] != 50:
        raise ValueError(
            "Expected stack shape (50, H, W) or (N, 50, H, W); "
            f"received {tuple(tensor.shape)}"
        )

    if device is None:
        device = next(model.parameters()).device
    output = model(tensor.to(device, non_blocking=True)).detach().cpu()
    output = output.clone()
    return output.squeeze(0) if squeeze_batch else output


def super_resolve_dn(
    model: nn.Module,
    stack: NDArray[np.integer] | NDArray[np.floating],
    *,
    max_range: float = 10_000,
    device: str | torch.device | None = None,
) -> NDArray[np.uint16]:
    """Convert Sentinel reflectance DNs to a ten-band, 10x uint16 result."""
    if max_range <= 0:
        raise ValueError("max_range must be positive")
    normalized = np.asarray(stack, dtype=np.float32) / max_range
    output = predict_normalized(model, normalized, device=device)
    output = torch.clamp(output, 0, 1) * max_range
    return torch.round(output).numpy().astype(np.uint16)
