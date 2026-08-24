"""Inference helpers: normalized tensor prediction and DN-level super-resolution."""
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
    output = output.clamp_(0, 1).mul_(max_range)
    return output.numpy().astype(np.uint16)
