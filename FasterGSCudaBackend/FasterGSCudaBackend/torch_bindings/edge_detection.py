import torch

from FasterGSCudaBackend import _C


def _to_bchw(image: torch.Tensor) -> torch.Tensor:
    if image.dim() == 2:
        return image.unsqueeze(0).unsqueeze(0)

    if image.dim() == 3:
        if image.shape[0] in (1, 3):
            return image.unsqueeze(0)
        if image.shape[-1] in (1, 3):
            return image.permute(2, 0, 1).unsqueeze(0)
        raise ValueError("3D input must be CHW or HWC with 1 or 3 channels.")

    if image.dim() == 4:
        if image.shape[1] in (1, 3):
            return image
        if image.shape[-1] in (1, 3):
            return image.permute(0, 3, 1, 2)
        raise ValueError("4D input must be BCHW or BHWC with 1 or 3 channels.")

    raise ValueError("Input must have shape [H,W], [C,H,W], [H,W,C], [B,C,H,W], or [B,H,W,C].")


def _normalize_input(image: torch.Tensor) -> torch.Tensor:
    if image.dtype == torch.uint8:
        image = image.float() / 255.0
    else:
        image = image.float()
        if torch.isfinite(image).all():
            max_val = float(image.detach().amax().item())
            if max_val > 1.5:
                image = image / 255.0
    return image.clamp(0.0, 1.0)


def compute_edge_scores(
    image: torch.Tensor,
    histogram_bins: int = 512,
    eps: float = 1e-6,
    return_intermediates: bool = False,
):
    if not image.is_cuda:
        raise ValueError("image must be a CUDA tensor")

    image_bchw = _to_bchw(image).contiguous()
    image_bchw = _normalize_input(image_bchw)

    scores, nms, grad_mag, blurred, median_per_image = _C.compute_edge_scores(
        image_bchw,
        int(histogram_bins),
        float(eps),
    )

    scores = scores.unsqueeze(1)
    nms = nms.unsqueeze(1)
    grad_mag = grad_mag.unsqueeze(1)
    blurred = blurred.unsqueeze(1)

    if return_intermediates:
        return scores, nms, grad_mag, blurred, median_per_image

    return scores
