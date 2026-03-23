"""FasterGS/utils.py"""

import io
import warnings
import contextlib
from pathlib import Path
from typing import Any

import numpy as np
import torch
from PIL import Image

from Datasets.Base import BaseDataset
from Logging import Logger


def enable_expandable_segments() -> bool:
    """Return True if 'expandable_segments' allocator feature is available on this device."""
    torch.cuda.memory._set_allocator_settings('expandable_segments:True')
    stderr_buffer = io.StringIO()
    with contextlib.redirect_stderr(stderr_buffer), warnings.catch_warnings(record=True) as caught_warnings:
        warnings.simplefilter('always')
        torch.empty(1, device='cuda')  # allocate gpu memory to trigger potential warning
    stderr_output = stderr_buffer.getvalue()
    for warning in caught_warnings:
        if 'expandable_segments' in str(warning.message):
            return False
    if 'expandable_segments' in stderr_output:
        return False
    return True


def carve(points: torch.Tensor, dataset: BaseDataset, in_all_frustums: bool, enforce_alpha: bool) -> torch.Tensor:
    """
    Carves away points based on visibility and alpha.
    - Points that are never in-frustum in any view are removed.
    - If in_all_frustums=True, points not in-frustum in all views are removed.
    - If enforce_alpha=True, points that project to a pixel with alpha=0 in any view (where the point is in-frustum) are removed.
    """
    Logger.log_info(f'removing points that would not be visible in any training view (in_all_frustums={in_all_frustums}, enforce_alpha={enforce_alpha})')
    in_frustum_any = torch.zeros(points.shape[0], dtype=torch.bool, device=points.device)
    in_frustum_all = torch.ones_like(in_frustum_any)
    in_alpha_all = torch.ones_like(in_frustum_any)
    dilation_kernel = torch.ones(1, 1, 3, 3) if enforce_alpha else None
    for view in dataset:
        xy_screen, _, in_frustum = view.project_points(points)
        in_frustum_any |= in_frustum
        if in_all_frustums:
            in_frustum_all &= in_frustum
        if enforce_alpha and in_frustum.any() and (alpha_gt := view.alpha) is not None:
            alpha_gt = torch.nn.functional.conv2d(alpha_gt[None], dilation_kernel, padding=1)[0] > 0
            xy_screen = torch.floor(xy_screen[in_frustum]).long()
            valid_alpha = alpha_gt[0, xy_screen[:, 1], xy_screen[:, 0]] > 0
            in_alpha_all[in_frustum] &= valid_alpha
    valid_mask = in_frustum_any & in_alpha_all & in_frustum_all
    return points[valid_mask].contiguous()


def _candidate_view_paths(view: Any) -> list[Path]:
    """Collect likely image path candidates from a training view."""
    candidates: list[Path] = []
    attribute_names = (
        'image_path',
        'rgb_path',
        'path',
        'file_path',
        'image_file',
        'filename',
        'name',
    )
    objects_to_check = [view, getattr(view, 'camera', None)]

    for obj in objects_to_check:
        if obj is None:
            continue
        for attribute_name in attribute_names:
            value = getattr(obj, attribute_name, None)
            if isinstance(value, (str, Path)) and str(value):
                candidates.append(Path(value))
    return candidates


def resolve_heatmap_path(
    view: Any,
    subdirectory: str = 'fvvdp_outputs',
    suffix: str = '_heatmap_gray.png',
) -> Path | None:
    """Return the VideoVDP heatmap path corresponding to a training image view."""
    for candidate in _candidate_view_paths(view):
        search_directories = [candidate.parent]
        if subdirectory:
            search_directories.insert(0, candidate.parent / subdirectory)

        for search_directory in search_directories:
            direct_candidate = search_directory / f'{candidate.stem}{suffix}'
            if direct_candidate.exists():
                return direct_candidate

            if candidate.suffix:
                suffix_replaced = search_directory / candidate.name.replace(candidate.suffix, suffix)
                if suffix_replaced.exists():
                    return suffix_replaced

    return None


def load_videovdp_heatmap(
    view: Any,
    device: torch.device | str,
    subdirectory: str = 'fvvdp_outputs',
    suffix: str = '_heatmap_gray.png',
) -> torch.Tensor | None:
    """Load a single-channel VideoVDP heatmap corresponding to the given training view."""
    heatmap_path = resolve_heatmap_path(view, subdirectory=subdirectory, suffix=suffix)
    if heatmap_path is None:
        return None

    heatmap_np = np.asarray(Image.open(heatmap_path).convert('L'), dtype=np.float32) / 255.0
    return torch.from_numpy(heatmap_np).to(device=device)
