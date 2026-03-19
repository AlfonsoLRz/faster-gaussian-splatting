"""FasterGS/Renderer.py"""

import math
import torch

import Framework
from Cameras.Perspective import PerspectiveCamera
from Datasets.Base import BaseDataset
from Datasets.utils import View
from Logging import Logger
from Methods.Base.Renderer import BaseModel, BaseRenderer
from Methods.FasterGS.Model import FasterGSModel
from Methods.FasterGS.FasterGSCudaBackend import (
    diff_rasterize,
    rasterize,
    update_pruning_scores,
    RasterizerSettings,
)


def extract_settings(
    view: View,
    active_sh_bases: int,
    bg_color: torch.Tensor,
    proper_antialiasing: bool,
) -> RasterizerSettings:
    if not isinstance(view.camera, PerspectiveCamera):
        raise Framework.RendererError('FasterGS renderer only supports perspective cameras')
    if view.camera.distortion is not None:
        Logger.log_warning('found distortion parameters that will be ignored by the rasterizer')
    return RasterizerSettings(
        view.w2c,
        view.position,
        bg_color,
        active_sh_bases,
        view.camera.width,
        view.camera.height,
        view.camera.focal_x,
        view.camera.focal_y,
        view.camera.center_x,
        view.camera.center_y,
        view.camera.near_plane,
        view.camera.far_plane,
        proper_antialiasing,
    )


def compute_clod_soft_and_hard_eta(
    means: torch.Tensor,
    raw_opacities: torch.Tensor,
    raw_distance_decay: torch.Tensor,
    camera_position: torch.Tensor,
    virtual_scale: float,
    tau: float,
    soft_k: float = 64.0,
    eps: float = 1e-8,
) -> tuple[torch.Tensor, torch.Tensor]:
    if means.shape[0] == 0:
        zero = torch.zeros((), dtype=raw_opacities.dtype, device=raw_opacities.device)
        return zero, zero

    d = torch.linalg.norm(means - camera_position[None, :], dim=1, keepdim=True)

    sigma = torch.relu(raw_distance_decay)
    alpha = raw_opacities.sigmoid()

    denom = (d * virtual_scale).square() + eps
    alpha_lod = alpha * torch.exp(-(2.0 * sigma.square() + eps) / denom)
    threshold = tau * virtual_scale

    hard_mask = (alpha_lod > threshold).float()
    soft_mask = torch.sigmoid(soft_k * (alpha_lod - threshold))

    eta_hard = hard_mask.mean()
    eta_soft = soft_mask.mean()
    return eta_soft, eta_hard


@Framework.Configurable.configure(
    SCALE_MODIFIER=1.0,
    PROPER_ANTIALIASING=False,
    FORCE_OPTIMIZED_INFERENCE=False,
    CLOD_VIRTUAL_SCALE=1.0,
    CLOD_TAU=1e-2,
)
class FasterGSRenderer(BaseRenderer):
    """Wrapper around the rasterization module from 3DGS."""

    def __init__(self, model: 'BaseModel') -> None:
        super().__init__(model, [FasterGSModel])
        if not Framework.config.GLOBAL.GPU_INDICES:
            raise Framework.RendererError('FasterGS renderer not implemented in CPU mode')
        if len(Framework.config.GLOBAL.GPU_INDICES) > 1:
            Logger.log_warning(f'FasterGS renderer not implemented in multi-GPU mode: using GPU {Framework.config.GLOBAL.GPU_INDICES[0]}')

    def render_image(self, view: View, to_chw: bool = False, benchmark: bool = False) -> dict[str, torch.Tensor]:
        if benchmark or self.FORCE_OPTIMIZED_INFERENCE:
            return self.render_image_benchmark(view, to_chw=to_chw or benchmark)
        if self.model.training:
            raise Framework.RendererError('please directly call render_image_training() instead of render_image() during training')
        return self.render_image_inference(view, to_chw)

    def render_image_training(
        self,
        view: View,
        update_densification_info: bool,
        bg_color: torch.Tensor,
        virtual_scale: float = 1.0,
        tau: float = 1e-3,
        use_clod: bool = False,
    ) -> tuple[torch.Tensor, dict[str, torch.Tensor | float]]:
        gaussians = self.model.gaussians
        device = gaussians.means.device

        effective_use_clod = use_clod and (not update_densification_info)
        actual_virtual_scale = virtual_scale if effective_use_clod else 1.0

        image = diff_rasterize(
            means=gaussians.means,
            scales=gaussians.raw_scales,
            rotations=gaussians.raw_rotations,
            opacities=gaussians.raw_opacities,
            distance_decay=gaussians.raw_distance_decay,
            sh_coefficients_0=gaussians.sh_coefficients_0,
            sh_coefficients_rest=gaussians.sh_coefficients_rest,
            densification_info=gaussians.densification_info if update_densification_info else torch.empty(0, device=device),
            rasterizer_settings=extract_settings(view, gaussians.active_sh_bases, bg_color, self.PROPER_ANTIALIASING),
            virtual_scale=actual_virtual_scale,
            tau=tau,
        )

        if effective_use_clod:
            eta_soft, eta_hard = compute_clod_soft_and_hard_eta(
                means=gaussians.means,
                raw_opacities=gaussians.raw_opacities,
                raw_distance_decay=gaussians.raw_distance_decay,
                camera_position=view.position,
                virtual_scale=virtual_scale,
                tau=tau,
            )
        else:
            eta_soft = torch.ones((), dtype=gaussians.means.dtype, device=device)
            eta_hard = eta_soft

        return image, {
            'eta_actual': eta_soft,
            'eta_actual_hard': eta_hard,
            'virtual_scale': virtual_scale,
        }

    @torch.no_grad()
    def render_image_inference(self, view: View, to_chw: bool = False) -> dict[str, torch.Tensor]:
        gaussians = self.model.gaussians

        image = rasterize(
            means=gaussians.means,
            scales=gaussians.raw_scales + math.log(max(self.SCALE_MODIFIER, 1e-6)),
            rotations=gaussians.raw_rotations,
            opacities=gaussians.raw_opacities,
            distance_decay=gaussians.raw_distance_decay,
            sh_coefficients_0=gaussians.sh_coefficients_0,
            sh_coefficients_rest=gaussians.sh_coefficients_rest,
            rasterizer_settings=extract_settings(
                view,
                gaussians.active_sh_bases,
                view.camera.background_color,
                self.PROPER_ANTIALIASING,
            ),
            to_chw=True,
            virtual_scale=self.CLOD_VIRTUAL_SCALE,
            tau=self.CLOD_TAU,
        )

        image = image.clamp(0.0, 1.0)
        return {'rgb': image if to_chw else image.permute(1, 2, 0)}

    @torch.inference_mode()
    def render_image_benchmark(self, view: View, to_chw: bool = False) -> dict[str, torch.Tensor]:
        image = rasterize(
            means=self.model.gaussians.means,
            scales=self.model.gaussians.raw_scales,
            rotations=self.model.gaussians.raw_rotations,
            opacities=self.model.gaussians.raw_opacities,
            distance_decay=self.model.gaussians.raw_distance_decay,
            sh_coefficients_0=self.model.gaussians.sh_coefficients_0,
            sh_coefficients_rest=self.model.gaussians.sh_coefficients_rest,
            rasterizer_settings=extract_settings(
                view,
                self.model.gaussians.active_sh_bases,
                view.camera.background_color,
                self.PROPER_ANTIALIASING,
            ),
            to_chw=True,
            virtual_scale=self.CLOD_VIRTUAL_SCALE,
            tau=self.CLOD_TAU,
        )

        image = image.clamp(0.0, 1.0)
        return {'rgb': image if to_chw else image.permute(1, 2, 0)}

    @torch.no_grad()
    def compute_pruning_scores(self, dataset: BaseDataset) -> torch.Tensor:
        gaussians = self.model.gaussians
        scores = torch.zeros(
            (gaussians.n_primitives, 1),
            dtype=gaussians.means.dtype,
            device=gaussians.means.device,
        )

        for view in dataset:
            update_pruning_scores(
                scores=scores,
                means=gaussians.means,
                scales=gaussians.raw_scales,
                rotations=gaussians.raw_rotations,
                opacities=gaussians.raw_opacities,
                sh_coefficients_0=gaussians.sh_coefficients_0,
                sh_coefficients_rest=gaussians.sh_coefficients_rest,
                rasterizer_settings=extract_settings(
                    view,
                    gaussians.active_sh_bases,
                    view.camera.background_color,
                    self.PROPER_ANTIALIASING,
                ),
            )

        return scores