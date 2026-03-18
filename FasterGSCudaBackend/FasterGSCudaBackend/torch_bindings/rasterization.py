from typing import Any, NamedTuple
import torch
from torch.autograd.function import once_differentiable

from FasterGSCudaBackend import _C


class RasterizerSettings(NamedTuple):
    w2c: torch.Tensor
    cam_position: torch.Tensor
    bg_color: torch.Tensor
    active_sh_bases: int
    width: int
    height: int
    focal_x: float
    focal_y: float
    center_x: float
    center_y: float
    near_plane: float
    far_plane: float
    proper_antialiasing: bool

    def as_tuple(self) -> tuple:
        return (
            self.w2c,
            self.cam_position,
            self.bg_color,
            self.active_sh_bases,
            self.width,
            self.height,
            self.focal_x,
            self.focal_y,
            self.center_x,
            self.center_y,
            self.near_plane,
            self.far_plane,
            self.proper_antialiasing,
        )


class _Rasterize(torch.autograd.Function):
    @staticmethod
    def forward(
        ctx: Any,
        means: torch.Tensor,
        scales: torch.Tensor,
        rotations: torch.Tensor,
        opacities: torch.Tensor,
        distance_decay: torch.Tensor,
        sh_coefficients_0: torch.Tensor,
        sh_coefficients_rest: torch.Tensor,
        densification_info: torch.Tensor,
        rasterizer_settings: RasterizerSettings,
        virtual_scale: float,
        tau: float,
    ) -> torch.Tensor:
        (
            image,
            primitive_buffers,
            tile_buffers,
            instance_buffers,
            bucket_buffers,
            n_instances,
            n_buckets,
            instance_primitive_indices_selector,
        ) = _C.forward(
            means,
            scales,
            rotations,
            opacities,
            distance_decay,
            sh_coefficients_0,
            sh_coefficients_rest,
            *rasterizer_settings.as_tuple(),
            virtual_scale,
            tau,
        )

        ctx.rasterizer_settings = rasterizer_settings
        ctx.buffer_state = (n_instances, n_buckets, instance_primitive_indices_selector)
        ctx.virtual_scale = float(virtual_scale)
        ctx.tau = float(tau)
        ctx.densification_info = densification_info
        ctx.mark_non_differentiable(densification_info)

        ctx.save_for_backward(
            image,
            means,
            scales,
            rotations,
            opacities,
            distance_decay,
            sh_coefficients_rest,
            primitive_buffers,
            tile_buffers,
            instance_buffers,
            bucket_buffers,
        )
        return image

    @staticmethod
    @once_differentiable
    def backward(
        ctx: Any,
        grad_image: torch.Tensor,
    ):
        (
            image,
            means,
            scales,
            rotations,
            opacities,
            distance_decay,
            sh_coefficients_rest,
            primitive_buffers,
            tile_buffers,
            instance_buffers,
            bucket_buffers,
        ) = ctx.saved_tensors

        (
            grad_means,
            grad_scales,
            grad_rotations,
            grad_opacities,
            grad_distance_decay,
            grad_sh_coefficients_0,
            grad_sh_coefficients_rest,
        ) = _C.backward(
            ctx.densification_info,
            grad_image,
            image,
            means,
            scales,
            rotations,
            opacities,
            distance_decay,
            sh_coefficients_rest,
            primitive_buffers,
            tile_buffers,
            instance_buffers,
            bucket_buffers,
            *ctx.rasterizer_settings.as_tuple(),
            ctx.virtual_scale,
            ctx.tau,
            *ctx.buffer_state,
        )

        return (
            grad_means,                 # means
            grad_scales,                # scales
            grad_rotations,             # rotations
            grad_opacities,             # opacities
            grad_distance_decay,        # distance_decay
            grad_sh_coefficients_0,     # sh_coefficients_0
            grad_sh_coefficients_rest,  # sh_coefficients_rest
            None,                       # densification_info
            None,                       # rasterizer_settings
            None,                       # virtual_scale
            None,                       # tau
        )


def diff_rasterize(
    means: torch.Tensor,
    scales: torch.Tensor,
    rotations: torch.Tensor,
    opacities: torch.Tensor,
    distance_decay: torch.Tensor,
    sh_coefficients_0: torch.Tensor,
    sh_coefficients_rest: torch.Tensor,
    densification_info: torch.Tensor,
    rasterizer_settings: RasterizerSettings,
    virtual_scale: float,
    tau: float,
) -> torch.Tensor:
    return _Rasterize.apply(
        means,
        scales,
        rotations,
        opacities,
        distance_decay,
        sh_coefficients_0,
        sh_coefficients_rest,
        densification_info,
        rasterizer_settings,
        virtual_scale,
        tau,
    )


def rasterize(
    means: torch.Tensor,
    scales: torch.Tensor,
    rotations: torch.Tensor,
    opacities: torch.Tensor,
    distance_decay: torch.Tensor,
    sh_coefficients_0: torch.Tensor,
    sh_coefficients_rest: torch.Tensor,
    rasterizer_settings: RasterizerSettings,
    to_chw: bool,
    virtual_scale: float,
    tau: float,
) -> torch.Tensor:
    return _C.inference(
        means,
        scales,
        rotations,
        opacities,
        distance_decay,
        sh_coefficients_0,
        sh_coefficients_rest,
        *rasterizer_settings.as_tuple(),
        to_chw,
        virtual_scale,
        tau,
    )


def update_pruning_scores(
    scores: torch.Tensor,
    means: torch.Tensor,
    scales: torch.Tensor,
    rotations: torch.Tensor,
    opacities: torch.Tensor,
    sh_coefficients_0: torch.Tensor,
    sh_coefficients_rest: torch.Tensor,
    rasterizer_settings: RasterizerSettings,
) -> torch.Tensor:
    return _C.pruning_scores(
        scores,
        means,
        scales,
        rotations,
        opacities,
        sh_coefficients_0,
        sh_coefficients_rest,
        *rasterizer_settings.as_tuple(),
    )