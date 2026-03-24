#pragma once

#include <torch/extension.h>
#include <tuple>

namespace faster_gs::edge_detection {

std::tuple<torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor>
compute_edge_scores_wrapper(
    const torch::Tensor& image_bchw,
    const int histogram_bins = 512,
    const float eps = 1e-6f
);

} // namespace faster_gs::edge_detection
