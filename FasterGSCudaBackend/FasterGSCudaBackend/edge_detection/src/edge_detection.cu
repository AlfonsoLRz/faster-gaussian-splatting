#include "edge_detection.h"

#if __has_include(<c10/cuda/CUDAGuard.h>)
    #include <c10/cuda/CUDAGuard.h>
    namespace torch_cuda_guard = c10::cuda;
#elif __has_include(<ATen/cuda/CUDAGuard.h>)
    #include <ATen/cuda/CUDAGuard.h>
    namespace torch_cuda_guard = at::cuda;
#else
    #error "Could not find CUDAGuard.h in either c10/cuda or ATen/cuda"
#endif

#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAException.h>
#include <cuda_runtime.h>
#include <cmath>

namespace faster_gs::edge_detection {

#define CHECK_CUDA(x) TORCH_CHECK((x).is_cuda(), #x " must be a CUDA tensor")
#define CHECK_CONTIGUOUS(x) TORCH_CHECK((x).is_contiguous(), #x " must be contiguous")
#define CHECK_FLOAT32(x) TORCH_CHECK((x).scalar_type() == torch::kFloat32, #x " must be float32")
#define CHECK_INPUT(x) \
    CHECK_CUDA(x); \
    CHECK_CONTIGUOUS(x); \
    CHECK_FLOAT32(x)

namespace {

__constant__ float kGaussian5x5[25] = {
    1.0f / 256.0f,  4.0f / 256.0f,  6.0f / 256.0f,  4.0f / 256.0f, 1.0f / 256.0f,
    4.0f / 256.0f, 16.0f / 256.0f, 24.0f / 256.0f, 16.0f / 256.0f, 4.0f / 256.0f,
    6.0f / 256.0f, 24.0f / 256.0f, 36.0f / 256.0f, 24.0f / 256.0f, 6.0f / 256.0f,
    4.0f / 256.0f, 16.0f / 256.0f, 24.0f / 256.0f, 16.0f / 256.0f, 4.0f / 256.0f,
    1.0f / 256.0f,  4.0f / 256.0f,  6.0f / 256.0f,  4.0f / 256.0f, 1.0f / 256.0f
};

__device__ __forceinline__ int clamp_int(const int v, const int lo, const int hi) {
    return max(lo, min(v, hi));
}

__device__ __forceinline__ int idx3(const int b, const int y, const int x, const int H, const int W) {
    return (b * H + y) * W + x;
}

__device__ __forceinline__ int idx4(const int b, const int c, const int y, const int x, const int C, const int H, const int W) {
    return ((b * C + c) * H + y) * W + x;
}

__device__ __forceinline__ float read_clamped_3d(
    const float* image,
    const int b,
    const int y,
    const int x,
    const int H,
    const int W
) {
    const int yy = clamp_int(y, 0, H - 1);
    const int xx = clamp_int(x, 0, W - 1);
    return image[idx3(b, yy, xx, H, W)];
}

__device__ __forceinline__ float bilinear_read_clamped_3d(
    const float* image,
    const int b,
    const float y,
    const float x,
    const int H,
    const int W
) {
    const float yy = fminf(fmaxf(y, 0.0f), static_cast<float>(H - 1));
    const float xx = fminf(fmaxf(x, 0.0f), static_cast<float>(W - 1));

    const int y0 = static_cast<int>(floorf(yy));
    const int x0 = static_cast<int>(floorf(xx));
    const int y1 = min(y0 + 1, H - 1);
    const int x1 = min(x0 + 1, W - 1);

    const float ty = yy - static_cast<float>(y0);
    const float tx = xx - static_cast<float>(x0);

    const float v00 = image[idx3(b, y0, x0, H, W)];
    const float v01 = image[idx3(b, y0, x1, H, W)];
    const float v10 = image[idx3(b, y1, x0, H, W)];
    const float v11 = image[idx3(b, y1, x1, H, W)];

    const float v0 = v00 + tx * (v01 - v00);
    const float v1 = v10 + tx * (v11 - v10);
    return v0 + ty * (v1 - v0);
}

__global__ void grayscale_kernel(
    const float* input,
    float* gray,
    const int B,
    const int C,
    const int H,
    const int W
) {
    const int x = blockIdx.x * blockDim.x + threadIdx.x;
    const int y = blockIdx.y * blockDim.y + threadIdx.y;
    const int b = blockIdx.z;

    if (x >= W || y >= H || b >= B) return;

    float value = 0.0f;
    if (C == 1) {
        value = input[idx4(b, 0, y, x, C, H, W)];
    } else {
        const float r = input[idx4(b, 0, y, x, C, H, W)];
        const float g = input[idx4(b, 1, y, x, C, H, W)];
        const float bl = input[idx4(b, 2, y, x, C, H, W)];
        value = 0.299f * r + 0.587f * g + 0.114f * bl;
    }

    gray[idx3(b, y, x, H, W)] = value;
}

__global__ void gaussian5x5_kernel(
    const float* gray,
    float* blurred,
    const int B,
    const int H,
    const int W
) {
    const int x = blockIdx.x * blockDim.x + threadIdx.x;
    const int y = blockIdx.y * blockDim.y + threadIdx.y;
    const int b = blockIdx.z;

    if (x >= W || y >= H || b >= B) return;

    float sum = 0.0f;
    #pragma unroll
    for (int ky = -2; ky <= 2; ++ky) {
        #pragma unroll
        for (int kx = -2; kx <= 2; ++kx) {
            const float sample = read_clamped_3d(gray, b, y + ky, x + kx, H, W);
            sum += kGaussian5x5[(ky + 2) * 5 + (kx + 2)] * sample;
        }
    }

    blurred[idx3(b, y, x, H, W)] = sum;
}

__global__ void sobel_kernel(
    const float* blurred,
    float* grad_x,
    float* grad_y,
    float* grad_mag,
    const int B,
    const int H,
    const int W
) {
    const int x = blockIdx.x * blockDim.x + threadIdx.x;
    const int y = blockIdx.y * blockDim.y + threadIdx.y;
    const int b = blockIdx.z;

    if (x >= W || y >= H || b >= B) return;

    const float p00 = read_clamped_3d(blurred, b, y - 1, x - 1, H, W);
    const float p01 = read_clamped_3d(blurred, b, y - 1, x + 0, H, W);
    const float p02 = read_clamped_3d(blurred, b, y - 1, x + 1, H, W);
    const float p10 = read_clamped_3d(blurred, b, y + 0, x - 1, H, W);
    const float p12 = read_clamped_3d(blurred, b, y + 0, x + 1, H, W);
    const float p20 = read_clamped_3d(blurred, b, y + 1, x - 1, H, W);
    const float p21 = read_clamped_3d(blurred, b, y + 1, x + 0, H, W);
    const float p22 = read_clamped_3d(blurred, b, y + 1, x + 1, H, W);

    const float gx = -p00 + p02 - 2.0f * p10 + 2.0f * p12 - p20 + p22;
    const float gy =  p00 + 2.0f * p01 + p02 - p20 - 2.0f * p21 - p22;
    const float mag = sqrtf(gx * gx + gy * gy);

    const int out_idx = idx3(b, y, x, H, W);
    grad_x[out_idx] = gx;
    grad_y[out_idx] = gy;
    grad_mag[out_idx] = mag;
}

__global__ void nms_along_gradient_kernel(
    const float* grad_x,
    const float* grad_y,
    const float* grad_mag,
    float* nms,
    const int B,
    const int H,
    const int W,
    const float eps
) {
    const int x = blockIdx.x * blockDim.x + threadIdx.x;
    const int y = blockIdx.y * blockDim.y + threadIdx.y;
    const int b = blockIdx.z;

    if (x >= W || y >= H || b >= B) return;

    const int out_idx = idx3(b, y, x, H, W);
    const float gx = grad_x[out_idx];
    const float gy = grad_y[out_idx];
    const float mag = grad_mag[out_idx];

    if (mag <= eps) {
        nms[out_idx] = 0.0f;
        return;
    }

    const float inv_mag = rsqrtf(gx * gx + gy * gy + eps);
    const float step_x = gx * inv_mag;
    const float step_y = gy * inv_mag;

    const float pos_val = bilinear_read_clamped_3d(grad_mag, b, static_cast<float>(y) + step_y, static_cast<float>(x) + step_x, H, W);
    const float neg_val = bilinear_read_clamped_3d(grad_mag, b, static_cast<float>(y) - step_y, static_cast<float>(x) - step_x, H, W);

    nms[out_idx] = (mag >= pos_val && mag >= neg_val) ? mag : 0.0f;
}

__global__ void histogram_positive_kernel(
    const float* nms,
    const float* image_max,
    int* histogram,
    int* positive_count,
    const int B,
    const int H,
    const int W,
    const int bins
) {
    const int x = blockIdx.x * blockDim.x + threadIdx.x;
    const int y = blockIdx.y * blockDim.y + threadIdx.y;
    const int b = blockIdx.z;

    if (x >= W || y >= H || b >= B) return;

    const int image_idx = idx3(b, y, x, H, W);
    const float value = nms[image_idx];
    if (value <= 0.0f) return;

    const float denom = fmaxf(image_max[b], 1e-12f);
    const float normalized = fminf(fmaxf(value / denom, 0.0f), 1.0f);
    int bin = static_cast<int>(normalized * static_cast<float>(bins - 1) + 0.5f);
    bin = clamp_int(bin, 0, bins - 1);

    atomicAdd(&histogram[b * bins + bin], 1);
    atomicAdd(&positive_count[b], 1);
}

__global__ void histogram_to_median_kernel(
    const int* histogram,
    const int* positive_count,
    const float* image_max,
    float* median_per_image,
    const int B,
    const int bins
) {
    const int b = blockIdx.x * blockDim.x + threadIdx.x;
    if (b >= B) return;

    const int count = positive_count[b];
    const float max_value = image_max[b];
    if (count <= 0 || max_value <= 0.0f) {
        median_per_image[b] = 1.0f;
        return;
    }

    const int target = (count - 1) / 2;
    int cumulative = 0;
    for (int i = 0; i < bins; ++i) {
        cumulative += histogram[b * bins + i];
        if (cumulative > target) {
            const float normalized_center = (static_cast<float>(i) + 0.5f) / static_cast<float>(bins);
            median_per_image[b] = fmaxf(normalized_center * max_value, 1e-12f);
            return;
        }
    }

    median_per_image[b] = fmaxf(max_value, 1e-12f);
}

__global__ void median_normalize_kernel(
    const float* nms,
    const float* median_per_image,
    float* scores,
    const int B,
    const int H,
    const int W,
    const float eps
) {
    const int x = blockIdx.x * blockDim.x + threadIdx.x;
    const int y = blockIdx.y * blockDim.y + threadIdx.y;
    const int b = blockIdx.z;

    if (x >= W || y >= H || b >= B) return;

    const int image_idx = idx3(b, y, x, H, W);
    const float denom = fmaxf(median_per_image[b], eps);
    scores[image_idx] = nms[image_idx] / denom;
}

} // namespace

std::tuple<torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor>
compute_edge_scores_wrapper(
    const torch::Tensor& image_bchw,
    const int histogram_bins,
    const float eps
) {
    CHECK_INPUT(image_bchw);
    TORCH_CHECK(image_bchw.dim() == 4, "image_bchw must have shape [B, C, H, W]");
    TORCH_CHECK(image_bchw.size(1) == 1 || image_bchw.size(1) == 3, "image_bchw must have 1 or 3 channels");
    TORCH_CHECK(histogram_bins >= 16, "histogram_bins must be >= 16");
    TORCH_CHECK(eps > 0.0f, "eps must be > 0");

    const torch_cuda_guard::CUDAGuard device_guard(image_bchw.device());
    const auto input = image_bchw.contiguous();

    const int B = static_cast<int>(input.size(0));
    const int C = static_cast<int>(input.size(1));
    const int H = static_cast<int>(input.size(2));
    const int W = static_cast<int>(input.size(3));

    auto float_opts = input.options().dtype(torch::kFloat32);
    auto int_opts = input.options().dtype(torch::kInt32);

    auto gray = torch::empty({B, H, W}, float_opts);
    auto blurred = torch::empty({B, H, W}, float_opts);
    auto grad_x = torch::empty({B, H, W}, float_opts);
    auto grad_y = torch::empty({B, H, W}, float_opts);
    auto grad_mag = torch::empty({B, H, W}, float_opts);
    auto nms = torch::empty({B, H, W}, float_opts);
    auto scores = torch::empty({B, H, W}, float_opts);
    auto histogram = torch::zeros({B, histogram_bins}, int_opts);
    auto positive_count = torch::zeros({B}, int_opts);
    auto median_per_image = torch::empty({B}, float_opts);

    const dim3 block(16, 16, 1);
    const dim3 grid((W + block.x - 1) / block.x, (H + block.y - 1) / block.y, B);
    cudaStream_t stream = at::cuda::getCurrentCUDAStream();

    grayscale_kernel<<<grid, block, 0, stream>>>(
        input.data_ptr<float>(),
        gray.data_ptr<float>(),
        B,
        C,
        H,
        W
    );
    C10_CUDA_KERNEL_LAUNCH_CHECK();

    gaussian5x5_kernel<<<grid, block, 0, stream>>>(
        gray.data_ptr<float>(),
        blurred.data_ptr<float>(),
        B,
        H,
        W
    );
    C10_CUDA_KERNEL_LAUNCH_CHECK();

    sobel_kernel<<<grid, block, 0, stream>>>(
        blurred.data_ptr<float>(),
        grad_x.data_ptr<float>(),
        grad_y.data_ptr<float>(),
        grad_mag.data_ptr<float>(),
        B,
        H,
        W
    );
    C10_CUDA_KERNEL_LAUNCH_CHECK();

    nms_along_gradient_kernel<<<grid, block, 0, stream>>>(
        grad_x.data_ptr<float>(),
        grad_y.data_ptr<float>(),
        grad_mag.data_ptr<float>(),
        nms.data_ptr<float>(),
        B,
        H,
        W,
        eps
    );
    C10_CUDA_KERNEL_LAUNCH_CHECK();

    auto nms_max = std::get<0>(nms.flatten(1).max(1, false)).contiguous();

    histogram_positive_kernel<<<grid, block, 0, stream>>>(
        nms.data_ptr<float>(),
        nms_max.data_ptr<float>(),
        histogram.data_ptr<int>(),
        positive_count.data_ptr<int>(),
        B,
        H,
        W,
        histogram_bins
    );
    C10_CUDA_KERNEL_LAUNCH_CHECK();

    const int median_threads = 64;
    const int median_blocks = (B + median_threads - 1) / median_threads;
    histogram_to_median_kernel<<<median_blocks, median_threads, 0, stream>>>(
        histogram.data_ptr<int>(),
        positive_count.data_ptr<int>(),
        nms_max.data_ptr<float>(),
        median_per_image.data_ptr<float>(),
        B,
        histogram_bins
    );
    C10_CUDA_KERNEL_LAUNCH_CHECK();

    median_normalize_kernel<<<grid, block, 0, stream>>>(
        nms.data_ptr<float>(),
        median_per_image.data_ptr<float>(),
        scores.data_ptr<float>(),
        B,
        H,
        W,
        eps
    );
    C10_CUDA_KERNEL_LAUNCH_CHECK();

    return {scores, nms, grad_mag, blurred, median_per_image};
}

} // namespace faster_gs::edge_detection
