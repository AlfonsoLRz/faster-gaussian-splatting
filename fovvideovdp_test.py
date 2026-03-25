import os
import sys
import numpy as np
from scipy.ndimage import gaussian_filter
import imageio.v3 as iio
import torch

this_dir = os.path.dirname(os.path.abspath(__file__))
fvvdp_root = os.path.join(this_dir, "FovVideoVDP")
sys.path.insert(0, fvvdp_root)

import pyfvvdp

def save_fvvdp_heatmap_gray(heatmap_tensor, out_path):
    hm = heatmap_tensor

    if isinstance(hm, torch.Tensor):
        hm = hm.detach().cpu()

    print("Heatmap shape:", tuple(hm.shape), "dtype:", hm.dtype)

    hm = hm.numpy()
    hm = np.squeeze(hm)

    # Expected with heatmap='raw' for image mode:
    # [1, 1, 1, H, W] -> [H, W]
    # but handle a few possible layouts robustly
    if hm.ndim == 2:
        pass
    elif hm.ndim == 3:
        # Could be [C,H,W] or [H,W,C]
        if hm.shape[0] <= 4:
            hm = hm.mean(axis=0)
        else:
            hm = hm.mean(axis=-1)
    elif hm.ndim > 3:
        # Collapse all non-spatial dims
        hm = hm.reshape((-1, hm.shape[-2], hm.shape[-1])).mean(axis=0)

    # Normalize to [0,1]
    hm_min = hm.min()
    hm_max = hm.max()
    if hm_max > hm_min:
        hm = (hm - hm_min) / (hm_max - hm_min)
    else:
        hm = np.zeros_like(hm, dtype=np.float32)

    hm_u8 = (255.0 * hm).astype(np.uint8)

    print("Saving to:", out_path)
    iio.imwrite(out_path, hm_u8)

dataset = 'C:/Github/Forks/nerficg/dataset/mipnerf360/garden/images/'
image_extension = '.JPG'
image_files = [f for f in os.listdir(dataset) if f.endswith(image_extension)]

output_dir = os.path.join(dataset, "fvvdp_outputs")
os.makedirs(output_dir, exist_ok=True)

# Use raw heatmap instead of threshold visualization
fv = pyfvvdp.fvvdp(display_name='standard_4k', heatmap='raw')

for image_file in image_files:
    print(f"Processing image: {image_file}")

    image_path = os.path.join(dataset, image_file)
    I_ref = pyfvvdp.load_image_as_array(image_path)
    print(f"Loaded image: {image_path}, shape: {I_ref.shape}")

    # Gaussian noise with variance 0.003
    sigma_noise = np.sqrt(0.003)
    I_test_noise = I_ref + np.random.normal(0.0, sigma_noise, I_ref.shape).astype(np.float32)
    I_test_noise = np.clip(I_test_noise, 0.0, 1.0)

    Q_JOD_noise, stats_noise = fv.predict(I_test_noise, I_ref, dim_order="HWC")
    print("Noise JOD:", Q_JOD_noise)

    # Gaussian blur with sigma=2
    I_test_blur = gaussian_filter(I_ref, sigma=(2, 2, 0))
    Q_JOD_blur, stats_blur = fv.predict(I_test_blur, I_ref, dim_order="HWC")
    print("Blur JOD:", Q_JOD_blur)

    base = os.path.splitext(image_file)[0]
    noise_path = os.path.join(output_dir, f"{base}_noise_heatmap_gray.png")
    blur_path  = os.path.join(output_dir, f"{base}_blur_heatmap_gray.png")

    save_fvvdp_heatmap_gray(stats_noise['heatmap'], noise_path)
    save_fvvdp_heatmap_gray(stats_blur['heatmap'], blur_path)