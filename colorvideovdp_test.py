import os
import numpy as np
from scipy.ndimage import gaussian_filter
import imageio.v3 as iio
import torch

import pycvvdp


def save_cvvdp_heatmap_gray(heatmap_tensor, out_path):
    hm = heatmap_tensor

    if isinstance(hm, torch.Tensor):
        hm = hm.detach().cpu()

    print("Heatmap shape:", tuple(hm.shape), "dtype:", hm.dtype)

    hm = hm.numpy()
    hm = np.squeeze(hm)

    # For heatmap='raw', ColorVideoVDP documents a raw difference map
    # mapped to black->white for visualization. We keep it grayscale. 
    # Handle a few possible layouts robustly.
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

    hm = hm.astype(np.float32)

    # Normalize to [0, 1]
    hm_min = float(hm.min())
    hm_max = float(hm.max())
    if hm_max > hm_min:
        hm = (hm - hm_min) / (hm_max - hm_min)
    else:
        hm = np.zeros_like(hm, dtype=np.float32)

    hm_u8 = (255.0 * hm).clip(0, 255).astype(np.uint8)

    print("Saving to:", out_path)
    iio.imwrite(out_path, hm_u8)


dataset = r"C:/Github/Forks/nerficg/dataset/mipnerf360/kitchen/images"
image_extension = ".JPG"
image_files = [f for f in os.listdir(dataset) if f.endswith(image_extension)]

output_dir = os.path.join(dataset, "cvvdp_outputs")
os.makedirs(output_dir, exist_ok=True)

# ColorVideoVDP Python interface:
# cvvdp = pycvvdp.cvvdp(display_name='standard_4k', heatmap='raw')
metric = pycvvdp.cvvdp(display_name="standard_4k", heatmap="raw")

for image_file in image_files:
    print(f"Processing image: {image_file}")

    image_path = os.path.join(dataset, image_file)
    I_ref = pycvvdp.load_image_as_array(image_path)
    print(f"Loaded image: {image_path}, shape: {I_ref.shape}, dtype: {I_ref.dtype}")

    # Gaussian noise with variance 0.003
    sigma_noise = np.sqrt(0.003)
    I_test_noise = I_ref + np.random.normal(0.0, sigma_noise, I_ref.shape).astype(np.float32)
    I_test_noise = np.clip(I_test_noise, 0.0, 1.0)

    Q_JOD_noise, stats_noise = metric.predict(I_test_noise, I_ref, dim_order="HWC")
    print("Noise JOD:", Q_JOD_noise)

    # Gaussian blur with sigma=2
    I_test_blur = gaussian_filter(I_ref, sigma=(2, 2, 0))
    I_test_blur = np.clip(I_test_blur, 0.0, 1.0).astype(np.float32)

    Q_JOD_blur, stats_blur = metric.predict(I_test_blur, I_ref, dim_order="HWC")
    print("Blur JOD:", Q_JOD_blur)

    base = os.path.splitext(image_file)[0]

    noise_img_path = os.path.join(output_dir, f"{base}_noise.png")
    blur_img_path  = os.path.join(output_dir, f"{base}_blur.png")
    noise_heatmap_path = os.path.join(output_dir, f"{base}_noise_heatmap_gray.png")
    blur_heatmap_path  = os.path.join(output_dir, f"{base}_blur_heatmap_gray.png")

    iio.imwrite(noise_img_path, (255.0 * I_test_noise).clip(0, 255).astype(np.uint8))
    iio.imwrite(blur_img_path,  (255.0 * I_test_blur).clip(0, 255).astype(np.uint8))

    if "heatmap" in stats_noise:
        save_cvvdp_heatmap_gray(stats_noise["heatmap"], noise_heatmap_path)
    else:
        print("Warning: stats_noise does not contain 'heatmap'")

    if "heatmap" in stats_blur:
        save_cvvdp_heatmap_gray(stats_blur["heatmap"], blur_heatmap_path)
    else:
        print("Warning: stats_blur does not contain 'heatmap'")