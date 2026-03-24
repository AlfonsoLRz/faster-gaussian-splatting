import argparse
import sys
from pathlib import Path

THIS_DIR = Path(__file__).resolve().parent
REPO_ROOT = THIS_DIR.parents[2]

sys.path.insert(0, str(REPO_ROOT / "src"))
sys.path.insert(0, str(REPO_ROOT / "src" / "Methods" / "FasterGS"))

import imageio.v3 as iio
import numpy as np
import torch

from FasterGSCudaBackend.FasterGSCudaBackend.torch_bindings import compute_edge_scores


def save_map(path: Path, tensor: torch.Tensor) -> None:
    image = tensor.squeeze().detach().float().cpu().numpy()
    image = image - image.min()
    vmax = image.max()
    if vmax > 0:
        image = image / vmax
    iio.imwrite(path, (255.0 * image).astype(np.uint8))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=str, required=True)
    parser.add_argument("--outdir", type=str, default="edge_score_debug")
    parser.add_argument("--bins", type=int, default=512)
    parser.add_argument("--eps", type=float, default=1e-6)
    args = parser.parse_args()

    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    image_np = iio.imread(args.input)
    image = torch.from_numpy(np.asarray(image_np)).cuda()

    scores, nms, grad_mag, blurred, median = compute_edge_scores(
        image,
        histogram_bins=args.bins,
        eps=args.eps,
        return_intermediates=True,
    )

    save_map(outdir / "scores.png", scores)
    save_map(outdir / "nms.png", nms)
    save_map(outdir / "grad_mag.png", grad_mag)
    save_map(outdir / "blurred.png", blurred)

    print(f"Saved outputs to: {outdir}")
    print("Median per image:", median.detach().cpu().tolist())


if __name__ == "__main__":
    main()
