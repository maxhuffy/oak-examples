#!/usr/bin/env python3
import argparse
import os
import numpy as np
from skimage import io, img_as_float, morphology
from skimage.segmentation import slic, find_boundaries


def load_image_and_alpha(path):
    img = io.imread(path)
    if img.ndim == 2:
        img = np.stack([img] * 3, axis=-1)

    if img.shape[2] == 4:
        rgb = img[:, :, :3]
        alpha = img[:, :, 3] > 0
    else:
        rgb = img[:, :, :3]
        alpha = np.ones(rgb.shape[:2], dtype=bool)
    return rgb, alpha


def get_union_bbox(alpha1, alpha2):
    union = alpha1 | alpha2
    ys, xs = np.where(union)
    if ys.size == 0 or xs.size == 0:
        h, w = alpha1.shape
        return 0, h, 0, w
    ymin, ymax = ys.min(), ys.max() + 1
    xmin, xmax = xs.min(), xs.max() + 1
    return ymin, ymax, xmin, xmax


def run_slic_like_notebook(img):
    img_float = img_as_float(img)
    labels = slic(
        img_float,
        n_segments=2200,
        compactness=20,
        sigma=0,
        start_label=1,
    )
    return labels


def local_nontransparent(alpha):
    """4-neighbor expansion to decide if a boundary pixel is near real content."""
    h, w = alpha.shape
    any_nt = alpha.copy()

    up = np.zeros_like(alpha)
    up[1:, :] = alpha[:-1, :]
    any_nt |= up

    down = np.zeros_like(alpha)
    down[:-1, :] = alpha[1:, :]
    any_nt |= down

    left = np.zeros_like(alpha)
    left[:, 1:] = alpha[:, :-1]
    any_nt |= left

    right = np.zeros_like(alpha)
    right[:, :-1] = alpha[:, 1:]
    any_nt |= right

    return any_nt


def thin_boundaries(boundary_mask, thickness=1):
    thin = morphology.skeletonize(boundary_mask)
    if thickness > 1:
        thin = morphology.binary_dilation(thin, morphology.disk(thickness - 1))
    return thin


def main():
    parser = argparse.ArgumentParser(
        description="Create a UNION/superset superpixelization of original+edited and output just the borders."
    )
    parser.add_argument("original", help="path to original.png")
    parser.add_argument("edited", help="path to edited.png")
    parser.add_argument(
        "--out",
        default="super_pixel_superset_borders.png",
        help="output image path (default: super_pixel_superset_borders.png)",
    )
    parser.add_argument(
        "--thickness",
        type=int,
        default=1,
        help="boundary thickness (1 = thinnest, default = 1)",
    )
    parser.add_argument(
        "--keep-transparent",
        action="store_true",
        help="keep borders even in fully transparent areas",
    )
    args = parser.parse_args()

    # 1) load
    orig_img, orig_alpha = load_image_and_alpha(args.original)
    edit_img, edit_alpha = load_image_and_alpha(args.edited)

    if orig_img.shape != edit_img.shape:
        raise ValueError(
            f"Images must have same shape. Got {orig_img.shape} vs {edit_img.shape}"
        )

    H, W, _ = orig_img.shape

    # 2) union bbox
    ymin, ymax, xmin, xmax = get_union_bbox(orig_alpha, edit_alpha)

    # 3) crop
    orig_img_c = orig_img[ymin:ymax, xmin:xmax, :]
    edit_img_c = edit_img[ymin:ymax, xmin:xmax, :]
    orig_alpha_c = orig_alpha[ymin:ymax, xmin:xmax]
    edit_alpha_c = edit_alpha[ymin:ymax, xmin:xmax]

    # 4) SLIC on both
    orig_labels = run_slic_like_notebook(orig_img_c)
    edit_labels = run_slic_like_notebook(edit_img_c)

    # 5) build superset labels: unique pairs → new id
    h_c, w_c = orig_labels.shape
    pairs = np.stack([orig_labels, edit_labels], axis=-1).reshape(-1, 2)
    unique_pairs, inverse = np.unique(pairs, axis=0, return_inverse=True)
    superset_labels = inverse.reshape(h_c, w_c) + 1  # start at 1

    # 6) boundaries on superset
    superset_bounds = find_boundaries(superset_labels, mode="inner")

    # 7) transparency filtering
    if not args.keep_transparent:
        union_alpha_c = local_nontransparent(orig_alpha_c | edit_alpha_c)
        superset_bounds &= union_alpha_c

    # 8) thin
    superset_bounds = thin_boundaries(superset_bounds, thickness=args.thickness)

    # 9) make an RGB image with just borders (white)
    out_cropped = np.zeros((h_c, w_c, 3), dtype=np.uint8)
    out_cropped[superset_bounds] = (255, 255, 255)

    # 10) paste back to full-size
    full_out = np.zeros((H, W, 3), dtype=np.uint8)
    full_out[ymin:ymax, xmin:xmax, :] = out_cropped

    # 11) save
    io.imsave(args.out, full_out)
    print(
        f"✅ Saved superset boundary image to {os.path.abspath(args.out)} "
        f"(crop y[{ymin}:{ymax}], x[{xmin}:{xmax}], regions={len(unique_pairs)})"
    )


if __name__ == "__main__":
    main()
