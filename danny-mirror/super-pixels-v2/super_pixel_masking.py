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
        n_segments=2000,
        compactness=20,
        sigma=1,
        start_label=1,
    )
    return labels


def local_nontransparent(alpha):
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


def boundaries_to_rgb(orig_bounds, edit_bounds):
    h, w = orig_bounds.shape
    out = np.zeros((h, w, 3), dtype=np.uint8)

    overlap = orig_bounds & edit_bounds
    orig_only = orig_bounds & ~edit_bounds
    edit_only = edit_bounds & ~orig_bounds

    out[orig_only] = (0, 0, 255)
    out[edit_only] = (255, 0, 0)
    out[overlap] = (255, 0, 255)
    return out


def main():
    parser = argparse.ArgumentParser(
        description="Crop to visible area, run superpixel boundaries, and paste result back into full-size image."
    )
    parser.add_argument("original", help="path to original.png")
    parser.add_argument("edited", help="path to edited.png")
    parser.add_argument(
        "--out",
        default="super_pixel_boundaries.png",
        help="output image path (default: super_pixel_boundaries.png)",
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
        help="keep all boundaries, even those fully inside transparent regions",
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

    # 4) SLIC on cropped
    orig_labels = run_slic_like_notebook(orig_img_c)
    edit_labels = run_slic_like_notebook(edit_img_c)

    # 5) boundaries on cropped
    orig_bounds = find_boundaries(orig_labels, mode="inner")
    edit_bounds = find_boundaries(edit_labels, mode="inner")

    # 6) transparency filtering on cropped
    if not args.keep_transparent:
        orig_touch_real = local_nontransparent(orig_alpha_c)
        edit_touch_real = local_nontransparent(edit_alpha_c)
        orig_bounds &= orig_touch_real
        edit_bounds &= edit_touch_real

    # 7) thin on cropped
    orig_bounds = thin_boundaries(orig_bounds, thickness=args.thickness)
    edit_bounds = thin_boundaries(edit_bounds, thickness=args.thickness)

    # 8) colorize CROPPED
    out_cropped = boundaries_to_rgb(orig_bounds, edit_bounds)

    # 9) paste back into FULL-SIZE canvas
    full_out = np.zeros((H, W, 3), dtype=np.uint8)
    full_out[ymin:ymax, xmin:xmax, :] = out_cropped

    # 10) save
    io.imsave(args.out, full_out)
    print(
        f"✅ Saved full-size boundary image to {os.path.abspath(args.out)} "
        f"(processed crop y[{ymin}:{ymax}], x[{xmin}:{xmax}])"
    )


if __name__ == "__main__":
    main()
