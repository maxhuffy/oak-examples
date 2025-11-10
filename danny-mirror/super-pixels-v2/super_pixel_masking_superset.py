#!/usr/bin/env python3
import argparse
import os
import numpy as np
from skimage import io, img_as_float
from skimage.segmentation import slic


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
        n_segments=2100,
        compactness=20,
        sigma=1,
        start_label=1,
    )
    return labels


def colorize_labels(label_map, bg_mask=None):
    """
    Turn an integer label map into an RGB image.
    bg_mask=False (or None) pixels will be black.
    """
    h, w = label_map.shape
    out = np.zeros((h, w, 3), dtype=np.uint8)

    unique_labels = np.unique(label_map)
    # 0 might be background (we'll just leave it black)
    rng = np.random.default_rng(42)
    colors = {}

    for lab in unique_labels:
        if lab == 0:
            continue
        # random color
        colors[lab] = rng.integers(0, 256, size=3, dtype=np.uint8)

    for lab, col in colors.items():
        out[label_map == lab] = col

    if bg_mask is not None:
        # ensure truly transparent pixels stay black
        out[~bg_mask] = 0

    return out


def main():
    parser = argparse.ArgumentParser(
        description="Create a superset/union superpixelization from original and edited images."
    )
    parser.add_argument("original", help="path to original.png")
    parser.add_argument("edited", help="path to edited.png")
    parser.add_argument(
        "--out",
        default="super_pixel_superset.png",
        help="output image path (default: super_pixel_superset.png)",
    )
    parser.add_argument(
        "--keep-transparent",
        action="store_true",
        help="show superset regions even in fully transparent areas",
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

    # 2) union bbox (so we don't SLIC massive whitespace)
    ymin, ymax, xmin, xmax = get_union_bbox(orig_alpha, edit_alpha)

    orig_img_c = orig_img[ymin:ymax, xmin:xmax, :]
    edit_img_c = edit_img[ymin:ymax, xmin:xmax, :]
    orig_alpha_c = orig_alpha[ymin:ymax, xmin:xmax]
    edit_alpha_c = edit_alpha[ymin:ymax, xmin:xmax]

    # 3) run SLIC separately
    orig_labels = run_slic_like_notebook(orig_img_c)
    edit_labels = run_slic_like_notebook(edit_img_c)

    # 4) build superset label map
    # We'll make a pair -> id map
    h_c, w_c = orig_labels.shape
    superset_labels = np.zeros((h_c, w_c), dtype=np.int32)

    # pack pairs into a single array so we can unique them
    # (orig_label, edit_label)
    pairs = np.stack([orig_labels, edit_labels], axis=-1).reshape(-1, 2)
    # get unique pairs and inverse index to map back
    unique_pairs, inverse = np.unique(pairs, axis=0, return_inverse=True)
    # we want labels to start at 1 (0 = background)
    superset_labels = inverse.reshape(h_c, w_c) + 1

    # 5) transparency handling:
    # if user does NOT want to keep transparent, mask out pixels where both are transparent
    both_transparent = ~(orig_alpha_c | edit_alpha_c)
    if not args.keep_transparent:
        superset_labels[both_transparent] = 0  # background

    # 6) colorize CROPPED
    # use union alpha as bg_mask so pure transparent stays black
    union_alpha_c = (orig_alpha_c | edit_alpha_c) if not args.keep_transparent else None
    out_cropped = colorize_labels(superset_labels, bg_mask=union_alpha_c)

    # 7) paste back to full size
    full_out = np.zeros((H, W, 3), dtype=np.uint8)
    full_out[ymin:ymax, xmin:xmax, :] = out_cropped

    # 8) save
    io.imsave(args.out, full_out)
    print(
        f"✅ Saved superset superpixel image to {os.path.abspath(args.out)} "
        f"(crop y[{ymin}:{ymax}], x[{xmin}:{xmax}], regions={len(unique_pairs)})"
    )


if __name__ == "__main__":
    main()
