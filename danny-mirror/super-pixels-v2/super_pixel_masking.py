#!/usr/bin/env python3
import argparse
import os
import numpy as np
from skimage import io, img_as_float
from skimage.segmentation import slic, find_boundaries


def load_image_and_alpha(path):
    """Loads an image, returning (rgb, alpha_mask_bool)."""
    img = io.imread(path)
    if img.ndim == 2:  # grayscale
        img = np.stack([img] * 3, axis=-1)

    if img.shape[2] == 4:
        rgb = img[:, :, :3]
        alpha = img[:, :, 3] > 0
    else:
        rgb = img[:, :, :3]
        alpha = np.ones(rgb.shape[:2], dtype=bool)

    return rgb, alpha


def run_slic_like_notebook(img):
    """Use the same params as your notebook UI."""
    img_float = img_as_float(img)
    labels = slic(
        img_float,
        n_segments=1400,
        compactness=19.9526,
        sigma=1,
        start_label=1,
    )
    return labels


def local_nontransparent(alpha):
    """
    For each pixel, tell me if there is ANY nontransparent pixel
    in its 4-neighborhood (including itself).
    This lets us say: “this boundary edge actually touches real content”.
    """
    h, w = alpha.shape
    # start with itself
    any_nt = alpha.copy()

    # up
    up = np.zeros_like(alpha)
    up[1:, :] = alpha[:-1, :]
    any_nt |= up

    # down
    down = np.zeros_like(alpha)
    down[:-1, :] = alpha[1:, :]
    any_nt |= down

    # left
    left = np.zeros_like(alpha)
    left[:, 1:] = alpha[:, :-1]
    any_nt |= left

    # right
    right = np.zeros_like(alpha)
    right[:, :-1] = alpha[:, 1:]
    any_nt |= right

    return any_nt


def boundaries_to_rgb(orig_bounds, edit_bounds):
    """
    Combine boolean boundary masks into RGB.
    Blue   = original only
    Red    = edited only
    Purple = both
    Black  = none
    """
    h, w = orig_bounds.shape
    out = np.zeros((h, w, 3), dtype=np.uint8)

    overlap = orig_bounds & edit_bounds
    orig_only = orig_bounds & ~edit_bounds
    edit_only = edit_bounds & ~orig_bounds

    out[orig_only] = (0, 0, 255)       # blue
    out[edit_only] = (255, 0, 0)       # red
    out[overlap] = (255, 0, 255)       # purple
    return out


def main():
    parser = argparse.ArgumentParser(
        description="Combine superpixel boundaries from original and edited images into a 3-color mask, ignoring edges that separate only transparent regions."
    )
    parser.add_argument("original", help="path to original.png")
    parser.add_argument("edited", help="path to edited.png")
    parser.add_argument(
        "--out",
        default="super_pixel_boundaries.png",
        help="output image path (default: super_pixel_boundaries.png)",
    )
    args = parser.parse_args()

    # load
    orig_img, orig_alpha = load_image_and_alpha(args.original)
    edit_img, edit_alpha = load_image_and_alpha(args.edited)

    if orig_img.shape != edit_img.shape:
        raise ValueError(
            f"Images must have the same shape. Got {orig_img.shape} vs {edit_img.shape}"
        )

    # slic
    orig_labels = run_slic_like_notebook(orig_img)
    edit_labels = run_slic_like_notebook(edit_img)

    # raw boundaries
    orig_bounds = find_boundaries(orig_labels, mode="outer")
    edit_bounds = find_boundaries(edit_labels, mode="outer")

    # figure out which boundary pixels actually touch real content
    orig_touch_real = local_nontransparent(orig_alpha)
    edit_touch_real = local_nontransparent(edit_alpha)

    # keep only boundary pixels that touch real content in that image
    orig_bounds &= orig_touch_real
    edit_bounds &= edit_touch_real

    # combine
    out_img = boundaries_to_rgb(orig_bounds, edit_bounds)

    io.imsave(args.out, out_img)
    print(f"Saved combined boundary image to {os.path.abspath(args.out)}")


if __name__ == "__main__":
    main()
