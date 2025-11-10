#!/usr/bin/env python3
import argparse
import os
import numpy as np
from skimage import io, img_as_float, color, morphology
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

    rgb = img_as_float(rgb)
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


def run_slic(img):
    return slic(
        img,
        n_segments=2200,
        compactness=20,
        sigma=0,
        start_label=1,
    )


def main():
    parser = argparse.ArgumentParser(
        description="Region-based diff mask from superset superpixels (color-composition only)."
    )
    parser.add_argument("original", help="path to original.png")
    parser.add_argument("edited", help="path to edited.png")
    parser.add_argument(
        "--out",
        default="super_pixel_mask.png",
        help="output mask path (default: super_pixel_mask.png)",
    )
    # ΔE threshold ~3-6 is noticeable; tune to your data
    parser.add_argument(
        "--lab-thresh",
        type=float,
        default=4.5,
        help="mean Lab ΔE threshold to mark region as changed (default: 4.5)",
    )
    parser.add_argument(
        "--pix-diff",
        type=float,
        default=0.06,
        help="per-pixel RGB abs-diff considered 'changed' (0..1) (default: 0.06 ≈ 15/255)",
    )
    parser.add_argument(
        "--pix-ratio",
        type=float,
        default=0.12,
        help="if this fraction of valid pixels exceed --pix-diff, keep region (default: 0.12 = 12%)",
    )
    parser.add_argument(
        "--min-area",
        type=int,
        default=5,
        help="ignore tiny regions (default: 5)",
    )
    parser.add_argument(
        "--alpha-dilate",
        type=int,
        default=1,
        help="dilate union alpha by this many pixels to catch border regions (default: 1)",
    )
    args = parser.parse_args()

    # 1) load
    orig_img, orig_alpha = load_image_and_alpha(args.original)
    edit_img, edit_alpha = load_image_and_alpha(args.edited)

    if orig_img.shape != edit_img.shape:
        raise ValueError("Images must be same size for this script.")

    H, W, _ = orig_img.shape

    # 2) work only inside union bbox
    ymin, ymax, xmin, xmax = get_union_bbox(orig_alpha, edit_alpha)
    orig_c = orig_img[ymin:ymax, xmin:xmax, :]
    edit_c = edit_img[ymin:ymax, xmin:xmax, :]
    orig_a_c = orig_alpha[ymin:ymax, xmin:xmax]
    edit_a_c = edit_alpha[ymin:ymax, xmin:xmax]
    h_c, w_c, _ = orig_c.shape

    # 3) SLIC on both, then combine → superset
    orig_labels = run_slic(orig_c)
    edit_labels = run_slic(edit_c)

    pairs = np.stack([orig_labels, edit_labels], axis=-1).reshape(-1, 2)
    unique_pairs, inverse = np.unique(pairs, axis=0, return_inverse=True)
    superset_labels = inverse.reshape(h_c, w_c) + 1
    n_regions = len(unique_pairs)

    # 4) union content mask, dilated, to avoid keeping empty background regions
    union_alpha_c = (orig_a_c | edit_a_c)
    if args.alpha_dilate > 0:
        selem = morphology.square(1 + 2 * args.alpha_dilate)
        union_alpha_c = morphology.dilation(union_alpha_c, selem)

    # 5) convert to Lab for perceptual diffs
    orig_lab = color.rgb2lab(orig_c)
    edit_lab = color.rgb2lab(edit_c)

    # 6) build final mask (cropped)
    keep_mask_c = np.zeros((h_c, w_c), dtype=bool)

    for region_id in range(1, n_regions + 1):
        region_pixels = (superset_labels == region_id)
        area = np.count_nonzero(region_pixels)
        if area < args.min_area:
            continue

        # only evaluate where region intersects real content
        valid_pixels = region_pixels & union_alpha_c
        valid_count = np.count_nonzero(valid_pixels)
        if valid_count == 0:
            # region is outside or transparent → skip
            continue

        # color composition difference: mean ΔE
        orig_lab_vals = orig_lab[valid_pixels]
        edit_lab_vals = edit_lab[valid_pixels]
        delta_lab = np.linalg.norm(orig_lab_vals - edit_lab_vals, axis=1)
        mean_delta = float(delta_lab.mean())

        # also: what fraction of pixels changed a lot in plain RGB?
        orig_rgb_vals = orig_c[valid_pixels]
        edit_rgb_vals = edit_c[valid_pixels]
        per_pix_diff = np.abs(orig_rgb_vals - edit_rgb_vals).mean(axis=1)  # (N,)
        changed_ratio = float(np.count_nonzero(per_pix_diff >= args.pix_diff)) / valid_count

        # decide to keep
        if (mean_delta >= args.lab_thresh) or (changed_ratio >= args.pix_ratio):
            # fill WHOLE REGION, not just valid pixels
            keep_mask_c[region_pixels] = True

    # 7) paste back to full-size
    full_mask = np.zeros((H, W), dtype=np.uint8)
    full_mask[ymin:ymax, xmin:xmax] = keep_mask_c.astype(np.uint8) * 255

    # 8) save
    io.imsave(args.out, full_mask)
    print(
        f"✅ Saved mask to {os.path.abspath(args.out)} | "
        f"regions={n_regions}, kept_pixels={keep_mask_c.sum()} "
        f"({keep_mask_c.sum()/(h_c*w_c):.2%} of crop)"
    )


if __name__ == "__main__":
    main()
