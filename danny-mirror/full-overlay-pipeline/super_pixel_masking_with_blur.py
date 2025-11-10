#!/usr/bin/env python3
import argparse
import os
import numpy as np
from skimage import io, img_as_float, color, morphology, filters, measure
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
        n_segments=3000,
        compactness=25,
        sigma=0,
        start_label=1,
    )


def trimmed_lab_deltaE_from_masks(orig_lab, edit_lab, mask, trim_ratio=0.05):
    idx = np.where(mask)
    if idx[0].size == 0:
        return None
    ol = orig_lab[idx]
    el = edit_lab[idx]
    delta = np.linalg.norm(ol - el, axis=1)
    delta_sorted = np.sort(delta)
    k = int(len(delta_sorted) * trim_ratio)
    if k > 0 and k < len(delta_sorted):
        delta_trim = delta_sorted[:-k]
    else:
        delta_trim = delta_sorted
    return float(delta_trim.mean())


def color_erosion(img, radius=1):
    selem = morphology.disk(radius)
    eroded = np.zeros_like(img)
    for c in range(3):
        eroded[..., c] = morphology.erosion(img[..., c], selem)
    return eroded


def keep_largest_components(mask_uint8, max_components=4, min_area=0):
    """
    mask_uint8: 0/255
    keep at most `max_components` biggest blobs
    and also drop anything smaller than `min_area`
    """
    bin_mask = mask_uint8 > 0
    labeled = measure.label(bin_mask, connectivity=2)
    if labeled.max() == 0:
        return mask_uint8

    # collect (label, area)
    areas = []
    for lab in range(1, labeled.max() + 1):
        a = np.sum(labeled == lab)
        if a >= min_area:
            areas.append((lab, a))

    # sort big → small
    areas.sort(key=lambda x: x[1], reverse=True)
    areas = areas[:max_components]  # keep only N biggest

    keep_labels = {lab for lab, _ in areas}
    cleaned = np.zeros_like(bin_mask)
    for lab in keep_labels:
        cleaned |= (labeled == lab)

    return (cleaned.astype(np.uint8) * 255)


def main():
    parser = argparse.ArgumentParser(
        description="Region-based diff mask from superset superpixels with pre-erosion smoothing."
    )
    parser.add_argument("original", help="path to original.png")
    parser.add_argument("edited", help="path to edited.png")
    parser.add_argument("--out", default="super_pixel_mask.png")
    parser.add_argument("--kernel-size", type=int, default=9)
    parser.add_argument("--kernel-trim", type=float, default=0.05)
    parser.add_argument("--kernel-thresh", type=float, default=8.0)
    parser.add_argument("--min-area", type=int, default=5)
    parser.add_argument("--alpha-dilate", type=int, default=1)
    parser.add_argument("--region-trim", type=float, default=0.05)
    parser.add_argument("--region-thresh", type=float, default=6.0)
    parser.add_argument("--blur-sigma", type=float, default=8)
    # NEW: artifact-removal controls
    parser.add_argument("--max-components", type=int, default=2,
                        help="keep at most this many big blobs at the very end")
    parser.add_argument("--min-component-area", type=int, default=200,
                        help="drop connected components smaller than this many pixels")
    args = parser.parse_args()

    # load
    orig_img, orig_alpha = load_image_and_alpha(args.original)
    edit_img, edit_alpha = load_image_and_alpha(args.edited)

    if orig_img.shape != edit_img.shape:
        raise ValueError("Images must be same size.")

    # small erosions
    orig_img = color_erosion(orig_img, radius=1)
    edit_img = color_erosion(edit_img, radius=1)
    orig_img = color_erosion(orig_img, radius=1)

    H, W, _ = orig_img.shape

    # crop to union bbox
    ymin, ymax, xmin, xmax = get_union_bbox(orig_alpha, edit_alpha)
    orig_c = orig_img[ymin:ymax, xmin:xmax, :]
    edit_c = edit_img[ymin:ymax, xmin:xmax, :]
    orig_a_c = orig_alpha[ymin:ymax, xmin:xmax]
    edit_a_c = edit_alpha[ymin:ymax, xmin:xmax]
    h_c, w_c, _ = orig_c.shape

    # blur
    if args.blur_sigma > 0:
        orig_c = filters.gaussian(orig_c, sigma=args.blur_sigma, channel_axis=-1)
        edit_c = filters.gaussian(edit_c, sigma=args.blur_sigma, channel_axis=-1)

    # slic → superset
    orig_labels = run_slic(orig_c)
    edit_labels = run_slic(edit_c)
    pairs = np.stack([orig_labels, edit_labels], axis=-1).reshape(-1, 2)
    unique_pairs, inverse = np.unique(pairs, axis=0, return_inverse=True)
    superset_labels = inverse.reshape(h_c, w_c) + 1
    n_regions = len(unique_pairs)

    # union alpha (dilated)
    union_alpha_c = (orig_a_c | edit_a_c)
    if args.alpha_dilate > 0:
        # Revert to square to match previously working behavior (may emit FutureWarning on newer skimage)
        selem = morphology.square(1 + 2 * args.alpha_dilate)
        union_alpha_c = morphology.dilation(union_alpha_c, selem)

    # lab
    orig_lab = color.rgb2lab(orig_c)
    edit_lab = color.rgb2lab(edit_c)

    # region loop
    keep_mask_c = np.zeros((h_c, w_c), dtype=bool)
    ks = args.kernel_size

    for region_id in range(1, n_regions + 1):
        region_pixels = (superset_labels == region_id)
        area = np.count_nonzero(region_pixels)
        if area < args.min_area:
            continue

        valid_pixels = region_pixels & union_alpha_c
        if not np.any(valid_pixels):
            continue

        ys, xs = np.where(region_pixels)
        rymin, rymax = ys.min(), ys.max()
        rxmin, rxmax = xs.min(), xs.max()

        region_is_changed = False

        # per-kernel
        for y0 in range(rymin, rymax + 1, ks):
            if region_is_changed:
                break
            y1 = min(y0 + ks, h_c)
            for x0 in range(rxmin, rxmax + 1, ks):
                x1 = min(x0 + ks, w_c)

                kernel_mask = region_pixels[y0:y1, x0:x1] & union_alpha_c[y0:y1, x0:x1]
                if not np.any(kernel_mask):
                    continue

                ol = orig_lab[y0:y1, x0:x1, :]
                el = edit_lab[y0:y1, x0:x1, :]

                idx = np.where(kernel_mask)
                ol_sel = ol[idx]
                el_sel = el[idx]
                delta = np.linalg.norm(ol_sel - el_sel, axis=1)
                delta_sorted = np.sort(delta)
                ktrim = int(len(delta_sorted) * args.kernel_trim)
                if ktrim > 0 and ktrim < len(delta_sorted):
                    delta_trim = delta_sorted[:-ktrim]
                else:
                    delta_trim = delta_sorted
                kernel_score = float(delta_trim.mean())

                if kernel_score > args.kernel_thresh:
                    region_is_changed = True
                    break

        # region-level fallback
        if not region_is_changed:
            region_score = trimmed_lab_deltaE_from_masks(
                orig_lab, edit_lab, valid_pixels, trim_ratio=args.region_trim
            )
            if region_score is not None and region_score > args.region_thresh:
                region_is_changed = True

        if region_is_changed:
            keep_mask_c[region_pixels] = True

    # close tiny seams in crop
    keep_mask_c = morphology.binary_closing(keep_mask_c, morphology.disk(1))

    # paste to full-size
    full_mask = np.zeros((H, W), dtype=np.uint8)
    full_mask[ymin:ymax, xmin:xmax] = keep_mask_c.astype(np.uint8) * 255

    # clip by edited.png transparency (this can create new floaters!)
    full_mask[~edit_alpha] = 0

    # NOW do artifact removal on the final mask
    if args.max_components > 0:
        full_mask = keep_largest_components(
            full_mask,
            max_components=args.max_components,
            min_area=args.min_component_area,
        )

    # save
    io.imsave(args.out, full_mask)
    # Avoid Unicode emoji that can trigger Windows cp1252 encode errors
    print(f"Saved mask to {os.path.abspath(args.out)} | regions={n_regions}")


if __name__ == "__main__":
    main()
