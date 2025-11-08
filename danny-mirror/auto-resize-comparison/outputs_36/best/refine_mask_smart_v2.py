#!/usr/bin/env python3
import cv2
import numpy as np
import sys
import os


def load_imgs(orig_path, edit_path, mask_path):
    orig = cv2.imread(orig_path)
    edit = cv2.imread(edit_path)
    mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)

    if orig is None or edit is None or mask is None:
        raise RuntimeError("Could not read one or more input images")

    h, w = orig.shape[:2]
    if edit.shape[:2] != (h, w):
        edit = cv2.resize(edit, (w, h), interpolation=cv2.INTER_LINEAR)
    if mask.shape[:2] != (h, w):
        mask = cv2.resize(mask, (w, h), interpolation=cv2.INTER_NEAREST)

    return orig, edit, mask


def make_diff_mask(orig, edit, blur_ksize=5, thresh=7):
    """binary: where did the edited image actually change vs original"""
    if blur_ksize > 1:
        orig_b = cv2.GaussianBlur(orig, (blur_ksize, blur_ksize), 0)
        edit_b = cv2.GaussianBlur(edit, (blur_ksize, blur_ksize), 0)
    else:
        orig_b = orig
        edit_b = edit

    diff = cv2.absdiff(orig_b, edit_b)
    gray = cv2.cvtColor(diff, cv2.COLOR_BGR2GRAY)
    _, diff_bin = cv2.threshold(gray, thresh, 255, cv2.THRESH_BINARY)
    return diff_bin


def largest_filled_region(mask):
    """Keep only the largest connected area and fill it."""
    mask_bin = (mask > 0).astype(np.uint8) * 255
    contours, _ = cv2.findContours(mask_bin, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return mask_bin

    biggest = max(contours, key=cv2.contourArea)
    filled = np.zeros_like(mask_bin)
    cv2.drawContours(filled, [biggest], -1, 255, thickness=cv2.FILLED)
    return filled


def grow_mask_with_diff(mask, diff_mask, steps=1, kernel_size=5):
    """
    Dilate mask but only add pixels that are in diff_mask.
    We can run it multiple times (steps).
    """
    kernel = np.ones((kernel_size, kernel_size), np.uint8)
    current = mask.copy()

    for _ in range(steps):
        dilated = cv2.dilate(current, kernel, iterations=1)
        new_pixels = (dilated == 255) & (current == 0)
        allowed = new_pixels & (diff_mask == 255)
        if not np.any(allowed):
            break
        current[allowed] = 255

    return current


def shrink_where_no_diff(mask, diff_mask, support_kernel=7, support_thresh_ratio=0.15):
    """
    For pixels currently in the mask, check their neighborhood in diff_mask.
    If a mask pixel has too little diff-support around it, we drop it.
    This is the "shrink" counterpart to the grow.
    """
    # compute local support from diff
    k = support_kernel
    kernel = np.ones((k, k), np.uint8)
    # this gives us, for each pixel, how many diff pixels are around
    diff_support = cv2.filter2D((diff_mask == 255).astype(np.uint8), -1, kernel)

    # max possible in neighborhood:
    max_support = k * k
    # threshold in counts
    support_thresh = int(max_support * support_thresh_ratio)

    mask_bin = (mask == 255)
    low_support = (diff_support < support_thresh)

    # only shrink where we currently have mask but very low diff around
    to_remove = mask_bin & low_support
    out = mask.copy()
    out[to_remove] = 0
    return out


def smooth_edges(mask, kernel_size=5):
    """light close + redraw biggest contour to round it out"""
    kernel = np.ones((kernel_size, kernel_size), np.uint8)
    closed = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)

    contours, _ = cv2.findContours(closed, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return closed
    biggest = max(contours, key=cv2.contourArea)
    smoothed = np.zeros_like(mask)
    cv2.drawContours(smoothed, [biggest], -1, 255, thickness=cv2.FILLED)
    return smoothed


def refine_mask_iterative(orig_path, edit_path, mask_path,
                          out_path="refined_mask.png",
                          steps_dir=None,
                          outer_iters=10):
    """
    More iterative version:
    - make diff
    - get solid base
    - for a few outer iters:
        * grow a bit (into diff)
        * shrink a bit (where diff doesn't support it)
        * smooth
    """
    if steps_dir:
        os.makedirs(steps_dir, exist_ok=True)

    orig, edit, start_mask = load_imgs(orig_path, edit_path, mask_path)

    diff_mask = make_diff_mask(orig, edit, blur_ksize=5, thresh=7)
    if steps_dir:
        cv2.imwrite(os.path.join(steps_dir, "step_00_diff.png"), diff_mask)

    current = largest_filled_region(start_mask)
    if steps_dir:
        cv2.imwrite(os.path.join(steps_dir, "step_01_solid.png"), current)

    for i in range(outer_iters):
        # 1) grow a little
        current = grow_mask_with_diff(current, diff_mask,
                                      steps=1,   # small grow each outer loop
                                      kernel_size=5)

        # 2) shrink where support is low
        current = shrink_where_no_diff(current, diff_mask,
                                       support_kernel=11,
                                       support_thresh_ratio=0.6)

        # 3) smooth
        current = smooth_edges(current, kernel_size=17)

        if steps_dir:
            cv2.imwrite(os.path.join(steps_dir, f"step_{i+2:02d}_iter.png"), current)

    cv2.imwrite(out_path, current)
    print(f"[ok] wrote {out_path}")


def main():
    if len(sys.argv) < 4:
        print("usage: python refine_mask_iter_grow_shrink.py original.png edited.png start_mask.png [out_mask.png] [steps_dir]")
        sys.exit(1)

    orig_path = sys.argv[1]
    edit_path = sys.argv[2]
    mask_path = sys.argv[3]
    out_path = sys.argv[4] if len(sys.argv) > 4 else "refined_mask.png"
    steps_dir = sys.argv[5] if len(sys.argv) > 5 else None

    refine_mask_iterative(orig_path, edit_path, mask_path, out_path, steps_dir)


if __name__ == "__main__":
    main()
