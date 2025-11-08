#!/usr/bin/env python3
import cv2
import numpy as np
import sys
import os


# -------------------- load & align --------------------
def load_imgs(orig_path, edit_path, markers_path):
    orig = cv2.imread(orig_path)
    edit = cv2.imread(edit_path)
    markers = cv2.imread(markers_path)

    if orig is None or edit is None or markers is None:
        raise RuntimeError("Could not read one or more input images")

    h, w = orig.shape[:2]
    if edit.shape[:2] != (h, w):
        edit = cv2.resize(edit, (w, h), interpolation=cv2.INTER_LINEAR)
    if markers.shape[:2] != (h, w):
        markers = cv2.resize(markers, (w, h), interpolation=cv2.INTER_LINEAR)

    return orig, edit, markers


# -------------------- diff mask (baseline clothes-ish area) --------------------
def make_diff_mask(orig, edit, blur_ksize=5, thresh=15):
    if blur_ksize > 1:
        orig_b = cv2.GaussianBlur(orig, (blur_ksize, blur_ksize), 0)
        edit_b = cv2.GaussianBlur(edit, (blur_ksize, blur_ksize), 0)
    else:
        orig_b = orig
        edit_b = edit

    diff = cv2.absdiff(orig_b, edit_b)
    gray = cv2.cvtColor(diff, cv2.COLOR_BGR2GRAY)
    _, diff_bin = cv2.threshold(gray, thresh, 255, cv2.THRESH_BINARY)

    # light cleanup so it’s not just salt & pepper
    k = np.ones((5, 5), np.uint8)
    diff_bin = cv2.morphologyEx(diff_bin, cv2.MORPH_OPEN, k)
    diff_bin = cv2.morphologyEx(diff_bin, cv2.MORPH_CLOSE, k)

    return diff_bin


# -------------------- markers → red seeds + blue forbids --------------------
def extract_red_blue(markers_bgr):
    """
    markers_bgr has skinny red lines (FF0000) and blue circles (0026FF)
    We detect them in HSV and then DILATE them so they become usable regions.
    """
    hsv = cv2.cvtColor(markers_bgr, cv2.COLOR_BGR2HSV)

    # red (two HSV ranges)
    lower_red1 = np.array([0, 120, 70])
    upper_red1 = np.array([10, 255, 255])
    lower_red2 = np.array([170, 120, 70])
    upper_red2 = np.array([180, 255, 255])
    mask_red1 = cv2.inRange(hsv, lower_red1, upper_red1)
    mask_red2 = cv2.inRange(hsv, lower_red2, upper_red2)
    red_mask = cv2.bitwise_or(mask_red1, mask_red2)

    # blue
    lower_blue = np.array([100, 80, 80])
    upper_blue = np.array([130, 255, 255])
    blue_mask = cv2.inRange(hsv, lower_blue, upper_blue)

    # thicken them so they’re usable
    red_mask = cv2.dilate(red_mask, np.ones((7, 7), np.uint8), iterations=1)
    blue_mask = cv2.dilate(blue_mask, np.ones((9, 9), np.uint8), iterations=1)

    return red_mask, blue_mask


# -------------------- biggest region helper --------------------
def biggest_region(mask):
    mask_bin = (mask > 0).astype(np.uint8) * 255
    contours, _ = cv2.findContours(mask_bin, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return mask_bin
    biggest = max(contours, key=cv2.contourArea)
    out = np.zeros_like(mask_bin)
    cv2.drawContours(out, [biggest], -1, 255, thickness=cv2.FILLED)
    return out


# -------------------- grow from red, but only into diff and not blue --------------------
def grow_from_seeds(seed_mask, diff_mask, blue_mask, steps=10, ksize=5):
    kernel = np.ones((ksize, ksize), np.uint8)
    current = seed_mask.copy()

    for _ in range(steps):
        dil = cv2.dilate(current, kernel, iterations=1)
        new_pixels = (dil == 255) & (current == 0)
        allowed = new_pixels & (diff_mask == 255) & (blue_mask == 0)
        if not np.any(allowed):
            break
        current[allowed] = 255

    return current


# -------------------- shrink where diff doesn’t support or near blue --------------------
def shrink_with_refs(mask, diff_mask, blue_mask,
                     support_kernel=7,
                     support_ratio=0.12,
                     protect_mask=None):
    k = support_kernel
    kernel = np.ones((k, k), np.uint8)
    diff_support = cv2.filter2D((diff_mask == 255).astype(np.uint8), -1, kernel)
    max_support = k * k
    need = int(max_support * support_ratio)

    mask_bin = (mask == 255)
    low_support = diff_support < need

    # expand blue a bit → stay away from it
    blue_dil = cv2.dilate((blue_mask == 255).astype(np.uint8) * 255,
                          np.ones((5, 5), np.uint8), iterations=1)
    near_blue = (blue_dil == 255)

    to_remove = mask_bin & (low_support | near_blue)

    if protect_mask is not None:
        to_remove = to_remove & (protect_mask == 0)

    out = mask.copy()
    out[to_remove] = 0
    return out


# -------------------- smoothing --------------------
def smooth(mask, ksize=5):
    kernel = np.ones((ksize, ksize), np.uint8)
    closed = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
    # redraw biggest to get rid of little edges
    return biggest_region(closed)


# -------------------- main pipeline --------------------
def refine_from_markers(orig_path, edit_path, markers_path,
                        out_path="refined_mask.png",
                        steps_dir=None,
                        outer_iters=4):
    if steps_dir:
        os.makedirs(steps_dir, exist_ok=True)

    orig, edit, markers = load_imgs(orig_path, edit_path, markers_path)

    # 1. baseline diff mask
    diff_mask = make_diff_mask(orig, edit, blur_ksize=5, thresh=15)
    if steps_dir:
        cv2.imwrite(os.path.join(steps_dir, "step_00_diff.png"), diff_mask)

    # 2. red/blue from marker image
    red_mask, blue_mask = extract_red_blue(markers)
    if steps_dir:
        cv2.imwrite(os.path.join(steps_dir, "step_01_red.png"), red_mask)
        cv2.imwrite(os.path.join(steps_dir, "step_01_blue.png"), blue_mask)

    # 3. initial mask:
    #    - start with diff area
    #    - ensure red seeds are included (these sit on the person)
    base = diff_mask.copy()
    base = cv2.bitwise_or(base, red_mask)
    base = biggest_region(base)
    if steps_dir:
        cv2.imwrite(os.path.join(steps_dir, "step_02_base.png"), base)

    current = base

    # 4. iterative grow/shrink using diff + markers
    for i in range(outer_iters):
        # grow *from what we currently have*
        current = grow_from_seeds(current, diff_mask, blue_mask,
                                  steps=2, ksize=5)
        # shrink where diff doesn't back it up or near blue
        current = shrink_with_refs(current, diff_mask, blue_mask,
                                   support_kernel=7,
                                   support_ratio=0.10,
                                   protect_mask=red_mask)
        # smooth
        current = smooth(current, ksize=5)

        if steps_dir:
            cv2.imwrite(os.path.join(steps_dir, f"step_{i+3:02d}.png"), current)

    cv2.imwrite(out_path, current)
    print(f"[ok] wrote {out_path}")


def main():
    if len(sys.argv) < 4:
        print("usage: python refine_from_markers.py original.png edited.png original_with_markers.png [out_mask.png] [steps_dir]")
        sys.exit(1)

    orig_path = sys.argv[1]
    edit_path = sys.argv[2]
    markers_path = sys.argv[3]
    out_path = sys.argv[4] if len(sys.argv) > 4 else "refined_mask_from_marker.png"
    steps_dir = sys.argv[5] if len(sys.argv) > 5 else None

    refine_from_markers(orig_path, edit_path, markers_path, out_path, steps_dir)


if __name__ == "__main__":
    main()
