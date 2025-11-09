#!/usr/bin/env python3
"""
first_ai_mask.py

Baseline "first guess" clothing / edited-region mask from:
    - original.png
    - edited.png

This uses the change-detection style approach we’ve been iterating on:
1. downscale original -> edited size (scale-to-cover + center-crop)
2. absdiff in a blurred space
3. threshold
4. morphological cleanup
5. keep biggest region
"""

import cv2
import numpy as np
import sys
from pathlib import Path


# -------------------------
# 1) load + align
# -------------------------
def load_images(orig_path, edit_path):
    orig = cv2.imread(str(orig_path))
    edit = cv2.imread(str(edit_path))

    if orig is None:
        raise RuntimeError(f"could not read original image: {orig_path}")
    if edit is None:
        raise RuntimeError(f"could not read edited image: {edit_path}")

    return orig, edit


def scale_cover_crop_down(orig, edit):
    """
    Make ORIGINAL match EDITED size without stretching:
    - scale original until it fully covers edited
    - center-crop to edited size
    This is the downscale direction that gave us the lowest MSE earlier.
    """
    target_h, target_w = edit.shape[:2]
    h, w = orig.shape[:2]

    # scale so that original fully covers edited
    scale = max(target_w / w, target_h / h)
    new_w = int(w * scale)
    new_h = int(h * scale)

    resized = cv2.resize(orig, (new_w, new_h), interpolation=cv2.INTER_LINEAR)

    # center-crop to edited size
    start_x = (new_w - target_w) // 2
    start_y = (new_h - target_h) // 2
    cropped = resized[start_y:start_y + target_h, start_x:start_x + target_w]

    return cropped


# -------------------------
# 2) change map
# -------------------------
def make_change_map(orig_aligned, edit, blur_ksize=5, diff_thresh=15):
    """
    Simple remote-sensing style change map:
    - blur both to reduce noise
    - absdiff
    - gray + threshold
    """
    if blur_ksize > 1:
        orig_b = cv2.GaussianBlur(orig_aligned, (blur_ksize, blur_ksize), 0)
        edit_b = cv2.GaussianBlur(edit, (blur_ksize, blur_ksize), 0)
    else:
        orig_b = orig_aligned
        edit_b = edit

    diff = cv2.absdiff(orig_b, edit_b)
    gray = cv2.cvtColor(diff, cv2.COLOR_BGR2GRAY)

    # fixed threshold for now (15 worked well in your tests)
    _, change = cv2.threshold(gray, diff_thresh, 255, cv2.THRESH_BINARY)

    return change


# -------------------------
# 3) postprocess to a single blob
# -------------------------
def clean_and_biggest(mask, open_ksize=5, close_ksize=5):
    """
    - open to remove specks
    - close to fill small holes
    - keep largest connected component
    """
    mask_u8 = mask.astype(np.uint8)

    if open_ksize > 1:
        k_open = np.ones((open_ksize, open_ksize), np.uint8)
        mask_u8 = cv2.morphologyEx(mask_u8, cv2.MORPH_OPEN, k_open)

    if close_ksize > 1:
        k_close = np.ones((close_ksize, close_ksize), np.uint8)
        mask_u8 = cv2.morphologyEx(mask_u8, cv2.MORPH_CLOSE, k_close)

    # keep biggest region
    contours, _ = cv2.findContours(mask_u8, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return mask_u8

    biggest = max(contours, key=cv2.contourArea)
    out = np.zeros_like(mask_u8)
    cv2.drawContours(out, [biggest], -1, 255, thickness=cv2.FILLED)
    return out



SCRIPT_DIR = Path(__file__).resolve().parent


def resolve_paths():
    """Resolve CLI or default paths relative to the project layout."""
    if len(sys.argv) >= 3:
        orig_path = Path(sys.argv[1])
        edit_path = Path(sys.argv[2])
        out_path = Path(sys.argv[3]) if len(sys.argv) > 3 else Path("first_ai_mask.png")
    else:
        # default to commonly used inputs that live alongside this script
        base_dir = SCRIPT_DIR.parent
        orig_path = base_dir / "calibration_images" / "3_square.png"
        edit_path = base_dir / "calibration_images" / "square_api_output.jpg"
        out_path = Path("first_ai_mask_square.png")

    if not orig_path.is_absolute():
        orig_path = (SCRIPT_DIR / orig_path).resolve()
    if not edit_path.is_absolute():
        edit_path = (SCRIPT_DIR / edit_path).resolve()
    if not out_path.is_absolute():
        out_path = (SCRIPT_DIR / out_path).resolve()

    return orig_path, edit_path, out_path


def main():
    # if len(sys.argv) < 3:
    #     print("usage: python first_ai_mask.py original.png edited.png [out_mask.png]")
    #     sys.exit(1)

    orig_path, edit_path, out_path = resolve_paths()

    orig, edit = load_images(orig_path, edit_path)

    # 1) downscale original to edited using our best-performing method
    orig_aligned = scale_cover_crop_down(orig, edit)

    # 2) change map
    change = make_change_map(orig_aligned, edit,
                             blur_ksize=5,
                             diff_thresh=15)

    # 3) cleanup -> single blob
    mask = clean_and_biggest(change,
                             open_ksize=5,
                             close_ksize=5)

    cv2.imwrite(str(out_path), mask)
    print(f"[ok] wrote {out_path}")


if __name__ == "__main__":
    main()
