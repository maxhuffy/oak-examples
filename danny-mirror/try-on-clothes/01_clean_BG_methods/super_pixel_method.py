#!/usr/bin/env python3
"""
extract_jacket_superpixels.py

Superpixel-based change detector for the "original vs AI-added jacket" problem.

Strategy:
- segment the EDITED image into superpixels
- for each superpixel, compare mean color in edited vs base
- if difference > threshold, mark that superpixel as "changed"
- output a binary mask + overlay

Requires: scikit-image, OpenCV
    pip install scikit-image opencv-python
"""

import cv2
import numpy as np
import sys
import argparse
from pathlib import Path

from skimage.segmentation import slic
from skimage.util import img_as_float


def make_overlay(base_img, mask, color=(0, 0, 255), alpha=0.6):
    overlay = base_img.copy()
    colored = np.zeros_like(base_img)
    colored[:] = color
    mask_3 = cv2.cvtColor(mask, cv2.COLOR_GRAY2BGR)
    overlay = np.where(mask_3 > 0,
                       cv2.addWeighted(colored, alpha, overlay, 1 - alpha, 0),
                       overlay)
    return overlay


def main():
    parser = argparse.ArgumentParser(
        description="Extract jacket mask using superpixel/segmentation comparison.")
    parser.add_argument("base_img", help="Path to original/base image (no jacket)")
    parser.add_argument("edited_img", help="Path to edited image (with jacket)")
    parser.add_argument("--out-mask", default="jacket_mask_superpixels.png",
                        help="Output path for binary mask")
    parser.add_argument("--out-overlay", default="jacket_overlay_superpixels.png",
                        help="Output path for overlay preview")
    args = parser.parse_args()

    # =========================
    # 🔧 KNOBS TO TUNE
    # =========================
    # Superpixel params
    N_SEGMENTS = 600         # try 300–600 depending on image size/detail
    COMPACTNESS = 6.0       # higher = more square, lower = more color-based
    SIGMA = 1.0              # pre-smoothing for SLIC

    # How different a segment must be to count as "changed"
    # (this is on 0–255 scale because we compare in LAB)
    SEG_DIFF_THRESH = 10.0   # try 8–15

    # Skip tiny superpixels (sometimes SLIC makes tiny ones at edges)
    MIN_REGION_PIXELS = 30

    # Morph cleanup
    MORPH_CLOSE_K = 5

    # =========================
    # LOAD
    # =========================
    base_path = Path(args.base_img)
    edited_path = Path(args.edited_img)
    if not base_path.exists():
        print(f"[ERR] Base image not found: {base_path}")
        sys.exit(1)
    if not edited_path.exists():
        print(f"[ERR] Edited image not found: {edited_path}")
        sys.exit(1)

    base_bgr = cv2.imread(str(base_path), cv2.IMREAD_COLOR)
    edited_bgr = cv2.imread(str(edited_path), cv2.IMREAD_COLOR)

    if base_bgr is None or edited_bgr is None:
        print("[ERR] Failed to read one or both images.")
        sys.exit(1)

    # match size (resize base to edited)
    if base_bgr.shape[:2] != edited_bgr.shape[:2]:
        base_bgr = cv2.resize(
            base_bgr,
            (edited_bgr.shape[1], edited_bgr.shape[0]),
            interpolation=cv2.INTER_LANCZOS4
        )

    h, w = edited_bgr.shape[:2]

    # =========================
    # CONVERT TO LAB FOR DIFF
    # =========================
    base_lab = cv2.cvtColor(base_bgr, cv2.COLOR_BGR2LAB)
    edited_lab = cv2.cvtColor(edited_bgr, cv2.COLOR_BGR2LAB)

    # =========================
    # SUPERPIXEL SEGMENTATION (on edited image)
    # skimage works on float images in RGB
    # =========================
    edited_rgb = cv2.cvtColor(edited_bgr, cv2.COLOR_BGR2RGB)
    edited_float = img_as_float(edited_rgb)

    segments = slic(
        edited_float,
        n_segments=N_SEGMENTS,
        compactness=COMPACTNESS,
        sigma=SIGMA,
        start_label=0
    )
    # segments is (H, W) with integer labels 0..K-1
    num_segments = segments.max() + 1
    # print(f"[INFO] Got {num_segments} segments")

    # =========================
    # PER-SEGMENT COMPARISON
    # =========================
    mask = np.zeros((h, w), dtype=np.uint8)

    # precompute flat views
    base_lab_flat = base_lab.reshape(-1, 3)
    edited_lab_flat = edited_lab.reshape(-1, 3)
    seg_flat = segments.reshape(-1)

    for seg_id in range(num_segments):
        idx = np.where(seg_flat == seg_id)[0]
        if idx.size < MIN_REGION_PIXELS:
            continue

        # mean LAB for this segment in base and edited
        base_mean = base_lab_flat[idx].mean(axis=0)
        edited_mean = edited_lab_flat[idx].mean(axis=0)

        # distance in LAB space (just L1 or L2 — we'll use L1 to keep it simple)
        diff = np.abs(base_mean - edited_mean).mean()

        if diff >= SEG_DIFF_THRESH:
            # mark whole segment
            mask.reshape(-1)[idx] = 255

    # =========================
    # MORPHOLOGICAL CLEANUP
    # =========================
    if MORPH_CLOSE_K > 0:
        k_close = cv2.getStructuringElement(
            cv2.MORPH_ELLIPSE, (MORPH_CLOSE_K, MORPH_CLOSE_K))
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, k_close)

    # =========================
    # SAVE
    # =========================
    cv2.imwrite(args.out_mask, mask)
    print(f"[OK] Saved mask to {args.out_mask}")

    overlay = make_overlay(edited_bgr, mask,
                           color=(0, 128, 255),
                           alpha=0.55)
    cv2.imwrite(args.out_overlay, overlay)
    print(f"[OK] Saved overlay to {args.out_overlay}")


if __name__ == "__main__":
    main()
