#!/usr/bin/env python3
"""
extract_jacket_mask.py

Given:
  1) an original / base person image (no jacket)
  2) an edited image (with AI-added jacket)
…produce a mask of "what changed", i.e. the jacket.

Assumptions:
- images are already background-removed and roughly aligned
- sizes are the same (if not, we resize base to edited)
"""

import cv2
import numpy as np
import sys
import argparse
from pathlib import Path


def bilateral_keep_edges(img, d=9, sigma_color=75, sigma_space=75):
    return cv2.bilateralFilter(img, d, sigma_color, sigma_space)


def color_diff_lab(img1, img2):
    """
    Return per-pixel color difference in LAB space (focus on a/b channels).
    This is more robust to lighting than RGB diff.
    """
    lab1 = cv2.cvtColor(img1, cv2.COLOR_BGR2LAB)
    lab2 = cv2.cvtColor(img2, cv2.COLOR_BGR2LAB)

    # split
    L1, a1, b1 = cv2.split(lab1)
    L2, a2, b2 = cv2.split(lab2)

    # focus on chroma difference
    da = cv2.absdiff(a1, a2)
    db = cv2.absdiff(b1, b2)

    # combine — you can tune this if needed
    diff = cv2.addWeighted(da, 0.5, db, 0.5, 0)
    return diff


def edge_diff(img1, img2, canny1=50, canny2=150):
    gray1 = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)
    gray2 = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY)
    e1 = cv2.Canny(gray1, canny1, canny2)
    e2 = cv2.Canny(gray2, canny1, canny2)
    ed = cv2.absdiff(e1, e2)
    return ed


def make_overlay(base_img, mask, color=(0, 0, 255), alpha=0.6):
    """Overlay the mask onto the base image for quick visual debugging."""
    overlay = base_img.copy()
    colored = np.zeros_like(base_img)
    colored[:] = color
    mask_3 = cv2.cvtColor(mask, cv2.COLOR_GRAY2BGR)
    overlay = np.where(mask_3 > 0, cv2.addWeighted(colored, alpha, overlay, 1 - alpha, 0), overlay)
    return overlay


def main():
    parser = argparse.ArgumentParser(description="Extract the 'added jacket' mask from two person images.")
    parser.add_argument("base_img", help="Path to original/base image (no jacket)")
    parser.add_argument("edited_img", help="Path to edited image (with jacket)")
    parser.add_argument("--out-mask", default="jacket_mask.png", help="Output path for binary mask")
    parser.add_argument("--out-overlay", default="jacket_overlay.png", help="Output path for overlay preview")
    args = parser.parse_args()

    # =========================
    # 🔧 KNOBS TO TUNE EASILY
    # =========================
    # 1. Bilateral filter strength (None to skip)
    USE_BILATERAL = True
    BILATERAL_D = 9
    BILATERAL_SIGMA_COLOR = 75
    BILATERAL_SIGMA_SPACE = 75

    # 2. Color-diff threshold (lower = more sensitive)
    COLOR_DIFF_THRESH = 12  # try 10–20

    # 3. Edge gating — helps tighten jacket outline
    USE_EDGE_GATE = True
    EDGE_GATE_MIN = 15  # pixels of edge diff to consider
    EDGE_GATE_DILATE = 3  # how much to grow the edge gate

    # 4. Morphological cleanup
    MORPH_OPEN_K = 3
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

    base = cv2.imread(str(base_path), cv2.IMREAD_COLOR)
    edited = cv2.imread(str(edited_path), cv2.IMREAD_COLOR)

    if base is None or edited is None:
        print("[ERR] Failed to read one or both images.")
        sys.exit(1)

    # match size
    if base.shape[:2] != edited.shape[:2]:
        # resize base to edited
        base = cv2.resize(base, (edited.shape[1], edited.shape[0]), interpolation=cv2.INTER_LANCZOS4)

    # =========================
    # PREPROCESS (edge-preserving)
    # =========================
    if USE_BILATERAL:
        base_sm = bilateral_keep_edges(base, BILATERAL_D, BILATERAL_SIGMA_COLOR, BILATERAL_SIGMA_SPACE)
        edited_sm = bilateral_keep_edges(edited, BILATERAL_D, BILATERAL_SIGMA_COLOR, BILATERAL_SIGMA_SPACE)
    else:
        base_sm = base
        edited_sm = edited

    # =========================
    # COLOR DIFF MAP
    # =========================
    col_diff = color_diff_lab(base_sm, edited_sm)
    # normalize to 0-255 just in case
    col_diff = cv2.normalize(col_diff, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)

    # threshold to get candidate jacket area
    _, col_mask = cv2.threshold(col_diff, COLOR_DIFF_THRESH, 255, cv2.THRESH_BINARY)

    # =========================
    # EDGE GATE (optional)
    # =========================
    if USE_EDGE_GATE:
        e_diff = edge_diff(base_sm, edited_sm)
        # threshold edge diff
        _, e_mask = cv2.threshold(e_diff, EDGE_GATE_MIN, 255, cv2.THRESH_BINARY)
        if EDGE_GATE_DILATE > 0:
            k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (EDGE_GATE_DILATE, EDGE_GATE_DILATE))
            e_mask = cv2.dilate(e_mask, k, iterations=1)

        # combine: only keep color-changed pixels that are also near an edge change
        combined = cv2.bitwise_and(col_mask, e_mask)
        mask = combined
    else:
        mask = col_mask

    # =========================
    # MORPH CLEANUP
    # =========================
    if MORPH_OPEN_K > 0:
        k_open = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (MORPH_OPEN_K, MORPH_OPEN_K))
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, k_open)

    if MORPH_CLOSE_K > 0:
        k_close = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (MORPH_CLOSE_K, MORPH_CLOSE_K))
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, k_close)

    # =========================
    # SAVE
    # =========================
    cv2.imwrite(args.out_mask, mask)
    print(f"[OK] Saved mask to {args.out_mask}")

    overlay = make_overlay(edited, mask, color=(128, 0, 255), alpha=0.55)
    cv2.imwrite(args.out_overlay, overlay)
    print(f"[OK] Saved overlay to {args.out_overlay}")


if __name__ == "__main__":
    main()
