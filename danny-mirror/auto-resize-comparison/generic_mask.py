#!/usr/bin/env python3
import cv2
import numpy as np
import os
import sys


# -------------------------
# load
# -------------------------
def load_images(orig_path, edit_path):
    orig = cv2.imread(orig_path)
    edit = cv2.imread(edit_path)
    if orig is None or edit is None:
        raise RuntimeError("Could not read one of the images")
    return orig, edit


# -------------------------
# 1) scale-cover-crop (downscale original → edited)
# -------------------------
def scale_cover_crop_down(orig, edit):
    target_h, target_w = edit.shape[:2]
    h, w = orig.shape[:2]

    scale = max(target_w / w, target_h / h)
    new_w = int(w * scale)
    new_h = int(h * scale)

    resized = cv2.resize(orig, (new_w, new_h), interpolation=cv2.INTER_LINEAR)

    start_x = (new_w - target_w) // 2
    start_y = (new_h - target_h) // 2
    cropped = resized[start_y:start_y+target_h, start_x:start_x+target_w]
    return cropped


# -------------------------
# ECC helpers
# -------------------------
def _ecc_align(template_gray, input_gray, motion_model, mask=None):
    if motion_model in (cv2.MOTION_TRANSLATION, cv2.MOTION_EUCLIDEAN, cv2.MOTION_AFFINE):
        warp_matrix = np.eye(2, 3, dtype=np.float32)
        warp_mode = motion_model
    else:
        warp_matrix = np.eye(3, 3, dtype=np.float32)
        warp_mode = motion_model

    criteria = (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 100, 1e-5)
    try:
        cc, warp_matrix = cv2.findTransformECC(
            templateImage=template_gray,
            inputImage=input_gray,
            warpMatrix=warp_matrix,
            motionType=warp_mode,
            criteria=criteria,
            inputMask=mask
        )
    except cv2.error as e:
        print(f"[ecc] alignment failed: {e}")
        return None
    return warp_matrix


# -------------------------
# 2) ecc-affine-border (downscale original → edited, then align using border)
# -------------------------
def ecc_affine_border_down(orig, edit, border=50):
    h, w = edit.shape[:2]
    orig_rs = cv2.resize(orig, (w, h), interpolation=cv2.INTER_LINEAR)

    edit_gray = cv2.cvtColor(edit, cv2.COLOR_BGR2GRAY)
    orig_gray = cv2.cvtColor(orig_rs, cv2.COLOR_BGR2GRAY)

    # mask: only border of EDITED
    mask = np.zeros((h, w), np.uint8)
    mask[:border, :] = 255
    mask[-border:, :] = 255
    mask[:, :border] = 255
    mask[:, -border:] = 255

    warp_matrix = _ecc_align(edit_gray, orig_gray, cv2.MOTION_AFFINE, mask=mask)
    if warp_matrix is None:
        # fall back to the resized version at least
        return orig_rs

    aligned = cv2.warpAffine(
        orig_rs,
        warp_matrix,
        (w, h),
        flags=cv2.INTER_LINEAR + cv2.WARP_INVERSE_MAP
    )
    return aligned


# -------------------------
# diff → mask generator
# -------------------------
def make_mask_from_pair(aligned_orig, edited,
                        blur_ksize=5,
                        thresh_val=25,
                        morph_ksize=5):
    """
    aligned_orig and edited must be same size.
    Returns a single-channel mask (0/255).
    """
    # small blur to reduce noise
    ao_b = cv2.GaussianBlur(aligned_orig, (blur_ksize, blur_ksize), 0)
    ed_b = cv2.GaussianBlur(edited, (blur_ksize, blur_ksize), 0)

    diff = cv2.absdiff(ao_b, ed_b)
    gray = cv2.cvtColor(diff, cv2.COLOR_BGR2GRAY)

    _, mask = cv2.threshold(gray, thresh_val, 255, cv2.THRESH_BINARY)

    # clean up mask
    kernel = np.ones((morph_ksize, morph_ksize), np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)

    return mask


def main(orig_path, edit_path, out_dir="outputs_masks"):
    os.makedirs(out_dir, exist_ok=True)
    orig, edit = load_images(orig_path, edit_path)

    # ---- method 1: scale-cover-crop down ----
    aligned_scc = scale_cover_crop_down(orig, edit)
    mask_scc = make_mask_from_pair(aligned_scc, edit)
    cv2.imwrite(os.path.join(out_dir, "aligned_scale_cover_crop_down.png"), aligned_scc)
    cv2.imwrite(os.path.join(out_dir, "mask_scale_cover_crop_down.png"), mask_scc)
    print("[ok] wrote mask_scale_cover_crop_down.png")

    # ---- method 2: ecc-affine-border down ----
    aligned_ecc = ecc_affine_border_down(orig, edit, border=50)
    mask_ecc = make_mask_from_pair(aligned_ecc, edit)
    cv2.imwrite(os.path.join(out_dir, "aligned_ecc_affine_border_down.png"), aligned_ecc)
    cv2.imwrite(os.path.join(out_dir, "mask_ecc_affine_border_down.png"), mask_ecc)
    print("[ok] wrote mask_ecc_affine_border_down.png")

    print(f"Done. Check directory: {out_dir}")


if __name__ == "__main__":
    if len(sys.argv) < 3:
        print("usage: python make_masks_downscale.py original.png edited.png [out_dir]")
        sys.exit(1)

    orig_path = sys.argv[1]
    edit_path = sys.argv[2]
    out_dir = sys.argv[3] if len(sys.argv) > 3 else "outputs_masks"
    main(orig_path, edit_path, out_dir)
