#!/usr/bin/env python3
import cv2
import numpy as np
import os
import sys


# ------------ basic loaders ------------
def load_images(orig_path, edit_path):
    orig = cv2.imread(orig_path)
    edit = cv2.imread(edit_path)
    if orig is None or edit is None:
        raise RuntimeError("Could not read one of the images")
    return orig, edit


# ------------ downscale: scale-to-cover, then crop ------------
def scale_cover_crop_down(orig, edit):
    target_h, target_w = edit.shape[:2]
    h, w = orig.shape[:2]

    scale = max(target_w / w, target_h / h)
    new_w = int(w * scale)
    new_h = int(h * scale)

    resized = cv2.resize(orig, (new_w, new_h), interpolation=cv2.INTER_LINEAR)
    start_x = (new_w - target_w) // 2
    start_y = (new_h - target_h) // 2
    cropped = resized[start_y:start_y + target_h, start_x:start_x + target_w]
    return cropped


# ------------ ECC helpers ------------
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


def ecc_affine_border_down(orig, edit, border=50):
    """resize orig -> edit size, then ECC using border as the reliable region"""
    h, w = edit.shape[:2]
    orig_rs = cv2.resize(orig, (w, h), interpolation=cv2.INTER_LINEAR)

    edit_gray = cv2.cvtColor(edit, cv2.COLOR_BGR2GRAY)
    orig_gray = cv2.cvtColor(orig_rs, cv2.COLOR_BGR2GRAY)

    mask = np.zeros((h, w), np.uint8)
    mask[:border, :] = 255
    mask[-border:, :] = 255
    mask[:, :border] = 255
    mask[:, -border:] = 255

    warp_matrix = _ecc_align(edit_gray, orig_gray, cv2.MOTION_AFFINE, mask=mask)
    if warp_matrix is None:
        # fall back
        return orig_rs

    aligned = cv2.warpAffine(
        orig_rs,
        warp_matrix,
        (w, h),
        flags=cv2.INTER_LINEAR + cv2.WARP_INVERSE_MAP
    )
    return aligned


# ------------ diff -> mask ------------
def make_mask_from_pair(
    aligned_orig,
    edited,
    blur_ksize=5,
    thresh_val=25,
    morph_ksize=5
):
    # blur
    if blur_ksize > 1:
        ao_b = cv2.GaussianBlur(aligned_orig, (blur_ksize, blur_ksize), 0)
        ed_b = cv2.GaussianBlur(edited, (blur_ksize, blur_ksize), 0)
    else:
        ao_b = aligned_orig
        ed_b = edited

    diff = cv2.absdiff(ao_b, ed_b)
    gray = cv2.cvtColor(diff, cv2.COLOR_BGR2GRAY)

    _, mask = cv2.threshold(gray, thresh_val, 255, cv2.THRESH_BINARY)

    if morph_ksize > 1:
        kernel = np.ones((morph_ksize, morph_ksize), np.uint8)
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)

    return mask


def main(orig_path, edit_path, out_dir="outputs_36"):
    os.makedirs(out_dir, exist_ok=True)

    orig, edit = load_images(orig_path, edit_path)

    # ---- precompute the two aligned bases ----
    aligned_scale = scale_cover_crop_down(orig, edit)

    # for ECC we will vary border per mask, so we won't just do one
    # but let's keep a resized version around
    h, w = edit.shape[:2]
    orig_resized = cv2.resize(orig, (w, h), interpolation=cv2.INTER_LINEAR)

    mask_count = 0

    # =========================================================
    # 1) 16 masks from scale_cover_crop_down
    # thresholds and kernels tuned for your flat background
    # =========================================================
    thresholds = [15, 25, 35, 45]
    blur_ks = [3, 5]
    morph_ks = [3, 5]

    for t in thresholds:
        for b in blur_ks:
            for m in morph_ks:
                mask = make_mask_from_pair(
                    aligned_scale,
                    edit,
                    blur_ksize=b,
                    thresh_val=t,
                    morph_ksize=m
                )
                fname = f"{mask_count:02d}_scale_t{t}_b{b}_m{m}.png"
                cv2.imwrite(os.path.join(out_dir, fname), mask)
                mask_count += 1

    # =========================================================
    # 2) 16 masks from ecc_affine_border_down
    # we'll vary border size a bit + the same diff params
    # =========================================================
    borders = [30, 50]           # 2 values
    thresholds_ecc = [15, 25, 35, 45]  # 4 values
    morph_ks_ecc = [3, 5]        # 2 values
    # 2 * 4 * 2 = 16

    for border in borders:
        aligned_ecc = ecc_affine_border_down(orig, edit, border=border)
        for t in thresholds_ecc:
            for m in morph_ks_ecc:
                # keep blur modest (5) for ECC
                mask = make_mask_from_pair(
                    aligned_ecc,
                    edit,
                    blur_ksize=5,
                    thresh_val=t,
                    morph_ksize=m
                )
                fname = f"{mask_count:02d}_ecc_bord{border}_t{t}_m{m}.png"
                cv2.imwrite(os.path.join(out_dir, fname), mask)
                mask_count += 1

    # =========================================================
    # 3) 4 extra “aggressive” masks (from scale alignment)
    # for cases where jacket edges are ragged
    # =========================================================
    extra_params = [
        (2, 10, 7),
        (3, 20, 7),
        (5, 20, 9),
        (5, 30, 9),
    ]
    for (b, t, m) in extra_params:
        mask = make_mask_from_pair(
            aligned_scale,
            edit,
            blur_ksize=b,
            thresh_val=t,
            morph_ksize=m
        )
        fname = f"{mask_count:02d}_extra_scale_t{t}_b{b}_m{m}.png"
        cv2.imwrite(os.path.join(out_dir, fname), mask)
        mask_count += 1

    print(f"Done. Wrote {mask_count} mask files to {out_dir}/")


if __name__ == "__main__":
    if len(sys.argv) < 3:
        print("usage: python make_masks_batch.py original.png edited.png [out_dir]")
        sys.exit(1)

    orig_path = sys.argv[1]
    edit_path = sys.argv[2]
    out_dir = sys.argv[3] if len(sys.argv) > 3 else "outputs_36"
    main(orig_path, edit_path, out_dir)
