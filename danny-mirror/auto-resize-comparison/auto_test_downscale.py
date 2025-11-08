#!/usr/bin/env python3
import cv2
import numpy as np
import os
import sys

try:
    from skimage.metrics import structural_similarity as ssim
    HAS_SKIMAGE = True
except ImportError:
    HAS_SKIMAGE = False
    print("[warn] skimage not found; SSIM scores will be skipped. `pip install scikit-image` to enable.")


def load_images(orig_path, edit_path):
    orig = cv2.imread(orig_path)
    edit = cv2.imread(edit_path)

    if orig is None or edit is None:
        raise RuntimeError("Could not read one of the images")

    return orig, edit


# ========== METHODS (this time: make ORIG match EDIT) ==========

def method_force_resize_down(orig, edit):
    """Resize the ORIGINAL to match the edited image size (downscale if needed)."""
    h, w = edit.shape[:2]
    aligned = cv2.resize(orig, (w, h), interpolation=cv2.INTER_LINEAR)
    return aligned


def method_letterbox_pad_down(orig, edit, pad_color=(0, 0, 0)):
    """Scale original to fit inside edited, then pad to edited size."""
    target_h, target_w = edit.shape[:2]
    h, w = orig.shape[:2]

    scale = min(target_w / w, target_h / h)
    new_w = int(w * scale)
    new_h = int(h * scale)

    resized = cv2.resize(orig, (new_w, new_h), interpolation=cv2.INTER_LINEAR)

    top = (target_h - new_h) // 2
    bottom = target_h - new_h - top
    left = (target_w - new_w) // 2
    right = target_w - new_w - left

    padded = cv2.copyMakeBorder(resized, top, bottom, left, right,
                                borderType=cv2.BORDER_CONSTANT, value=pad_color)
    return padded


def method_scale_cover_crop_down(orig, edit):
    """Scale original to cover edited, then center-crop to edited size."""
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


# -------- ECC helpers --------
def _ecc_align(template_gray, input_gray, motion_model, mask=None):
    # template_gray: edited (target)
    # input_gray: resized original (we want to warp this to template)
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


def method_ecc_affine_down(orig, edit):
    """Resize orig to edit size, then ECC-align orig→edit."""
    h, w = edit.shape[:2]
    orig_rs = cv2.resize(orig, (w, h), interpolation=cv2.INTER_LINEAR)

    edit_gray = cv2.cvtColor(edit, cv2.COLOR_BGR2GRAY)
    orig_gray = cv2.cvtColor(orig_rs, cv2.COLOR_BGR2GRAY)

    warp_matrix = _ecc_align(edit_gray, orig_gray, cv2.MOTION_AFFINE, mask=None)
    if warp_matrix is None:
        return None

    aligned = cv2.warpAffine(
        orig_rs,
        warp_matrix,
        (w, h),
        flags=cv2.INTER_LINEAR + cv2.WARP_INVERSE_MAP
    )
    return aligned


def method_ecc_affine_border_down(orig, edit, border=50):
    """Same as above but only trust the border of the edited image for alignment."""
    h, w = edit.shape[:2]
    orig_rs = cv2.resize(orig, (w, h), interpolation=cv2.INTER_LINEAR)

    edit_gray = cv2.cvtColor(edit, cv2.COLOR_BGR2GRAY)
    orig_gray = cv2.cvtColor(orig_rs, cv2.COLOR_BGR2GRAY)

    # border mask on the EDIT (template)
    mask = np.zeros((h, w), np.uint8)
    mask[:border, :] = 255
    mask[-border:, :] = 255
    mask[:, :border] = 255
    mask[:, -border:] = 255

    warp_matrix = _ecc_align(edit_gray, orig_gray, cv2.MOTION_AFFINE, mask=mask)
    if warp_matrix is None:
        return None

    aligned = cv2.warpAffine(
        orig_rs,
        warp_matrix,
        (w, h),
        flags=cv2.INTER_LINEAR + cv2.WARP_INVERSE_MAP
    )
    return aligned


def method_feature_homography_down(orig, edit):
    """Feature-based alignment: warp original onto edited."""
    h, w = edit.shape[:2]
    orig_rs = cv2.resize(orig, (w, h), interpolation=cv2.INTER_LINEAR)

    edit_gray = cv2.cvtColor(edit, cv2.COLOR_BGR2GRAY)
    orig_gray = cv2.cvtColor(orig_rs, cv2.COLOR_BGR2GRAY)

    orb = cv2.ORB_create(3000)
    kp1, des1 = orb.detectAndCompute(edit_gray, None)   # template (edited)
    kp2, des2 = orb.detectAndCompute(orig_gray, None)   # input (orig)

    if des1 is None or des2 is None:
        print("[homography] no keypoints detected")
        return None

    bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
    matches = bf.match(des1, des2)
    if len(matches) < 10:
        print("[homography] not enough matches")
        return None

    matches = sorted(matches, key=lambda x: x.distance)[:200]

    pts_edit = np.float32([kp1[m.queryIdx].pt for m in matches]).reshape(-1, 1, 2)
    pts_orig = np.float32([kp2[m.trainIdx].pt for m in matches]).reshape(-1, 1, 2)

    H, mask = cv2.findHomography(pts_orig, pts_edit, cv2.RANSAC, 5.0)
    if H is None:
        print("[homography] findHomography failed")
        return None

    aligned = cv2.warpPerspective(orig_rs, H, (w, h))
    return aligned


# ========== SCORING (edited is the reference now) ==========

def make_border_mask(shape, border=50):
    h, w = shape[:2]
    mask = np.zeros((h, w), np.uint8)
    mask[:border, :] = 1
    mask[-border:, :] = 1
    mask[:, :border] = 1
    mask[:, -border:] = 1
    return mask


def score_alignment(reference_img, aligned_img, border=50):
    """
    reference_img = edited image (target)
    aligned_img   = original mapped to edited
    """
    mask = make_border_mask(reference_img.shape, border)
    ref_g = cv2.cvtColor(reference_img, cv2.COLOR_BGR2GRAY)
    ali_g = cv2.cvtColor(aligned_img, cv2.COLOR_BGR2GRAY)

    diff = (ref_g.astype(np.float32) - ali_g.astype(np.float32)) ** 2
    mse_border = diff[mask == 1].mean()

    res = {"mse_border": mse_border}
    if HAS_SKIMAGE:
        ssim_val = ssim(ref_g, ali_g, data_range=255)
        res["ssim_full"] = ssim_val
    return res


def main(orig_path, edit_path, out_dir="outputs_down"):
    os.makedirs(out_dir, exist_ok=True)
    orig, edit = load_images(orig_path, edit_path)

    methods = [
        ("force_resize_down", method_force_resize_down),
        ("letterbox_pad_down", method_letterbox_pad_down),
        ("scale_cover_crop_down", method_scale_cover_crop_down),
        ("ecc_affine_down", method_ecc_affine_down),
        ("ecc_affine_border_down", method_ecc_affine_border_down),
        ("feature_homography_down", method_feature_homography_down),
    ]

    results = []

    for name, func in methods:
        print(f"--- running {name} ---")
        aligned = func(orig, edit)
        if aligned is None:
            print(f"[{name}] failed or returned None")
            continue

        out_path = os.path.join(out_dir, f"aligned_{name}.png")
        cv2.imwrite(out_path, aligned)
        print(f"[{name}] saved to {out_path}")

        scores = score_alignment(edit, aligned, border=50)
        scores["name"] = name
        results.append(scores)

    results_sorted = sorted(results, key=lambda x: x["mse_border"])
    print("\n===== SCORE LEADERBOARD (lower MSE better) =====")
    for r in results_sorted:
        line = f"{r['name']}: MSE(border)={r['mse_border']:.2f}"
        if "ssim_full" in r:
            line += f", SSIM(full)={r['ssim_full']:.4f}"
        print(line)


if __name__ == "__main__":
    if len(sys.argv) < 3:
        print("usage: python align_compare_downscale.py original.png edited.png [out_dir]")
        sys.exit(1)

    orig_path = sys.argv[1]
    edit_path = sys.argv[2]
    out_dir = sys.argv[3] if len(sys.argv) > 3 else "outputs_down"
    main(orig_path, edit_path, out_dir)
