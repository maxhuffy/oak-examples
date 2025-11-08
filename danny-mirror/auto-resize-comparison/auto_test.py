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


# -------------------------
# Utility: load images
# -------------------------
def load_images(orig_path, edit_path):
    orig = cv2.imread(orig_path)
    edit = cv2.imread(edit_path)

    if orig is None or edit is None:
        raise RuntimeError("Could not read one of the images")

    return orig, edit


# -------------------------
# 1) Force resize (current method)
# -------------------------
def method_force_resize(orig, edit):
    h, w = orig.shape[:2]
    aligned = cv2.resize(edit, (w, h), interpolation=cv2.INTER_LINEAR)
    return aligned


# -------------------------
# 2) Letterbox / pad to fit
# -------------------------
def method_letterbox_pad(orig, edit, pad_color=(0, 0, 0)):
    target_h, target_w = orig.shape[:2]
    h, w = edit.shape[:2]

    scale = min(target_w / w, target_h / h)
    new_w = int(w * scale)
    new_h = int(h * scale)

    resized = cv2.resize(edit, (new_w, new_h), interpolation=cv2.INTER_LINEAR)

    top = (target_h - new_h) // 2
    bottom = target_h - new_h - top
    left = (target_w - new_w) // 2
    right = target_w - new_w - left

    padded = cv2.copyMakeBorder(resized, top, bottom, left, right,
                                borderType=cv2.BORDER_CONSTANT, value=pad_color)
    return padded


# -------------------------
# 3) Scale to cover, then center-crop
# (no padding, no stretching)
# -------------------------
def method_scale_cover_crop(orig, edit):
    target_h, target_w = orig.shape[:2]
    h, w = edit.shape[:2]

    scale = max(target_w / w, target_h / h)
    new_w = int(w * scale)
    new_h = int(h * scale)

    resized = cv2.resize(edit, (new_w, new_h), interpolation=cv2.INTER_LINEAR)

    # center crop
    start_x = (new_w - target_w) // 2
    start_y = (new_h - target_h) // 2
    cropped = resized[start_y:start_y+target_h, start_x:start_x+target_w]
    return cropped


# -------------------------
# ECC alignment helpers
# -------------------------
def _ecc_align(orig_gray, edit_gray, motion_model, mask=None):
    # initial warp
    if motion_model in (cv2.MOTION_TRANSLATION, cv2.MOTION_EUCLIDEAN, cv2.MOTION_AFFINE):
        warp_matrix = np.eye(2, 3, dtype=np.float32)
        warp_mode = motion_model
    else:
        # homography
        warp_matrix = np.eye(3, 3, dtype=np.float32)
        warp_mode = motion_model

    criteria = (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 100, 1e-5)

    try:
        cc, warp_matrix = cv2.findTransformECC(
            templateImage=orig_gray,
            inputImage=edit_gray,
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
# 4) ECC affine on whole image
# -------------------------
def method_ecc_affine(orig, edit):
    h, w = orig.shape[:2]
    # start from something already close in size
    edit_rs = cv2.resize(edit, (w, h), interpolation=cv2.INTER_LINEAR)

    orig_gray = cv2.cvtColor(orig, cv2.COLOR_BGR2GRAY)
    edit_gray = cv2.cvtColor(edit_rs, cv2.COLOR_BGR2GRAY)

    warp_matrix = _ecc_align(orig_gray, edit_gray, cv2.MOTION_AFFINE, mask=None)
    if warp_matrix is None:
        return None

    aligned = cv2.warpAffine(
        edit_rs,
        warp_matrix,
        (w, h),
        flags=cv2.INTER_LINEAR + cv2.WARP_INVERSE_MAP
    )
    return aligned


# -------------------------
# 5) ECC affine but only using border mask
# -------------------------
def method_ecc_affine_border(orig, edit, border=50):
    h, w = orig.shape[:2]
    edit_rs = cv2.resize(edit, (w, h), interpolation=cv2.INTER_LINEAR)

    orig_gray = cv2.cvtColor(orig, cv2.COLOR_BGR2GRAY)
    edit_gray = cv2.cvtColor(edit_rs, cv2.COLOR_BGR2GRAY)

    # build border mask
    mask = np.zeros((h, w), np.uint8)
    mask[:border, :] = 255
    mask[-border:, :] = 255
    mask[:, :border] = 255
    mask[:, -border:] = 255

    warp_matrix = _ecc_align(orig_gray, edit_gray, cv2.MOTION_AFFINE, mask=mask)
    if warp_matrix is None:
        return None

    aligned = cv2.warpAffine(
        edit_rs,
        warp_matrix,
        (w, h),
        flags=cv2.INTER_LINEAR + cv2.WARP_INVERSE_MAP
    )
    return aligned


# -------------------------
# 6) Feature-based homography (ORB)
# -------------------------
def method_feature_homography(orig, edit):
    h, w = orig.shape[:2]
    edit_rs = cv2.resize(edit, (w, h), interpolation=cv2.INTER_LINEAR)

    orig_gray = cv2.cvtColor(orig, cv2.COLOR_BGR2GRAY)
    edit_gray = cv2.cvtColor(edit_rs, cv2.COLOR_BGR2GRAY)

    orb = cv2.ORB_create(3000)
    kp1, des1 = orb.detectAndCompute(orig_gray, None)
    kp2, des2 = orb.detectAndCompute(edit_gray, None)

    if des1 is None or des2 is None:
        print("[homography] no keypoints detected")
        return None

    bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
    matches = bf.match(des1, des2)
    if len(matches) < 10:
        print("[homography] not enough matches")
        return None

    matches = sorted(matches, key=lambda x: x.distance)[:200]

    pts1 = np.float32([kp1[m.queryIdx].pt for m in matches]).reshape(-1, 1, 2)
    pts2 = np.float32([kp2[m.trainIdx].pt for m in matches]).reshape(-1, 1, 2)

    H, mask = cv2.findHomography(pts2, pts1, cv2.RANSAC, 5.0)
    if H is None:
        print("[homography] findHomography failed")
        return None

    aligned = cv2.warpPerspective(edit_rs, H, (w, h))
    return aligned


# -------------------------
# Scoring: compare border-only area
# lower MSE = better
# higher SSIM = better
# -------------------------
def make_border_mask(shape, border=50):
    h, w = shape[:2]
    mask = np.zeros((h, w), np.uint8)
    mask[:border, :] = 1
    mask[-border:, :] = 1
    mask[:, :border] = 1
    mask[:, -border:] = 1
    return mask


def score_alignment(orig, aligned, border=50):
    """
    Return dict with mse_border, ssim_border (if available)
    """
    mask = make_border_mask(orig.shape, border)
    # convert to gray for scoring
    orig_g = cv2.cvtColor(orig, cv2.COLOR_BGR2GRAY)
    aligned_g = cv2.cvtColor(aligned, cv2.COLOR_BGR2GRAY)

    # MSE on border
    diff = (orig_g.astype(np.float32) - aligned_g.astype(np.float32)) ** 2
    mse_border = diff[mask == 1].mean()

    res = {"mse_border": mse_border}

    if HAS_SKIMAGE:
        # we can feed only border, but easier to zero out non-border
        # or we can just compute global ssim
        ssim_val = ssim(orig_g, aligned_g, data_range=255)
        res["ssim_full"] = ssim_val

    return res


def main(orig_path, edit_path, out_dir="outputs"):
    os.makedirs(out_dir, exist_ok=True)
    orig, edit = load_images(orig_path, edit_path)

    methods = [
        ("force_resize", method_force_resize),
        ("letterbox_pad", method_letterbox_pad),
        ("scale_cover_crop", method_scale_cover_crop),
        ("ecc_affine", method_ecc_affine),
        ("ecc_affine_border", method_ecc_affine_border),
        ("feature_homography", method_feature_homography),
    ]

    results = []

    for name, func in methods:
        print(f"--- running {name} ---")
        aligned = func(orig, edit)
        if aligned is None:
            print(f"[{name}] failed or returned None")
            continue

        # save aligned image
        out_path = os.path.join(out_dir, f"aligned_{name}.png")
        cv2.imwrite(out_path, aligned)
        print(f"[{name}] saved to {out_path}")

        # score it
        scores = score_alignment(orig, aligned, border=50)
        scores["name"] = name
        results.append(scores)

    # sort by mse_border ascending
    results_sorted = sorted(results, key=lambda x: x["mse_border"])
    print("\n===== SCORE LEADERBOARD (lower MSE better) =====")
    for r in results_sorted:
        line = f"{r['name']}: MSE(border)={r['mse_border']:.2f}"
        if "ssim_full" in r:
            line += f", SSIM(full)={r['ssim_full']:.4f}"
        print(line)


if __name__ == "__main__":
    if len(sys.argv) < 3:
        print("usage: python align_compare.py original.png edited.png [out_dir]")
        sys.exit(1)

    orig_path = sys.argv[1]
    edit_path = sys.argv[2]
    out_dir = sys.argv[3] if len(sys.argv) > 3 else "outputs"
    main(orig_path, edit_path, out_dir)
