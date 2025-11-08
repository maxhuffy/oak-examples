#!/usr/bin/env python3
import cv2
import numpy as np
import os
import sys


def load_imgs(orig_path, edit_path, mask_path):
    orig = cv2.imread(orig_path)
    edit = cv2.imread(edit_path)
    mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)

    if orig is None or edit is None or mask is None:
        raise RuntimeError("Could not read one or more input images")

    # make sure edit & mask match orig size
    h, w = orig.shape[:2]
    if edit.shape[:2] != (h, w):
        edit = cv2.resize(edit, (w, h), interpolation=cv2.INTER_LINEAR)
    if mask.shape[:2] != (h, w):
        mask = cv2.resize(mask, (w, h), interpolation=cv2.INTER_NEAREST)

    return orig, edit, mask


def make_diff_map(orig, edit, blur_ksize=5, diff_thresh=15):
    """Return a binary map of where the two images differ."""
    if blur_ksize > 1:
        orig_b = cv2.GaussianBlur(orig, (blur_ksize, blur_ksize), 0)
        edit_b = cv2.GaussianBlur(edit, (blur_ksize, blur_ksize), 0)
    else:
        orig_b = orig
        edit_b = edit

    diff = cv2.absdiff(orig_b, edit_b)
    gray = cv2.cvtColor(diff, cv2.COLOR_BGR2GRAY)
    _, diff_bin = cv2.threshold(gray, diff_thresh, 255, cv2.THRESH_BINARY)
    return diff_bin


def score_mask(mask, diff_map, over_penalty=0.4):
    """
    Score how well mask fits diff_map.
    We want high overlap with diff_map, but not tons of extra.
    score = TP - over_penalty * FP, normalized.
    """
    # make sure binary
    mask_bin = (mask > 0).astype(np.uint8)
    diff_bin = (diff_map > 0).astype(np.uint8)

    tp = np.sum((mask_bin == 1) & (diff_bin == 1))  # correctly covered change
    fp = np.sum((mask_bin == 1) & (diff_bin == 0))  # mask spills outside change
    fn = np.sum((mask_bin == 0) & (diff_bin == 1))  # missed change

    # basic weighted score
    score = tp - over_penalty * fp - 0.1 * fn
    return score, tp, fp, fn


def refine_mask_iterative(orig, edit, start_mask,
                          max_iters=30,
                          kernel_sizes=(3, 5, 7),
                          over_penalty=0.4,
                          save_steps_dir=None):
    """
    Iteratively try to improve the mask using morphological ops,
    guided by a diff map computed from orig+edit.
    """
    diff_map = make_diff_map(orig, edit, blur_ksize=5, diff_thresh=15)

    current = start_mask.copy()
    best_score, _, _, _ = score_mask(current, diff_map, over_penalty=over_penalty)

    if save_steps_dir is not None:
        os.makedirs(save_steps_dir, exist_ok=True)
        cv2.imwrite(os.path.join(save_steps_dir, f"step_00_start.png"), current)

    for it in range(1, max_iters + 1):
        improved = False
        best_candidate = current
        best_candidate_score = best_score

        # try several kernels, several ops
        for k in kernel_sizes:
            kernel = np.ones((k, k), np.uint8)

            # 1) dilate
            cand_dil = cv2.dilate(current, kernel, iterations=1)
            s_dil, _, _, _ = score_mask(cand_dil, diff_map, over_penalty=over_penalty)
            if s_dil > best_candidate_score:
                best_candidate_score = s_dil
                best_candidate = cand_dil
                improved = True

            # 2) erode
            cand_er = cv2.erode(current, kernel, iterations=1)
            s_er, _, _, _ = score_mask(cand_er, diff_map, over_penalty=over_penalty)
            if s_er > best_candidate_score:
                best_candidate_score = s_er
                best_candidate = cand_er
                improved = True

            # 3) close (dilate then erode) - good for filling jacket interior
            cand_close = cv2.morphologyEx(current, cv2.MORPH_CLOSE, kernel)
            s_close, _, _, _ = score_mask(cand_close, diff_map, over_penalty=over_penalty)
            if s_close > best_candidate_score:
                best_candidate_score = s_close
                best_candidate = cand_close
                improved = True

            # 4) open (erode then dilate) - good for removing specks
            cand_open = cv2.morphologyEx(current, cv2.MORPH_OPEN, kernel)
            s_open, _, _, _ = score_mask(cand_open, diff_map, over_penalty=over_penalty)
            if s_open > best_candidate_score:
                best_candidate_score = s_open
                best_candidate = cand_open
                improved = True

        current = best_candidate
        best_score = best_candidate_score

        if save_steps_dir is not None:
            cv2.imwrite(os.path.join(save_steps_dir, f"step_{it:02d}.png"), current)

        if not improved:
            # no op made it better → we're done
            break

    return current


def main(orig_path, edit_path, mask_path, out_path="refined_mask.png", steps_dir=None):
    orig, edit, start_mask = load_imgs(orig_path, edit_path, mask_path)
    refined = refine_mask_iterative(
        orig,
        edit,
        start_mask,
        max_iters=300,
        kernel_sizes=(3, 5, 7),
        over_penalty=0.07,
        save_steps_dir=steps_dir
    )
    cv2.imwrite(out_path, refined)
    print(f"[ok] wrote refined mask to {out_path}")
    if steps_dir:
        print(f"[ok] intermediate steps in {steps_dir}/")


if __name__ == "__main__":
    if len(sys.argv) < 4:
        print("usage: python refine_mask_iterative.py original.png edited.png start_mask.png [out_mask.png] [steps_dir]")
        sys.exit(1)

    orig_path = sys.argv[1]
    edit_path = sys.argv[2]
    mask_path = sys.argv[3]

    out_path = sys.argv[4] if len(sys.argv) > 4 else "refined_mask.png"
    steps_dir = sys.argv[5] if len(sys.argv) > 5 else None

    main(orig_path, edit_path, mask_path, out_path, steps_dir)
