import os
import cv2
import numpy as np


def load_images_and_seed():
    script_dir = os.path.dirname(os.path.abspath(__file__))
    tryon_clothes_root = os.path.abspath(os.path.join(script_dir, os.pardir, os.pardir, os.pardir))  # .../try-on-clothes
    me_path = os.path.join(tryon_clothes_root, "me.png")
    tryon_path = os.path.join(tryon_clothes_root, "isolate-clothing", "scaler", "output", "tryon_result_scaled.png")
    seed_path = os.path.join(script_dir, "clothes_mask.png")

    me = cv2.imread(me_path, cv2.IMREAD_COLOR)
    if me is None:
        raise FileNotFoundError(f"Could not read: {me_path}")
    tr = cv2.imread(tryon_path, cv2.IMREAD_COLOR)
    if tr is None:
        raise FileNotFoundError(f"Could not read: {tryon_path}")
    seed = cv2.imread(seed_path, cv2.IMREAD_GRAYSCALE)
    if seed is None:
        raise FileNotFoundError(f"Could not read seed mask: {seed_path}")
    if me.shape[:2] != tr.shape[:2]:
        raise ValueError(f"Size mismatch me={me.shape[:2]} tryon={tr.shape[:2]}")
    if seed.shape[:2] != tr.shape[:2]:
        # Allow small mismatches by resizing seed to image size if necessary
        seed = cv2.resize(seed, (tr.shape[1], tr.shape[0]), interpolation=cv2.INTER_NEAREST)
    return me, tr, seed, script_dir


def lab_delta(a_bgr: np.ndarray, b_bgr: np.ndarray) -> np.ndarray:
    a = cv2.cvtColor(a_bgr, cv2.COLOR_BGR2LAB).astype(np.float32)
    b = cv2.cvtColor(b_bgr, cv2.COLOR_BGR2LAB).astype(np.float32)
    d = a - b
    return np.sqrt(np.sum(d * d, axis=2))


def build_grabcut_mask(seed_mask: np.ndarray, delta: np.ndarray,
                       sure_bg_thr: float = 6.0,
                       prob_bg_thr: float = 12.0) -> np.ndarray:
    """
    Build a trimap for GrabCut from a seed mask and Lab delta between try-on and original.
      - sure FG: eroded seed
      - prob FG: seed
      - prob BG: pixels with low delta (< prob_bg_thr) outside seed
      - sure BG: pixels with very low delta (< sure_bg_thr) outside seed + a small border band
    Returns a mask with values in {GC_BGD(0), GC_FGD(1), GC_PR_BGD(2), GC_PR_FGD(3)}
    """
    h, w = seed_mask.shape[:2]
    seed_bin = (seed_mask > 0).astype(np.uint8) * 255

    # sure FG = eroded seed (shrink to high-confidence interior)
    k_erode = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7))
    sure_fg = cv2.erode(seed_bin, k_erode, iterations=1)

    # probable FG = original seed
    prob_fg = seed_bin.copy()

    # Background from delta (outside seed)
    non_seed = (seed_bin == 0)
    sure_bg = ((delta < sure_bg_thr) & non_seed)
    prob_bg = ((delta < prob_bg_thr) & non_seed)

    # Add a thin border band to sure BG to prevent edge leakage
    band = 5
    border = np.zeros_like(sure_bg, dtype=bool)
    border[:band, :] = True
    border[-band:, :] = True
    border[:, :band] = True
    border[:, -band:] = True
    sure_bg = sure_bg | border

    # Compose GrabCut mask
    GC_BGD, GC_FGD, GC_PR_BGD, GC_PR_FGD = 0, 1, 2, 3
    gc = np.full((h, w), GC_PR_BGD, dtype=np.uint8)  # default probable background
    gc[prob_bg] = GC_PR_BGD
    gc[sure_bg] = GC_BGD
    gc[prob_fg > 0] = GC_PR_FGD
    gc[sure_fg > 0] = GC_FGD
    return gc


def refine_with_grabcut(tr: np.ndarray, gc_mask: np.ndarray, iters: int = 5) -> np.ndarray:
    bgdModel = np.zeros((1, 65), np.float64)
    fgdModel = np.zeros((1, 65), np.float64)
    mask = gc_mask.copy()
    cv2.grabCut(tr, mask, None, bgdModel, fgdModel, iters, cv2.GC_INIT_WITH_MASK)
    # Foreground if labeled FGD or PR_FGD
    result = np.where((mask == cv2.GC_FGD) | (mask == cv2.GC_PR_FGD), 255, 0).astype(np.uint8)
    return result


def keep_connected_to_seed(mask: np.ndarray, seed: np.ndarray) -> np.ndarray:
    # Only keep components that overlap the seed (prevents stray regions)
    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(mask, connectivity=8)
    if num_labels <= 1:
        return mask
    seed_bin = (seed > 0).astype(np.uint8)
    kept = np.zeros_like(mask)
    for i in range(1, num_labels):
        comp = (labels == i)
        if (seed_bin[comp] > 0).any():
            kept[comp] = 255
    # Fallback: if nothing kept, keep the largest
    if cv2.countNonZero(kept) == 0:
        areas = stats[1:, cv2.CC_STAT_AREA]
        if areas.size > 0:
            largest = 1 + int(np.argmax(areas))
            kept[labels == largest] = 255
    return kept


def main():
    me, tr, seed, outdir = load_images_and_seed()

    # Compute difference map
    d = lab_delta(tr, me)
    d_blur = cv2.GaussianBlur(d, (5, 5), 0)

    # Build grabcut trimap from seed and delta
    gc_mask = build_grabcut_mask(seed, d_blur, sure_bg_thr=3.0, prob_bg_thr=12.0)

    # Refine
    refined = refine_with_grabcut(tr, gc_mask, iters=5)

    # Keep only regions connected to initial seed
    refined = keep_connected_to_seed(refined, seed)

    # Light smoothing
    k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    refined = cv2.morphologyEx(refined, cv2.MORPH_CLOSE, k)

    # Save outputs
    refined_path = os.path.join(outdir, "refined_shirt_mask.png")
    cv2.imwrite(refined_path, refined)

    # Also save alpha cutout for quick review
    b, g, r = cv2.split(tr)
    alpha = refined
    cutout = cv2.merge((b, g, r, alpha))
    cutout_path = os.path.join(outdir, "refined_shirt_alpha.png")
    cv2.imwrite(cutout_path, cutout)

    # Optional: visualize delta for debugging
    d_vis = np.uint8(np.clip(d_blur * (255.0 / max(d_blur.max(), 1e-6)), 0, 255))
    cv2.imwrite(os.path.join(outdir, "delta_debug.png"), d_vis)

    print(f"Saved refined mask: {refined_path}")
    print(f"Saved refined alpha: {cutout_path}")


if __name__ == "__main__":
    main()
