import os
import argparse
from typing import List, Tuple

import cv2
import numpy as np


def load_pair(me_path: str, tryon_scaled_path: str) -> Tuple[np.ndarray, np.ndarray]:
    me = cv2.imread(me_path, cv2.IMREAD_COLOR)
    if me is None:
        raise FileNotFoundError(f"Could not read me image: {me_path}")
    tr = cv2.imread(tryon_scaled_path, cv2.IMREAD_COLOR)
    if tr is None:
        raise FileNotFoundError(f"Could not read try-on image: {tryon_scaled_path}")
    if me.shape[:2] != tr.shape[:2]:
        raise ValueError(f"Images must be same size. me={me.shape[:2]} tryon={tr.shape[:2]}")
    return me, tr


def lab_delta(tryon_bgr: np.ndarray, me_bgr: np.ndarray) -> np.ndarray:
    a = cv2.cvtColor(tryon_bgr, cv2.COLOR_BGR2LAB).astype(np.float32)
    b = cv2.cvtColor(me_bgr, cv2.COLOR_BGR2LAB).astype(np.float32)
    d = a - b
    delta = np.sqrt(np.sum(d * d, axis=2))
    return delta


def choose_seed_points(delta: np.ndarray, center_frac_h: float, center_frac_w: float, num_seeds: int, min_dist: int = 25) -> List[Tuple[int, int]]:
    h, w = delta.shape
    ch = int(h * center_frac_h)
    cw = int(w * center_frac_w)
    y0 = (h - ch) // 2
    x0 = (w - cw) // 2
    roi = delta[y0:y0+ch, x0:x0+cw].copy()

    flat_idx = np.argsort(roi.reshape(-1))[::-1]
    seeds: List[Tuple[int, int]] = []

    def far_enough(y: int, x: int) -> bool:
        for sy, sx in seeds:
            if (sy - y) ** 2 + (sx - x) ** 2 < (min_dist ** 2):
                return False
        return True

    for idx in flat_idx:
        ry = idx // cw
        rx = idx % cw
        y = y0 + ry
        x = x0 + rx
        if far_enough(y, x):
            seeds.append((y, x))
            if len(seeds) >= num_seeds:
                break
    return seeds


def flood_by_delta(delta: np.ndarray,
                   seeds: List[Tuple[int, int]],
                   thresh: float,
                   candidate_margin: float = 5.0,
                   connectivity: int = 4,
                   max_area_frac: float = 0.8) -> np.ndarray:
    h, w = delta.shape
    cand = (delta >= (thresh - candidate_margin))
    visited = np.zeros((h, w), dtype=np.uint8)
    out = np.zeros((h, w), dtype=np.uint8)

    max_area = int(h * w * max_area_frac)

    if connectivity == 8:
        neighbors = [(-1,-1),(-1,0),(-1,1),(0,-1),(0,1),(1,-1),(1,0),(1,1)]
    else:
        neighbors = [(-1,0),(1,0),(0,-1),(0,1)]

    total = 0
    for sy, sx in seeds:
        if not cand[sy, sx]:
            continue
        if visited[sy, sx]:
            continue
        q: List[Tuple[int,int]] = [(sy, sx)]
        visited[sy, sx] = 1
        while q:
            y, x = q.pop()
            if delta[y, x] >= thresh:
                out[y, x] = 255
                total += 1
                if total >= max_area:
                    break
            for dy, dx in neighbors:
                ny, nx = y + dy, x + dx
                if 0 <= ny < h and 0 <= nx < w and not visited[ny, nx] and cand[ny, nx]:
                    visited[ny, nx] = 1
                    q.append((ny, nx))
        if total >= max_area:
            break
    return out


def postprocess_mask(mask: np.ndarray, morph_kernel: int = 7, fill_holes: bool = True) -> np.ndarray:
    k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (morph_kernel, morph_kernel))
    m = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, k, iterations=2)
    if fill_holes:
        h, w = m.shape[:2]
        flood = m.copy()
        flood_mask = np.zeros((h + 2, w + 2), np.uint8)
        cv2.floodFill(flood, flood_mask, (0, 0), 255)
        m = cv2.bitwise_or(m, cv2.bitwise_not(flood))
    return m


def apply_alpha(bgr: np.ndarray, mask: np.ndarray) -> np.ndarray:
    if mask.ndim == 3:
        mask = cv2.cvtColor(mask, cv2.COLOR_BGR2GRAY)
    mask = np.clip(mask, 0, 255).astype(np.uint8)
    b, g, r = cv2.split(bgr)
    return cv2.merge((b, g, r, mask))


def run(me_path: str,
        tryon_scaled_path: str,
        outdir: str,
        seed_center_frac_h: float = 0.35,
        seed_center_frac_w: float = 0.35,
        num_seeds: int = 3,
        delta_thresh: float = 22.0,
        candidate_margin: float = 5.0,
        morph_kernel: int = 7,
        connectivity: int = 4,
        fill_holes: bool = True) -> Tuple[str, str]:
    os.makedirs(outdir, exist_ok=True)
    me, tr = load_pair(me_path, tryon_scaled_path)
    d = lab_delta(tr, me)

    seeds = choose_seed_points(d, seed_center_frac_h, seed_center_frac_w, num_seeds=num_seeds)
    mask = flood_by_delta(d, seeds, thresh=delta_thresh, candidate_margin=candidate_margin, connectivity=connectivity)
    mask = postprocess_mask(mask, morph_kernel=morph_kernel, fill_holes=fill_holes)

    mask_path = os.path.join(outdir, "flood_mask.png")
    cv2.imwrite(mask_path, mask)

    clothing = apply_alpha(tr, mask)
    clothing_path = os.path.join(outdir, "tryon_clothing_flood.png")
    cv2.imwrite(clothing_path, clothing)

    return mask_path, clothing_path


def main():
    script_dir = os.path.dirname(os.path.abspath(__file__))
    parent_dir = os.path.abspath(os.path.join(script_dir, os.pardir, os.pardir))
    default_me = os.path.join(parent_dir, "me.png")
    default_tr = os.path.join(parent_dir, "scaler", "output", "tryon_result_scaled.png")
    default_out = os.path.join(script_dir, "output")

    parser = argparse.ArgumentParser(description="Extract clothing by flood-growth using tryon vs me difference.")
    parser.add_argument("--me", type=str, default=default_me)
    parser.add_argument("--tryon-scaled", type=str, default=default_tr)
    parser.add_argument("--outdir", type=str, default=default_out, help="Directory to save flood outputs (defaults to flood/output)")
    parser.add_argument("--delta-thresh", type=float, default=22.0)
    parser.add_argument("--candidate-margin", type=float, default=5.0)
    parser.add_argument("--num-seeds", type=int, default=3)
    parser.add_argument("--seed-center-frac-h", type=float, default=0.35)
    parser.add_argument("--seed-center-frac-w", type=float, default=0.35)
    parser.add_argument("--morph-kernel", type=int, default=7)
    parser.add_argument("--connectivity", type=int, choices=[4, 8], default=4)
    parser.add_argument("--no-fill-holes", action="store_true")

    args = parser.parse_args()

    os.makedirs(args.outdir, exist_ok=True)
    print("[flood] Running flood extraction...")
    mask_path, clothing_path = run(
        me_path=args.me,
        tryon_scaled_path=args.tryon_scaled,
        outdir=args.outdir,
        seed_center_frac_h=args.seed_center_frac_h,
        seed_center_frac_w=args.seed_center_frac_w,
        num_seeds=args.num_seeds,
        delta_thresh=args.delta_thresh,
        candidate_margin=args.candidate_margin,
        morph_kernel=args.morph_kernel,
        connectivity=args.connectivity,
        fill_holes= not args.no_fill_holes,
    )
    print(f"[flood] Saved mask: {mask_path}")
    print(f"[flood] Saved clothing: {clothing_path}")


if __name__ == "__main__":
    main()
