import os
import argparse
from typing import Tuple

import cv2
import numpy as np


def _resolve_default_paths() -> Tuple[str, str, str]:
    """Return (me.png path, tryon_result.jpg path, output_dir) with output_dir pointing to scaler/output."""
    script_dir = os.path.dirname(os.path.abspath(__file__))
    parent_dir = os.path.abspath(os.path.join(script_dir, os.pardir, os.pardir))
    me_path = os.path.join(parent_dir, "me.png")
    tryon_path = os.path.join(parent_dir, "tryon_result.jpg")
    output_dir = os.path.join(script_dir, "output")
    return me_path, tryon_path, output_dir


def load_images(me_path: str, tryon_path: str) -> Tuple[np.ndarray, np.ndarray]:
    me = cv2.imread(me_path, cv2.IMREAD_COLOR)
    if me is None:
        raise FileNotFoundError(f"Could not read me image: {me_path}")
    tryon = cv2.imread(tryon_path, cv2.IMREAD_COLOR)
    if tryon is None:
        raise FileNotFoundError(f"Could not read try-on image: {tryon_path}")
    return me, tryon


def _letterbox_center(img: np.ndarray, target_h: int, target_w: int, scale: float) -> np.ndarray:
    h, w = img.shape[:2]
    new_w, new_h = int(round(w * scale)), int(round(h * scale))
    resized = cv2.resize(img, (new_w, new_h), interpolation=cv2.INTER_LINEAR)
    canvas = np.zeros((target_h, target_w, 3), dtype=img.dtype)
    y0 = max(0, (target_h - new_h) // 2)
    x0 = max(0, (target_w - new_w) // 2)
    y1 = min(target_h, y0 + new_h)
    x1 = min(target_w, x0 + new_w)
    canvas[y0:y1, x0:x1] = resized[0:(y1 - y0), 0:(x1 - x0)]
    return canvas


def scale_tryon_to_me(me_bgr: np.ndarray, tryon_bgr: np.ndarray, mode: str = "auto") -> np.ndarray:
    th, tw = me_bgr.shape[:2]
    h, w = tryon_bgr.shape[:2]
    scale_h = th / float(h)
    scale_w = tw / float(w)

    if mode == "height":
        return _letterbox_center(tryon_bgr, th, tw, scale_h)
    if mode == "width":
        return _letterbox_center(tryon_bgr, th, tw, scale_w)
    if mode == "min":
        return _letterbox_center(tryon_bgr, th, tw, min(scale_h, scale_w))
    if mode == "max":
        return _letterbox_center(tryon_bgr, th, tw, max(scale_h, scale_w))

    cand_h = _letterbox_center(tryon_bgr, th, tw, scale_h)
    cand_w = _letterbox_center(tryon_bgr, th, tw, scale_w)
    me_gray = cv2.cvtColor(me_bgr, cv2.COLOR_BGR2GRAY)
    h_gray = cv2.cvtColor(cand_h, cv2.COLOR_BGR2GRAY)
    w_gray = cv2.cvtColor(cand_w, cv2.COLOR_BGR2GRAY)
    mse_h = float(np.mean((me_gray.astype(np.float32) - h_gray.astype(np.float32)) ** 2))
    mse_w = float(np.mean((me_gray.astype(np.float32) - w_gray.astype(np.float32)) ** 2))
    return cand_h if mse_h <= mse_w else cand_w


def main():
    default_me, default_tryon, default_out = _resolve_default_paths()

    parser = argparse.ArgumentParser(description="Scale try-on result to match original (no masking).")
    parser.add_argument("--me", type=str, default=default_me, help="Path to original image (me.png)")
    parser.add_argument("--tryon", type=str, default=default_tryon, help="Path to try-on result image")
    parser.add_argument("--outdir", type=str, default=default_out, help="Directory to save outputs (defaults to scaler/output)")
    parser.add_argument("--mode", type=str, choices=["auto", "height", "width", "min", "max"], default="auto",
                        help="Scaling strategy (default: auto)")
    args = parser.parse_args()

    os.makedirs(args.outdir, exist_ok=True)

    print("[scaler] Loading images...")
    me_bgr, tryon_bgr = load_images(args.me, args.tryon)
    print(f"[scaler] me.png shape: {me_bgr.shape}, tryon_result shape: {tryon_bgr.shape}")

    print(f"[scaler] Scaling try-on image to me.png (mode: {args.mode})...")
    scaled_tryon = scale_tryon_to_me(me_bgr, tryon_bgr, mode=args.mode)

    scaled_path = os.path.join(args.outdir, "tryon_result_scaled.png")
    cv2.imwrite(scaled_path, scaled_tryon)
    print(f"[scaler] Saved scaled try-on: {scaled_path}")

    overlay = cv2.addWeighted(me_bgr, 0.5, scaled_tryon, 0.5, 0)
    overlay_path = os.path.join(args.outdir, "comparison_overlay.png")
    cv2.imwrite(overlay_path, overlay)
    print(f"[scaler] Saved overlay comparison: {overlay_path}")


if __name__ == "__main__":
    main()
