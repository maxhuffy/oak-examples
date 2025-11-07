import os
import sys
import argparse

import cv2
import importlib.util

# Ensure local imports work when running as a script
CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
if CURRENT_DIR not in sys.path:
    sys.path.append(CURRENT_DIR)

def _load_scaler_module():
    # Look in new scaler/ subdirectory first
    candidates = [
        ("scale_tryon_result_to_original", os.path.join(CURRENT_DIR, "scaler", "scale_tryon_result_to_original.py")),
        ("create_clothing_mask", os.path.join(CURRENT_DIR, "create_clothing_mask.py")),  # legacy fallback
        ("scale_tryon_result_to_original_legacy", os.path.join(CURRENT_DIR, "scale_tryon_result_to_original.py")),
    ]
    last_err = None
    for mod_name, path in candidates:
        if not os.path.isfile(path):
            continue
        spec = importlib.util.spec_from_file_location(mod_name, path)
        if spec is None or spec.loader is None:
            last_err = ImportError(f"Failed to load spec from {path}")
            continue
        module = importlib.util.module_from_spec(spec)
        try:
            spec.loader.exec_module(module)
            return module
        except Exception as e:
            last_err = e
    if last_err is not None:
        raise last_err
    raise FileNotFoundError("Could not find scaler module (create_clothing_mask.py or scale_tryon_result_to_original.py)")

_cm = _load_scaler_module()

def _load_flood_module():
    candidates = [
        ("isolate_clothing_flood", os.path.join(CURRENT_DIR, "flood", "isolate_clothing_flood.py")),
        ("isolate_clothing_flood_legacy", os.path.join(CURRENT_DIR, "isolate_clothing_flood.py")),
    ]
    last_err = None
    for mod_name, path in candidates:
        if not os.path.isfile(path):
            continue
        spec = importlib.util.spec_from_file_location(mod_name, path)
        if spec is None or spec.loader is None:
            last_err = ImportError(f"Failed to load spec from {path}")
            continue
        module = importlib.util.module_from_spec(spec)
        try:
            spec.loader.exec_module(module)
            return module
        except Exception as e:
            last_err = e
    if last_err is not None:
        raise last_err
    raise FileNotFoundError("Could not find flood module (flood/isolate_clothing_flood.py)")

flood = _load_flood_module()


def main():
    # Use scaler's path resolution to get me.png and tryon_result.jpg; override outdir to scaler/output
    default_me, default_tryon, scaler_out = _cm._resolve_default_paths()
    flood_out = os.path.join(CURRENT_DIR, "flood", "output")

    parser = argparse.ArgumentParser(description="Run scaling and clothing isolation (flood-growth)")
    parser.add_argument("--me", type=str, default=default_me)
    parser.add_argument("--tryon", type=str, default=default_tryon)
    parser.add_argument("--scaler-out", type=str, default=scaler_out, help="Output directory for scaled images (scaler/output)")
    parser.add_argument("--flood-out", type=str, default=flood_out, help="Output directory for flood outputs (flood/output)")
    parser.add_argument("--mode", type=str, choices=["auto", "height", "width", "min", "max"], default="auto")
    parser.add_argument("--skip-scale", action="store_true", help="Skip scaling step and use existing tryon_result_scaled.png")

    # Flood params passthrough
    parser.add_argument("--delta-thresh", type=float, default=22.0)
    parser.add_argument("--candidate-margin", type=float, default=5.0)
    parser.add_argument("--num-seeds", type=int, default=3)
    parser.add_argument("--seed-center-frac-h", type=float, default=0.35)
    parser.add_argument("--seed-center-frac-w", type=float, default=0.35)
    parser.add_argument("--morph-kernel", type=int, default=19)
    parser.add_argument("--connectivity", type=int, choices=[4, 8], default=4)
    parser.add_argument("--no-fill-holes", action="store_true")

    args = parser.parse_args()

    os.makedirs(args.scaler_out, exist_ok=True)
    os.makedirs(args.flood_out, exist_ok=True)

    # Ensure we have a scaled try-on image
    scaled_path = os.path.join(args.scaler_out, "tryon_result_scaled.png")
    if not args.skip_scale:
        print(f"Scaling try-on -> {scaled_path} (mode={args.mode})")
        me_bgr, tryon_bgr = _cm.load_images(args.me, args.tryon)
        scaled_tryon = _cm.scale_tryon_to_me(me_bgr, tryon_bgr, mode=args.mode)
        cv2.imwrite(scaled_path, scaled_tryon)
        overlay = cv2.addWeighted(me_bgr, 0.5, scaled_tryon, 0.5, 0)
        cv2.imwrite(os.path.join(args.scaler_out, "comparison_overlay.png"), overlay)
    else:
        print("Skipping scaling step; expecting tryon_result_scaled.png to exist.")

    # Flood extraction
    print("Running flood-based clothing extraction...")
    mask_path, clothing_path = flood.run(
        me_path=args.me,
        tryon_scaled_path=scaled_path,
        outdir=args.flood_out,
        seed_center_frac_h=args.seed_center_frac_h,
        seed_center_frac_w=args.seed_center_frac_w,
        num_seeds=args.num_seeds,
        delta_thresh=args.delta_thresh,
        candidate_margin=args.candidate_margin,
        morph_kernel=args.morph_kernel,
        connectivity=args.connectivity,
        fill_holes=not args.no_fill_holes,
    )
    print(f"Saved mask: {mask_path}")
    print(f"Saved clothing: {clothing_path}")


if __name__ == "__main__":
    main()
