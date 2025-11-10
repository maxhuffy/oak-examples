#!/usr/bin/env python3
import argparse
import os
import sys
import json
import subprocess
from pathlib import Path

import numpy as np
import requests
from skimage import io
from skimage.transform import resize


BASE_DIR = Path(__file__).resolve().parent
TRY_ON_URL = "https://api.developer.pixelcut.ai/v1/try-on"
REMOVE_BG_URL = "https://api.developer.pixelcut.ai/v1/remove-background"


def read_image_rgb(path: Path) -> np.ndarray:
	"""Read an image and return HxWx3 RGB array. If grayscale, stack to 3 channels.
	If RGBA, drop alpha for padding/resizing stages.
	Keeps dtype; many skimage readers return uint8.
	"""
	img = io.imread(str(path))
	if img.ndim == 2:
		img = np.stack([img] * 3, axis=-1)
	if img.shape[2] == 4:
		img = img[:, :, :3]
	return img


def make_square_with_red_padding(img: np.ndarray) -> np.ndarray:
	"""Pad image to square with bright red background (255,0,0 or 1,0,0 depending on dtype)."""
	h, w, c = img.shape
	assert c == 3, "Expected RGB image"
	size = max(h, w)

	if np.issubdtype(img.dtype, np.integer):
		red_val = (np.iinfo(img.dtype).max, 0, 0)
	else:
		red_val = (1.0, 0.0, 0.0)

	bg = np.zeros((size, size, 3), dtype=img.dtype)
	bg[:, :, 0] = red_val[0]
	bg[:, :, 1] = red_val[1]
	bg[:, :, 2] = red_val[2]

	y0 = (size - h) // 2
	x0 = (size - w) // 2
	bg[y0:y0 + h, x0:x0 + w, :] = img
	return bg


def resize_to(img: np.ndarray, target_h: int, target_w: int) -> np.ndarray:
	"""Resize image to (target_h, target_w) preserving dtype using skimage.resize."""
	out = resize(img, (target_h, target_w), order=1, anti_aliasing=True, preserve_range=True)
	return out.astype(img.dtype)


def pixelcut_try_on(person_path: Path, garment_path: Path, api_key: str, out_path: Path) -> bool:
	headers = {
		"X-API-KEY": api_key,
		"Accept": "application/json",
	}
	with open(person_path, "rb") as pf, open(garment_path, "rb") as gf:
		files = {
			"person_image": (person_path.name, pf, "image/png"),
			"garment_image": (garment_path.name, gf, "image/png"),
		}
		data = {
			"preprocess_garment": "true",
			"remove_background": "false",
			"wait_for_result": "true",
		}
		resp = requests.post(TRY_ON_URL, headers=headers, files=files, data=data, timeout=120)
	if resp.status_code != 200:
		print("[try-on] Request failed:", resp.status_code, resp.text)
		return False
	try:
		result_url = resp.json().get("result_url")
	except Exception as e:
		print("[try-on] Failed parsing JSON:", e, resp.text)
		return False
	if not result_url:
		print("[try-on] No result_url in response")
		return False
	img_resp = requests.get(result_url, timeout=120)
	if img_resp.status_code != 200:
		print("[try-on] Failed to download result image:", img_resp.status_code)
		return False
	out_path.parent.mkdir(parents=True, exist_ok=True)
	with open(out_path, "wb") as f:
		f.write(img_resp.content)
	print(f"[try-on] Saved EDITED to {out_path}")
	return True


def pixelcut_remove_bg(image_path: Path, api_key: str, out_path: Path) -> bool:
	headers = {
		"Accept": "application/json",
		"X-API-KEY": api_key,
	}
	with open(image_path, "rb") as f:
		files = {"image": (image_path.name, f, "image/png")}
		data = {
			"format": "png",
			# Keep explicit structure for compatibility, though shadows disabled
			"shadow": json.dumps({
				"enabled": False,
				"opacity": 0,
				"light_source": {"size": 0, "position": {"x": 0, "y": 0, "z": 0}},
			}),
		}
		resp = requests.post(REMOVE_BG_URL, headers=headers, files=files, data=data, timeout=120)
	if resp.status_code != 200:
		print("[remove-bg] Request failed:", resp.status_code, resp.text)
		return False
	try:
		result_url = resp.json().get("result_url")
	except Exception as e:
		print("[remove-bg] Failed parsing JSON:", e, resp.text)
		return False
	if not result_url:
		print("[remove-bg] No result_url in response")
		return False
	img_resp = requests.get(result_url, timeout=120)
	if img_resp.status_code != 200:
		print("[remove-bg] Failed to download image:", img_resp.status_code)
		return False
	out_path.parent.mkdir(parents=True, exist_ok=True)
	with open(out_path, "wb") as f:
		f.write(img_resp.content)
	print(f"[remove-bg] Saved no-background image to {out_path}")
	return True


def run_superpixel_mask(original_no_bg: Path, edited_no_bg: Path, out_mask: Path, extra_args=None) -> bool:
	py = sys.executable
	script = BASE_DIR / "super_pixel_masking_with_blur.py"
	cmd = [py, str(script), str(original_no_bg), str(edited_no_bg), "--out", str(out_mask)]
	if extra_args:
		cmd += extra_args
	print("[superpixel] Running:", " ".join(cmd))
	try:
		res = subprocess.run(cmd, capture_output=True, text=True, check=False)
	except Exception as e:
		print("[superpixel] Failed to run:", e)
		return False
	if res.returncode != 0:
		print("[superpixel] Non-zero exit code:", res.returncode)
		if res.stdout:
			print("stdout:\n", res.stdout)
		if res.stderr:
			print("stderr:\n", res.stderr)
		return False
	if res.stdout:
		print(res.stdout.strip())
	return True


def main():
	parser = argparse.ArgumentParser(description="Full overlay pipeline to produce final black/white mask")
	parser.add_argument("input_image", help="Path to the person input image")
	parser.add_argument("garment_image", help="Path to the garment image for try-on")
	parser.add_argument("--out", default=str(BASE_DIR / "FINAL_EXTRACTED_BLACK_WHITE_MASK.png"), help="Output mask path")
	parser.add_argument("--workdir", default=str(BASE_DIR / "_pipeline_artifacts"), help="Directory to store intermediate artifacts")
	args = parser.parse_args()

	api_key = os.environ.get("PIXELCUT_API_KEY")
	if not api_key or api_key == "YOUR_PIXELCUT_API_KEY_HERE":
		raise SystemExit("Set PIXELCUT_API_KEY in environment to use Pixelcut APIs.")

	in_path = Path(args.input_image)
	garment_path = Path(args.garment_image)
	out_mask = Path(args.out)
	workdir = Path(args.workdir)
	workdir.mkdir(parents=True, exist_ok=True)

	# Filenames for intermediates
	input_sqr = workdir / "INPUT_IMAGE_sqr.png"
	edited_path = workdir / "EDITED.png"
	input_sqr_down = workdir / "INPUT_IMAGE_sqr_downscaled.png"
	input_sqr_down_no_bg = workdir / "INPUT_IMAGE_sqr_downscaled_no_background.png"
	edited_no_bg = workdir / "EDITED_no_background.png"

	print("[step 1] Square-padding input with bright red background …")
	img = read_image_rgb(in_path)
	img_sqr = make_square_with_red_padding(img)
	io.imsave(str(input_sqr), img_sqr)
	print(f"[step 1] Saved {input_sqr}")

	print("[step 2] Pixelcut Try-On to generate EDITED …")
	ok = pixelcut_try_on(input_sqr, garment_path, api_key, edited_path)
	if not ok:
		raise SystemExit("Try-On step failed")

	# Determine EDITED size and downscale input to match exactly
	edited_img = io.imread(str(edited_path))
	if edited_img.ndim == 2:
		edited_img = np.stack([edited_img] * 3, axis=-1)
	if edited_img.shape[2] > 3:
		edited_img_rgb = edited_img[:, :, :3]
	else:
		edited_img_rgb = edited_img

	th, tw = edited_img_rgb.shape[:2]
	print(f"[step 2] Downscaling squared input to match EDITED size {tw}x{th} …")
	img_sqr_down = resize_to(img_sqr, th, tw)
	io.imsave(str(input_sqr_down), img_sqr_down)
	print(f"[step 2] Saved {input_sqr_down}")

	print("[step 3] Removing backgrounds from both images …")
	if not pixelcut_remove_bg(input_sqr_down, api_key, input_sqr_down_no_bg):
		raise SystemExit("Remove background failed for original")
	if not pixelcut_remove_bg(edited_path, api_key, edited_no_bg):
		raise SystemExit("Remove background failed for edited")

	print("[step 4] Running super-pixel masking to produce final mask …")
	if not run_superpixel_mask(input_sqr_down_no_bg, edited_no_bg, out_mask):
		raise SystemExit("Super-pixel masking failed")

	print(f"✅ Done. Final mask at: {out_mask}")


if __name__ == "__main__":
	main()

