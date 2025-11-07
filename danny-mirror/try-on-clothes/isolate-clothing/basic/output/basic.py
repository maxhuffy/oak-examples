import os
import cv2
import numpy as np


def main():
	# Paths: original is me.png at try-on-clothes root; edited is isolate-clothing/scaler/output/tryon_result_scaled.png
	script_dir = os.path.dirname(os.path.abspath(__file__))
	tryon_clothes_root = os.path.abspath(os.path.join(script_dir, os.pardir, os.pardir, os.pardir))  # .../try-on-clothes
	orig_path = os.path.join(tryon_clothes_root, "me.png")
	edited_path = os.path.join(tryon_clothes_root, "isolate-clothing", "scaler", "output", "tryon_result_scaled.png")

	outdir = script_dir  # write outputs next to this script (basic/output)
	os.makedirs(outdir, exist_ok=True)

	orig = cv2.imread(orig_path)
	if orig is None:
		raise FileNotFoundError(f"Could not read original image at {orig_path}")
	edit = cv2.imread(edited_path)
	if edit is None:
		raise FileNotFoundError(f"Could not read edited image at {edited_path}")

	if orig.shape[:2] != edit.shape[:2]:
		raise ValueError(f"Images must be same size, got orig={orig.shape[:2]} edited={edit.shape[:2]}.")

	orig_b = cv2.GaussianBlur(orig, (5, 5), 0)
	edit_b = cv2.GaussianBlur(edit, (5, 5), 0)

	diff = cv2.absdiff(orig_b, edit_b)
	gray = cv2.cvtColor(diff, cv2.COLOR_BGR2GRAY)

	# pick a threshold that isolates the jacket area
	_, mask = cv2.threshold(gray, 10, 255, cv2.THRESH_BINARY)

	# clean it
	kernel = np.ones((5, 5), np.uint8)
	mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
	mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)

	# Save outputs
	cv2.imwrite(os.path.join(outdir, "diff_gray.png"), gray)
	out_mask = os.path.join(outdir, "clothes_mask.png")
	cv2.imwrite(out_mask, mask)
	print(f"Saved mask to {out_mask}")


if __name__ == "__main__":
	main()

