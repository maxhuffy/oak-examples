"""Iteratively expand the red mask into blue regions based on real edits.

The process:
1. Load `original.png`, `edited.png`, and `blue_red_mask_blue_dilated.png`.
2. Measure per-pixel color differences between original and edited.
3. Repeatedly grow the red region into adjacent blue pixels when the
   underlying color difference is strong enough to suggest a real edit.

The resulting mask is written to `blue_red_mask_red_expanded.png`.
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Tuple

import numpy as np
from PIL import Image, ImageFilter


BLUE_RGB = (0, 105, 255)
RED_RGB = (255, 32, 32)
BLUE_RGBA = (0, 105, 255, 255)
RED_RGBA = (255, 32, 32, 255)


def load_rgba(path: Path) -> Image.Image:
	image = Image.open(path)
	if image.mode != "RGBA":
		image = image.convert("RGBA")
	return image


def load_mask_layers(path: Path) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
	"""Return boolean arrays for red, blue, and opaque pixels."""

	mask = load_rgba(path)
	mask_arr = np.array(mask, dtype=np.uint8)

	alpha = mask_arr[..., 3] > 0
	red = (
		(mask_arr[..., 0] == RED_RGB[0])
		& (mask_arr[..., 1] == RED_RGB[1])
		& (mask_arr[..., 2] == RED_RGB[2])
		& alpha
	)
	blue = (
		(mask_arr[..., 0] == BLUE_RGB[0])
		& (mask_arr[..., 1] == BLUE_RGB[1])
		& (mask_arr[..., 2] == BLUE_RGB[2])
		& alpha
	)
	return red, blue, alpha


def compute_difference_map(original: Path, edited: Path) -> Tuple[np.ndarray, np.ndarray]:
	"""Return color-difference and validity masks between two images."""

	orig_img = load_rgba(original)
	edit_img = load_rgba(edited)

	if orig_img.size != edit_img.size:
		raise ValueError(
			"Input images must share the same dimensions: "
			f"{orig_img.size} != {edit_img.size}"
		)

	orig = np.array(orig_img, dtype=np.float32)
	edit = np.array(edit_img, dtype=np.float32)

	valid = (orig[..., 3] > 0) | (edit[..., 3] > 0)

	orig_rgb = orig[..., :3]
	edit_rgb = edit[..., :3]

	color_diff = np.linalg.norm(edit_rgb - orig_rgb, axis=-1)
	intensity_diff = np.abs(edit_rgb.mean(axis=-1) - orig_rgb.mean(axis=-1))
	combined = (0.7 * color_diff) + (0.3 * intensity_diff)
	combined = smooth_difference_map(combined, radius=1)

	combined[~valid] = 0.0

	return combined, valid


def smooth_difference_map(data: np.ndarray, radius: int = 1) -> np.ndarray:
	"""Apply a simple box blur implemented in numpy to reduce jitter."""

	if radius <= 0:
		return data

	ksize = (2 * radius) + 1
	pad = np.pad(data, radius, mode="edge")
	result = np.zeros_like(data)

	for dy in range(ksize):
		for dx in range(ksize):
			result += pad[dy : dy + data.shape[0], dx : dx + data.shape[1]]

	return result / float(ksize * ksize)


def dilate_mask(mask: np.ndarray, size: int = 3) -> np.ndarray:
	"""Dilate a boolean mask using Pillow's MaxFilter for convenience."""

	mask_img = Image.fromarray(mask.astype(np.uint8) * 255, mode="L")
	dilated = mask_img.filter(ImageFilter.MaxFilter(size=size))
	return np.array(dilated, dtype=np.uint8) > 0


def iterative_expand_red(
	red: np.ndarray,
	blue: np.ndarray,
	opaque: np.ndarray,
	diff_map: np.ndarray,
	max_iterations: int,
	decay: float,
	min_margin: float,
	frontier_percentile: float,
) -> np.ndarray:
	"""Grow the red mask into blue pixels when the diff map strongly differs."""

	red_mask = red.copy()
	blue_mask = blue.copy()

	blue_values = diff_map[blue_mask]
	red_values = diff_map[red_mask]

	noise_floor = np.percentile(blue_values, 85) if blue_values.size else 0.0
	signal_floor = (
		np.percentile(red_values, 30) if red_values.size else np.percentile(diff_map[opaque], 75)
	)

	threshold = max(signal_floor, noise_floor + min_margin)
	min_threshold = noise_floor + min_margin

	for iteration in range(max_iterations):
		frontier = dilate_mask(red_mask, size=3)
		frontier &= ~red_mask
		frontier &= blue_mask
		if not frontier.any():
			break

		candidate_indices = np.where(frontier)
		candidate_values = diff_map[candidate_indices]

		candidate_threshold = threshold
		if candidate_values.size:
			percentile_cutoff = np.percentile(
				candidate_values, np.clip(frontier_percentile, 0.0, 100.0)
			)
			candidate_threshold = min(
				candidate_threshold, max(min_threshold, percentile_cutoff)
			)

		accept = candidate_values >= candidate_threshold
		accepted_pixels = tuple(idx[accept] for idx in candidate_indices)

		if not accept.any():
			next_threshold = max(threshold * decay, min_threshold)
			if np.isclose(next_threshold, threshold):
				break
			threshold = next_threshold
			continue

		red_mask[accepted_pixels] = True
		blue_mask[accepted_pixels] = False

		threshold = max(threshold * decay, min_threshold)

	return red_mask, blue_mask


def compose_output(red: np.ndarray, blue: np.ndarray) -> Image.Image:
	height, width = red.shape
	output = np.zeros((height, width, 4), dtype=np.uint8)

	output[blue] = BLUE_RGBA
	output[red] = RED_RGBA

	return Image.fromarray(output, mode="RGBA")


def parse_args() -> argparse.Namespace:
	parser = argparse.ArgumentParser(description=__doc__)
	parser.add_argument(
		"--original",
		type=Path,
		default=Path("original.png"),
		help="Path to the original image",
	)
	parser.add_argument(
		"--edited",
		type=Path,
		default=Path("edited.png"),
		help="Path to the edited image",
	)
	parser.add_argument(
		"--mask",
		type=Path,
		default=Path("blue_red_mask_blue_dilated.png"),
		help="Source mask file containing blue and red regions",
	)
	parser.add_argument(
		"--output",
		type=Path,
		default=Path("blue_red_mask_red_expanded_v4_aggressive.png"),
		help="Filename for the expanded mask",
	)
	parser.add_argument(
		"--max-iterations",
		type=int,
		default=600,
		help="Maximum number of expansion iterations",
	)
	parser.add_argument(
		"--decay",
		type=float,
		default=0.4,
		help="Multiplier applied to the threshold after each successful iteration",
	)
	parser.add_argument(
		"--min-margin",
		type=float,
		default=0.75,
		help="Minimum difference above the noise floor to accept new red pixels",
	)
	parser.add_argument(
		"--frontier-percentile",
		type=float,
		default=40.0,
		help=(
			"Percentile of current frontier differences to guarantee acceptance;"
			" helps expansion continue when thresholds plateau"
		),
	)
	return parser.parse_args()


def main() -> None:
	args = parse_args()

	if not args.original.exists() or not args.edited.exists() or not args.mask.exists():
		missing = [
			str(path)
			for path in (args.original, args.edited, args.mask)
			if not path.exists()
		]
		raise FileNotFoundError("Missing required input file(s): " + ", ".join(missing))

	diff_map, valid = compute_difference_map(args.original, args.edited)
	red, blue, opaque = load_mask_layers(args.mask)

	red &= valid
	blue &= valid
	opaque &= valid

	expanded_red, updated_blue = iterative_expand_red(
		red,
		blue,
		opaque,
		diff_map,
		max_iterations=args.max_iterations,
		decay=args.decay,
		min_margin=args.min_margin,
		frontier_percentile=args.frontier_percentile,
	)

	output_img = compose_output(expanded_red, updated_blue)
	output_img.save(args.output)


if __name__ == "__main__":
	main()

