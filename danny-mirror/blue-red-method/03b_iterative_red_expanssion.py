"""Magic-wand style expansion of red regions into blue regions.

This script grows the red mask by absorbing nearby blue pixels whose
original/edited color deltas match the current red region within a user
defined tolerance, similar to how Photoshop's magic wand accumulates
adjacent pixels.
"""

from __future__ import annotations

import argparse
from collections import deque
from pathlib import Path
from typing import Iterable, Tuple

import numpy as np
from PIL import Image


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


def compute_diff_vectors(original: Path, edited: Path) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
	"""Return rgb deltas, delta magnitudes, and valid mask."""

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

	diff_vec = edit[..., :3] - orig[..., :3]
	diff_mag = np.linalg.norm(diff_vec, axis=-1)

	diff_vec[~valid] = 0.0
	diff_mag[~valid] = 0.0

	return diff_vec, diff_mag, valid


def neighborhood(connectivity: int) -> Iterable[Tuple[int, int]]:
	if connectivity == 4:
		return ((1, 0), (-1, 0), (0, 1), (0, -1))
	if connectivity == 8:
		return (
			(1, 0),
			(-1, 0),
			(0, 1),
			(0, -1),
			(1, 1),
			(1, -1),
			(-1, 1),
			(-1, -1),
		)
	raise ValueError("Connectivity must be 4 or 8")


def magic_wand_expand(
	red: np.ndarray,
	blue: np.ndarray,
	valid: np.ndarray,
	diff_vec: np.ndarray,
	diff_mag: np.ndarray,
	tolerance_start: float,
	tolerance_step: float,
	tolerance_max: float,
	min_diff: float,
	connectivity: int,
	max_passes: int,
) -> Tuple[np.ndarray, np.ndarray]:
	"""Grow the red mask using a tolerance-guided flood fill."""

	height, width = red.shape
	red_mask = red.copy()
	blue_mask = blue.copy()

	offsets = tuple(neighborhood(connectivity))

	tolerance = tolerance_start
	passes = 0

	while tolerance <= tolerance_max and passes < max_passes:
		passes += 1
		additions = 0

		queue = deque(zip(*np.nonzero(red_mask)))
		visited = red_mask.copy()

		while queue:
			y, x = queue.popleft()
			base_vec = diff_vec[y, x]

			for dy, dx in offsets:
				ny = y + dy
				nx = x + dx

				if ny < 0 or ny >= height or nx < 0 or nx >= width:
					continue
				if visited[ny, nx]:
					continue
				if not valid[ny, nx] or not blue_mask[ny, nx]:
					visited[ny, nx] = True
					continue

				candidate_mag = diff_mag[ny, nx]
				if candidate_mag < min_diff:
					continue

				candidate_vec = diff_vec[ny, nx]
				similarity = np.linalg.norm(candidate_vec - base_vec)

				if similarity <= tolerance:
					red_mask[ny, nx] = True
					blue_mask[ny, nx] = False
					visited[ny, nx] = True
					queue.append((ny, nx))
					additions += 1
				else:
					visited[ny, nx] = True

		if additions == 0:
			tolerance += tolerance_step
		else:
			tolerance = min(tolerance + (tolerance_step * 0.5), tolerance_max)

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
		help="Combined mask with red and blue regions",
	)
	parser.add_argument(
		"--output",
		type=Path,
		default=Path("blue_red_mask_red_expanded_magic.png"),
		help="Filename for the expanded mask",
	)
	parser.add_argument(
		"--tolerance-start",
		type=float,
		default=8.0,
		help="Initial tolerance for color-difference similarity",
	)
	parser.add_argument(
		"--tolerance-step",
		type=float,
		default=3.0,
		help="Increment added when no pixels are gained in a pass",
	)
	parser.add_argument(
		"--tolerance-max",
		type=float,
		default=30.0,
		help="Upper bound for the tolerance",
	)
	parser.add_argument(
		"--min-diff",
		type=float,
		default=5.0,
		help="Minimum difference magnitude required to consider a pixel",
	)
	parser.add_argument(
		"--connectivity",
		type=int,
		choices=(4, 8),
		default=8,
		help="Neighbor connectivity for flood fill",
	)
	parser.add_argument(
		"--max-passes",
		type=int,
		default=20,
		help="Maximum number of tolerance passes",
	)
	return parser.parse_args()


def main() -> None:
	args = parse_args()

	for path in (args.original, args.edited, args.mask):
		if not path.exists():
			raise FileNotFoundError(f"Missing required input file: {path}")

	diff_vec, diff_mag, valid = compute_diff_vectors(args.original, args.edited)
	red, blue, opaque = load_mask_layers(args.mask)

	red &= opaque
	blue &= opaque
	valid &= opaque

	expanded_red, updated_blue = magic_wand_expand(
		red,
		blue,
		valid,
		diff_vec,
		diff_mag,
		tolerance_start=args.tolerance_start,
		tolerance_step=args.tolerance_step,
		tolerance_max=args.tolerance_max,
		min_diff=args.min_diff,
		connectivity=args.connectivity,
		max_passes=args.max_passes,
	)

	output_img = compose_output(expanded_red, updated_blue)
	output_img.save(args.output)


if __name__ == "__main__":
	main()

