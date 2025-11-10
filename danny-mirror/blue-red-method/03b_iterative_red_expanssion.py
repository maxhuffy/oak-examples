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
	local_bias: float,
	min_red_neighbors: int,
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

				local_similarity = similarity
				if local_bias > 0.0:
					y0 = max(0, ny - 1)
					y1 = min(height, ny + 2)
					x0 = max(0, nx - 1)
					x1 = min(width, nx + 2)
					local_patch = red_mask[y0:y1, x0:x1]
					if local_patch.any():
						local_vectors = diff_vec[y0:y1, x0:x1][local_patch]
						local_mean = local_vectors.mean(axis=0)
						local_similarity = np.linalg.norm(candidate_vec - local_mean)
						similarity = (similarity * (1 - local_bias)) + (local_similarity * local_bias)

				if min_red_neighbors > 1:
					y0 = max(0, ny - 1)
					y1 = min(height, ny + 2)
					x0 = max(0, nx - 1)
					x1 = min(width, nx + 2)
					neighbor_count = int(red_mask[y0:y1, x0:x1].sum())
					if neighbor_count < min_red_neighbors:
						continue

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


def morphological_open(mask: np.ndarray, size: int) -> np.ndarray:
	if size <= 1:
		return mask
	if size % 2 == 0:
		size += 1
	img = Image.fromarray(mask.astype(np.uint8) * 255, mode="L")
	img = img.filter(ImageFilter.MinFilter(size=size))
	img = img.filter(ImageFilter.MaxFilter(size=size))
	return np.array(img, dtype=np.uint8) > 0


def convolve_boolean(mask: np.ndarray, kernel: np.ndarray) -> np.ndarray:
	kh, kw = kernel.shape
	if kh % 2 == 0 or kw % 2 == 0:
		raise ValueError("Kernel dimensions must be odd")

	pad_h = kh // 2
	pad_w = kw // 2
	padded = np.pad(mask.astype(np.int32), ((pad_h, pad_h), (pad_w, pad_w)), mode="constant")
	result = np.zeros_like(mask, dtype=np.int32)

	for y in range(kh):
		for x in range(kw):
			weight = int(kernel[y, x])
			if weight == 0:
				continue
			result += weight * padded[y : y + mask.shape[0], x : x + mask.shape[1]]

	return result


def remove_small_components(mask: np.ndarray, seeds: np.ndarray, min_size: int) -> np.ndarray:
	if min_size <= 1:
		return mask

	height, width = mask.shape
	visited = np.zeros_like(mask, dtype=bool)
	cleaned = mask.copy()

	for start_y, start_x in zip(*np.nonzero(mask)):
		if visited[start_y, start_x]:
			continue

		stack = [(start_y, start_x)]
		component = []
		has_seed = False

		while stack:
			y, x = stack.pop()
			if visited[y, x]:
				continue
			visited[y, x] = True
			if not mask[y, x]:
				continue

			component.append((y, x))
			if seeds[y, x]:
				has_seed = True

			for dy, dx in neighborhood(8):
				ny, nx = y + dy, x + dx
				if 0 <= ny < height and 0 <= nx < width and not visited[ny, nx] and mask[ny, nx]:
					stack.append((ny, nx))

		if not has_seed and len(component) < min_size:
			for y, x in component:
				cleaned[y, x] = False

	return cleaned


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
	parser.add_argument(
		"--local-bias",
		type=float,
		default=0.5,
		help="Weight (0-1) for local neighbor similarity influence",
	)
	parser.add_argument(
		"--min-red-neighbors",
		type=int,
		default=1,
		help="Minimum red neighbors required for accepting/keeping new red pixels",
	)
	parser.add_argument(
		"--morph-open-size",
		type=int,
		default=3,
		help="Size of morphological opening kernel to smooth stray tendrils",
	)
	parser.add_argument(
		"--min-component-size",
		type=int,
		default=40,
		help="Remove detached red components smaller than this (except original seeds)",
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
		local_bias=np.clip(args.local_bias, 0.0, 1.0),
		min_red_neighbors=max(args.min_red_neighbors, 1),
	)

	if args.min_red_neighbors > 1:
		kernel = np.ones((3, 3), dtype=int)
		red_counts = convolve_boolean(expanded_red, kernel)
		expanded_red &= red_counts >= args.min_red_neighbors
		expanded_red |= red

	expanded_red = morphological_open(expanded_red, size=args.morph_open_size)
	expanded_red |= red

	expanded_red = remove_small_components(expanded_red, red, args.min_component_size)
	updated_blue = blue & ~expanded_red

	output_img = compose_output(expanded_red, updated_blue)
	output_img.save(args.output)


if __name__ == "__main__":
	main()

