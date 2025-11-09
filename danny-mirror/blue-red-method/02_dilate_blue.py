"""Dilate only the blue region from `blue_red_mask.png`.

This script expands the blue mask slightly while leaving the red mask
untouched and writes the result to `blue_red_mask_blue_dilated.png`.
"""

from __future__ import annotations

from pathlib import Path

from PIL import Image, ImageFilter


BLUE_RGB = (0, 105, 255)
BLUE_RGBA = (0, 105, 255, 255)
RED_RGB = (255, 32, 32)
RED_RGBA = (255, 32, 32, 255)


def load_mask_layers(path: Path) -> tuple[Image.Image, Image.Image]:
	"""Return separate blue-mask (L) and red-layer (RGBA) images."""

	base = Image.open(path).convert("RGBA")
	width, height = base.size

	blue_mask = Image.new("L", (width, height), 0)
	red_layer = Image.new("RGBA", (width, height), (0, 0, 0, 0))

	base_pixels = base.load()
	blue_pixels = blue_mask.load()
	red_pixels = red_layer.load()

	for y in range(height):
		for x in range(width):
			r, g, b, a = base_pixels[x, y]
			if a == 0:
				continue
			if (r, g, b) == BLUE_RGB:
				blue_pixels[x, y] = 255
			elif (r, g, b) == RED_RGB:
				red_pixels[x, y] = RED_RGBA

	return blue_mask, red_layer


def dilate_mask(mask: Image.Image, size: int = 3) -> Image.Image:
	"""Dilate a binary mask using a square structuring element."""

	if size % 2 == 0 or size < 1:
		raise ValueError("Dilation size must be an odd positive integer")
	return mask.filter(ImageFilter.MaxFilter(size=size))


def build_blue_layer(mask: Image.Image) -> Image.Image:
	"""Create a blue RGBA layer whose alpha is driven by the mask."""

	layer = Image.new("RGBA", mask.size, BLUE_RGBA)
	layer.putalpha(mask)
	return layer


def dilate_blue_region(source: Path, output: Path, dilation_size: int = 3) -> None:
	"""Expand the blue region from `source` and write the combined output."""

	blue_mask, red_layer = load_mask_layers(source)
	dilated_blue_mask = dilate_mask(blue_mask, size=dilation_size)
	blue_layer = build_blue_layer(dilated_blue_mask)

	combined = Image.alpha_composite(red_layer, blue_layer)
	combined.save(output)


def main() -> None:
	here = Path(__file__).resolve().parent
	source = here / "blue_red_mask.png"
	output = here / "blue_red_mask_blue_dilated.png"

	if not source.exists():
		raise FileNotFoundError(f"Missing source mask: {source.name}")

	dilate_blue_region(source, output, dilation_size=5)


if __name__ == "__main__":
	main()

