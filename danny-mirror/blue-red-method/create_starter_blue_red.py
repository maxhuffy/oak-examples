"""Create a combined blue/red mask for the starter images.

This script expects two PNG assets living alongside it:
	- original.png : pixels to be shown in blue
	- edited.png   : pixels to be shown in red

The output `blue_red_mask.png` stacks the blue layer on top of the red layer.
"""

from __future__ import annotations

from pathlib import Path

from PIL import Image


BLUE_RGBA = (0, 105, 255, 255)  # bright blue that stands out well
RED_RGBA = (255, 32, 32, 255)  # bold red for strong contrast


def load_rgba(path: Path) -> Image.Image:
	"""Load an image as RGBA so we can reason about transparency."""

	image = Image.open(path)
	if image.mode != "RGBA":
		image = image.convert("RGBA")
	return image


def solid_color_mask(source: Image.Image, color: tuple[int, int, int, int]) -> Image.Image:
	"""Return a solid-color mask using the source image's alpha channel."""

	rgba = source.copy()
	data = rgba.getdata()

	mask_data = []
	for r, g, b, a in data:
		if a == 0:
			mask_data.append((0, 0, 0, 0))
		else:
			mask_data.append((color[0], color[1], color[2], color[3]))

	rgba.putdata(mask_data)
	return rgba


def generate_blue_red_mask(original: Path, edited: Path, output: Path) -> None:
	"""Produce `output` by stacking blue and red masks derived from the sources."""

	original_img = load_rgba(original)
	edited_img = load_rgba(edited)

	if original_img.size != edited_img.size:
		raise ValueError(
			"Input images must share the same dimensions: "
			f"{original_img.size} != {edited_img.size}"
		)

	blue_layer = solid_color_mask(original_img, BLUE_RGBA)
	red_layer = solid_color_mask(edited_img, RED_RGBA)

	combined = Image.alpha_composite(red_layer, blue_layer)
	combined.save(output)


def main() -> None:
	here = Path(__file__).resolve().parent
	original = here / "original.png"
	edited = here / "edited.png"
	output = here / "blue_red_mask.png"

	if not original.exists() or not edited.exists():
		missing = [p.name for p in (original, edited) if not p.exists()]
		raise FileNotFoundError(
			"Missing required input file(s): " + ", ".join(missing)
		)

	generate_blue_red_mask(original, edited, output)


if __name__ == "__main__":
	main()

