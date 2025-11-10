# Full Overlay Pipeline

End‑to‑end script (`main.py`) that:
1. Pads an input person image to a red square.
2. Sends the squared image + a garment image to Pixelcut Try‑On API to generate an edited image.
3. Downscales the squared original to the edited size.
4. Removes backgrounds from both images via Pixelcut Remove Background API.
5. Computes a super‑pixel based difference mask (`FINAL_EXTRACTED_BLACK_WHITE_MASK.png`).
6. Applies that mask to the edited no‑background image, crops back to the original aspect ratio, and upscales to the original resolution -> `FINAL_JUST_EDITS.png` (RGBA).

---
## Folder Contents
| File | Purpose |
|------|---------|
| `main.py` | Orchestrates the entire pipeline (full or mask‑only mode). |
| `pixel-cut_remove_background.py` | Standalone helper to remove background for a single image (binary upload). |
| `pixel-cut.py` | Standalone Pixelcut Try‑On example (person + garment). |
| `super_pixel_masking_with_blur.py` | Generates change mask between two RGBA images. |
| `requirements.txt` | Python dependencies pinned for this submodule. |

Intermediate artifacts are written to `_pipeline_artifacts/` by default.

---
## Installation
Create / activate a virtual environment (recommended) and install deps:

```bash
python -m venv .venv
source .venv/Scripts/activate  # Windows Git Bash
# or: .venv\Scripts\activate   # Windows CMD/PowerShell
pip install -r requirements.txt
```

---
## Pixelcut API Key
Set your Pixelcut key in the environment (the script currently has a placeholder; set it properly for production):
```bash
export PIXELCUT_API_KEY=sk_your_real_key_here
```
(Use `setx` on Windows if you want it persisted, or just `export` per session.)

---
## Full Pipeline Usage
Generates mask + final edits (calls both Try‑On and Remove Background endpoints):
```bash
python main.py person_input.png garment_input.png \
  --out FINAL_EXTRACTED_BLACK_WHITE_MASK.png \
  --workdir _pipeline_artifacts
```
Outputs:
- `FINAL_EXTRACTED_BLACK_WHITE_MASK.png` – black/white diff mask.
- `_pipeline_artifacts/FINAL_JUST_EDITS.png` – RGBA image containing only edited regions at the original resolution.

---
## Mask‑Only Mode (Skip API Calls)
If you already have the two no‑background images from a previous run, you can regenerate the mask (after tweaking `super_pixel_masking_with_blur.py`) without spending credits.

Expected existing files (default locations):
- `_pipeline_artifacts/INPUT_IMAGE_sqr_downscaled_no_background.png`
- `_pipeline_artifacts/EDITED_no_background.png`

Run:
```bash
python main.py original_input.png placeholder_garment.png --mask-only
```
The garment path is ignored in mask‑only mode.

If your artifact paths differ:
```bash
python main.py original_input.png placeholder.png --mask-only \
  --original-no-bg path/to/original_no_bg.png \
  --edited-no-bg path/to/edited_no_bg.png \
  --out NEW_MASK.png
```

---
## super_pixel_masking_with_blur.py Parameters
Useful flags when tuning mask sensitivity:
- `--kernel-size` (default 9)
- `--kernel-thresh` (default 8.0) – lower -> more sensitive
- `--region-thresh` (default 6.0) – fallback region-level threshold
- `--blur-sigma` (default 8) – pre-blur images; reduce for sharper regions
- `--max-components` (default 1 here) – keep only largest N blobs
- `--min-component-area` – discard tiny connected regions

You can modify defaults (e.g. set `max-components` back to 4) directly in the script or by passing flags in mask-only runs (edit `main.py` to forward them if needed).

---
## Output Files Summary
| Artifact | When Created | Description |
|----------|--------------|-------------|
| `INPUT_IMAGE_sqr.png` | Step 1 | Square padded original. |
| `EDITED.png` | Step 2 | Pixelcut Try‑On result. |
| `INPUT_IMAGE_sqr_downscaled.png` | Step 2 | Padded original resized to edited dimensions. |
| `*_no_background.png` | Step 3 | Background-removed versions of both images. |
| `FINAL_EXTRACTED_BLACK_WHITE_MASK.png` | Step 4 | Binary mask (white = changed). |
| `FINAL_JUST_EDITS.png` | Step 5 | RGBA composited edits only, cropped + upscaled to original size. |

---
## Troubleshooting
| Issue | Cause | Fix |
|-------|-------|-----|
| UnicodeEncodeError on checkmark | Windows cp1252 console | Removed emoji already; or enable UTF‑8 (`chcp 65001`). |
| 401 / 403 from Pixelcut | Bad / missing API key | Verify `PIXELCUT_API_KEY` env var. |
| Mask empty | Thresholds too high | Lower `--kernel-thresh` / `--region-thresh`. |
| Too many speckles | Thresholds too low or max-components | Increase thresholds, raise `--min-component-area`, or reduce `--max-components`. |
| Slow SLIC step | High n_segments (3000) | Lower `n_segments` inside `run_slic`. |

---
## Version Notes
- `morphology.square` is used; pinned `scikit-image < 0.27` where it still exists.
- Upgrade path: replace with `morphology.footprint_rectangle(size)` after verifying scikit-image API.

---
## Extending
Ideas:
- Add flag passthrough from `main.py` to `super_pixel_masking_with_blur.py` for threshold tuning.
- Cache Pixelcut responses to disk keyed by image hash.
- Add optional inpainting / blending step for smoother edit boundaries.

---
## License
Follows parent repository license.

---
## Quick One-Liner
Full run:
```bash
python main.py person.png garment.png && echo "Mask + edits ready." 
```
Mask only regeneration:
```bash
python main.py person.png dummy.png --mask-only && echo "Mask regenerated." 
```
