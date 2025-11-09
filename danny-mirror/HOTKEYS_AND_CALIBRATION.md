# Magic Mirror – Hotkeys & Calibration Guide

This document summarizes all interactive controls, what they do, and how to run each calibration reliably. It reflects the current behavior of `danny-mirror/magic_mirror.py` with RawFace depth used for key features.

## Quick Start
- Launch the app (GUI OpenCV build required). The main window is the “Magic Mirror”.
- Most hotkeys act immediately; some display overlays or require simple steps.

## Hotkeys

- q
  - Quit the app.

- f
  - Toggle fullscreen for the main window.

- m
  - Move window to the next monitor (requires `screeninfo`).

- p
  - Save a screenshot of the current mirror output (e.g., `mirror_screenshot_<ts>.jpg`).

- d
  - Toggle a small depth inset overlay (debug view) when available.

- z
  - Freeze/unfreeze scaling at 1.0.
  - When freezing: resets image offsets to zero and clears smoothed offsets so you can align cleanly.

- c  (Calibration mode)
  - Enter/exit the on‑screen calibration wizard.
  - Follow on‑screen text; use SPACE to set reference distance in step 1, then finish.

- SPACE (inside Calibration mode)
  - Step 1: Set reference distance. Prefers RawFace depth (median around face center in raw depth) and falls back to ROI depth only if needed. Then proceed to step 2 (alignment).
  - Step 2: Finish calibration and print the chosen values.

- Arrow Keys or WASD (while aligning during calibration)
  - Nudge image offsets (X/Y) to align reflection.

- = or +;  - or _
  - Adjust numeric parameters as printed in the on‑screen instructions (e.g., reference distance). The UI text in the app indicates what they target.

- x
  - Toggle Fixed ROI mode. Uses a centered ROI to read distance (used for phone camera/non‑face targets debugging).

- n
  - Manual ROI select for calibration. Click two corners of a region to steer depth sampling during specific calibration flows (auto‑cal/multi‑depth). Useful for non‑face targets.

- g  (Auto‑Cal 150mm)
  - One‑shot: Click two points exactly 150mm apart on the screen; the app adjusts the life‑size multiplier accordingly. During this flow, scaling is frozen and parallax disabled for clean measurement. Uses raw depth sampling to set reference when possible.
  - Press g again to exit.

- i / o / ENTER / BACKSPACE  (Multi‑Depth Calibration)
  - i: Toggle the multi‑depth calibration wizard (freezes scale at 1.0 while active).
  - o: Capture one sample: click two points 150mm apart; the app records measured px and current depth.
  - ENTER: Solve for a quadratic scale model (k, q) across samples and save to `calibration.json`.
  - BACKSPACE: Remove last sample.

- e  (Eye‑Offset Auto‑Calibration)
  - Hold steady and press e. Samples face/eye spatials for ~1.2s, inverts them to compute `CAMERA_OFFSET_X_MM/Y_MM`, resets smoothing, and saves to `calibration.json`.

- [ and ]  (Parallax distance exponent)
  - Decrease/increase `parallax_distance_scale` by 0.05. Persists to `calibration.json`.

- b  (Viewer Band Mask)
  - Toggle ON/OFF. Continuously estimates viewer depth from raw depth at face center (requires recent face confidence ≥ 0.8) and blacks out everything outside ±200mm around that depth.
  - Uses connected components + a vertical corridor below the face to prefer the body and reject background at similar depth.

- y  (Persistent Depth Readout)
  - Captures the current RawFace depth (median of a 13×13 window at the face center) if a confident face (≥ 0.8) was seen in the last ~2s. Displays large, centered text until replaced with another y press.

- t  (Depth Diagnostics Overlay)
  - Shows a large, centered overlay with both values:
    - ROI Z: smoothed ROI distance (legacy; used as fallback only).
    - RawFace Z: raw median around face center (primary).
  - Helps identify any divergence between the two.

## Calibration Workflows

### 1) Reference Distance (SPACE within Calibration Mode)
- Press c to enter calibration.
- Step 1: Stand at your normal viewing distance and press SPACE.
  - The app prefers RawFace depth for setting `REFERENCE_DISTANCE_MM` (falls back to ROI Z only if RawFace isn’t available).
- Step 2: Use Arrow Keys/WASD to align the reflection. Press SPACE again to finish.

Tips:
- Use z to freeze scaling and reset offsets for alignment runs.
- If face confidence is unstable, toggle t to confirm RawFace Z is sensible; reposition if needed.

### 2) Auto‑Cal 150mm (g)
- Press g to enable. Scaling/parallax are suppressed for accurate measurement.
- Click two points on the screen that are exactly 150mm apart (use a ruler on‑screen). The app adjusts the life‑size multiplier and, when possible, updates the reference distance from raw depth at the segment midpoint.
- Press g again to exit.

### 3) Multi‑Depth Calibration (i/o/ENTER/BACKSPACE)
- Press i to start (freezes scale at 1.0). For each sample:
- Press o, then click two points 150mm apart.
  - The app records depth (preferring raw depth) and measured pixels.
- Press ENTER to solve for a (k, q) scale model and save to `calibration.json`.
- Press BACKSPACE to remove the last sample; press i again to exit.

### 4) Eye‑Offset Calibration (e)
- Ensure your face is detected (confidence ≥ 0.8). Press e; hold steady for ~1.2s.
- The app estimates `CAMERA_OFFSET_X_MM/Y_MM` from RawFace spatials, applies them immediately, and saves to `calibration.json`.

## Persistence & Files
- `calibration.json` may store:
  - `CAMERA_OFFSET_X_MM`, `CAMERA_OFFSET_Y_MM`
  - `IMAGE_OFFSET_X`, `IMAGE_OFFSET_Y`
  - `parallax_distance_scale`
  - Fitted `scale_model_k`, `scale_model_q`
  - (Optional) `DISPLAY_CAMERA_OFFSET_X_MM`, `DISPLAY_CAMERA_OFFSET_Y_MM`
- RawFace Z is now used for:
  - Scaling (primary depth source with smoothing/outlier rejection)
  - SPACE reference distance (preferred)
  - Y capture
  - B viewer band
- ROI Z remains as a fallback and as a diagnostic via the t overlay.

## Troubleshooting
- No overlay text? Ensure the OpenCV window has focus and you’re running GUI OpenCV (not headless). Use `t` and `y` to verify live depth.
- Face not detected: check lighting/angle; the viewer band and Y capture require a recent face confidence ≥ 0.8.
- Window/monitor issues: toggle fullscreen with f or move between monitors with m (install `screeninfo`).

---

Keep this file handy while calibrating. If you want any hotkeys to be re-bound, or to persist additional settings automatically, ask and we can wire it up.
