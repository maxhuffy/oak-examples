# Stereo Dynamic Calibration

This example demonstrates **runtime stereo camera calibration** with the `DynamicCalibration` node, plus a host-side controller/visualizer that overlays helpful UI (help panel, coverage bar, quality/recalibration modals, and a depth ROI HUD). It integrates with the [RemoteConnection](https://rvc4.docs.luxonis.com/software/depthai-components/tools/remote_connection/) service to visualize streams in real time.

> Works in **peripheral mode**: the device performs calibration; the host sends commands and renders overlays.

## Features

- **Interactive commands**: start/force recalibration, load images, run quality checks, apply/rollback calibrations, and **flash** (EEPROM) new/previous/factory calibration.
- **Coverage bar**: centered, large progress bar while collecting frames (or briefly after `l`).
- **Quality modal**: big 3-color bar (GOOD / COULD BE IMPROVED / NEEDS RECALIBRATION) with a pointer based on rotation change and a summary of depth-error deltas.
- **Recalibration modal**: summary with Euler angle deltas and depth-error deltas; prompts to flash if there is a significant change.
- **Depth HUD**: optional, shows depth/disp at a movable ROI (center + mean), with a small “tiny box” indicator.
- **Always-on help panel** (toggleable).

## Demo

<p align="center">
  <img src="media/dcl.gif" alt="demo" />
</p>

## Requirements

- A **Luxonis device** connected via USB/Ethernet.
- Python **3.10+** (tested with 3.12).
- Packages:
  - `depthai`
  - `depthai-nodes`
  - `opencv-python`
  - `numpy`

Install via:
```bash
pip install -r requirements.txt
```

## Run

```bash
python3 main.py
# or
python3 main.py --fps_limit 10
# or
python3 main.py --device 18443010C1BA9D1200
```

When launched, the app starts a RemoteConnection server. Open the visualizer at:
```
http://localhost:8082
```
Replace `localhost` with your host IP if viewing from another machine.

## Wiring Notes (snippet)

Make sure to:
1) Link a preview stream for timestamped overlays (e.g., colormapped disparity/depth),
2) Link a raw depth OR disparity stream to the controller (for the ROI HUD),
3) Provide the **device** handle to enable flashing.

```python
visualizer = dai.RemoteConnection(httpPort=8082)
device = pipeline.getDefaultDevice()

stereo = pipeline.create(dai.node.StereoDepth)
# ... set stereo configs, link left/right, etc.

# Colormap for display
depth_color = pipeline.create(ApplyColormap).build(stereo.disparity)  # or stereo.depth
depth_color.setColormap(cv2.COLORMAP_JET)

# Dynamic calibration node
dyn_calib = pipeline.create(dai.node.DynamicCalibration)
left_out.link(dyn_calib.left)
right_out.link(dyn_calib.right)

# Controller
dyn_ctrl = pipeline.create(DynamicCalibrationControler).build(
    preview=depth_color.out,     # used for overlay timing
    depth=stereo.depth           # or .disparity; call set_depth_units_is_mm(False) if disparity
)
dyn_ctrl.set_command_input(dyn_calib.inputControl.createInputQueue())
dyn_ctrl.set_quality_output(dyn_calib.qualityOutput.createOutputQueue())
dyn_ctrl.set_calibration_output(dyn_calib.calibrationOutput.createOutputQueue())
dyn_ctrl.set_coverage_output(dyn_calib.coverageOutput.createOutputQueue())
dyn_ctrl.set_device(device)  # enables flashing p/k/f
```

## Controls

Use these keys while the app is running (focus the browser visualizer window):

| Key | Action |
| --- | ------ |
| `q` | Quit the app |
| `h` | Toggle help panel |
| `g` | Toggle Depth HUD (ROI readout) |
| `r` | Start recalibration |
| `d` | **Force** recalibration |
| `l` | Load image(s) for calibration (shows coverage bar for ~2s) |
| `c` | Calibration quality check |
| `v` | **Force** calibration quality check |
| `n` | Apply **NEW** calibration (when available) |
| `o` | Apply **PREVIOUS** calibration (rollback) |
| `p` | **Flash NEW/current** calibration to EEPROM |
| `k` | **Flash PREVIOUS** calibration to EEPROM |
| `f` | **Flash FACTORY** calibration to EEPROM |
| `w / a / s` | Move ROI up/left/down (Depth HUD).<br>**Note:** `d` is reserved for *Force recalibrate*. |
| `z / x` | ROI size − / + |

> **Status banners** appear in the **center** after critical actions (e.g., applying/ flashing calibration) and auto-hide after ~2s.  
> **Modals** (quality/recalibration) also appear centered and auto-hide after ~3.5s or on any key press.

## On‑screen UI Cheat Sheet

- **Help panel** (top-left): quick reference of all keys (toggle with `h`).  
- **Coverage bar** (center): big progress bar while collecting frames; also shown briefly (≈2s) after pressing `l`.  
- **Quality modal** (center): three colored segments (green/yellow/red) with a **downward** pointer (`▼`) indicating rotation-change severity; optional line with depth-error deltas (@1m/2m/5m/10m).  
- **Recalibration modal** (center): “Recalibration complete”, significant-axis warning (if any), Euler angles, and depth-error deltas; suggests flashing if the change is significant.  
- **Depth HUD** (inline): shows depth/disp at the ROI center and mean within a tiny box; move with `w/a/s` (and resize with `z/x`).

## Output (console)

- **Coverage**: per-cell coverage and acquisition status when emitted by the device.
- **Calibration results**: prints when a new calibration is produced and shows deltas:
  - Rotation delta `|| r_current - r_new ||` in degrees,
  - Mean Sampson error (new vs. current),
  - Theoretical **Depth Error Difference** at 1/2/5/10 meters.
- **Quality checks**: same metrics as above without actually applying a new calibration.

## Tips & Notes

- To **flash** (EEPROM) from the UI you must pass the `device` into the controller (`dyn_ctrl.set_device(device)`).  
- If you link **disparity** instead of **depth** to the controller, call `dyn_ctrl.set_depth_units_is_mm(False)` so the HUD labels use “Disp” instead of meters.
- The coverage percentage accepts either `[0..1]` or `[0..100]` from the device; the controller auto-detects and normalizes.
- The **Collecting frames** bar hides automatically 2s after pressing `l`; during active recalibration (`r`/`d`) it stays up until calibration finishes.

## Installation (dev quick start)

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -U pip
pip install -r requirements.txt
python3 main.py
```

---

If you use this as a base for your own app, the heart of the UX is `utils/dynamic_controler.py` — it wires `DynamicCalibration` queues and renders all overlays via `ImgAnnotations` so you don’t need `cv2.imshow()`.