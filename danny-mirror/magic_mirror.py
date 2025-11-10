#!/usr/bin/env python3
"""
Magic Mirror Effect - Scales the video feed based on user distance to create
a realistic mirror reflection effect.

The image scales proportionally: as you move closer, your face appears larger
at the same rate it would in a real mirror.

IMAGE PROCESSING FLOW:
1. Camera captures: 640x480 (landscape - optimal for YuNet face detection)
2. Flip horizontal: mirror effect
3. Scale by distance: closer = larger (inversely proportional)
4. Scale to display: 640x480 â†’ varies (fits portrait display 2160x3840)
5. Apply offsets: align with real reflection
6. Crop to display: 2160x3840 (final output)
"""

import os
import json
import time
from dotenv import load_dotenv

import cv2
import numpy as np
import depthai as dai
from depthai_nodes.node import ApplyColormap, ParsingNeuralNetwork, ImgFrameOverlay
from depthai_nodes import ImgDetectionsExtended, ImgDetectionExtended

from utils.arguments import initialize_argparser
from utils.input import create_input_node
from utils.measure_distance import MeasureDistance, RegionOfInterest
from utils.roi_from_face import ROIFromFace


# Mouse callback for auto-calibration point picking (accepts 5 args)
def _on_mouse(event, x, y, flags, param):
    state = param if isinstance(param, dict) else None
    if state is None or not state.get("active", False):
        return
    if event == cv2.EVENT_LBUTTONDOWN:
        pts = state.setdefault("points", [])
        if len(pts) < 2:
            pts.append((int(x), int(y)))
        if len(pts) > 2:
            state["points"] = pts[-2:]

load_dotenv(override=True)

_, args = initialize_argparser()

if args.api_key:
    os.environ["DEPTHAI_HUB_API_KEY"] = args.api_key

device = dai.Device(dai.DeviceInfo(args.device)) if args.device else dai.Device()
platform = device.getPlatformAsString()
print(f"Platform: {platform}")

# Magic Mirror Configuration
REFERENCE_DISTANCE_MM = 1800  # Distance where scale = 1.0 (TRUE life-size at this distance)
MIN_DISTANCE_MM = 300        # Minimum distance to prevent extreme scaling
MAX_DISTANCE_MM = 6000       # Maximum distance to track (increased to handle far detections)
MIN_SCALE = 0.2              # Minimum allowed scale (allows smaller for far distances)
MAX_SCALE = 3.0              # Maximum allowed scale (prevents image from becoming too large)

# CRITICAL FOR TRUE 1:1 SCALING: Measure your display's physical dimensions
DISPLAY_PHYSICAL_WIDTH_MM = 620    # Physical width of display in millimeters  
DISPLAY_PHYSICAL_HEIGHT_MM = 1100   # Physical height of display in millimeters

# Scale multiplier for 1:1 life-size at reference distance
# CALIBRATED FROM IMAGE: At 1500mm with scale frozen at 1.0, image filled ~50% of screen
# This means we need roughly 2Ã— larger â†’ setting to 2.2 for safety margin
LIFE_SIZE_SCALE_MULTIPLIER = 1.55

# Camera/Display Alignment Configuration
# MEASURED: Camera is ~7cm above display top + ~3cm tape = ~100mm total
CAMERA_OFFSET_Y_MM = 0     # Camera position relative to EYE LINE (mm). +Y = camera above eyes
CAMERA_OFFSET_X_MM = 0     # Camera position relative to EYE LINE (mm). +X = camera to your right

# Parallax gain for camera-to-eye baseline compensation.
# Based on your observation, doubling the effect aligns the image correctly.
# This gain scales how CAMERA_OFFSET_* translates into pixel parallax.
CAMERA_PARALLAX_GAIN = 2.0

# Optional: Camera mount relative to DISPLAY CENTER (mm). +Y = camera above display center, +X = camera right of center
# If you measure these and set them (or provide in calibration.json), the code will convert to pixels
# and add as a static image offset automatically, so you don't have to hand-tune IMAGE_OFFSET_*.
DISPLAY_CAMERA_OFFSET_X_MM = 0
DISPLAY_CAMERA_OFFSET_Y_MM = 0

# Calibration factor: Measured from ruler in image
# 22 inches (558.8mm) vertical ruler spans roughly 1700 pixels
# Actual pixels/mm â‰ˆ 3.04 (vs theoretical 3.49 from 3840/1100)
PIXELS_PER_MM = 3.04         

# Depth clipping / viewer band configuration
DEPTH_CLIP_MAX_MM = 3000  # legacy single-threshold clip (unused in viewer band mode)
VIEWER_BAND_TOL_MM = 200  # when enabled, keep only pixels within +/- this tolerance of viewer depth

# Lateral follow configuration (viewer moves left/right)
LATERAL_FOLLOW_GAIN = 1.0

# Vertical follow configuration (viewer moves up/down)
VERTICAL_FOLLOW_GAIN = 1.0

# Alignment target (relative screen position)
# X is fraction of width; Y is fraction of height from top.
# Portrait mode: place target higher on the screen so it's over content.
ALIGN_TARGET_X_REL = 0.5   # center horizontally
ALIGN_TARGET_Y_REL = 0.25  # 25% from top ("75% up" from bottom)


 

# Optional calibrated distance scaling model: s(z) = 1 + k*(r-1) + q*(r-1)^2
# where r = REFERENCE_DISTANCE_MM / z. Loaded from calibration.json if present.
SCALE_MODEL_K = None
SCALE_MODEL_Q = None
# Default persisted tuning values (can be overridden by CLI)
PARALLAX_DISTANCE_EXP_DEFAULT = 1.0
try:
    _cal_path = os.path.join(os.path.dirname(__file__), "calibration.json")
    if os.path.exists(_cal_path):
        with open(_cal_path, "r", encoding="utf-8") as _f:
            _cal = json.load(_f)
        if isinstance(_cal, dict):
            if "scale_model_k" in _cal:
                SCALE_MODEL_K = float(_cal["scale_model_k"])  # may raise
            if "scale_model_q" in _cal:
                SCALE_MODEL_Q = float(_cal["scale_model_q"])  # may raise
            # Load persisted offsets if present
            if "CAMERA_OFFSET_Y_MM" in _cal:
                CAMERA_OFFSET_Y_MM = int(_cal["CAMERA_OFFSET_Y_MM"])  # type: ignore[name-defined]
            if "CAMERA_OFFSET_X_MM" in _cal:
                CAMERA_OFFSET_X_MM = int(_cal["CAMERA_OFFSET_X_MM"])  # type: ignore[name-defined]
            if "IMAGE_OFFSET_X" in _cal:
                IMAGE_OFFSET_X = int(_cal["IMAGE_OFFSET_X"])  # type: ignore[name-defined]
            if "IMAGE_OFFSET_Y" in _cal:
                IMAGE_OFFSET_Y = int(_cal["IMAGE_OFFSET_Y"])  # type: ignore[name-defined]
            if "REFERENCE_DISTANCE_MM" in _cal:
                REFERENCE_DISTANCE_MM = int(_cal["REFERENCE_DISTANCE_MM"])  # type: ignore[name-defined]
            if "LIFE_SIZE_SCALE_MULTIPLIER" in _cal:
                LIFE_SIZE_SCALE_MULTIPLIER = float(_cal["LIFE_SIZE_SCALE_MULTIPLIER"])  # type: ignore[name-defined]
            # Load optional display-center-to-camera offsets (mm)
            if "DISPLAY_CAMERA_OFFSET_X_MM" in _cal:
                DISPLAY_CAMERA_OFFSET_X_MM = int(_cal["DISPLAY_CAMERA_OFFSET_X_MM"])  # type: ignore[name-defined]
            if "DISPLAY_CAMERA_OFFSET_Y_MM" in _cal:
                DISPLAY_CAMERA_OFFSET_Y_MM = int(_cal["DISPLAY_CAMERA_OFFSET_Y_MM"])  # type: ignore[name-defined]
            # Load parallax exponent default if present
            if "parallax_distance_scale" in _cal:
                PARALLAX_DISTANCE_EXP_DEFAULT = float(_cal["parallax_distance_scale"])  # type: ignore[assignment]
            # Optional override for camera parallax gain
            if "CAMERA_PARALLAX_GAIN" in _cal:
                CAMERA_PARALLAX_GAIN = float(_cal["CAMERA_PARALLAX_GAIN"])  # type: ignore[name-defined]
            # Optional alignment target overrides
            if "ALIGN_TARGET_X_REL" in _cal:
                ALIGN_TARGET_X_REL = float(_cal["ALIGN_TARGET_X_REL"])  # type: ignore[name-defined]
            if "ALIGN_TARGET_Y_REL" in _cal:
                ALIGN_TARGET_Y_REL = float(_cal["ALIGN_TARGET_Y_REL"])  # type: ignore[name-defined]
            
            # Optional gain for vertical follow
            if "VERTICAL_FOLLOW_GAIN" in _cal:
                VERTICAL_FOLLOW_GAIN = float(_cal["VERTICAL_FOLLOW_GAIN"])  # type: ignore[name-defined]
except Exception:
    pass

# Log current display-camera offsets so theyâ€™re visible at startup
try:
    print(
        f"DisplayCamera offset (mm): X={DISPLAY_CAMERA_OFFSET_X_MM}, Y={DISPLAY_CAMERA_OFFSET_Y_MM}"
    )
except Exception:
    pass

# Fine-tune offset (in pixels) - use arrow keys during calibration to adjust
IMAGE_OFFSET_X = 0           # Additional shift left (-) or right (+)
IMAGE_OFFSET_Y = 0          # Shift image up (-) or down (+)  - POSITIVE moves display DOWN

with dai.Pipeline(device) as pipeline:
    print("Creating Magic Mirror pipeline...")

    # RGB input for NN (camera by default, or media if provided)
    # Use 640x480 to match YuNet model blob
    VIDEO_RESOLUTION = (640, 480)
    if args.media_path:
        input_node = create_input_node(
            pipeline,
            platform,
            args.media_path,
        )
        use_camera = False
    else:
        color = pipeline.create(dai.node.Camera).build(dai.CameraBoardSocket.CAM_A)
        frame_type = (
            dai.ImgFrame.Type.BGR888p if platform == "RVC2" else dai.ImgFrame.Type.BGR888i
        )
        input_node = color.requestOutput(
            VIDEO_RESOLUTION, frame_type, fps=args.fps_limit
        )
        use_camera = True

    # Load model from Zoo - use YuNet face detection (smaller, uses fewer shaves)
    model_description = dai.NNModelDescription(f"yunet_n_640x640.{platform}.yaml")
    if model_description.model != args.model:
        model_description = dai.NNModelDescription(args.model, platform=platform)
    nn_archive = dai.NNArchive(dai.getModelFromZoo(model_description))

    # NN with parser
    nn_with_parser: ParsingNeuralNetwork = pipeline.create(ParsingNeuralNetwork).build(
        input_node, nn_archive, fps=args.fps_limit
    )

    # Prefer dropping frames to reduce latency
    try:
        nn_with_parser.input.setMaxSize(1)
        nn_with_parser.input.setBlocking(False)
        nn_with_parser.setNumPoolFrames(4)
    except Exception:
        pass

    # Hint the runtime to use fewer SHAVEs if the blob default is too high for this pipeline
    try:
        nn_with_parser.setNNArchive(nn_archive, numShaves=7)
    except Exception:
        # If the API isn't available on this node version, we'll rely on resource allocation below
        pass

    # Stereo depth for distance measurement
    monoLeft = (
        pipeline.create(dai.node.Camera)
        .build(dai.CameraBoardSocket.CAM_B)
        .requestOutput((640, 400), type=dai.ImgFrame.Type.NV12)
    )
    monoRight = (
        pipeline.create(dai.node.Camera)
        .build(dai.CameraBoardSocket.CAM_C)
        .requestOutput((640, 400), type=dai.ImgFrame.Type.NV12)
    )

    stereo = pipeline.create(dai.node.StereoDepth).build(
        monoLeft, monoRight, presetMode=dai.node.StereoDepth.PresetMode.DEFAULT
    )
    
    # Reduce StereoDepth SHAVE usage to make room for NN
    try:
        stereo.setNumShaves(2)  # Reduce from default 4 to 2
    except Exception:
        pass
    
    # Align depth to RGB camera
    try:
        stereo.setDepthAlign(dai.CameraBoardSocket.CAM_A)
        if use_camera and platform == "RVC2":
            stereo.setOutputSize(*VIDEO_RESOLUTION)
    except Exception:
        pass

    depth_color_transform = pipeline.create(ApplyColormap).build(stereo.disparity)
    depth_color_transform.setColormap(cv2.COLORMAP_JET)

    # Distance measurement
    initial_roi = RegionOfInterest(300, 150, 340, 190)
    measure_distance = pipeline.create(MeasureDistance).build(
        stereo.depth, device.readCalibration(), initial_roi
    )

    # Auto-update ROI from face detections
    roi_from_face = pipeline.create(ROIFromFace).build(
        disparity_frames=depth_color_transform.out,
        parser_output=nn_with_parser.out,
        use_eye_roi=args.eye_roi,
    )
    roi_from_face.output_roi.link(measure_distance.roi_input)

    # Create output queues for host-side processing
    video_queue = nn_with_parser.passthrough.createOutputQueue(maxSize=4, blocking=False)
    detections_queue = nn_with_parser.out.createOutputQueue(maxSize=4, blocking=False)
    distance_queue = measure_distance.output.createOutputQueue(maxSize=4, blocking=False)
    depth_vis_queue = depth_color_transform.out.createOutputQueue(maxSize=2, blocking=False)
    depth_raw_queue = stereo.depth.createOutputQueue(maxSize=2, blocking=False)

    print("Pipeline created. Starting Magic Mirror...")
    pipeline.start()

    # OpenCV window - setup for second monitor
    window_name = "Magic Mirror"
    cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
    # Auto-calibrate (15cm) mouse state
    mouse_state = {"active": False, "points": []}
    cv2.setMouseCallback(window_name, _on_mouse, mouse_state)
    
    # Get display size - default to portrait 4K
    DISPLAY_WIDTH = 2160   # Portrait width
    DISPLAY_HEIGHT = 3840  # Portrait height
    
    # Try to detect and select the desired monitor
    current_monitor_index = 0
    try:
        import screeninfo
        screens = screeninfo.get_monitors()

        def choose_monitor():
            # 1) Match by name/device substring if provided
            if getattr(args, "monitor_name", None):
                name_query = str(args.monitor_name).lower()
                for i, s in enumerate(screens):
                    name = getattr(s, "name", None) or getattr(s, "device", None) or ""
                    if name_query in str(name).lower():
                        return i
            # 2) Use explicit index if provided and valid
            if getattr(args, "display_index", None) is not None:
                idx = int(args.display_index)
                if 0 <= idx < len(screens):
                    return idx
            # 3) Default: prefer portrait monitors (largest portrait by area) if any exist
            portraits = [i for i, s in enumerate(screens) if s.height > s.width]
            if portraits:
                return max(portraits, key=lambda i: screens[i].width * screens[i].height)
            # 4) Fallback: choose the largest monitor by area (landscape-only setups)
            return max(range(len(screens)), key=lambda i: screens[i].width * screens[i].height) if screens else 0

        if screens:
            current_monitor_index = choose_monitor()
            target = screens[current_monitor_index]
            DISPLAY_WIDTH = target.width
            DISPLAY_HEIGHT = target.height
            print(
                f"Using monitor {current_monitor_index}: {DISPLAY_WIDTH}x{DISPLAY_HEIGHT} at ({target.x}, {target.y})"
            )
            cv2.moveWindow(window_name, target.x, target.y)
            if not getattr(args, "windowed", False):
                cv2.setWindowProperty(
                    window_name, cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN
                )
        else:
            print("No monitors detected by screeninfo; using defaults on primary screen")
    except ImportError:
        print("screeninfo not installed. Install with: pip install screeninfo")
        print("Window will open on primary monitor")
    except Exception as e:
        print(f"Could not auto-detect monitor: {e}")
    
    print(f"Display output size: {DISPLAY_WIDTH}x{DISPLAY_HEIGHT}")
    
    # Manual fullscreen toggle (for single monitor or if auto-detect fails):
    # cv2.setWindowProperty(window_name, cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)

    # For FPS calculation
    fps_counter = 0
    fps_start_time = time.time()
    current_fps = 0
    
    # Enhanced smoothing for scale changes (avoid jitter)
    current_scale = 1.0
    SMOOTHING_FACTOR = 0.15  # Even faster response to recover from bad readings
    
    # Moving average filter for depth - reduced for faster response
    from collections import deque
    depth_history = deque(maxlen=5)  # Keep only last 3 readings for faster response

    # Robust median history for outlier rejection
    z_median_history = deque(maxlen=15)
    Z_OUTLIER_MM = 300.0
    Z_OUTLIER_RATIO = 1.25
    # Keep track of last valid distance - start at reference so image stays at 1.0 scale initially
    # Relative deadband to ignore tiny scale changes
    SCALE_DEADBAND_REL = 0.02
    # Only allow decreases when large and sustained
    DECREASE_HYSTERESIS_REL = 0.04
    DECREASE_HYSTERESIS_FRAMES = 3
    last_valid_distance = REFERENCE_DISTANCE_MM
    frames_without_detection = 0
    MAX_FRAMES_WITHOUT_DETECTION = 999999  # Keep scale indefinitely until face detected
    
    # Smooth position offsets to prevent jitter
    current_offset_x = 0.0
    current_offset_y = 0.0
    OFFSET_SMOOTHING = 0.3  # Faster response
    
    # Store recent detections for visualization (lingering circles)
    from collections import deque
    recent_detections = deque(maxlen=90)  # Keep last 90 frames (3 seconds at 30fps) for longer visibility
    
    # Live-tunable parameters (ensure defined for auto-calibration and UI)
    # Latest raw depth frame (uint16 mm) for direct sampling
    last_depth_raw = None
    try:
        distance_scale_weight_live = float(getattr(args, "distance_scale_weight", 0.3))
    except Exception:
        distance_scale_weight_live = 0.3
    try:
        parallax_weight_x_live = float(getattr(args, "parallax_weight_x", 1.0))
    except Exception:
        parallax_weight_x_live = 0.0
    try:
        parallax_weight_y_live = float(getattr(args, "parallax_weight_y", 1.0))
    except Exception:
        parallax_weight_y_live = 0.0
    try:
        parallax_distance_exp_live = float(getattr(args, "parallax_distance_scale", PARALLAX_DISTANCE_EXP_DEFAULT))
    except Exception:
        parallax_distance_exp_live = PARALLAX_DISTANCE_EXP_DEFAULT
    use_world_parallax = True
    show_ruler = False
    show_content_ruler = False
    lock_mode = False
    show_depth_inset = False
    # Hysteresis counter for allowing decreases
    decrease_pending_frames = 0
    show_depth_inset = False
    # Throttled console prints for fixed ROI depth sanity check
    last_cam_roi = None
    last_fixed_roi_print_z = None
    last_fixed_roi_print_time = 0.0
    last_raw_z = None
    last_clamped_z = None
    
    # Calibration mode
    calibration_mode = False
    calibration_step = 0  # 0=not started, 1=set reference distance, 2=adjust offset
    saved_distance = None
    fixed_roi_mode = False  # Toggle for using fixed center ROI instead of face tracking
    freeze_scale = False  # Toggle to lock scale at 1.0 for calibration
    # Auto-calibrate 150mm state
    auto_calibrate_mode = False
    auto_saved_state = None
    # Manual ROI selection for calibration (non-face targets)
    manual_roi_select_mode = False
    manual_roi_points = []  # two display-space points
    manual_roi_active = False
    # Multi-depth calibration wizard state
    multi_cal_mode = False
    multi_samples = []  # list of dicts: {"z": z_mm, "measured_px": px, "axis": "x"|"y"}
    collecting_sample = False
    # Persistent depth readout (set by 'y' hotkey)
    stored_depth_display_mm = None
    last_face_center_cam = None  # (x,y) in camera coords (pre-flip)
    last_face_conf_seen = 0.0
    last_face_conf_time = 0.0
    # Lateral/Vertical follow state
    viewer_x_ema = None  # mm (legacy EMA, kept for reference)
    viewer_x_last_time = 0.0
    viewer_x_locked = None  # mm (last known good; sticky)
    viewer_x_locked_time = 0.0
    viewer_y_locked = None  # mm (last known good; sticky)
    viewer_y_locked_time = 0.0
    # Layout cache for mapping camera coords -> display coords (used by auto alignment calib)
    last_final_w = None
    last_final_h = None
    last_src_x_start = 0
    last_src_y_start = 0
    last_dst_x_start = 0
    last_dst_y_start = 0
    # Two-point baseline calibration state (solves CAMERA_OFFSET_X/Y_MM)
    baseline_cal_mode = False
    baseline_samples = []  # list of dicts with keys: z, dx, dy, p
    baseline_prev_freeze = None
    show_depth_diag = False  # toggle to visualize ROI Z vs raw-face Z
    # Depth clip state (B toggle)
    depth_clip_enabled = False
    viewer_band_target_z = None
    viewer_band_last_time = 0.0
    
    print("\n=== MAGIC MIRROR CALIBRATION ===")
    print("Press 'C' to enter calibration mode")
    print("Press 'X' to toggle Fixed ROI mode (for phone camera calibration)")
    print("Press 'Z' to toggle FREEZE SCALE at 1.0 (for calibration)")
    print("Press 'Q' to quit")
    print("================================\n")
    
    while pipeline.isRunning():
        # Override ROI with fixed center position if in fixed mode
        if fixed_roi_mode:
            # Send fixed center ROI to measure_distance (for phone camera calibration)
            # Use center 25% of frame
            center_roi = RegionOfInterest(240, 180, 400, 300)  # Center region of 640x480
            measure_distance.roi_input.send(center_roi)
        
        # Get video frame
        video_msg = video_queue.tryGet()
        det_msg = detections_queue.tryGet()
        distance_msg = distance_queue.tryGet()
        
        # Capture detections for visualization (before any transforms)
        last_face_conf_max = 0.0
        if det_msg is not None:
            # Store detection info for lingering visualization
            for detection in det_msg.detections:
                try:
                    # Filter by confidence - only keep high-confidence detections (>75%)
                    confidence = detection.confidence if hasattr(detection, 'confidence') else 1.0
                    if confidence > last_face_conf_max:
                        last_face_conf_max = float(confidence)
                    if confidence < 0.75:
                        continue  # Skip low-confidence detections (false positives)
                    
                    # YuNet returns keypoints (eyes, nose, mouth corners) instead of bbox
                    if hasattr(detection, 'keypoints') and len(detection.keypoints) > 0:
                        # Keypoints are normalized [0-1], need to denormalize to 640x480
                        keypoints = []
                        for kp in detection.keypoints:
                            x = kp.x * 640  # Denormalize to camera resolution
                            y = kp.y * 480
                            keypoints.append((x, y))
                        
                        # Calculate bounding box from keypoints
                        xs = [kp[0] for kp in keypoints]
                        ys = [kp[1] for kp in keypoints]
                        x1, x2 = min(xs), max(xs)
                        y1, y2 = min(ys), max(ys)
                        
                        # Add padding around keypoints to encompass full face
                        width = x2 - x1
                        height = y2 - y1
                        padding = max(width, height) * 0.3  # 30% padding
                        
                        center_x = (x1 + x2) / 2
                        center_y = (y1 + y2) / 2
                        radius = max(width, height) / 2 + padding
                        # Track last face center in camera/depth coords for direct depth sampling
                        last_face_center_cam = (int(round(center_x)), int(round(center_y)))
                        # Persist last confident face time (>=0.8) for Y hotkey gating
                        try:
                            if float(confidence) >= 0.8:
                                last_face_conf_seen = float(confidence)
                                last_face_conf_time = time.time()
                        except Exception:
                            pass
                        
                        recent_detections.append({
                            'center_x': center_x,
                            'center_y': center_y,
                            'radius': radius,
                            'confidence': detection.confidence if hasattr(detection, 'confidence') else 1.0
                        })
                        
                except Exception as e:
                    # Skip detections that fail to process
                    pass
                    pass
        
        if video_msg is not None:
            # Convert to OpenCV format
            frame = video_msg.getCvFrame()
            # Reset per-frame diagnostics
            z_face_raw = None
            # Update latest raw depth frame for host-side sampling
            try:
                depth_raw_msg = depth_raw_queue.tryGet()
                if depth_raw_msg is not None:
                    last_depth_raw = depth_raw_msg.getFrame()
            except Exception:
                pass
            
            # Camera is landscape orientation (correct for face detection)
            # Apply horizontal flip for mirror effect
            frame = cv2.flip(frame, 1)
            
            # Get original frame dimensions (640w x 480h landscape)
            orig_h, orig_w = frame.shape[:2]

            # Compute raw-face depth (median around last face center in raw depth)
            try:
                if last_face_center_cam is not None and last_depth_raw is not None and orig_w > 0 and orig_h > 0:
                    cx_cam, cy_cam = last_face_center_cam
                    dh, dw = last_depth_raw.shape[:2]
                    scale_x = dw / float(orig_w)
                    scale_y = dh / float(orig_h)
                    sx = int(round(cx_cam * scale_x))
                    sy = int(round(cy_cam * scale_y))
                    rwin = 6
                    x1s = max(0, sx - rwin)
                    x2s = min(dw - 1, sx + rwin)
                    y1s = max(0, sy - rwin)
                    y2s = min(dh - 1, sy + rwin)
                    win = last_depth_raw[y1s:y2s+1, x1s:x2s+1].astype(np.float32)
                    valid = (win >= MIN_DISTANCE_MM) & (win <= MAX_DISTANCE_MM)
                    if np.any(valid):
                        z_face_raw = float(np.median(win[valid]))
            except Exception:
                pass

            # Calculate scale based on depth FIRST (before any resizing)
            target_scale = current_scale  # Default to current scale
            
            if (z_face_raw is not None and z_face_raw > 0 and not np.isnan(z_face_raw)) or (distance_msg is not None):
                if z_face_raw is not None and z_face_raw > 0 and not np.isnan(z_face_raw):
                    z_mm = float(z_face_raw)
                    z_raw_for_print = z_mm
                else:
                    z_mm = distance_msg.spatials.z
                    z_raw_for_print = z_mm
                # Track centroid for mapping seed (in depth/camera space 640x480, pre-flip)
                
                
                # Only use valid depth readings (not NaN or 0)
                if z_mm > 0 and not np.isnan(z_mm):
                    # Robust outlier rejection using running median
                    candidate_z = float(z_mm)
                    if len(z_median_history) > 3:
                        med = float(np.median(z_median_history))
                        if med > 0:
                            if abs(candidate_z - med) > Z_OUTLIER_MM or (candidate_z > med * Z_OUTLIER_RATIO or candidate_z < med / Z_OUTLIER_RATIO):
                                candidate_z = med
                    z_median_history.append(candidate_z)
                    depth_history.append(candidate_z)
                    z_mm = sum(depth_history) / len(depth_history)
                    
                    # Clamp distance to reasonable range
                    z_mm = max(MIN_DISTANCE_MM, min(MAX_DISTANCE_MM, z_mm))
                    # Record latest raw/clamped values for overlays (for both fixed/manual ROI)
                    last_raw_z = z_raw_for_print
                    last_clamped_z = z_mm
                    
                    # Update last valid distance
                    last_valid_distance = z_mm
                    frames_without_detection = 0
                    
                    # Scale vs distance (unless frozen)
                    if not freeze_scale:
                        r = REFERENCE_DISTANCE_MM / z_mm
                        if SCALE_MODEL_K is not None or SCALE_MODEL_Q is not None:
                            k = SCALE_MODEL_K or 0.0
                            q = SCALE_MODEL_Q or 0.0
                            target_scale = 1.0 + k * (r - 1.0) + q * ((r - 1.0) ** 2)
                        else:
                            # Fallback to simple inverse
                            target_scale = r
                        # Clamp scale to prevent extreme values
                        target_scale = max(MIN_SCALE, min(MAX_SCALE, float(target_scale)))
                        if target_scale > current_scale:
                            max_step_up = 0.03
                            target_scale = min(current_scale + max_step_up, target_scale)
                        else:
                            max_step_down = 0.02
                            target_scale = max(current_scale - max_step_down, target_scale)

                    # Print raw vs smoothed depth when fixed ROI mode is active (sanity check)
                    if fixed_roi_mode:
                        now_ts = time.time()
                        if (
                            last_fixed_roi_print_z is None
                            or abs(z_raw_for_print - last_fixed_roi_print_z) >= 10
                            or (now_ts - last_fixed_roi_print_time) > 0.5
                        ):
                            try:
                                print(f"[Fixed ROI] RawZ:{z_raw_for_print:.0f} mm  SmoothedZ:{z_mm:.0f} mm")
                            except Exception:
                                pass
                            last_fixed_roi_print_z = z_raw_for_print
                            last_fixed_roi_print_time = now_ts
                            last_raw_z = z_raw_for_print
                            last_clamped_z = z_mm
                    else:
                        # Scale frozen at 1.0 for calibration
                        target_scale = 1.0
                else:
                    # Invalid depth, use last known distance
                    frames_without_detection += 1
                    if frames_without_detection < MAX_FRAMES_WITHOUT_DETECTION:
                        if not freeze_scale:
                            r = REFERENCE_DISTANCE_MM / last_valid_distance
                            if SCALE_MODEL_K is not None or SCALE_MODEL_Q is not None:
                                k = SCALE_MODEL_K or 0.0
                                q = SCALE_MODEL_Q or 0.0
                                target_scale = 1.0 + k * (r - 1.0) + q * ((r - 1.0) ** 2)
                            else:
                                target_scale = r
                            target_scale = max(MIN_SCALE, min(MAX_SCALE, float(target_scale)))
                        if target_scale > current_scale:
                            max_step_up = 0.03
                            target_scale = min(current_scale + max_step_up, target_scale)
                        else:
                            max_step_down = 0.02
                            target_scale = max(current_scale - max_step_down, target_scale)
                        if freeze_scale:
                            target_scale = 1.0
            else:
                # No distance message, use last known distance
                frames_without_detection += 1
                if frames_without_detection < MAX_FRAMES_WITHOUT_DETECTION:
                    if not freeze_scale:
                        r = REFERENCE_DISTANCE_MM / last_valid_distance
                        if SCALE_MODEL_K is not None or SCALE_MODEL_Q is not None:
                            k = SCALE_MODEL_K or 0.0
                            q = SCALE_MODEL_Q or 0.0
                            target_scale = 1.0 + k * (r - 1.0) + q * ((r - 1.0) ** 2)
                        else:
                            target_scale = r
                        target_scale = max(MIN_SCALE, min(MAX_SCALE, float(target_scale)))
                        if target_scale > current_scale:
                            max_step_up = 0.03
                            target_scale = min(current_scale + max_step_up, target_scale)
                        else:
                            max_step_down = 0.02
                            target_scale = max(current_scale - max_step_down, target_scale)
                    else:
                        target_scale = 1.0
            
            # Apply deadband and asymmetric hysteresis before smoothing
            try:
                rel = abs(target_scale - current_scale) / max(current_scale, 1e-6)
            except Exception:
                rel = 0.0
            if rel < SCALE_DEADBAND_REL:
                target_scale = current_scale

            if target_scale < current_scale:
                rel_dec = (current_scale - target_scale) / max(current_scale, 1e-6)
                if rel_dec >= DECREASE_HYSTERESIS_REL:
                    decrease_pending_frames += 1
                else:
                    decrease_pending_frames = 0
                if decrease_pending_frames < DECREASE_HYSTERESIS_FRAMES:
                    target_scale = current_scale
            else:
                decrease_pending_frames = 0

            # Smooth the scale transition
            # Apply deadband to avoid micro jitter
            try:
                rel = abs(target_scale - current_scale) / max(current_scale, 1e-6)
            except Exception:
                rel = 0.0
            if rel < SCALE_DEADBAND_REL:
                target_scale = current_scale
            current_scale += (target_scale - current_scale) * SMOOTHING_FACTOR
            
            # === SIMPLIFIED TRUE 1:1 SCALING ===
            # Goal: At reference distance, make face life-size (1:1)
            # 
            # Calculate pixels per mm on the display
            pixels_per_mm_x = DISPLAY_WIDTH / DISPLAY_PHYSICAL_WIDTH_MM
            pixels_per_mm_y = DISPLAY_HEIGHT / DISPLAY_PHYSICAL_HEIGHT_MM
            avg_pixels_per_mm = (pixels_per_mm_x + pixels_per_mm_y) / 2
            
            # Base scale: fill a reasonable portion of the display
            # Use the smaller dimension to ensure we don't crop too much
            base_scale = min(DISPLAY_WIDTH / orig_w, DISPLAY_HEIGHT / orig_h) * LIFE_SIZE_SCALE_MULTIPLIER
            
            # Apply distance-based scaling
            # At reference distance: current_scale = 1.0 â†’ use base_scale (calibrated for 1:1)
            # Closer/farther: scales proportionally
            combined_scale = base_scale * current_scale
            
            final_w = int(orig_w * combined_scale)
            final_h = int(orig_h * combined_scale)
            frame = cv2.resize(frame, (final_w, final_h), interpolation=cv2.INTER_LINEAR)
            
            # === CORRECT MIRROR GEOMETRY ===
            # The key insight: Camera position relative to YOUR EYES, not mirror center!
            #
            # When you look in a real mirror:
            # - Your eyes see the reflection at YOUR eye level
            # - Camera is mounted above/below your eyes (typically above)
            # - Camera sees you from that elevated/lowered angle
            #
            # To simulate a real mirror reflection:
            # - If camera is ABOVE your eyes â†’ shift image UP (so it appears at eye level)
            # - If camera is BELOW your eyes â†’ shift image DOWN
            # - The offset scales with distance (closer = bigger offset needed)
            #
            # Example: Camera 150mm above your eyes, you're 500mm away
            # - At 500mm: offset = baseline
            # - At 250mm (closer): offset doubles (parallax effect stronger)
            # - At 1000mm (farther): offset halves (parallax effect weaker)
            
            # Start from static image offset plus static display-center-to-camera offset converted to pixels
            # Use axis-specific pixels-per-mm (portrait/landscape-safe)
            ppmm_x_disp = float(DISPLAY_WIDTH) / max(1e-6, float(DISPLAY_PHYSICAL_WIDTH_MM))
            ppmm_y_disp = float(DISPLAY_HEIGHT) / max(1e-6, float(DISPLAY_PHYSICAL_HEIGHT_MM))
            static_disp_x = -float(DISPLAY_CAMERA_OFFSET_X_MM) * float(ppmm_x_disp)
            static_disp_y = -float(DISPLAY_CAMERA_OFFSET_Y_MM) * float(ppmm_y_disp)
            target_offset_x = float(IMAGE_OFFSET_X) + static_disp_x
            target_offset_y = float(IMAGE_OFFSET_Y) + static_disp_y
            
            if last_valid_distance > 0:
                # Dynamic parallax from calibrated camera-vs-eye baseline (scaled by CAMERA_PARALLAX_GAIN)
                # Use axis-specific px/mm to avoid anisotropy
                base_y_px = -float(CAMERA_PARALLAX_GAIN) * float(CAMERA_OFFSET_Y_MM) * float(ppmm_y_disp)
                base_x_px = -float(CAMERA_PARALLAX_GAIN) * float(CAMERA_OFFSET_X_MM) * float(ppmm_x_disp)
                if freeze_scale:
                    parallax_scale = 0.0
                else:
                    try:
                        r_parallax = float(REFERENCE_DISTANCE_MM) / max(1e-6, float(last_valid_distance))
                    except Exception:
                        r_parallax = 1.0
                    try:
                        parallax_scale = float(r_parallax) ** float(parallax_distance_exp_live)
                    except Exception:
                        parallax_scale = r_parallax
                    parallax_scale = max(0.0, min(3.0, parallax_scale))
                target_offset_x += base_x_px * parallax_scale
                target_offset_y += base_y_px * parallax_scale

                # Lateral follow: make image follow viewer left/right (mirror-equivalent)
                try:
                    if False and not freeze_scale and distance_msg is not None:
                        viewer_x_mm = float(getattr(distance_msg.spatials, "x", 0.0))
                        # Mirror-equivalent screen shift (mm) ≈ 0.5 * viewer_x_mm
                        screen_shift_mm = 0.5 * viewer_x_mm * float(LATERAL_FOLLOW_GAIN)
                        ppmm_x_disp = float(DISPLAY_WIDTH) / max(1e-6, float(DISPLAY_PHYSICAL_WIDTH_MM))
                        target_offset_x += screen_shift_mm * ppmm_x_disp
                except Exception:
                    pass

                # Lateral follow using last-known-good position (sticky) with sanity gating
                try:
                    now_ts = time.time()
                    ppmm_x_disp = float(DISPLAY_WIDTH) / max(1e-6, float(DISPLAY_PHYSICAL_WIDTH_MM))
                    recent_face_ok = (last_face_conf_seen >= 0.8) and ((now_ts - float(last_face_conf_time)) <= 1.0)
                    vx = None
                    if not freeze_scale and distance_msg is not None and hasattr(distance_msg, "spatials"):
                        _vx = float(getattr(distance_msg.spatials, "x", float("nan")))  # mm
                        if np.isfinite(_vx):
                            vx = _vx

                    # Update the locked position only on recent, confident face + valid depth
                    if recent_face_ok and (vx is not None):
                        if viewer_x_locked is None:
                            viewer_x_locked = vx
                        else:
                            # Deadband to ignore small jitters
                            if abs(vx - viewer_x_locked) < 4.0:
                                pass
                            else:
                                # Limit per-frame step to avoid jumps
                                max_step_mm = 100.0
                                delta = max(-max_step_mm, min(max_step_mm, vx - viewer_x_locked))
                                viewer_x_locked += delta
                        viewer_x_locked_time = now_ts

                    # If we have a lock, use it; decay gently to center if stale
                    if viewer_x_locked is not None:
                        if (now_ts - viewer_x_locked_time) > 0.8:
                            viewer_x_locked *= 0.98
                            # Snap to zero if extremely small to avoid drift
                            if abs(viewer_x_locked) < 1.0:
                                viewer_x_locked = 0.0
                        screen_shift_mm = 0.5 * float(viewer_x_locked) * float(LATERAL_FOLLOW_GAIN)
                        # Flip sign because the display feed is mirrored (cv2.flip(...,1))
                        # Positive camera X (viewer moves right) should shift image left in buffer coords
                        target_offset_x -= screen_shift_mm * ppmm_x_disp
                except Exception:
                    pass

                # Vertical follow using last-known-good position (sticky) with sanity gating
                try:
                    now_ts = time.time()
                    ppmm_y_disp = float(DISPLAY_HEIGHT) / max(1e-6, float(DISPLAY_PHYSICAL_HEIGHT_MM))
                    recent_face_ok = (last_face_conf_seen >= 0.8) and ((now_ts - float(last_face_conf_time)) <= 1.0)
                    vy = None
                    if not freeze_scale and distance_msg is not None and hasattr(distance_msg, "spatials"):
                        _vy = float(getattr(distance_msg.spatials, "y", float("nan")))  # mm
                        if np.isfinite(_vy):
                            vy = _vy

                    # Update the locked position only on recent, confident face + valid depth
                    if recent_face_ok and (vy is not None):
                        if viewer_y_locked is None:
                            viewer_y_locked = vy
                        else:
                            # Deadband to ignore small jitters
                            if abs(vy - viewer_y_locked) < 4.0:
                                pass
                            else:
                                # Limit per-frame step to avoid jumps
                                max_step_mm = 100.0
                                delta = max(-max_step_mm, min(max_step_mm, vy - viewer_y_locked))
                                viewer_y_locked += delta
                        viewer_y_locked_time = now_ts

                    # If we have a lock, use it; decay gently to center if stale
                    if viewer_y_locked is not None:
                        if (now_ts - viewer_y_locked_time) > 0.8:
                            viewer_y_locked *= 0.98
                            if abs(viewer_y_locked) < 1.0:
                                viewer_y_locked = 0.0
                        screen_shift_mm_y = 0.5 * float(viewer_y_locked) * float(VERTICAL_FOLLOW_GAIN)
                        # Mirror view: align perceived motion. In practice, invert sign to match on-screen reflection.
                        # Positive camera Y (viewer moves down) should shift image up in buffer coords.
                        target_offset_y -= screen_shift_mm_y * ppmm_y_disp
                except Exception:
                    pass

            
            # Clamp target offsets to prevent extreme values
            MAX_OFFSET = DISPLAY_HEIGHT * 2
            target_offset_x = max(-MAX_OFFSET, min(MAX_OFFSET, target_offset_x))
            target_offset_y = max(-MAX_OFFSET, min(MAX_OFFSET, target_offset_y))
            
            # Smooth the offset transitions to prevent jitter
            current_offset_x += (target_offset_x - current_offset_x) * OFFSET_SMOOTHING
            current_offset_y += (target_offset_y - current_offset_y) * OFFSET_SMOOTHING
            
            # Convert to integers for pixel operations
            pixel_offset_x = int(current_offset_x)
            pixel_offset_y = int(current_offset_y)
            
            # Create output canvas at display size
            output_frame = np.zeros((DISPLAY_HEIGHT, DISPLAY_WIDTH, 3), dtype=np.uint8)
            
            # Calculate where to place the frame on the canvas
            # Start with centering the frame
            current_h, current_w = frame.shape[:2]
            
            # Center position
            paste_y = (DISPLAY_HEIGHT - current_h) // 2
            paste_x = (DISPLAY_WIDTH - current_w) // 2
            
            # Apply offset (negative Y = shift up, positive Y = shift down)
            paste_y += pixel_offset_y
            paste_x += pixel_offset_x
            
            # Calculate what part of the frame to copy and where to paste it
            # Source coordinates (from frame)
            src_y_start = 0
            src_y_end = current_h
            src_x_start = 0
            src_x_end = current_w
            
            # Destination coordinates (in output_frame)
            dst_y_start = paste_y
            dst_y_end = paste_y + current_h
            dst_x_start = paste_x
            dst_x_end = paste_x + current_w
            
            # Clip to output bounds and adjust source accordingly
            if dst_y_start < 0:
                src_y_start = -dst_y_start
                dst_y_start = 0
            if dst_y_end > DISPLAY_HEIGHT:
                src_y_end -= (dst_y_end - DISPLAY_HEIGHT)
                dst_y_end = DISPLAY_HEIGHT
            if dst_x_start < 0:
                src_x_start = -dst_x_start
                dst_x_start = 0
            if dst_x_end > DISPLAY_WIDTH:
                src_x_end -= (dst_x_end - DISPLAY_WIDTH)
                dst_x_end = DISPLAY_WIDTH
            
            # Copy the visible portion
            if src_y_end > src_y_start and src_x_end > src_x_start:
                output_frame[dst_y_start:dst_y_end, dst_x_start:dst_x_end] = frame[src_y_start:src_y_end, src_x_start:src_x_end]
            
            frame = output_frame

            # Cache layout for auto alignment calibration
            last_final_w = final_w
            last_final_h = final_h
            last_src_x_start = src_x_start
            last_src_y_start = src_y_start
            last_dst_x_start = dst_x_start
            last_dst_y_start = dst_y_start

            # Optional on-screen ruler (150mm) for visual calibration
            if show_ruler:
                try:
                    ppmm_x = DISPLAY_WIDTH / max(1e-6, float(DISPLAY_PHYSICAL_WIDTH_MM))
                    ppmm_y = DISPLAY_HEIGHT / max(1e-6, float(DISPLAY_PHYSICAL_HEIGHT_MM))
                    ruler_mm = 150
                    len_px_h = int(round(ruler_mm * ppmm_x))
                    len_px_v = int(round(ruler_mm * ppmm_y))
                    cx = DISPLAY_WIDTH // 2
                    cy = DISPLAY_HEIGHT // 2
                    overlay = frame.copy()
                    # Background box
                    box_w = max(420, len_px_h + 120)
                    box_h = max(300, len_px_v + 120)
                    bx1 = max(10, cx - box_w // 2)
                    by1 = max(10, cy - box_h // 2)
                    bx2 = min(DISPLAY_WIDTH - 10, bx1 + box_w)
                    by2 = min(DISPLAY_HEIGHT - 10, by1 + box_h)
                    cv2.rectangle(overlay, (bx1, by1), (bx2, by2), (0, 0, 0), -1)
                    # Horizontal ruler (centered cross)
                    hx1 = cx - len_px_h // 2
                    hx2 = hx1 + len_px_h
                    hy = cy
                    cv2.line(overlay, (hx1, hy), (hx2, hy), (0, 255, 255), 8)
                    for tx in (hx1, (hx1 + hx2) // 2, hx2):
                        cv2.line(overlay, (tx, hy - 20), (tx, hy + 20), (0, 255, 255), 6)
                    # Vertical ruler (centered cross)
                    vy1 = cy - len_px_v // 2
                    vy2 = vy1 + len_px_v
                    vx = cx
                    cv2.line(overlay, (vx, vy1), (vx, vy2), (0, 255, 255), 8)
                    for ty in (vy1, (vy1 + vy2) // 2, vy2):
                        cv2.line(overlay, (vx - 20, ty), (vx + 20, ty), (0, 255, 255), 6)
                    # Labels
                    label = f"{ruler_mm} mm"
                    cv2.putText(overlay, label, (hx1, hy - 30), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 255, 255), 3, cv2.LINE_AA)
                    cv2.putText(overlay, label, (vx + 30, vy1 + 40), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 255, 255), 3, cv2.LINE_AA)
                    # Blend overlay
                    cv2.addWeighted(overlay, 0.35, frame, 0.65, 0, frame)
                except Exception:
                    pass

            # Alignment target overlay (for baseline two-point calibration)
            if 'baseline_cal_mode' in locals() and baseline_cal_mode:
                try:
                    cx = int(round(DISPLAY_WIDTH * float(ALIGN_TARGET_X_REL)))
                    cy = int(round(DISPLAY_HEIGHT * float(ALIGN_TARGET_Y_REL)))
                    # Big bullseye + crosshair
                    cv2.circle(frame, (cx, cy), 28, (0, 255, 0), 4)
                    cv2.circle(frame, (cx, cy), 8, (0, 255, 0), -1)
                    cv2.line(frame, (cx - 70, cy), (cx + 70, cy), (0, 255, 0), 3)
                    cv2.line(frame, (cx, cy - 70), (cx, cy + 70), (0, 255, 0), 3)
                    cv2.putText(frame, "BASELINE CAL: stand steady; press 1 to capture, move in/out, press 2 to capture",
                                (max(10, cx - 620), min(DISPLAY_HEIGHT - 20, cy + 110)),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2, cv2.LINE_AA)
                except Exception:
                    pass

            # Viewer band mask: when enabled, keep only the viewer's connected depth region within +/- VIEWER_BAND_TOL_MM
            if depth_clip_enabled and last_depth_raw is not None:
                try:
                    now_ts = time.time()
                    # Update viewer target depth from recent confident face if available
                    recent_face_ok = False
                    try:
                        recent_face_ok = (last_face_conf_seen >= 0.8) and ((now_ts - float(last_face_conf_time)) <= 1.5)
                    except Exception:
                        recent_face_ok = False
                    if recent_face_ok and (last_face_center_cam is not None):
                        cx_cam, cy_cam = last_face_center_cam
                        dh, dw = last_depth_raw.shape[:2]
                        try:
                            scale_x = dw / float(orig_w)
                            scale_y = dh / float(orig_h)
                        except Exception:
                            scale_x = 1.0
                            scale_y = 1.0
                        sx = int(round(cx_cam * scale_x))
                        sy = int(round(cy_cam * scale_y))
                        rwin = 6
                        x1s = max(0, sx - rwin)
                        x2s = min(dw - 1, sx + rwin)
                        y1s = max(0, sy - rwin)
                        y2s = min(dh - 1, sy + rwin)
                        win = last_depth_raw[y1s:y2s+1, x1s:x2s+1].astype(np.float32)
                        valid_win = (win >= MIN_DISTANCE_MM) & (win <= MAX_DISTANCE_MM)
                        if np.any(valid_win):
                            viewer_band_target_z = float(np.median(win[valid_win]))
                            viewer_band_last_time = now_ts
                    # Fallback: if no recent confident face, keep last target; if none yet, try last_clamped_z
                    if (viewer_band_target_z is None or not np.isfinite(viewer_band_target_z)) and (last_clamped_z is not None and last_clamped_z > 0):
                        viewer_band_target_z = float(last_clamped_z)
                        viewer_band_last_time = now_ts

                    if viewer_band_target_z is not None and np.isfinite(viewer_band_target_z):
                        depth_img = last_depth_raw.astype(np.float32)
                        depth_mir = cv2.flip(depth_img, 1)
                        depth_resized = cv2.resize(depth_mir, (final_w, final_h), interpolation=cv2.INTER_NEAREST)
                        depth_crop = depth_resized[src_y_start:src_y_end, src_x_start:src_x_end]
                        if depth_crop.size > 0:
                            valid = (depth_crop >= MIN_DISTANCE_MM) & (depth_crop <= MAX_DISTANCE_MM)
                            in_band = np.abs(depth_crop - float(viewer_band_target_z)) <= float(VIEWER_BAND_TOL_MM)
                            band_mask = (valid & in_band)
                            if band_mask.any():
                                band_u8 = (band_mask.astype(np.uint8) * 255)
                                # Clean the band to connect body parts
                                k_close = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7))
                                band_u8 = cv2.morphologyEx(band_u8, cv2.MORPH_CLOSE, k_close, iterations=1)
                                # Add a bit of vertical dilation to connect torso/legs
                                k_vert = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 17))
                                band_u8 = cv2.dilate(band_u8, k_vert, iterations=1)
                                # Choose connected component that overlaps a vertical corridor beneath face center
                                num_labels, labels = cv2.connectedComponents(band_u8)
                                chosen = None
                                best_score = -1
                                try:
                                    if last_face_center_cam is not None:
                                        cx_cam, cy_cam = last_face_center_cam
                                        # Map face center to cropped-resized coords
                                        fx_mir = (orig_w - 1) - int(cx_cam)
                                        rx = int(round(fx_mir * combined_scale))
                                        ry = int(round(int(cy_cam) * combined_scale))
                                        sx = rx - src_x_start
                                        sy = ry - src_y_start
                                        # Build a corridor mask around face X extending downward
                                        corridor = np.zeros_like(band_u8, dtype=np.uint8)
                                        half_w = max(40, int(0.12 * corridor.shape[1]))
                                        x1c = max(0, sx - half_w)
                                        x2c = min(corridor.shape[1] - 1, sx + half_w)
                                        y1c = max(0, sy - int(0.5 * half_w))
                                        y2c = corridor.shape[0] - 1
                                        corridor[y1c:y2c+1, x1c:x2c+1] = 255
                                        for lbl in range(1, num_labels):
                                            comp = (labels == lbl)
                                            # Score by overlap with corridor and area (favor larger overlap, then area)
                                            overlap = int(np.count_nonzero(comp & (corridor > 0)))
                                            if overlap <= 0:
                                                continue
                                            area = int(np.count_nonzero(comp))
                                            score = overlap * 10 + area
                                            if score > best_score:
                                                best_score = score
                                                chosen = comp
                                except Exception:
                                    chosen = None
                                # Fallback: choose largest component if no seeded overlap
                                if chosen is None and num_labels > 1:
                                    max_area = -1
                                    for lbl in range(1, num_labels):
                                        comp = (labels == lbl)
                                        area = int(np.count_nonzero(comp))
                                        if area > max_area:
                                            max_area = area
                                            chosen = comp
                                if chosen is not None:
                                    roi = frame[dst_y_start:dst_y_end, dst_x_start:dst_x_end]
                                    if roi.shape[:2] == chosen.shape:
                                        inv = ~chosen
                                        roi[inv] = 0
                except Exception:
                    pass

            # Compute raw-face depth (for diagnostics and capture): sample median around last face center
            try:
                if last_face_center_cam is not None and last_depth_raw is not None:
                    cx_cam, cy_cam = last_face_center_cam
                    dh, dw = last_depth_raw.shape[:2]
                    scale_x = dw / float(orig_w) if orig_w else 1.0
                    scale_y = dh / float(orig_h) if orig_h else 1.0
                    sx = int(round(cx_cam * scale_x))
                    sy = int(round(cy_cam * scale_y))
                    rwin = 6
                    x1s = max(0, sx - rwin)
                    x2s = min(dw - 1, sx + rwin)
                    y1s = max(0, sy - rwin)
                    y2s = min(dh - 1, sy + rwin)
                    win = last_depth_raw[y1s:y2s+1, x1s:x2s+1].astype(np.float32)
                    valid = (win >= MIN_DISTANCE_MM) & (win <= MAX_DISTANCE_MM)
                    if np.any(valid):
                        z_face_raw = float(np.median(win[valid]))
            except Exception:
                z_face_raw = None

            # Draw persistent depth readout if set
            try:
                if stored_depth_display_mm is not None:
                    text = f"{int(stored_depth_display_mm)} mm"
                    font = cv2.FONT_HERSHEY_SIMPLEX
                    scale = 3.0
                    thickness = 8
                    (tw, th), _ = cv2.getTextSize(text, font, scale, thickness)
                    x = max(20, (DISPLAY_WIDTH - tw) // 2)
                    y = 200
                    # Background rectangle for readability
                    cv2.rectangle(frame, (x - 20, y - th - 20), (x + tw + 20, y + 20), (0, 0, 0), -1)
                    cv2.putText(frame, text, (x, y), font, scale, (0, 255, 255), thickness, cv2.LINE_AA)
            except Exception:
                pass

            # Optional depth diagnostics overlay (big, centered like 'y' readout)
            if show_depth_diag:
                try:
                    lines = []
                    if last_clamped_z is not None:
                        lines.append(f"ROI Z: {last_clamped_z:.0f} mm")
                    if z_face_raw is not None:
                        lines.append(f"RawFace Z: {z_face_raw:.0f} mm")
                    if lines:
                        font = cv2.FONT_HERSHEY_SIMPLEX
                        scale = 2.6
                        thickness = 7
                        total_h = 0
                        widths = []
                        heights = []
                        for s in lines:
                            (tw, th), _ = cv2.getTextSize(s, font, scale, thickness)
                            widths.append(tw)
                            heights.append(th)
                            total_h += th + 20
                        total_h -= 20
                        box_w = max(widths) + 80
                        box_h = total_h + 80
                        x = max(20, (DISPLAY_WIDTH - box_w) // 2)
                        y = 200
                        # Background rectangle
                        cv2.rectangle(frame, (x, y - 60), (x + box_w, y - 60 + box_h), (0, 0, 0), -1)
                        # Draw lines centered within box
                        cur_y = y
                        for idx, s in enumerate(lines):
                            (tw, th), _ = cv2.getTextSize(s, font, scale, thickness)
                            tx = x + (box_w - tw) // 2
                            cv2.putText(frame, s, (tx, cur_y), font, scale, (0, 255, 255), thickness, cv2.LINE_AA)
                            cur_y += th + 20
                except Exception:
                    pass

            # Manual ROI override during calibration: map display rectangle to camera ROI
            if manual_roi_active and (auto_calibrate_mode or multi_cal_mode) and len(manual_roi_points) == 2:
                try:
                    (dx1, dy1), (dx2, dy2) = manual_roi_points
                    dx1 = max(0, min(DISPLAY_WIDTH - 1, int(dx1)))
                    dx2 = max(0, min(DISPLAY_WIDTH - 1, int(dx2)))
                    dy1 = max(0, min(DISPLAY_HEIGHT - 1, int(dy1)))
                    dy2 = max(0, min(DISPLAY_HEIGHT - 1, int(dy2)))
                    x_min_disp, x_max_disp = sorted([dx1, dx2])
                    y_min_disp, y_max_disp = sorted([dy1, dy2])
                    # Map display coords to resized-frame coords (account paste + clipping)
                    def map_disp_to_resized(xd, yd):
                        xr = src_x_start + max(0, min(current_w, xd - dst_x_start))
                        yr = src_y_start + max(0, min(current_h, yd - dst_y_start))
                        return xr, yr
                    rx1, ry1 = map_disp_to_resized(x_min_disp, y_min_disp)
                    rx2, ry2 = map_disp_to_resized(x_max_disp, y_max_disp)
                    # Back to original (pre-resize) camera coords
                    sx1 = int(rx1 / max(1e-6, combined_scale))
                    sy1 = int(ry1 / max(1e-6, combined_scale))
                    sx2 = int(rx2 / max(1e-6, combined_scale))
                    sy2 = int(ry2 / max(1e-6, combined_scale))
                    # Undo horizontal flip
                    cam_x1 = max(0, min(orig_w - 1, orig_w - sx2))
                    cam_x2 = max(0, min(orig_w - 1, orig_w - sx1))
                    cam_y1 = max(0, min(orig_h - 1, sy1))
                    cam_y2 = max(0, min(orig_h - 1, sy2))
                    if cam_x2 > cam_x1 and cam_y2 > cam_y1:
                        measure_distance.roi_input.send(RegionOfInterest(cam_x1, cam_y1, cam_x2, cam_y2))
                        cv2.rectangle(frame, (x_min_disp, y_min_disp), (x_max_disp, y_max_disp), (0, 200, 255), 4)
                        last_cam_roi = (int(cam_x1), int(cam_y1), int(cam_x2), int(cam_y2))
                        if last_raw_z is not None:
                            cv2.putText(frame, f'RawZ:{last_raw_z:.0f} Clamped:{(last_clamped_z or 0):.0f}',
                                        (x_min_disp, min(DISPLAY_HEIGHT-10, y_max_disp+30)),
                                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 200, 255), 2)
                except Exception:
                    pass

            # Show manual ROI rectangle even when not actively calibrating (visual confirmation)
            if manual_roi_active and not (auto_calibrate_mode or multi_cal_mode) and len(manual_roi_points) == 2:
                try:
                    (dx1, dy1), (dx2, dy2) = manual_roi_points
                    x_min_disp, x_max_disp = sorted([int(dx1), int(dx2)])
                    y_min_disp, y_max_disp = sorted([int(dy1), int(dy2)])
                    cv2.rectangle(frame, (x_min_disp, y_min_disp), (x_max_disp, y_max_disp), (0, 200, 255), 3)
                    cv2.putText(frame, "Manual ROI armed (press G or I)",
                                (x_min_disp, max(30, y_min_disp - 10)),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 200, 255), 2)
                except Exception:
                    pass

            # Draw fixed center ROI rectangle when fixed ROI mode is active
            if fixed_roi_mode:
                try:
                    # Center ROI is in camera coords before flip
                    cx1, cy1, cx2, cy2 = 240, 180, 400, 300
                    # Mirror horizontally to display camera
                    mx1, mx2 = (orig_w - cx2), (orig_w - cx1)
                    my1, my2 = cy1, cy2
                    # Scale to resized
                    sx1 = int(mx1 * combined_scale)
                    sy1 = int(my1 * combined_scale)
                    sx2 = int(mx2 * combined_scale)
                    sy2 = int(my2 * combined_scale)
                    # Adjust for centering/cropping
                    dx1 = sx1 - (final_w - DISPLAY_WIDTH) // 2
                    dy1 = sy1 - (final_h - DISPLAY_HEIGHT) // 2
                    dx2 = sx2 - (final_w - DISPLAY_WIDTH) // 2
                    dy2 = sy2 - (final_h - DISPLAY_HEIGHT) // 2
                    cv2.rectangle(frame, (dx1, dy1), (dx2, dy2), (255, 200, 0), 2)
                    cv2.putText(frame, "Fixed ROI (X)", (dx1, max(30, dy1 - 10)),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 200, 0), 2)
                    last_cam_roi = (240, 180, 400, 300)
                    if last_raw_z is not None:
                        cv2.putText(frame, f"RawZ:{last_raw_z:.0f} Clamped:{(last_clamped_z or 0):.0f}",
                                    (dx1, min(DISPLAY_HEIGHT-10, dy2+30)),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 200, 0), 2)
                except Exception:
                    pass
            
            # Draw detection circles (lingering visualization)
            if len(recent_detections) > 0:
                # Calculate scale factor from original camera size to final display size
                scale_factor = combined_scale
                
                for i, det in enumerate(recent_detections):
                    # Calculate opacity based on age (newer = brighter)
                    age_factor = (len(recent_detections) - i) / len(recent_detections)
                    
                    # Scale detection coordinates to match final frame size
                    # Note: After horizontal flip, x coordinates are inverted
                    scaled_x = int((orig_w - det['center_x']) * scale_factor)  # Flip x
                    scaled_y = int(det['center_y'] * scale_factor)
                    scaled_radius = int(det['radius'] * scale_factor)
                    
                    # Adjust for cropping offset
                    display_x = scaled_x - (final_w - DISPLAY_WIDTH) // 2
                    display_y = scaled_y - (final_h - DISPLAY_HEIGHT) // 2
                    
                    # Draw even if partially offscreen - simpler and more visible
                    thickness = max(5, int(12 * age_factor))
                    color_intensity = int(255 * age_factor)
                    cv2.circle(frame, (display_x, display_y), scaled_radius, (0, color_intensity, 0), thickness)
                    cv2.circle(frame, (display_x, display_y), 10, (0, 255, 0), -1)  # Larger center dot
                    
                    # Draw confidence text
                    conf_text = f"{det['confidence']:.2f}"
                    cv2.putText(frame, conf_text, (display_x + scaled_radius + 20, display_y),
                              cv2.FONT_HERSHEY_SIMPLEX, 2.0, (0, 255, 0), 4)
            
            # Optional: Draw debug info
            # Always show debug info to diagnose issues - centered for visibility
            center_x = DISPLAY_WIDTH // 2 - 400
            center_y = DISPLAY_HEIGHT // 2 - 300
            info_y = center_y
            
            # DRAW GIANT TEST MARKER TO CONFIRM DRAWING WORKS
            cv2.circle(frame, (DISPLAY_WIDTH // 2, DISPLAY_HEIGHT // 2), 100, (255, 0, 255), 20)  # MAGENTA circle at center
            cv2.line(frame, (0, DISPLAY_HEIGHT // 2), (DISPLAY_WIDTH, DISPLAY_HEIGHT // 2), (255, 0, 255), 5)  # Horizontal line
            cv2.line(frame, (DISPLAY_WIDTH // 2, 0), (DISPLAY_WIDTH // 2, DISPLAY_HEIGHT), (255, 0, 255), 5)  # Vertical line
            
            # Show calibration instructions if in calibration mode
            if calibration_mode:
                if calibration_step == 0:
                    cv2.putText(frame, "CALIBRATION MODE", (center_x, info_y),
                              cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0, 255, 255), 4)
                    cv2.putText(frame, "Step 1: Stand at your normal viewing distance", (center_x, info_y + 60),
                              cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 255, 255), 3)
                    cv2.putText(frame, "Make sure your face is detected", (center_x, info_y + 110),
                              cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 255, 255), 3)
                    cv2.putText(frame, "Press SPACE to set reference distance", (center_x, info_y + 160),
                              cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 255, 0), 3)
                    cv2.putText(frame, "Press ESC to cancel", (center_x, info_y + 210),
                              cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 0, 255), 3)
                elif calibration_step == 1:
                    cv2.putText(frame, f"Reference distance set to: {REFERENCE_DISTANCE_MM}mm", (center_x, info_y),
                              cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 255, 0), 3)
                    cv2.putText(frame, "Step 2: Align the display with your reflection", (center_x, info_y + 60),
                              cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 255, 255), 3)
                    cv2.putText(frame, "Use Arrow Keys or WASD to move image", (center_x, info_y + 110),
                              cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 255, 255), 3)
                    cv2.putText(frame, f"Offset: X={IMAGE_OFFSET_X} Y={IMAGE_OFFSET_Y}", (center_x, info_y + 160),
                              cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 255, 255), 3)
                    cv2.putText(frame, "Press SPACE to finish calibration", (center_x, info_y + 210),
                              cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 255, 0), 3)
            else:
                cv2.putText(frame, f"Frame size: {frame.shape[1]}x{frame.shape[0]}", (center_x, info_y),
                          cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 255, 0), 3)
                cv2.putText(frame, f"Scale: {current_scale:.2f}x (base:{base_scale:.2f})", (center_x, info_y + 50),
                          cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 255, 0), 3)
                cv2.putText(frame, f"Offset: X={pixel_offset_x} Y={pixel_offset_y}", (center_x, info_y + 100),
                          cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 0, 255), 3)
                cv2.putText(frame, f"Cam offset: {CAMERA_OFFSET_Y_MM}mm above", (center_x, info_y + 150),
                          cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 255, 255), 3)
            
            if distance_msg is not None:
                x_mm = distance_msg.spatials.x
                y_mm = distance_msg.spatials.y
                z_mm = distance_msg.spatials.z
                
                if not calibration_mode:
                    # Debug overlay (comment out for clean mirror)
                    cv2.putText(frame, f"Distance: {z_mm:.0f}mm", (center_x, info_y + 150),
                              cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 255, 0), 3)
                else:
                    # Show current distance during calibration
                    cv2.putText(frame, f"Current Distance: {z_mm:.0f}mm", (center_x, info_y + 260),
                              cv2.FONT_HERSHEY_SIMPLEX, 1.2, (255, 255, 0), 3)
                cv2.putText(frame, f"Target Scale: {target_scale:.2f}x", (center_x, info_y + 200),
                          cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 255, 0), 3)
                cv2.putText(frame, f"Ref Dist: {REFERENCE_DISTANCE_MM}mm", (center_x, info_y + 250),
                          cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 255, 0), 3)
                cv2.putText(frame, f"Offset: X={pixel_offset_x} Y={pixel_offset_y}", (center_x, info_y + 300),
                          cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 255, 0), 3)
            else:
                cv2.putText(frame, "NO FACE DETECTED", (center_x, info_y + 150),
                          cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 0, 255), 3)
            
            # Calculate and display FPS
            fps_counter += 1
            elapsed = time.time() - fps_start_time
            if elapsed > 1.0:
                current_fps = fps_counter / elapsed
                fps_counter = 0
                fps_start_time = time.time()
            
            cv2.putText(frame, f"FPS: {current_fps:.1f}", (DISPLAY_WIDTH - 120, 30),
                      cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
            
            # Display the frame
            # Draw auto-calibrate overlay if active
            if auto_calibrate_mode:
                try:
                    pts = mouse_state.get("points", [])
                    for p in pts:
                        cv2.circle(frame, p, 10, (0, 255, 0), -1)
                    if len(pts) == 2:
                        cv2.line(frame, pts[0], pts[1], (0, 255, 0), 4)
                    cv2.putText(frame, f"AUTO-CAL 150mm @ {REFERENCE_DISTANCE_MM}mm: Click two points 15cm apart (press G to cancel)",
                                 (40, 80), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 255, 0), 3)
                    cv2.putText(frame, "Stand steady at reference distance; parallax and distance scaling are disabled",
                                (40, 120), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
                except Exception:
                    pass
            # Depth inset with camera-space ROI (toggle with 'd')
            if show_depth_inset:
                try:
                    depth_vis_msg = depth_vis_queue.tryGet()
                    if depth_vis_msg is not None:
                        inset = depth_vis_msg.getCvFrame()
                        if last_cam_roi is not None:
                            x1, y1, x2, y2 = last_cam_roi
                            cv2.rectangle(inset, (x1, y1), (x2, y2), (0, 255, 255), 2)
                            if last_raw_z is not None:
                                cv2.putText(
                                    inset,
                                    f"Raw:{last_raw_z:.0f} Clp:{(last_clamped_z or 0):.0f}",
                                    (x1, max(15, y1 - 5)),
                                    cv2.FONT_HERSHEY_SIMPLEX,
                                    0.5,
                                    (0, 255, 255),
                                    1,
                                )
                        # Also draw current click points mapped into camera space
                        try:
                            pts_disp = mouse_state.get("points", []) if isinstance(mouse_state, dict) else []
                            if pts_disp:
                                for (px, py) in pts_disp:
                                    px = int(px); py = int(py)
                                    # Map display -> resized content coords
                                    rx = src_x_start + max(0, min(current_w, px - dst_x_start))
                                    ry = src_y_start + max(0, min(current_h, py - dst_y_start))
                                    # Resized -> original camera coords
                                    sx = int(rx / max(1e-6, combined_scale))
                                    sy = int(ry / max(1e-6, combined_scale))
                                    # Undo horizontal flip to camera space
                                    cam_x = max(0, min(orig_w - 1, orig_w - sx))
                                    cam_y = max(0, min(orig_h - 1, sy))
                                    cv2.circle(inset, (cam_x, cam_y), 6, (255, 0, 255), -1)
                        except Exception:
                            pass
                        h, w = inset.shape[:2]
                        new_w = 480
                        new_h = int(h * (new_w / w))
                        inset_resized = cv2.resize(inset, (new_w, new_h), interpolation=cv2.INTER_AREA)
                        h_clip = min(new_h, frame.shape[0] - 10)
                        w_clip = min(new_w, frame.shape[1] - 10)
                        y0 = max(10, frame.shape[0] - h_clip - 10)
                        x0 = 10
                        frame[y0:y0 + h_clip, x0:x0 + w_clip] = inset_resized[0:h_clip, 0:w_clip]
                except Exception:
                    pass
            cv2.imshow(window_name, frame)
            # Handle auto-calibration apply when two points are present
            if auto_calibrate_mode and len(mouse_state.get("points", [])) == 2:
                try:
                    (x1, y1), (x2, y2) = mouse_state["points"]
                    dx = abs(x2 - x1)
                    dy = abs(y2 - y1)
                    # Require mostly horizontal or vertical selection to reduce diagonal error
                    dom = max(dx, dy)
                    if dom <= 0:
                        raise RuntimeError("Selection too small; click points farther apart.")
                    if min(dx, dy) / float(dom) > 0.25:
                        raise RuntimeError("Please click a mostly horizontal or vertical segment (avoid diagonal).")
                    # Convert HALF of 150mm to display pixels on dominant axis (mirror shows half angular size)
                    ppmm_x = DISPLAY_WIDTH / DISPLAY_PHYSICAL_WIDTH_MM
                    ppmm_y = DISPLAY_HEIGHT / DISPLAY_PHYSICAL_HEIGHT_MM
                    if dx >= dy:
                        target_px = 75.0 * ppmm_x
                        measured_px = max(1.0, dx)
                    else:
                        target_px = 75.0 * ppmm_y
                        measured_px = max(1.0, dy)


                    do_update = True
                    try:
                        rel_err = abs(target_px - measured_px) / max(1.0, float(target_px))
                    except Exception:
                        rel_err = 0.0
                    if rel_err <= 0.03:
                        print("Auto-calibrate 150mm: within 3% tolerance; multiplier unchanged.")
                        do_update = False
                    # Also set reference distance from raw depth at segment midpoint (like multi-depth)
                    try:
                        if last_depth_raw is not None:
                            def _map_disp_to_cam(px, py):
                                rx = src_x_start + max(0, min(current_w, int(px) - dst_x_start))
                                ry = src_y_start + max(0, min(current_h, int(py) - dst_y_start))
                                sx = int(rx / max(1e-6, combined_scale))
                                sy = int(ry / max(1e-6, combined_scale))
                                cam_x = max(0, min(orig_w - 1, orig_w - sx))
                                cam_y = max(0, min(orig_h - 1, sy))
                                return cam_x, cam_y
                            cpx = (x1 + x2) / 2.0
                            cpy = (y1 + y2) / 2.0
                            cx, cy = _map_disp_to_cam(cpx, cpy)
                            rwin = 5
                            x1s = max(0, cx - rwin)
                            x2s = min(last_depth_raw.shape[1] - 1, cx + rwin)
                            y1s = max(0, cy - rwin)
                            y2s = min(last_depth_raw.shape[0] - 1, cy + rwin)
                            win = last_depth_raw[y1s:y2s+1, x1s:x2s+1].astype(np.float32)
                            valid = (win > MIN_DISTANCE_MM) & (win < MAX_DISTANCE_MM)
                            if np.any(valid):
                                z_click = float(np.median(win[valid]))
                                REFERENCE_DISTANCE_MM = int(z_click)
                                print(f"Auto-cal: Reference distance set from raw depth to {REFERENCE_DISTANCE_MM}mm")
                    except Exception:
                        pass
                    # Compute ratio and clamp to avoid runaway scaling if clicks are off
                    if do_update:
                        # Compute ratio and clamp to avoid runaway scaling if clicks are off
                        ratio = float(target_px) / float(measured_px)
                        ratio = max(0.85, min(1.15, ratio))
                        old = LIFE_SIZE_SCALE_MULTIPLIER
                        LIFE_SIZE_SCALE_MULTIPLIER = old * ratio
                        print(f"Auto-calibrate 150mm: measured {measured_px:.1f}px, target {target_px:.1f}px -> multiplier {old:.4f} -> {LIFE_SIZE_SCALE_MULTIPLIER:.4f}")
                        # Persist updated multiplier and current reference distance
                        try:
                            cal_path = os.path.join(os.path.dirname(__file__), "calibration.json")
                            cal = {}
                            if os.path.exists(cal_path):
                                with open(cal_path, "r", encoding="utf-8") as f:
                                    cal = json.load(f) or {}
                            cal["LIFE_SIZE_SCALE_MULTIPLIER"] = float(LIFE_SIZE_SCALE_MULTIPLIER)
                            cal["REFERENCE_DISTANCE_MM"] = int(REFERENCE_DISTANCE_MM)
                            with open(cal_path, "w", encoding="utf-8") as f:
                                json.dump(cal, f, indent=2)
                            print("Saved multiplier and reference distance to calibration.json")
                        except Exception as e:
                            print(f"Save error: {e}")
                finally:
                    # Reset and restore state
                    auto_calibrate_mode = False
                    mouse_state["active"] = False
                    mouse_state["points"] = []
                    if auto_saved_state:
                        freeze_scale = auto_saved_state.get("freeze_scale", freeze_scale)
                        current_scale = auto_saved_state.get("current_scale", current_scale)
                        auto_saved_state = None
                    # Auto-disable manual ROI after one-shot auto-cal completes
                    manual_roi_active = False

            # Handle multi-depth sample capture when two points are picked
            if multi_cal_mode and collecting_sample and len(mouse_state.get("points", [])) == 2:
                try:
                    (x1, y1), (x2, y2) = mouse_state["points"]
                    dx = abs(x2 - x1)
                    dy = abs(y2 - y1)
                    # Refresh distance reading after ROI change: actively settle and read newest raw z
                    z_val = None
                    settle_deadline = time.time() + 0.25  # up to 250ms to allow ROI to take effect
                    while time.time() < settle_deadline:
                        # Re-send ROI to ensure takeover during calibration
                        try:
                            if 'last_cam_roi' in locals() or 'last_cam_roi' in globals():
                                if last_cam_roi is not None and len(last_cam_roi) == 4:
                                    cx1, cy1, cx2, cy2 = last_cam_roi
                                    measure_distance.roi_input.send(RegionOfInterest(int(cx1), int(cy1), int(cx2), int(cy2)))
                        except Exception:
                            pass
                        m = distance_queue.tryGet()
                        if m is not None and getattr(m.spatials, "z", 0) > 0:
                            z_val = float(m.spatials.z)  # use RAW z for calibration
                    # Final fallback if nothing fresh arrived
                    if z_val is None:
                        if distance_msg is not None and getattr(distance_msg.spatials, "z", 0) > 0:
                            z_val = float(distance_msg.spatials.z)
                        elif last_valid_distance and last_valid_distance > 0:
                            z_val = float(last_valid_distance)
                    # Prefer direct sampling from raw depth at the clicked segment center
                    if last_depth_raw is not None:
                        try:
                            def map_disp_to_cam(px, py):
                                rx = src_x_start + max(0, min(current_w, int(px) - dst_x_start))
                                ry = src_y_start + max(0, min(current_h, int(py) - dst_y_start))
                                sx = int(rx / max(1e-6, combined_scale))
                                sy = int(ry / max(1e-6, combined_scale))
                                cam_x = max(0, min(orig_w - 1, orig_w - sx))
                                cam_y = max(0, min(orig_h - 1, sy))
                                return cam_x, cam_y
                            cpx = (x1 + x2) / 2.0
                            cpy = (y1 + y2) / 2.0
                            cx, cy = map_disp_to_cam(cpx, cpy)
                            r = 5
                            x1s = max(0, cx - r)
                            x2s = min(last_depth_raw.shape[1] - 1, cx + r)
                            y1s = max(0, cy - r)
                            y2s = min(last_depth_raw.shape[0] - 1, cy + r)
                            win = last_depth_raw[y1s:y2s+1, x1s:x2s+1].astype(np.float32)
                            valid = (win > MIN_DISTANCE_MM) & (win < MAX_DISTANCE_MM)
                            if np.any(valid):
                                z_click = float(np.median(win[valid]))
                                z_val = z_click
                        except Exception:
                            pass
                    if not z_val or z_val <= 0:
                        raise RuntimeError("No valid distance for sample (ensure face or enable fixed ROI with 'x')")
                    z = z_val
                    ppmm_x = DISPLAY_WIDTH / DISPLAY_PHYSICAL_WIDTH_MM
                    ppmm_y = DISPLAY_HEIGHT / DISPLAY_PHYSICAL_HEIGHT_MM
                    if dx >= dy:
                        measured_px = max(1.0, dx)
                        axis = "x"
                        target_px = 75.0 * ppmm_x
                    else:
                        measured_px = max(1.0, dy)
                        axis = "y"
                        target_px = 75.0 * ppmm_y
                    multi_samples.append({"z": z, "measured_px": measured_px, "axis": axis})
                    print(f"Captured sample: z={z:.0f}mm, measured={measured_px:.1f}px (axis {axis})")
                except Exception as e:
                    print(f"Sample capture error: {e}")
                finally:
                    collecting_sample = False
                    mouse_state["active"] = False
                    mouse_state["points"] = []
        
        # Handle key presses
        key = cv2.waitKey(1)
        
        # Debug: print key code for any key press (comment out once calibrated)
        if key != -1 and key != 255:
            print(f"Key pressed: {key}")
        
        if key == ord("q"):
            print("Quitting...")
            break
        elif key == ord("c"):
            # Toggle calibration mode
            if not calibration_mode:
                calibration_mode = True
                calibration_step = 0
                print("\n=== CALIBRATION MODE STARTED ===")
                print("Stand at your normal viewing distance and press SPACE")
            else:
                calibration_mode = False
                calibration_step = 0
                print("Calibration mode exited")
        elif key == 32 and calibration_mode:  # SPACE key
            if calibration_step == 0:
                # Set reference distance
                # Prefer raw-face depth for reference; fallback to ROI depth
                ref_z = None
                try:
                    if last_face_center_cam is not None and last_depth_raw is not None and orig_w > 0 and orig_h > 0:
                        cx_cam, cy_cam = last_face_center_cam
                        dh, dw = last_depth_raw.shape[:2]
                        scale_x = dw / float(orig_w)
                        scale_y = dh / float(orig_h)
                        sx = int(round(cx_cam * scale_x))
                        sy = int(round(cy_cam * scale_y))
                        r = 6
                        x1s = max(0, sx - r)
                        x2s = min(dw - 1, sx + r)
                        y1s = max(0, sy - r)
                        y2s = min(dh - 1, sy + r)
                        win = last_depth_raw[y1s:y2s+1, x1s:x2s+1].astype(np.float32)
                        valid = (win >= MIN_DISTANCE_MM) & (win <= MAX_DISTANCE_MM)
                        if np.any(valid):
                            ref_z = float(np.median(win[valid]))
                except Exception:
                    ref_z = None
                if ref_z is None and distance_msg is not None and distance_msg.spatials.z > 0:
                    ref_z = float(distance_msg.spatials.z)
                if ref_z is not None and ref_z > 0:
                    REFERENCE_DISTANCE_MM = int(ref_z)
                    calibration_step = 1
                    current_scale = 1.0  # Reset scale
                    print(f"\nReference distance set to {REFERENCE_DISTANCE_MM}mm (raw-face preferred)")
                    # Persist to calibration.json
                    try:
                        cal_path = os.path.join(os.path.dirname(__file__), "calibration.json")
                        cal = {}
                        if os.path.exists(cal_path):
                            with open(cal_path, "r", encoding="utf-8") as f:
                                cal = json.load(f) or {}
                        cal["REFERENCE_DISTANCE_MM"] = int(REFERENCE_DISTANCE_MM)
                        with open(cal_path, "w", encoding="utf-8") as f:
                            json.dump(cal, f, indent=2)
                        print(f"Saved REFERENCE_DISTANCE_MM={REFERENCE_DISTANCE_MM} to calibration.json")
                    except Exception as e:
                        print(f"Save error: {e}")
                    print("Now adjust image position with arrow keys or WASD")
                    print("Press SPACE when aligned")
                else:
                    print("ERROR: No valid face detected. Move closer and try again.")
            elif calibration_step == 1:
                # Finish calibration
                calibration_mode = False
                calibration_step = 0
                print("\nCALIBRATION COMPLETE")
                print(f"Reference Distance: {REFERENCE_DISTANCE_MM}mm")
                print(f"Image Offset: X={IMAGE_OFFSET_X}, Y={IMAGE_OFFSET_Y}")
                print("\nYou can fine-tune anytime with:")
                print("  +/- keys: Adjust reference distance")
                print("  Arrow keys or WASD: Adjust offset")
        elif key == 27 and calibration_mode:  # ESC key
            calibration_mode = False
            calibration_step = 0
            print("Calibration cancelled")
        elif key == ord("r"):
            # Reset scale to 1.0 - useful if bad detection causes stuck scale
            current_scale = 1.0
            depth_history.clear()
            print("Scale reset to 1.0x")
        elif key == ord("x"):
            # Toggle fixed ROI mode (for phone camera calibration)
            fixed_roi_mode = not fixed_roi_mode
            if fixed_roi_mode:
                print("\n*** FIXED ROI MODE ENABLED ***")
                print("Distance measured from CENTER of frame (not face)")
                print("Use this mode for phone camera calibration")
                print("Press 'x' again to disable")
            else:
                print("*** Fixed ROI mode disabled - back to face tracking ***")
        elif key == ord("z"):
            # Toggle freeze scale mode and reset offsets when toggled
            freeze_scale = not freeze_scale
            if freeze_scale:
                current_scale = 1.0  # Lock at 1.0
                # Reset image offsets to zero for clean calibration alignment
                IMAGE_OFFSET_X = 0
                IMAGE_OFFSET_Y = 0
                # Also reset smoothed offsets to avoid lingering shift
                current_offset_x = 0.0
                current_offset_y = 0.0
                print("\n*** SCALE FROZEN AT 1.0 ***")
                print("No distance-based scaling - image stays constant size")
                print("Image offsets reset to 0 (X=0, Y=0)")
                print("Press 'z' again to re-enable scaling")
            else:
                print("*** Scale unfrozen - distance-based scaling enabled ***")
        elif key == ord("d"):
            # Toggle depth inset overlay
            show_depth_inset = not show_depth_inset
            print("Depth inset: " + ("ON" if show_depth_inset else "OFF"))
        elif key == ord("f"):
            # Toggle fullscreen
            is_fullscreen = cv2.getWindowProperty(window_name, cv2.WND_PROP_FULLSCREEN)
            if is_fullscreen == cv2.WINDOW_FULLSCREEN:
                cv2.setWindowProperty(window_name, cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_NORMAL)
            else:
                cv2.setWindowProperty(window_name, cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)
        elif key == ord("u"):
            # Toggle on-screen 150mm ruler overlay
            show_ruler = not show_ruler
            print("Ruler (150mm):", "ON" if show_ruler else "OFF")
        elif key == ord("b"):
            # Toggle viewer band masking (keep only +/- VIEWER_BAND_TOL_MM around viewer depth)
            depth_clip_enabled = not depth_clip_enabled
            if not depth_clip_enabled:
                viewer_band_target_z = None
            print(f"Viewer band mask (Â±{VIEWER_BAND_TOL_MM}mm): {'ON' if depth_clip_enabled else 'OFF'}")
        elif key == ord("["):
            # Nudge parallax distance exponent down and persist
            try:
                parallax_distance_exp_live = float(parallax_distance_exp_live) - 0.05
            except Exception:
                parallax_distance_exp_live = 1.0
            # Clamp to reasonable range
            if parallax_distance_exp_live < 0.0:
                parallax_distance_exp_live = 0.0
            print(f"parallax_distance_scale set to {parallax_distance_exp_live:.3f}")
            try:
                cal_path = os.path.join(os.path.dirname(__file__), "calibration.json")
                cal = {}
                if os.path.exists(cal_path):
                    with open(cal_path, "r", encoding="utf-8") as f:
                        cal = json.load(f) or {}
                cal["parallax_distance_scale"] = float(parallax_distance_exp_live)
                with open(cal_path, "w", encoding="utf-8") as f:
                    json.dump(cal, f, indent=2)
                print("Saved parallax_distance_scale to calibration.json")
            except Exception as e:
                print(f"Save error: {e}")
        elif key == ord("]"):
            # Nudge parallax distance exponent up and persist
            try:
                parallax_distance_exp_live = float(parallax_distance_exp_live) + 0.05
            except Exception:
                parallax_distance_exp_live = 1.0
            # Clamp to reasonable range
            if parallax_distance_exp_live > 3.0:
                parallax_distance_exp_live = 3.0
            print(f"parallax_distance_scale set to {parallax_distance_exp_live:.3f}")
            try:
                cal_path = os.path.join(os.path.dirname(__file__), "calibration.json")
                cal = {}
                if os.path.exists(cal_path):
                    with open(cal_path, "r", encoding="utf-8") as f:
                        cal = json.load(f) or {}
                cal["parallax_distance_scale"] = float(parallax_distance_exp_live)
                with open(cal_path, "w", encoding="utf-8") as f:
                    json.dump(cal, f, indent=2)
                print("Saved parallax_distance_scale to calibration.json")
            except Exception as e:
                print(f"Save error: {e}")
        elif key == ord("m"):
            # Move window to the next monitor
            try:
                import screeninfo
                screens = screeninfo.get_monitors()
                if screens:
                    current_monitor_index = (current_monitor_index + 1) % len(screens)
                    target = screens[current_monitor_index]
                    DISPLAY_WIDTH = target.width
                    DISPLAY_HEIGHT = target.height
                    cv2.moveWindow(window_name, target.x, target.y)
                    if not getattr(args, "windowed", False):
                        cv2.setWindowProperty(
                            window_name, cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN
                        )
                    print(
                        f"Moved to monitor {current_monitor_index}: {DISPLAY_WIDTH}x{DISPLAY_HEIGHT} at ({target.x}, {target.y})"
                    )
                else:
                    print("No monitors detected to switch")
            except Exception as e:
                print(f"Could not switch monitor: {e}")
        elif key == ord("p"):
            # Save screenshot (changed from 's' to avoid conflict)
            if video_msg is not None:
                filename = f"mirror_screenshot_{int(time.time())}.jpg"
                cv2.imwrite(filename, frame)
                print(f"Screenshot saved: {filename}")
        elif key == ord("t"):
            # Toggle depth diagnostics overlay (ROI vs raw-face depth)
            show_depth_diag = not show_depth_diag
            print("Depth diagnostics:", "ON" if show_depth_diag else "OFF")
        elif key == ord("y"):
            # Capture and persist current viewer depth using raw depth at face center
            try:
                # Accept a recently-seen confident face (within 2 seconds)
                recent_face_ok = False
                try:
                    recent_face_ok = (last_face_conf_seen >= 0.8) and ((time.time() - float(last_face_conf_time)) <= 2.0)
                except Exception:
                    recent_face_ok = False
                if recent_face_ok and last_face_center_cam is not None and last_depth_raw is not None:
                    cx_cam, cy_cam = last_face_center_cam
                    dh, dw = last_depth_raw.shape[:2]
                    # Map camera coords (orig_w x orig_h) to depth size in case of mismatch
                    try:
                        scale_x = dw / float(orig_w)
                        scale_y = dh / float(orig_h)
                    except Exception:
                        scale_x = 1.0
                        scale_y = 1.0
                    sx = int(round(cx_cam * scale_x))
                    sy = int(round(cy_cam * scale_y))
                    rwin = 6
                    x1s = max(0, sx - rwin)
                    x2s = min(dw - 1, sx + rwin)
                    y1s = max(0, sy - rwin)
                    y2s = min(dh - 1, sy + rwin)
                    win = last_depth_raw[y1s:y2s+1, x1s:x2s+1].astype(np.float32)
                    valid = (win >= MIN_DISTANCE_MM) & (win <= MAX_DISTANCE_MM)
                    if np.any(valid):
                        z_med = float(np.median(win[valid]))
                        stored_depth_display_mm = int(round(z_med))
                        print(f"Depth captured: {stored_depth_display_mm} mm (face conf {last_face_conf_seen:.2f})")
                    else:
                        print("Cannot capture depth: no valid depth at face position.")
                else:
                    print("Cannot capture depth: no recent confident face (>=0.80) or missing depth/position.")
            except Exception as e:
                print(f"Capture error: {e}")
        elif key == ord("e"):
            # Eye-offset auto-calibration: derive camera offset from eye/face spatials
            try:
                prev_freeze = freeze_scale
                freeze_scale = True
                print("\nEye-offset calibration: Hold steady and look at your reflection...")
                if getattr(args, "eye_roi", False):
                    print("Using eye ROI for calibration")
                else:
                    print("Using face ROI (enable --eye_roi for tighter calibration)")
                xs = []
                ys = []
                zs = []
                start = time.time()
                duration_s = 1.2
                while time.time() - start < duration_s:
                    msg = distance_queue.tryGet()
                    if msg is not None:
                        try:
                            _x = float(msg.spatials.x)
                            _y = float(msg.spatials.y)
                            _z = float(msg.spatials.z)
                            if _z > 0 and not np.isnan(_z):
                                xs.append(_x)
                                ys.append(_y)
                                zs.append(_z)
                        except Exception:
                            pass
                    time.sleep(0.005)
                if len(xs) >= 5:
                    cx = float(np.median(xs))
                    cy = float(np.median(ys))
                    cz = float(np.median(zs)) if zs else float('nan')
                    CAMERA_OFFSET_X_MM = int(round(-cx))
                    CAMERA_OFFSET_Y_MM = int(round(-cy))
                    current_offset_x = 0.0
                    current_offset_y = 0.0
                    # Persist to calibration.json (merge/update)
                    try:
                        cal_path = os.path.join(os.path.dirname(__file__), "calibration.json")
                        cal = {}
                        if os.path.exists(cal_path):
                            with open(cal_path, "r", encoding="utf-8") as f:
                                cal = json.load(f) or {}
                        cal["CAMERA_OFFSET_X_MM"] = int(CAMERA_OFFSET_X_MM)
                        cal["CAMERA_OFFSET_Y_MM"] = int(CAMERA_OFFSET_Y_MM)
                        cal["IMAGE_OFFSET_X"] = int(IMAGE_OFFSET_X)
                        cal["IMAGE_OFFSET_Y"] = int(IMAGE_OFFSET_Y)
                        cal["parallax_distance_scale"] = float(parallax_distance_exp_live)
                        with open(cal_path, "w", encoding="utf-8") as f:
                            json.dump(cal, f, indent=2)
                        print(f"Saved camera offsets to {cal_path}")
                    except Exception as se:
                        print(f"Save error: {se}")
                    x_px = -CAMERA_OFFSET_X_MM * PIXELS_PER_MM
                    y_px = -CAMERA_OFFSET_Y_MM * PIXELS_PER_MM
                    try:
                        print(
                            f"Calibrated CAMERA_OFFSET: X={CAMERA_OFFSET_X_MM}mm, Y={CAMERA_OFFSET_Y_MM}mm (px shift ~ X={x_px:.0f}, Y={y_px:.0f}); Zâ‰ˆ{cz:.0f}mm"
                        )
                    except Exception:
                        print(
                            f"Calibrated CAMERA_OFFSET: X={CAMERA_OFFSET_X_MM}mm, Y={CAMERA_OFFSET_Y_MM}mm (px shift ~ X={x_px}, Y={y_px})"
                        )
                else:
                    print("Eye-offset calibration failed: not enough samples. Ensure face/eyes are detected.")
            except Exception as e:
                print(f"Eye-offset calibration error: {e}")
            finally:
                freeze_scale = prev_freeze if 'prev_freeze' in locals() else freeze_scale
        elif key == ord("n"):
            # Toggle manual ROI selection for calibration
            if not manual_roi_select_mode and not auto_calibrate_mode and not multi_cal_mode:
                manual_roi_select_mode = True
                manual_roi_points = []
                mouse_state["active"] = True
                mouse_state["points"] = []
                print("Manual ROI select: Click two opposite corners of your target region (then start G or I/O calibration)")
            elif manual_roi_select_mode:
                manual_roi_select_mode = False
                mouse_state["active"] = False
                manual_roi_points = mouse_state.get("points", manual_roi_points)
                mouse_state["points"] = []
                if len(manual_roi_points) == 2:
                    manual_roi_active = True
                    print("Manual ROI locked for calibration (auto-disables after calibration)")
                else:
                    manual_roi_active = False
                    print("Manual ROI cancelled")

        elif key == ord("i"):
            # Toggle multi-depth calibration wizard
            multi_cal_mode = not multi_cal_mode
            if multi_cal_mode:
                freeze_scale = True
                current_scale = 1.0
                collecting_sample = False
                mouse_state["active"] = False
                mouse_state["points"] = []
                multi_samples = []
                print("Multi-depth calibration: ON. Press O to capture a sample, ENTER to solve, BACKSPACE to delete last, I to exit.")
            else:
                freeze_scale = False
                collecting_sample = False
                mouse_state["active"] = False
                mouse_state["points"] = []
                print("Multi-depth calibration: OFF")
        elif key == ord("o"):
            # Begin capture of one sample (two clicks)
            if multi_cal_mode and not collecting_sample:
                collecting_sample = True
                mouse_state["active"] = True
                mouse_state["points"] = []
            print("Click two content points 150mm apart to capture sample")
        elif key == 8:  # BACKSPACE
            if multi_cal_mode and multi_samples:
                removed = multi_samples.pop()
                print(f"Removed sample: z={removed['z']:.0f}mm, measured={removed['measured_px']:.1f}px")
        elif key == 13:  # ENTER
            if multi_cal_mode and multi_samples:
                try:
                    import numpy as _np
                    A = []
                    b = []
                    ppmm_x = DISPLAY_WIDTH / DISPLAY_PHYSICAL_WIDTH_MM
                    ppmm_y = DISPLAY_HEIGHT / DISPLAY_PHYSICAL_HEIGHT_MM
                    for s in multi_samples:
                        z = float(s["z"])
                        d = (REFERENCE_DISTANCE_MM / z) - 1.0
                        target_px = 75.0 * (ppmm_x if s["axis"] == "x" else ppmm_y)
                        ratio = float(target_px) / float(s["measured_px"])  # desired scale at z
                        A.append([d, d * d])
                        b.append([ratio - 1.0])
                    A = _np.array(A, dtype=_np.float64)
                    b = _np.array(b, dtype=_np.float64)
                    sol, *_ = _np.linalg.lstsq(A, b, rcond=None)
                    k = float(sol[0][0])
                    q = float(sol[1][0])
                    globals()["SCALE_MODEL_K"] = k
                    globals()["SCALE_MODEL_Q"] = q
                    print(f"Fitted scale model: k={k:.4f}, q={q:.4f} (samples={len(multi_samples)})")
                    # Save to calibration.json
                    try:
                        cal_path = os.path.join(os.path.dirname(__file__), "calibration.json")
                        cal = {}
                        if os.path.exists(cal_path):
                            with open(cal_path, "r", encoding="utf-8") as f:
                                cal = json.load(f) or {}
                        cal["scale_model_k"] = k
                        cal["scale_model_q"] = q
                        with open(cal_path, "w", encoding="utf-8") as f:
                            json.dump(cal, f, indent=2)
                        print(f"Saved scale model to {cal_path}")
                    except Exception as e:
                        print(f"Save error: {e}")
                    # Exit wizard
                    multi_cal_mode = False
                    freeze_scale = False
                    collecting_sample = False
                    mouse_state["active"] = False
                    mouse_state["points"] = []
                except Exception as e:
                    print(f"Solve error: {e}")
        elif key == ord("g"):
            # Toggle Auto-calibrate 150mm mode
            auto_calibrate_mode = not auto_calibrate_mode
            if auto_calibrate_mode:
                # Save and then freeze scale, disable parallax and distance scaling for a clean measurement
                auto_saved_state = {
                    "freeze_scale": freeze_scale,
                    "current_scale": current_scale,
                    "distance_scale_weight_live": distance_scale_weight_live,
                    "parallax_weight_x_live": parallax_weight_x_live,
                    "parallax_weight_y_live": parallax_weight_y_live,
                }
                freeze_scale = True
                current_scale = 1.0
                distance_scale_weight_live = 0.0
                parallax_weight_x_live = 0.0
                parallax_weight_y_live = 0.0
                mouse_state["active"] = True
                mouse_state["points"] = []
                print("Auto-calibrate started: stand at reference distance and Click two content points 150mm apart")
            else:
                mouse_state["active"] = False
                mouse_state["points"] = []
                if auto_saved_state:
                    freeze_scale = auto_saved_state.get("freeze_scale", freeze_scale)
                    current_scale = auto_saved_state.get("current_scale", current_scale)
                    distance_scale_weight_live = auto_saved_state.get("distance_scale_weight_live", distance_scale_weight_live)
                    parallax_weight_x_live = auto_saved_state.get("parallax_weight_x_live", parallax_weight_x_live)
                    parallax_weight_y_live = auto_saved_state.get("parallax_weight_y_live", parallax_weight_y_live)
                    auto_saved_state = None
                print("Auto-calibrate cancelled")
        
        # Real-time calibration controls
        # Arrow keys (codes may vary by system)
        elif key == 82 or key == 2490368:  # Up arrow
            IMAGE_OFFSET_Y -= 5
            print(f"Image offset: X={IMAGE_OFFSET_X}, Y={IMAGE_OFFSET_Y}")
        elif key == 84 or key == 2621440:  # Down arrow
            IMAGE_OFFSET_Y += 5
            print(f"Image offset: X={IMAGE_OFFSET_X}, Y={IMAGE_OFFSET_Y}")
        elif key == 81 or key == 2424832:  # Left arrow
            IMAGE_OFFSET_X -= 5
            print(f"Image offset: X={IMAGE_OFFSET_X}, Y={IMAGE_OFFSET_Y}")
        elif key == 83 or key == 2555904:  # Right arrow
            IMAGE_OFFSET_X += 5
            print(f"Image offset: X={IMAGE_OFFSET_X}, Y={IMAGE_OFFSET_Y}")
        
        # Alternative letter keys for adjustment
        elif key == ord("w"):  # W = move display UP
            IMAGE_OFFSET_Y -= 5
            print(f"Image offset: X={IMAGE_OFFSET_X}, Y={IMAGE_OFFSET_Y}")
        elif key == ord("s"):  # S = move display DOWN
            IMAGE_OFFSET_Y += 5
            print(f"Image offset: X={IMAGE_OFFSET_X}, Y={IMAGE_OFFSET_Y}")
        elif key == ord("a"):  # A = move display LEFT
            IMAGE_OFFSET_X -= 5
            print(f"Image offset: X={IMAGE_OFFSET_X}, Y={IMAGE_OFFSET_Y}")
        elif key == ord("d"):  # D = move display RIGHT
            IMAGE_OFFSET_X += 5
            print(f"Image offset: X={IMAGE_OFFSET_X}, Y={IMAGE_OFFSET_Y}")
        
        # Scale adjustment controls
        elif key == ord("=") or key == ord("+"):  # + or = to increase scale (make image larger)
            REFERENCE_DISTANCE_MM = min(3000, REFERENCE_DISTANCE_MM + 50)  # Cap at 3000mm
            print(f"Reference distance: {REFERENCE_DISTANCE_MM}mm (larger image)")
        elif key == ord("-") or key == ord("_"):  # - to decrease scale (make image smaller)
            REFERENCE_DISTANCE_MM = max(100, REFERENCE_DISTANCE_MM - 50)
            print(f"Reference distance: {REFERENCE_DISTANCE_MM}mm (smaller image)")
        
        elif key == ord("r"):
            # Reset offsets
            IMAGE_OFFSET_X = 0
            IMAGE_OFFSET_Y = 0
            print("Image offsets reset to 0")
        elif key == ord("v"):
            # Toggle two-point baseline calibration (solves CAMERA_OFFSET_X/Y_MM)
            try:
                baseline_cal_mode = not baseline_cal_mode
                if baseline_cal_mode:
                    baseline_prev_freeze = freeze_scale
                    freeze_scale = True
                    baseline_samples = []
                    print("Baseline calibration: Press 1 at first distance, then move closer/farther and press 2.")
                else:
                    # Exit and restore
                    freeze_scale = baseline_prev_freeze if baseline_prev_freeze is not None else freeze_scale
                    baseline_prev_freeze = None
                    baseline_samples = []
                    print("Baseline calibration cancelled")
            except Exception as e:
                print(f"Baseline calibration toggle error: {e}")
        # Handle two-point baseline calibration captures
        if baseline_cal_mode and (key == ord('1') or key == ord('2')):
            try:
                # Compute face position on display and depth z
                cx_disp = None; cy_disp = None; z_val = None
                if last_face_center_cam is not None and last_depth_raw is not None and orig_w > 0 and orig_h > 0:
                    cx_cam, cy_cam = last_face_center_cam
                    # Map to mirrored resized coords
                    fx_mir = (orig_w - 1) - float(cx_cam)
                    fy = float(cy_cam)
                    if last_final_w and last_final_h:
                        scale_x = float(last_final_w) / float(orig_w)
                        scale_y = float(last_final_h) / float(orig_h)
                        fx_res = int(round(fx_mir * scale_x))
                        fy_res = int(round(fy * scale_y))
                    else:
                        fx_res = int(round(fx_mir))
                        fy_res = int(round(fy))
                    cx_disp = int(last_dst_x_start + (fx_res - last_src_x_start))
                    cy_disp = int(last_dst_y_start + (fy_res - last_src_y_start))
                    # Raw depth at face center (median window)
                    dh, dw = last_depth_raw.shape[:2]
                    sx = int(round(cx_cam * (dw / float(orig_w))))
                    sy = int(round(cy_cam * (dh / float(orig_h))))
                    r = 6
                    x1s = max(0, sx - r); x2s = min(dw - 1, sx + r)
                    y1s = max(0, sy - r); y2s = min(dh - 1, sy + r)
                    win = last_depth_raw[y1s:y2s+1, x1s:x2s+1].astype(np.float32)
                    valid = (win >= MIN_DISTANCE_MM) & (win <= MAX_DISTANCE_MM)
                    if np.any(valid):
                        z_val = float(np.median(win[valid]))
                if cx_disp is None or cy_disp is None or z_val is None or z_val <= 0:
                    print("Baseline capture failed: need face and valid depth")
                else:
                    target_x = int(round(DISPLAY_WIDTH * float(ALIGN_TARGET_X_REL)))
                    target_y = int(round(DISPLAY_HEIGHT * float(ALIGN_TARGET_Y_REL)))
                    dx = float(target_x - cx_disp)
                    dy = float(target_y - cy_disp)
                    # Compute p(z) with current exponent
                    p = 1.0
                    try:
                        p = (float(REFERENCE_DISTANCE_MM) / float(z_val)) ** float(parallax_distance_exp_live)
                    except Exception:
                        p = float(REFERENCE_DISTANCE_MM) / max(1e-6, float(z_val))
                    baseline_samples.append({"z": z_val, "dx": dx, "dy": dy, "p": p})
                    print(f"Baseline sample {len(baseline_samples)}: z={z_val:.0f}mm, dx={dx:.1f}px, dy={dy:.1f}px, p={p:.3f}")
                    if len(baseline_samples) >= 2:
                        s1, s2 = baseline_samples[0], baseline_samples[1]
                        dp = float(s1["p"]) - float(s2["p"])
                        if abs(dp) < 1e-3:
                            print("Baseline solve failed: move to a different distance before second capture")
                        else:
                            # Solve base_px for each axis from difference
                            base_px_x = (float(s1["dx"]) - float(s2["dx"])) / dp
                            base_px_y = (float(s1["dy"]) - float(s2["dy"])) / dp
                            # Convert to CAMERA_OFFSET_MM (note: base_px = -GAIN * OFFSET_MM * ppmm)
                            ppmm_x = float(DISPLAY_WIDTH) / max(1e-6, float(DISPLAY_PHYSICAL_WIDTH_MM))
                            ppmm_y = float(DISPLAY_HEIGHT) / max(1e-6, float(DISPLAY_PHYSICAL_HEIGHT_MM))
                            gain = float(CAMERA_PARALLAX_GAIN) if 'CAMERA_PARALLAX_GAIN' in globals() else 2.0
                            try:
                                off_x_mm = - base_px_x / max(1e-6, gain * ppmm_x)
                                off_y_mm = - base_px_y / max(1e-6, gain * ppmm_y)
                            except Exception:
                                off_x_mm = 0.0; off_y_mm = 0.0
                            CAMERA_OFFSET_X_MM = int(round(off_x_mm))
                            CAMERA_OFFSET_Y_MM = int(round(off_y_mm))
                            # Persist
                            try:
                                cal_path = os.path.join(os.path.dirname(__file__), "calibration.json")
                                cal = {}
                                if os.path.exists(cal_path):
                                    with open(cal_path, "r", encoding="utf-8") as f:
                                        cal = json.load(f) or {}
                                cal["CAMERA_OFFSET_X_MM"] = int(CAMERA_OFFSET_X_MM)
                                cal["CAMERA_OFFSET_Y_MM"] = int(CAMERA_OFFSET_Y_MM)
                                with open(cal_path, "w", encoding="utf-8") as f:
                                    json.dump(cal, f, indent=2)
                            except Exception:
                                pass
                            print(f"Baseline solved: CAMERA_OFFSET_X_MM={CAMERA_OFFSET_X_MM}mm, Y={CAMERA_OFFSET_Y_MM}mm (from base_px_x={base_px_x:.1f}, base_px_y={base_px_y:.1f})")
                            # Exit wizard
                            baseline_cal_mode = False
                            freeze_scale = baseline_prev_freeze if baseline_prev_freeze is not None else freeze_scale
                            baseline_prev_freeze = None
                            baseline_samples = []
            except Exception as e:
                print(f"Baseline capture error: {e}")

        elif key == ord("h"):
            # Show help
            print("\n=== CALIBRATION CONTROLS ===")
            print("Arrow Keys OR W/A/S/D: Adjust image position")
            print("+/=: Increase scale (make image larger)")
            print("-: Decrease scale (make image smaller)")
            print("R: Reset position offsets")
            print("F: Toggle fullscreen")
            print("M: Move window to next monitor")
            print("I: Multi-depth calibration (O capture, ENTER solve, BACKSPACE undo)")
            print("G: Auto-calibrate 150mm (click two points)")
            print("P: Save screenshot")
            print("Q: Quit (and show final values)")
            print("H: Show this help")
            print("===========================\n")
    
    cv2.destroyAllWindows()
    print("Magic Mirror closed!")
    print(f"Final calibration values:")
    print(f"  IMAGE_OFFSET_X = {IMAGE_OFFSET_X}")
    print(f"  IMAGE_OFFSET_Y = {IMAGE_OFFSET_Y}")
    print(f"  REFERENCE_DISTANCE_MM = {REFERENCE_DISTANCE_MM}")



    # Throttled console prints for fixed ROI depth sanity check
    last_fixed_roi_print_z = None
    last_fixed_roi_print_time = 0.0
    last_raw_z = None
    last_clamped_z = None


















