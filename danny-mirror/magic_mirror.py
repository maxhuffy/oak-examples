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
4. Scale to display: 640x480 → varies (fits portrait display 2160x3840)
5. Apply offsets: align with real reflection
6. Crop to display: 2160x3840 (final output)
"""

import os
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
MAX_DISTANCE_MM = 2500       # Maximum distance to track (increased to handle far detections)
MIN_SCALE = 0.2              # Minimum allowed scale (allows smaller for far distances)
MAX_SCALE = 3.0              # Maximum allowed scale (prevents image from becoming too large)

# CRITICAL FOR TRUE 1:1 SCALING: Measure your display's physical dimensions
DISPLAY_PHYSICAL_WIDTH_MM = 620    # Physical width of display in millimeters  
DISPLAY_PHYSICAL_HEIGHT_MM = 1100   # Physical height of display in millimeters

# Scale multiplier for 1:1 life-size at reference distance
# CALIBRATED FROM IMAGE: At 1500mm with scale frozen at 1.0, image filled ~50% of screen
# This means we need roughly 2× larger → setting to 2.2 for safety margin
LIFE_SIZE_SCALE_MULTIPLIER = 2.2

# Camera/Display Alignment Configuration
# MEASURED: Camera is ~7cm above display top + ~3cm tape = ~100mm total
CAMERA_OFFSET_Y_MM = 60      # Camera/eyes vertical delta; positive moves image up
CAMERA_OFFSET_X_MM = 0       # Centered horizontally

# Calibration factor: Measured from ruler in image
# 22 inches (558.8mm) vertical ruler spans roughly 1700 pixels
# Actual pixels/mm ≈ 3.04 (vs theoretical 3.49 from 3840/1100)
PIXELS_PER_MM = 3.49  # Fallback; dynamic per-axis computed from display size below

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

    print("Pipeline created. Starting Magic Mirror...")
    pipeline.start()

    # OpenCV window - setup for second monitor
    window_name = "Magic Mirror"
    cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
    
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
    SMOOTHING_FACTOR = 0.5  # Even faster response to recover from bad readings
    
    # Moving average filter for depth - reduced for faster response
    from collections import deque
    depth_history = deque(maxlen=3)  # Keep only last 3 readings for faster response

    # Keep track of last valid distance - start at reference so image stays at 1.0 scale initially
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
    # Track last reliable face center in camera pixels (640x480, pre-flip)
    last_face_center_px = None
    
    # Calibration mode
    calibration_mode = False
    calibration_step = 0  # 0=not started, 1=set reference distance, 2=adjust offset
    saved_distance = None
    fixed_roi_mode = False  # Toggle for using fixed center ROI instead of face tracking
    freeze_scale = False  # Toggle to lock scale at 1.0 for calibration
    
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
        if det_msg is not None:
            # Store detection info for lingering visualization
            for detection in det_msg.detections:
                try:
                    # Filter by confidence - only keep high-confidence detections (>75%)
                    confidence = detection.confidence if hasattr(detection, 'confidence') else 1.0
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

                        recent_detections.append({
                            'center_x': center_x,
                            'center_y': center_y,
                            'radius': radius,
                            'confidence': detection.confidence if hasattr(detection, 'confidence') else 1.0
                        })
                        # Update last face center for parallax tracking
                        last_face_center_px = (center_x, center_y)
                        
                except Exception as e:
                    # Skip detections that fail to process
                    pass
                    pass
        
        if video_msg is not None:
            # Convert to OpenCV format
            frame = video_msg.getCvFrame()
            
            # Camera is landscape orientation (correct for face detection)
            # Apply horizontal flip for mirror effect
            frame = cv2.flip(frame, 1)
            
            # Get original frame dimensions (640w x 480h landscape)
            orig_h, orig_w = frame.shape[:2]
            
            # Calculate scale based on depth FIRST (before any resizing)
            target_scale = current_scale  # Default to current scale
            
            distance_scale_weight = getattr(args, "distance_scale_weight", 0.0)
            if distance_msg is not None:
                z_mm = distance_msg.spatials.z
                
                # Only use valid depth readings (not NaN or 0)
                if z_mm > 0 and not np.isnan(z_mm):
                    # Add to history and compute moving average
                    depth_history.append(z_mm)
                    z_mm = sum(depth_history) / len(depth_history)
                    
                    # Clamp distance to reasonable range
                    z_mm = max(MIN_DISTANCE_MM, min(MAX_DISTANCE_MM, z_mm))
                    
                    # Update last valid distance
                    last_valid_distance = z_mm
                    frames_without_detection = 0
                    
                    # Scale inversely proportional to distance, softened via weight (unless frozen)
                    if not freeze_scale:
                        distance_factor = REFERENCE_DISTANCE_MM / z_mm
                        # Clamp factor range before weighting
                        distance_factor = max(MIN_SCALE, min(MAX_SCALE, distance_factor))
                        # Interpolate toward 1.0 based on weight
                        target_scale = 1.0 + distance_scale_weight * (distance_factor - 1.0)
                    else:
                        # Scale frozen at 1.0 for calibration
                        target_scale = 1.0
                else:
                    # Invalid depth, use last known distance
                    frames_without_detection += 1
                    if frames_without_detection < MAX_FRAMES_WITHOUT_DETECTION:
                        if not freeze_scale:
                            distance_factor = REFERENCE_DISTANCE_MM / last_valid_distance
                            distance_factor = max(MIN_SCALE, min(MAX_SCALE, distance_factor))
                            target_scale = 1.0 + distance_scale_weight * (distance_factor - 1.0)
                        else:
                            target_scale = 1.0
            else:
                # No distance message, use last known distance
                frames_without_detection += 1
                if frames_without_detection < MAX_FRAMES_WITHOUT_DETECTION:
                    if not freeze_scale:
                        distance_factor = REFERENCE_DISTANCE_MM / last_valid_distance
                        distance_factor = max(MIN_SCALE, min(MAX_SCALE, distance_factor))
                        target_scale = 1.0 + distance_scale_weight * (distance_factor - 1.0)
                    else:
                        target_scale = 1.0
            
            # Smooth the scale transition
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
            # At reference distance: current_scale = 1.0 → use base_scale (calibrated for 1:1)
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
            # - If camera is ABOVE your eyes → shift image UP (so it appears at eye level)
            # - If camera is BELOW your eyes → shift image DOWN
            # - The offset scales with distance (closer = bigger offset needed)
            #
            # Example: Camera 150mm above your eyes, you're 500mm away
            # - At 500mm: offset = baseline
            # - At 250mm (closer): offset doubles (parallax effect stronger)
            # - At 1000mm (farther): offset halves (parallax effect weaker)
            
            target_offset_x = float(IMAGE_OFFSET_X)
            target_offset_y = float(IMAGE_OFFSET_Y)
            
            if last_valid_distance > 0:
                # Camera above your eye line means image should shift UP (negative Y in OpenCV)
                # Compute pixels-per-mm dynamically from the chosen display
                ppmm_y = (DISPLAY_HEIGHT / DISPLAY_PHYSICAL_HEIGHT_MM) if DISPLAY_PHYSICAL_HEIGHT_MM else PIXELS_PER_MM
                ppmm_x = (DISPLAY_WIDTH / DISPLAY_PHYSICAL_WIDTH_MM) if DISPLAY_PHYSICAL_WIDTH_MM else PIXELS_PER_MM

                # Baseline static camera/display offset
                y_offset_pixels = -CAMERA_OFFSET_Y_MM * ppmm_y
                x_offset_pixels = -CAMERA_OFFSET_X_MM * ppmm_x

                target_offset_x += x_offset_pixels
                target_offset_y += y_offset_pixels

                # Dynamic parallax from eye/head position (use face center in pixels for stability)
                try:
                    parallax_wx = getattr(args, "parallax_weight_x", 1.0)
                    parallax_wy = getattr(args, "parallax_weight_y", 1.0)
                    parallax_exp = getattr(args, "parallax_distance_scale", 1.0)
                    # Prefer the filtered last_valid_distance for stability
                    parallax_factor = (REFERENCE_DISTANCE_MM / last_valid_distance)
                    if parallax_exp != 1.0:
                        parallax_factor = parallax_factor ** parallax_exp
                    if last_face_center_px is not None:
                        # Convert detection center to mirrored camera coords
                        cam_cx, cam_cy = last_face_center_px
                        mirror_cx = (orig_w - cam_cx)
                        mirror_cy = cam_cy
                        # delta from image center in camera pixels
                        dx_cam = (orig_w / 2.0) - mirror_cx
                        dy_cam = (orig_h / 2.0) - mirror_cy
                        # scale to final display pixels
                        dx_disp = dx_cam * combined_scale
                        dy_disp = dy_cam * combined_scale
                        # apply distance-aware parallax
                        target_offset_x += parallax_wx * parallax_factor * dx_disp
                        target_offset_y += parallax_wy * parallax_factor * dy_disp
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
            cv2.imshow(window_name, frame)
        
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
                if distance_msg is not None and distance_msg.spatials.z > 0:
                    REFERENCE_DISTANCE_MM = int(distance_msg.spatials.z)
                    calibration_step = 1
                    current_scale = 1.0  # Reset scale
                    print(f"\n✓ Reference distance set to {REFERENCE_DISTANCE_MM}mm")
                    print("Now adjust image position with arrow keys or WASD")
                    print("Press SPACE when aligned")
                else:
                    print("ERROR: No valid face detected. Move closer and try again.")
            elif calibration_step == 1:
                # Finish calibration
                calibration_mode = False
                calibration_step = 0
                print(f"\n✓ CALIBRATION COMPLETE ✓")
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
            # Toggle freeze scale mode
            freeze_scale = not freeze_scale
            if freeze_scale:
                current_scale = 1.0  # Lock at 1.0
                print("\n*** SCALE FROZEN AT 1.0 ***")
                print("No distance-based scaling - image stays constant size")
                print("Perfect for phone camera calibration")
                print("Press 'z' again to re-enable scaling")
            else:
                print("*** Scale unfrozen - distance-based scaling enabled ***")
        elif key == ord("f"):
            # Toggle fullscreen
            is_fullscreen = cv2.getWindowProperty(window_name, cv2.WND_PROP_FULLSCREEN)
            if is_fullscreen == cv2.WINDOW_FULLSCREEN:
                cv2.setWindowProperty(window_name, cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_NORMAL)
            else:
                cv2.setWindowProperty(window_name, cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)
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
        elif key == ord("h"):
            # Show help
            print("\n=== CALIBRATION CONTROLS ===")
            print("Arrow Keys OR W/A/S/D: Adjust image position")
            print("+/=: Increase scale (make image larger)")
            print("-: Decrease scale (make image smaller)")
            print("R: Reset position offsets")
            print("F: Toggle fullscreen")
            print("M: Move window to next monitor")
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
