#!/usr/bin/env python3
"""
Magic Mirror Effect - Scales the video feed based on user distance to create
a realistic mirror reflection effect.

The image scales proportionally: as you move closer, your face appears larger
at the same rate it would in a real mirror.
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
REFERENCE_DISTANCE_MM = 600  # Distance where scale = 1.0 (600mm = ~2 feet)
MIN_DISTANCE_MM = 300        # Minimum distance to prevent extreme scaling
MAX_DISTANCE_MM = 2000       # Maximum distance to track

with dai.Pipeline(device) as pipeline:
    print("Creating Magic Mirror pipeline...")

    # RGB input for NN (camera by default, or media if provided)
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

    # OpenCV window - fullscreen for mirror effect
    window_name = "Magic Mirror"
    cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
    
    # Uncomment next line for fullscreen mode:
    # cv2.setWindowProperty(window_name, cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)

    # For FPS calculation
    fps_counter = 0
    fps_start_time = time.time()
    current_fps = 0
    
    # Enhanced smoothing for scale changes (avoid jitter)
    current_scale = 1.0
    SMOOTHING_FACTOR = 0.01  # Lower = smoother but slower response
    
    # Moving average filter for depth
    from collections import deque
    depth_history = deque(maxlen=100)  # Keep last 100 depth readings
    
    # Keep track of last valid distance
    last_valid_distance = REFERENCE_DISTANCE_MM
    frames_without_detection = 0
    MAX_FRAMES_WITHOUT_DETECTION = 30  # Hold scale for ~1 second at 30fps
    
    while pipeline.isRunning():
        # Get video frame
        video_msg = video_queue.tryGet()
        det_msg = detections_queue.tryGet()
        distance_msg = distance_queue.tryGet()
        
        if video_msg is not None:
            # Convert to OpenCV format
            frame = video_msg.getCvFrame()
            
            # Flip horizontally for mirror effect
            frame = cv2.flip(frame, 1)
            
            h, w = frame.shape[:2]
            
            # Calculate scale based on depth
            target_scale = current_scale  # Default to current scale
            
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
                    
                    # Scale inversely proportional to distance
                    # At reference distance (600mm), scale = 1.0
                    # At 300mm (half distance), scale = 2.0
                    # At 1200mm (double distance), scale = 0.5
                    target_scale = REFERENCE_DISTANCE_MM / z_mm
                else:
                    # Invalid depth, use last known distance
                    frames_without_detection += 1
                    if frames_without_detection < MAX_FRAMES_WITHOUT_DETECTION:
                        target_scale = REFERENCE_DISTANCE_MM / last_valid_distance
            else:
                # No distance message, use last known distance
                frames_without_detection += 1
                if frames_without_detection < MAX_FRAMES_WITHOUT_DETECTION:
                    target_scale = REFERENCE_DISTANCE_MM / last_valid_distance
            
            # Smooth the scale transition
            current_scale += (target_scale - current_scale) * SMOOTHING_FACTOR
            
            # Apply scaling to the frame
            if current_scale != 1.0:
                # Calculate new dimensions
                new_w = int(w * current_scale)
                new_h = int(h * current_scale)
                
                # Resize the frame
                scaled_frame = cv2.resize(frame, (new_w, new_h), interpolation=cv2.INTER_LINEAR)
                
                # Create output frame (same size as original)
                output_frame = np.zeros((h, w, 3), dtype=np.uint8)
                
                # Calculate crop/paste position (center the scaled image)
                if current_scale > 1.0:
                    # Image is larger - crop from center
                    start_x = (new_w - w) // 2
                    start_y = (new_h - h) // 2
                    output_frame = scaled_frame[start_y:start_y+h, start_x:start_x+w]
                else:
                    # Image is smaller - paste in center
                    start_x = (w - new_w) // 2
                    start_y = (h - new_h) // 2
                    output_frame[start_y:start_y+new_h, start_x:start_x+new_w] = scaled_frame
                
                frame = output_frame
            
            # Optional: Draw debug info
            if distance_msg is not None:
                x_mm = distance_msg.spatials.x
                y_mm = distance_msg.spatials.y
                z_mm = distance_msg.spatials.z
                
                # Debug overlay (comment out for clean mirror)
                info_y = 30
                cv2.putText(frame, f"Distance: {z_mm:.0f}mm", (10, info_y),
                          cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
                cv2.putText(frame, f"Scale: {current_scale:.2f}x", (10, info_y + 30),
                          cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            
            # Calculate and display FPS
            fps_counter += 1
            elapsed = time.time() - fps_start_time
            if elapsed > 1.0:
                current_fps = fps_counter / elapsed
                fps_counter = 0
                fps_start_time = time.time()
            
            cv2.putText(frame, f"FPS: {current_fps:.1f}", (w - 120, 30),
                      cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
            
            # Display the frame
            cv2.imshow(window_name, frame)
        
        # Handle key presses
        key = cv2.waitKey(1)
        if key == ord("q"):
            print("Quitting...")
            break
        elif key == ord("f"):
            # Toggle fullscreen
            is_fullscreen = cv2.getWindowProperty(window_name, cv2.WND_PROP_FULLSCREEN)
            if is_fullscreen == cv2.WINDOW_FULLSCREEN:
                cv2.setWindowProperty(window_name, cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_NORMAL)
            else:
                cv2.setWindowProperty(window_name, cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)
        elif key == ord("s"):
            # Save screenshot
            if video_msg is not None:
                filename = f"mirror_screenshot_{int(time.time())}.jpg"
                cv2.imwrite(filename, frame)
                print(f"Screenshot saved: {filename}")
    
    cv2.destroyAllWindows()
    print("Magic Mirror closed!")
