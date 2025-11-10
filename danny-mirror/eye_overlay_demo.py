#!/usr/bin/env python3

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

with dai.Pipeline(device) as pipeline:
    print("Creating pipeline with face detection + eye overlay...")

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

    print("Pipeline created. Starting...")
    pipeline.start()

    # OpenCV window
    cv2.namedWindow("Eye Tracking Overlay", cv2.WINDOW_NORMAL)

    # For FPS calculation
    fps_counter = 0
    fps_start_time = time.time()
    current_fps = 0

    while pipeline.isRunning():
        # Get video frame
        video_msg = video_queue.tryGet()
        det_msg = detections_queue.tryGet()
        distance_msg = distance_queue.tryGet()
        
        if video_msg is not None:
            # Convert to OpenCV format
            frame = video_msg.getCvFrame()
            h, w = frame.shape[:2]
            
            # Process detections if available
            if det_msg is not None:
                for detection in det_msg.detections:
                    # Try to draw bounding box if available
                    try:
                        # Check for different possible bbox attributes
                        if hasattr(detection, 'roi'):
                            bbox = detection.roi.denormalize(w, h)
                            x1, y1 = int(bbox.topLeft().x), int(bbox.topLeft().y)
                            x2, y2 = int(bbox.bottomRight().x), int(bbox.bottomRight().y)
                        else:
                            # Skip bbox if we can't find it
                            x1 = y1 = x2 = y2 = 0
                        
                        if x1 > 0 and y1 > 0:
                            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                            
                            # Label
                            label = f"{detection.label}" if hasattr(detection, 'label') else "Face"
                            if hasattr(detection, 'confidence'):
                                label += f" {detection.confidence:.2f}"
                            cv2.putText(frame, label, (x1, y1 - 10),
                                      cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
                    except Exception as e:
                        # If bbox fails, just continue with keypoints
                        pass
                    
                    # Draw keypoints (eyes, nose, etc.)
                    if hasattr(detection, 'keypoints') and detection.keypoints:
                        for idx, kp in enumerate(detection.keypoints):
                            kp_x = int(kp.x * w)
                            kp_y = int(kp.y * h)
                            
                            # Draw different colors for different keypoints
                            # Assuming: 0-1 are eyes, 2 is nose, 3-4 are mouth corners
                            if idx < 2:  # Eyes
                                color = (0, 255, 255)  # Yellow for eyes
                                radius = 8
                                # Draw a glow effect around eyes
                                cv2.circle(frame, (kp_x, kp_y), radius + 4, (0, 200, 200), 2)
                                cv2.circle(frame, (kp_x, kp_y), radius, color, -1)
                                cv2.circle(frame, (kp_x, kp_y), radius, (255, 255, 255), 1)
                            elif idx == 2:  # Nose
                                color = (255, 0, 255)  # Magenta for nose
                                radius = 5
                                cv2.circle(frame, (kp_x, kp_y), radius, color, -1)
                            else:  # Mouth corners
                                color = (255, 0, 0)  # Blue for mouth
                                radius = 5
                                cv2.circle(frame, (kp_x, kp_y), radius, color, -1)
                            
                            # Add keypoint index label
                            cv2.putText(frame, str(idx), (kp_x + 10, kp_y - 10),
                                      cv2.FONT_HERSHEY_SIMPLEX, 0.4, color, 1)
            
            # Display depth information if available
            if distance_msg is not None:
                x_mm = distance_msg.spatials.x
                y_mm = distance_msg.spatials.y
                z_mm = distance_msg.spatials.z
                
                # Create info overlay
                info_y = 30
                cv2.putText(frame, f"Depth: {z_mm:.0f}mm", (10, info_y),
                          cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
                cv2.putText(frame, f"X: {x_mm:.0f}mm  Y: {y_mm:.0f}mm", (10, info_y + 30),
                          cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
            
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
            cv2.imshow("Eye Tracking Overlay", frame)
        
        # Handle key presses
        key = cv2.waitKey(1)
        if key == ord("q"):
            print("Quitting...")
            break
        elif key == ord("s"):
            # Save screenshot
            if video_msg is not None:
                filename = f"screenshot_{int(time.time())}.jpg"
                cv2.imwrite(filename, frame)
                print(f"Screenshot saved: {filename}")
    
    cv2.destroyAllWindows()
    print("Done!")