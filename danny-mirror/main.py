#!/usr/bin/env python3

import os
import time
from dotenv import load_dotenv

import cv2
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

visualizer = dai.RemoteConnection(httpPort=8082)
device = dai.Device(dai.DeviceInfo(args.device)) if args.device else dai.Device()
platform = device.getPlatformAsString()
print(f"Platform: {platform}")

with dai.Pipeline(device) as pipeline:
    print("Creating pipeline with face detection + auto-ROI...")

    # RGB input for NN (camera by default, or media if provided)
    # When using camera, explicitly request 640x480 and align StereoDepth to the RGB camera
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

    # Load model (default yunet) from Zoo
    model_description = dai.NNModelDescription(f"yolov6_nano_r2_coco.{platform}.yaml")
    if model_description.model != args.model:
        model_description = dai.NNModelDescription(args.model, platform=platform)
    nn_archive = dai.NNArchive(dai.getModelFromZoo(model_description))

    # NN with parser
    nn_with_parser: ParsingNeuralNetwork = pipeline.create(ParsingNeuralNetwork).build(
        input_node, nn_archive, fps=args.fps_limit
    )

    # Prefer dropping frames to reduce end-to-end latency
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
    # Align depth/disparity to the RGB camera so detections and depth share the same viewpoint
    try:
        stereo.setDepthAlign(dai.CameraBoardSocket.CAM_A)
        if use_camera and platform == "RVC2":
            stereo.setOutputSize(*VIDEO_RESOLUTION)
    except Exception:
        pass

    depth_color_transform = pipeline.create(ApplyColormap).build(stereo.disparity)
    depth_color_transform.setColormap(cv2.COLORMAP_JET)

    # initial ROI; will be overridden as soon as a face is detected
    initial_roi = RegionOfInterest(300, 150, 340, 190)
    measure_distance = pipeline.create(MeasureDistance).build(
        stereo.depth, device.readCalibration(), initial_roi
    )

    # Auto-update ROI from face detections (normalized -> pixel coords in disparity space)
    roi_from_face = pipeline.create(ROIFromFace).build(
        disparity_frames=depth_color_transform.out,
        parser_output=nn_with_parser.out,
    )
    roi_from_face.output_roi.link(measure_distance.roi_input)

    # Visualizer topics (Video + Detections)
    if args.overlay_mode:
        apply_colormap_node = pipeline.create(ApplyColormap).build(nn_with_parser.out)
        overlay_frames_node = pipeline.create(ImgFrameOverlay).build(
            nn_with_parser.passthrough, apply_colormap_node.out
        )
        visualizer.addTopic("Video", overlay_frames_node.out, "images")
    else:
        visualizer.addTopic("Video", nn_with_parser.passthrough, "images")
    visualizer.addTopic("Detections", nn_with_parser.out, "detections")

    # Annotate ROI on disparity for visual verification
    from utils.roi_annotator import ROIAnnotator
    roi_annot = pipeline.create(ROIAnnotator).build(
        disparity_frames=depth_color_transform.out
    )
    roi_from_face.output_roi.link(roi_annot.roi_input)
    measure_distance.output.link(roi_annot.distance_input)
    visualizer.addTopic("Disparity", roi_annot.passthrough)
    visualizer.addTopic("ROI", roi_annot.annotation_output)

    # Create a host-side queue to inspect detections (for debugging/inspection)
    parser_output_queue = nn_with_parser.out.createOutputQueue()
    # Modularized keypoints printer (throttled)
    from utils.keypoints_debug import KeypointsPrinter
    keypoints_printer = KeypointsPrinter(interval_seconds=0.5)

    print("Pipeline created.")
    pipeline.start()
    visualizer.registerPipeline(pipeline)


    while pipeline.isRunning():
        # Non-blocking read of detections to print face keypoints
        det_msg = parser_output_queue.tryGet()
        if det_msg is not None:
            keypoints_printer.maybe_print(det_msg)

        key = visualizer.waitKey(1)
        if key == ord("q"):
            print("Got q key from the remote connection!")
            break
