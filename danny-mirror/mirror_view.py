#!/usr/bin/env python3

import os
from dotenv import load_dotenv

import cv2
import depthai as dai
from depthai_nodes.node import ParsingNeuralNetwork

from utils.arguments import initialize_argparser
from utils.measure_distance import MeasureDistance, RegionOfInterest
from utils.roi_from_face import ROIFromFace
from utils.color_xyz_annotator import ColorXYZAnnotator


load_dotenv(override=True)

_, args = initialize_argparser()

if args.api_key:
    os.environ["DEPTHAI_HUB_API_KEY"] = args.api_key

visualizer = dai.RemoteConnection(httpPort=8082)
device = dai.Device(dai.DeviceInfo(args.device)) if args.device else dai.Device()
platform = device.getPlatformAsString()
print(f"Platform: {platform}")

with dai.Pipeline(device) as pipeline:
    print("Creating mirror view pipeline (1080p color + low-res compute)...")

    # Color camera at 1080p for display, plus a lower-res BGR stream for NN
    COLOR_RES = (1920, 1080)
    NN_RES = (640, 480)

    color = pipeline.create(dai.node.Camera).build(dai.CameraBoardSocket.CAM_A)
    # 1080p preview for streaming - use BGR to ensure perspective transform compatibility
    preview_type = dai.ImgFrame.Type.BGR888p if platform == "RVC2" else dai.ImgFrame.Type.BGR888i
    color_preview = color.requestOutput(COLOR_RES, preview_type, fps=args.fps_limit)

    # Lower-res BGR for the NN (format differs per platform)
    frame_type = dai.ImgFrame.Type.BGR888p if platform == "RVC2" else dai.ImgFrame.Type.BGR888i
    nn_input = color.requestOutput(NN_RES, frame_type, fps=args.fps_limit)

    # Mono cameras + StereoDepth aligned to color
    left = pipeline.create(dai.node.Camera).build(dai.CameraBoardSocket.CAM_B)
    right = pipeline.create(dai.node.Camera).build(dai.CameraBoardSocket.CAM_C)

    stereo = pipeline.create(dai.node.StereoDepth).build(
        left=left.requestOutput(NN_RES, fps=args.fps_limit),
        right=right.requestOutput(NN_RES, fps=args.fps_limit),
        presetMode=dai.node.StereoDepth.PresetMode.DEFAULT,
    )
    try:
        stereo.setDepthAlign(dai.CameraBoardSocket.CAM_A)
        if platform == "RVC2":
            stereo.setOutputSize(*NN_RES)
    except Exception:
        pass

    # Face detector (default Yunet)
    model_description = dai.NNModelDescription(f"yolov6_nano_r2_coco.{platform}.yaml")
    if model_description.model != args.model:
        model_description = dai.NNModelDescription(args.model, platform=platform)
    nn_archive = dai.NNArchive(dai.getModelFromZoo(model_description))

    nn: ParsingNeuralNetwork = pipeline.create(ParsingNeuralNetwork).build(
        nn_input, nn_archive, fps=args.fps_limit
    )
    try:
        nn.input.setMaxSize(1)
        nn.input.setBlocking(False)
        nn.setNumPoolFrames(4)
        nn.setNNArchive(nn_archive, numShaves=7)
    except Exception:
        pass

    # Measure distance from ROI over depth
    initial_roi = RegionOfInterest(300, 150, 340, 190)
    measure_distance = pipeline.create(MeasureDistance).build(
        stereo.depth, device.readCalibration(), initial_roi
    )

    # ROI from detections (eyes if flag enabled), computed in NN/disparity resolution space
    roi_from_face = pipeline.create(ROIFromFace).build(
        disparity_frames=stereo.disparity,
        parser_output=nn.out,
        use_eye_roi=args.eye_roi,
    )
    roi_from_face.output_roi.link(measure_distance.roi_input)

    # Apply a vertical flip using a 3x3 perspective transform matrix (normalized coordinates)
    # Matrix H maps [x, y, 1]^T -> [x', y', 1]^T with x' = x, y' = 1 - y:
    #   H = [[1, 0, 0],
    #        [0,-1, 1],
    #        [0, 0, 1]]
    flip = pipeline.create(dai.node.ImageManip)
    H_horiz_flip = [
        -1.0, 0.0, 1.0,
        0.0, 1.0, 0.0,
        0.0, 0.0, 1.0,
    ]
    flip.initialConfig.addTransformPerspective(H_horiz_flip)
    flip.setMaxOutputFrameSize(COLOR_RES[0] * COLOR_RES[1] * 3)
    color_preview.link(flip.inputImage)

    # Annotate XYZ onto vertically flipped 1080p color stream only
    annot = pipeline.create(ColorXYZAnnotator).build(
        video_frames=flip.out,
        show_roi=args.show_roi,
        roi_src_size=NN_RES,
    )
    measure_distance.output.link(annot.distance_input)
    roi_from_face.output_roi.link(annot.roi_input)

    # Visualizer topics
    visualizer.addTopic("Video", annot.passthrough, "images")
    visualizer.addTopic("XYZ", annot.annotation_output)

    print("Pipeline created.")
    pipeline.start()
    visualizer.registerPipeline(pipeline)

    while pipeline.isRunning():
        key = visualizer.waitKey(1)
        if key == ord("q"):
            print("Got q key from the remote connection!")
            break
