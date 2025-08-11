from pathlib import Path

import depthai as dai
from depthai_nodes.node import ParsingNeuralNetwork, GatherData

from utils.arguments import initialize_argparser
from utils.annotation_node import AnnotationNode

DET_MODEL = "luxonis/yunet:320x240"
HEAD_POSE_MODEL = "luxonis/head-pose-estimation:60x60"
GAZE_MODEL = "luxonis/gaze-estimation-adas:60x60"
REQ_WIDTH, REQ_HEIGHT = (
    320,
    240,
)  # we are requesting larger input size than required because we want to keep some resolution for the second stage model

_, args = initialize_argparser()

visualizer = dai.RemoteConnection(httpPort=8082)
device = dai.Device(dai.DeviceInfo(args.device)) if args.device else dai.Device()
platform = device.getPlatform().name
print(f"Platform: {platform}")

if platform == "RVC4":
    #DET_MODEL = "luxonis/scrfd-face-detection:10g-640x640"
    REQ_WIDTH, REQ_HEIGHT = (768, 768)

frame_type = (
    dai.ImgFrame.Type.BGR888i if platform == "RVC4" else dai.ImgFrame.Type.BGR888p
)

if args.fps_limit is None:
    args.fps_limit = 15 if platform == "RVC2" else 30
    print(
        f"\nFPS limit set to {args.fps_limit} for {platform} platform. If you want to set a custom FPS limit, use the --fps_limit flag.\n"
    )

with dai.Pipeline(device) as pipeline:
    print("Creating pipeline...")

    # face detection model
    det_model_description = dai.NNModelDescription(DET_MODEL)
    det_model_description.platform = platform
    det_model_nn_archive = dai.NNArchive(dai.getModelFromZoo(det_model_description))

    # head pose model
    head_pose_model_description = dai.NNModelDescription(HEAD_POSE_MODEL)
    head_pose_model_description.platform = platform
    head_pose_model_nn_archive = dai.NNArchive(
        dai.getModelFromZoo(head_pose_model_description)
    )

    # gaze estimation model
    gaze_model_description = dai.NNModelDescription(GAZE_MODEL)
    gaze_model_description.platform = platform
    gaze_model_nn_archive = dai.NNArchive(dai.getModelFromZoo(gaze_model_description))

    # media/camera input
    if args.media_path:
        replay = pipeline.create(dai.node.ReplayVideo)
        replay.setReplayVideoFile(Path(args.media_path))
        replay.setOutFrameType(frame_type)
        replay.setLoop(True)
        if args.fps_limit:
            replay.setFps(args.fps_limit)
        replay.setSize(REQ_WIDTH, REQ_HEIGHT)
    else:
        cam = pipeline.create(dai.node.Camera).build()
        cam_out = cam.requestOutput(
            size=(REQ_WIDTH, REQ_HEIGHT), type=frame_type, fps=args.fps_limit
        )
    input_node_out = replay.out if args.media_path else cam_out

    # resize to det model input size
    resize_node = pipeline.create(dai.node.ImageManip)
    resize_node.initialConfig.setOutputSize(
        det_model_nn_archive.getInputWidth(), det_model_nn_archive.getInputHeight()
    )
    resize_node.setMaxOutputFrameSize(
        det_model_nn_archive.getInputWidth() * det_model_nn_archive.getInputHeight() * 3
    )
    resize_node.initialConfig.setReusePreviousImage(False)
    resize_node.inputImage.setBlocking(True)
    input_node_out.link(resize_node.inputImage)

    det_nn: ParsingNeuralNetwork = pipeline.create(ParsingNeuralNetwork).build(
        resize_node.out, det_model_nn_archive
    )
    det_nn.input.setBlocking(True)

    # annotation
    non_focused_annotation = pipeline.create(AnnotationNode).build_non_focused(det_nn.out)

    # visualization
    visualizer.addTopic("NN Input", resize_node.out, "images")
    visualizer.addTopic("Eyes (Non-Focused)", non_focused_annotation.out, "annotations")
    visualizer.addTopic("Video", input_node_out, "images")


    print("Pipeline created.")

    pipeline.start()
    visualizer.registerPipeline(pipeline)

    while pipeline.isRunning():
        key = visualizer.waitKey(1)
        if key == ord("q"):
            print("Got q key. Exiting...")
            break
