from pathlib import Path

import depthai as dai
from depthai_nodes.node import GatherData, ParsingNeuralNetwork
from utils.annotation_node import AnnotationNode
from utils.arguments import initialize_argparser
from utils.grid_layout_node import GridLayoutNode
from utils.process import ProcessDetections

_, args = initialize_argparser()

INPUT_WIDTH, INPUT_HEIGHT = 2560, 1440
OUTPUT_WIDTH, OUTPUT_HEIGHT = 640, 360
CROP_OUT_W, CROP_OUT_H = 2560, 1440
CROP_NN_W, CROP_NN_H = 320, 240
DET_MODEL = "luxonis/yunet:320x240"
PADDING = 0.0

visualizer = dai.RemoteConnection(httpPort=8082)
device = dai.Device(dai.DeviceInfo(args.device)) if args.device else dai.Device()
platform = device.getPlatform().name
print(f"[MAIN] Platform: {platform}")

frame_type = dai.ImgFrame.Type.BGR888i if platform == "RVC4" else dai.ImgFrame.Type.BGR888p
print(f"[MAIN] Frame type selected: {frame_type.name}")

if not args.fps_limit:
    args.fps_limit = 8 if platform == "RVC2" else 18
print(f"[MAIN] FPS limit: {args.fps_limit}")

with dai.Pipeline(device) as pipeline:
    print("[MAIN] Creating pipeline...")

    # detector
    det_desc = dai.NNModelDescription(DET_MODEL)
    det_desc.platform = platform
    det_archive = dai.NNArchive(dai.getModelFromZoo(det_desc))
    print(f"[MAIN] Detector: {DET_MODEL} ({det_archive.getInputWidth()}x{det_archive.getInputHeight()})")

    # source
    if args.media_path:
        print(f"[MAIN] Using ReplayVideo: {args.media_path}")
        replay = pipeline.create(dai.node.ReplayVideo)
        replay.setReplayVideoFile(Path(args.media_path))
        replay.setOutFrameType(frame_type)
        replay.setLoop(True)
        if args.fps_limit:
            replay.setFps(args.fps_limit)
        input_node = replay.out
    else:
        print("[MAIN] Using live camera")
        cam = pipeline.create(dai.node.Camera).build()
        cam_out = cam.requestOutput(size=(INPUT_WIDTH, INPUT_HEIGHT), type=frame_type, fps=args.fps_limit)

        input_node = cam_out  # encoder branch omitted for brevity

    # stage1 resize + NN
    resize1 = pipeline.create(dai.node.ImageManip)
    resize1.setMaxOutputFrameSize(det_archive.getInputWidth() * det_archive.getInputHeight() * 3)
    resize1.initialConfig.setOutputSize(
        det_archive.getInputWidth(),
        det_archive.getInputHeight(),
        mode=dai.ImageManipConfig.ResizeMode.STRETCH,
    )
    resize1.initialConfig.setFrameType(frame_type)
    input_node.link(resize1.inputImage)

    stage1_nn: ParsingNeuralNetwork = pipeline.create(ParsingNeuralNetwork).build(resize1.out, det_archive)

    # make configs (+ count) for crops
    proc = pipeline.create(ProcessDetections).build(
        detections_input=stage1_nn.out,
        padding=PADDING,
        target_size=(CROP_OUT_W, CROP_OUT_H),
    )

    script = pipeline.create(dai.node.Script)
    script.setScriptPath(str(Path(__file__).parent / "utils/script.py"))
    input_node.link(script.inputs["frame_input"])
    proc.config_output.link(script.inputs["config_input"])
    proc.num_configs_output.link(script.inputs["num_configs_input"])

    # ImageManip crops (N outputs per frame)
    face_crop_disp = pipeline.create(dai.node.ImageManip)
    face_crop_disp.setMaxOutputFrameSize(CROP_OUT_W * CROP_OUT_H * 3)
    face_crop_disp.initialConfig.setOutputSize(CROP_OUT_W, CROP_OUT_H)
    face_crop_disp.initialConfig.setFrameType(frame_type)
    face_crop_disp.inputConfig.setMaxSize(300)
    face_crop_disp.inputImage.setMaxSize(300)
    face_crop_disp.setNumFramesPool(300)
    face_crop_disp.inputConfig.setWaitForMessage(True)

    script.outputs["output_config"].link(face_crop_disp.inputConfig)
    script.outputs["output_frame"].link(face_crop_disp.inputImage)

    # stage2 on SAME crops
    face_crop_to_nn = pipeline.create(dai.node.ImageManip)
    face_crop_to_nn.setMaxOutputFrameSize(CROP_NN_W * CROP_NN_H * 3)
    face_crop_to_nn.initialConfig.setOutputSize(CROP_NN_W, CROP_NN_H)
    face_crop_disp.out.link(face_crop_to_nn.inputImage)

    stage2_nn: ParsingNeuralNetwork = pipeline.create(ParsingNeuralNetwork).build(face_crop_to_nn.out, det_archive)

    # gather for full-frame eye overlay
    gather = pipeline.create(GatherData).build(args.fps_limit)
    stage2_nn.out.link(gather.input_data)
    stage1_nn.out.link(gather.input_reference)

    gather_crops = pipeline.create(GatherData).build(args.fps_limit)
    face_crop_disp.out.link(gather_crops.input_data)
    stage1_nn.out.link(gather_crops.input_reference)

    # grid from crop stream
    grid_layout = pipeline.create(GridLayoutNode).build(
        crops_input=gather_crops.out,
        num_configs_input=proc.num_configs_output,   # count seq = gid
        target_size=(CROP_OUT_W, CROP_OUT_H),
    )
    grid_layout.frame_type = frame_type

    # annotations
    eye_full = pipeline.create(AnnotationNode).build_focused(gather.out, padding=PADDING)
    eye_crop = pipeline.create(AnnotationNode).build_crop(stage2_nn.out)

    # full-frame viz
    full_viz = pipeline.create(dai.node.ImageManip)
    full_viz.setMaxOutputFrameSize(OUTPUT_WIDTH * OUTPUT_HEIGHT * 3)
    full_viz.initialConfig.setOutputSize(OUTPUT_WIDTH, OUTPUT_HEIGHT)
    full_viz.initialConfig.setFrameType(frame_type)
    input_node.link(full_viz.inputImage)

    # visualizer
    visualizer.addTopic("Video", full_viz.out, "images")
    # visualizer.addTopic("Face Crops (debug)", face_crop_disp.out, "images")
    # visualizer.addTopic("Face Mosaic", grid_layout.output, "images")
    visualizer.addTopic("Eyes (Full)", eye_full.out, "annotations")
    visualizer.addTopic("Eyes (Crop)", eye_crop.out, "annotations")

    print("[MAIN] Pipeline created.")
    pipeline.start()
    visualizer.registerPipeline(pipeline)
    print("[MAIN] Running. Press 'q' in the visualizer window to quit.")

    while pipeline.isRunning():
        key = visualizer.waitKey(1)
        if key == ord("q"):
            print("[MAIN] Got q key. Exiting...")
            break
