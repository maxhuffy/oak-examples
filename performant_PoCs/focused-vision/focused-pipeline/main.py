from pathlib import Path

import depthai as dai
from depthai_nodes.node import GatherData, ParsingNeuralNetwork
from utils.annotation_node import AnnotationNode
from utils.grid_eyes_annotation_node import GridEyesAnnotationNode
from utils.arguments import initialize_argparser
from utils.grid_layout_node import GridLayoutNode
from utils.process import ProcessDetections

_, args = initialize_argparser()

INPUT_WIDTH, INPUT_HEIGHT = 3840, 2160
CROP_OUT_W, CROP_OUT_H = 320, 240
DET_MODEL = "luxonis/yunet:320x240"
PADDING = 0.0

visualizer = dai.RemoteConnection(httpPort=8082)
device = dai.Device(dai.DeviceInfo(args.device)) if args.device else dai.Device()
platform = device.getPlatform().name

frame_type = dai.ImgFrame.Type.BGR888i if platform == "RVC4" else dai.ImgFrame.Type.BGR888p

if not args.fps_limit:
    args.fps_limit = 8 if platform == "RVC2" else 30

with dai.Pipeline(device) as pipeline:
    det_desc = dai.NNModelDescription(DET_MODEL)
    det_desc.platform = platform
    det_archive = dai.NNArchive(dai.getModelFromZoo(det_desc))

    if args.media_path:
        replay = pipeline.create(dai.node.ReplayVideo)
        replay.setReplayVideoFile(Path(args.media_path))
        replay.setOutFrameType(frame_type)
        replay.setLoop(True)
        if args.fps_limit:
            replay.setFps(args.fps_limit)
        input_node = replay.out
    else:
        cam = pipeline.create(dai.node.Camera).build()
        cam_out = cam.requestOutput(size=(INPUT_WIDTH, INPUT_HEIGHT), type=frame_type, fps=args.fps_limit)
        input_node = cam_out

        out_NV12 = cam.requestOutput(size=(INPUT_WIDTH, INPUT_HEIGHT), type=dai.ImgFrame.Type.NV12, fps=args.fps_limit)

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

    # make configs for crops
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
    face_crop_to_nn.setMaxOutputFrameSize(CROP_OUT_W * CROP_OUT_H * 3)
    face_crop_to_nn.initialConfig.setOutputSize(CROP_OUT_W, CROP_OUT_H)
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
        target_size=(CROP_OUT_W, CROP_OUT_H),
        frame_type = frame_type,
    )
    grid_layout.frame_type = frame_type

    # annotations
    eye_full = pipeline.create(AnnotationNode).build(gather.out, padding=PADDING)

    grid_eyes = pipeline.create(GridEyesAnnotationNode).build(
        gathered_pair_out=gather.out,
        mosaic_size=(CROP_OUT_W, CROP_OUT_H),
        crop_size=(CROP_OUT_W, CROP_OUT_H),
        eye_scale=0.35,
    )

    video_enc = pipeline.create(dai.node.VideoEncoder)
    video_enc.setDefaultProfilePreset(
        fps=args.fps_limit, profile=dai.VideoEncoderProperties.Profile.H264_MAIN
    )

    if args.media_path:
        out_NV12 = pipeline.create(dai.node.ImageManip)
        out_NV12.initialConfig.setOutputSize(INPUT_WIDTH, INPUT_HEIGHT)
        out_NV12.initialConfig.setFrameType(dai.ImgFrame.Type.NV12)
        out_NV12.setMaxOutputFrameSize(int(INPUT_WIDTH * INPUT_HEIGHT * 3))
        replay.out.link(out_NV12.inputImage)
        out_NV12.out.link(video_enc.input)
    else:
        out_NV12.link(video_enc.input)

    visualizer.addTopic("Video", video_enc.out, "images")
    visualizer.addTopic("Face Mosaic", grid_layout.output, "images")
    visualizer.addTopic("Eyes (Full)", eye_full.out, "annotations")
    visualizer.addTopic("Eyes (Crops)", grid_eyes.out, "annotations")

    pipeline.start()
    visualizer.registerPipeline(pipeline)

    while pipeline.isRunning():
        key = visualizer.waitKey(1)
        if key == ord("q"):
            print("[MAIN] Got q key. Exiting...")
            break
