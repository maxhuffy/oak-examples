from pathlib import Path

import depthai as dai
from depthai_nodes.node import GatherData, ParsingNeuralNetwork, ImgDetectionsBridge
from depthai_nodes.node.utils import generate_script_content

from utils.annotation_node import AnnotationNode
from utils.grid_eyes_annotation_node import GridEyesAnnotationNode
from utils.arguments import initialize_argparser
from utils.grid_layout_node import GridLayoutNode

_, args = initialize_argparser()

INPUT_WIDTH, INPUT_HEIGHT = 3840, 2160
DET_MODEL = "luxonis/yunet:320x240"
EYE_MODEL = "luxonis-ml-team/eye-detection:eye-detection-512x512"

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

    det_w = det_archive.getInputWidth()
    det_h = det_archive.getInputHeight()

    eye_model_description = dai.NNModelDescription(EYE_MODEL)
    eye_model_description.platform = platform
    eye_model_nn_archive = dai.NNArchive(dai.getModelFromZoo(eye_model_description,
                                                             apiKey=args.api_key))

    crop_h = eye_model_nn_archive.getInputHeight()
    crop_w = eye_model_nn_archive.getInputWidth()

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
    resize1.setMaxOutputFrameSize(det_w * det_h * 3)
    resize1.initialConfig.setOutputSize(
        det_w,
        det_h,
        mode=dai.ImageManipConfig.ResizeMode.STRETCH,
    )
    resize1.initialConfig.setFrameType(frame_type)
    input_node.link(resize1.inputImage)

    stage1_nn: ParsingNeuralNetwork = pipeline.create(ParsingNeuralNetwork).build(resize1.out, det_archive)

    det_bridge = pipeline.create(ImgDetectionsBridge).build(
        stage1_nn.out
    )  # TODO: remove once we have it working with ImgDetectionsExtended
    script = pipeline.create(dai.node.Script)
    det_bridge.out.link(script.inputs["det_in"])
    input_node.link(script.inputs["preview"])
    script_content = generate_script_content(
        resize_width=crop_w,
        resize_height=crop_h,
    )
    script.setScript(script_content)

    face_crop_disp = pipeline.create(dai.node.ImageManip)
    face_crop_disp.setMaxOutputFrameSize(crop_w * crop_h * 3)
    face_crop_disp.initialConfig.setOutputSize(crop_w, crop_h)
    face_crop_disp.initialConfig.setFrameType(frame_type)
    face_crop_disp.inputConfig.setMaxSize(30)
    face_crop_disp.inputImage.setMaxSize(30)
    face_crop_disp.setNumFramesPool(30)
    face_crop_disp.inputConfig.setWaitForMessage(True)

    script.outputs["manip_cfg"].link(face_crop_disp.inputConfig)
    script.outputs["manip_img"].link(face_crop_disp.inputImage)

    # stage2 on SAME crops
    face_crop_to_nn = pipeline.create(dai.node.ImageManip)
    face_crop_to_nn.setMaxOutputFrameSize(crop_w * crop_h * 3)
    face_crop_to_nn.initialConfig.setOutputSize(crop_w, crop_h)
    face_crop_disp.out.link(face_crop_to_nn.inputImage)

    stage2_nn: ParsingNeuralNetwork = pipeline.create(ParsingNeuralNetwork).build(face_crop_to_nn.out, eye_model_nn_archive)

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
        target_size=(crop_w, crop_h),
        frame_type = frame_type,
    )
    grid_layout.frame_type = frame_type

    # annotations
    eye_full = pipeline.create(AnnotationNode).build(gather.out)

    grid_eyes = pipeline.create(GridEyesAnnotationNode).build(
        gathered_pair_out=gather.out,
        mosaic_size=(crop_w, crop_h),
        crop_size=(crop_w, crop_h),
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
    visualizer.addTopic("Face stage 1", eye_full.out, "annotations")
    visualizer.addTopic("Eyes (Crops)", grid_eyes.out, "annotations")

    pipeline.start()
    visualizer.registerPipeline(pipeline)

    while pipeline.isRunning():
        key = visualizer.waitKey(1)
        if key == ord("q"):
            print("[MAIN] Got q key. Exiting...")
            break
