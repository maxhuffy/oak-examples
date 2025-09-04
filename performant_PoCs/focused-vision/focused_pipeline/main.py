from pathlib import Path

import depthai as dai
from depthai_nodes.node import GatherData, ParsingNeuralNetwork, ImgDetectionsBridge
from depthai_nodes.node.utils import generate_script_content

from utils.annotation_node import AnnotationNode
from utils.mosaic_eyes_annotation_node import MosaicEyesAnnotationNode
from utils.arguments import initialize_argparser
from utils.mosaic_layout_node import MosaicLayoutNode

_, args = initialize_argparser()

INPUT_WIDTH, INPUT_HEIGHT = 3840, 2160
FACE_DETECTION_MODEL = "luxonis/yunet:320x240"
EYE_DETECTION_MODEL = "luxonis-ml-team/eye-detection:eye-detection-512x512"

visualizer = dai.RemoteConnection(httpPort=8082)
device = dai.Device(dai.DeviceInfo(args.device)) if args.device else dai.Device()
platform = device.getPlatform().name

frame_type = dai.ImgFrame.Type.BGR888i if platform == "RVC4" else dai.ImgFrame.Type.BGR888p

if not args.fps_limit:
    args.fps_limit = 8 if platform == "RVC2" else 30


def convert_to_nv12(original_video, width, height):
    video_NV12 = pipeline.create(dai.node.ImageManip)
    video_NV12.initialConfig.setOutputSize(width, height)
    video_NV12.initialConfig.setFrameType(dai.ImgFrame.Type.NV12)
    video_NV12.setMaxOutputFrameSize(int(width * height * 3))
    original_video.link(video_NV12.inputImage)

    return video_NV12


with dai.Pipeline(device) as pipeline:
    face_model_desc = dai.NNModelDescription(FACE_DETECTION_MODEL)
    face_model_desc.platform = platform
    face_model_archive = dai.NNArchive(dai.getModelFromZoo(face_model_desc))

    face_model_input_width = face_model_archive.getInputWidth()
    face_model_input_height = face_model_archive.getInputHeight()

    eye_model_desc = dai.NNModelDescription(EYE_DETECTION_MODEL)
    eye_model_desc.platform = platform
    eye_model_archive = dai.NNArchive(dai.getModelFromZoo(eye_model_desc, apiKey=args.api_key))

    eye_model_input_height = eye_model_archive.getInputHeight()
    eye_model_input_width = eye_model_archive.getInputWidth()

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
    resize1.setMaxOutputFrameSize(face_model_input_width * face_model_input_height * 3)
    resize1.initialConfig.setOutputSize(
        face_model_input_width,
        face_model_input_height,
        mode=dai.ImageManipConfig.ResizeMode.STRETCH,
    )
    resize1.initialConfig.setFrameType(frame_type)
    input_node.link(resize1.inputImage)

    stage1_nn: ParsingNeuralNetwork = pipeline.create(ParsingNeuralNetwork).build(resize1.out, face_model_archive)

    det_bridge = pipeline.create(ImgDetectionsBridge).build(
        stage1_nn.out
    )  # TODO: remove once we have it working with ImgDetectionsExtended
    script = pipeline.create(dai.node.Script)
    det_bridge.out.link(script.inputs["det_in"])
    input_node.link(script.inputs["preview"])
    script_content = generate_script_content(
        resize_width=eye_model_input_width,
        resize_height=eye_model_input_height,
    )
    script.setScript(script_content)

    face_crop_disp = pipeline.create(dai.node.ImageManip)
    face_crop_disp.setMaxOutputFrameSize(eye_model_input_width * eye_model_input_height * 3)
    face_crop_disp.initialConfig.setOutputSize(eye_model_input_width, eye_model_input_height)
    face_crop_disp.initialConfig.setFrameType(frame_type)
    face_crop_disp.inputConfig.setMaxSize(30)
    face_crop_disp.inputImage.setMaxSize(30)
    face_crop_disp.setNumFramesPool(30)
    face_crop_disp.inputConfig.setWaitForMessage(True)

    script.outputs["manip_cfg"].link(face_crop_disp.inputConfig)
    script.outputs["manip_img"].link(face_crop_disp.inputImage)

    # stage2 on SAME crops
    face_crop_to_nn = pipeline.create(dai.node.ImageManip)
    face_crop_to_nn.setMaxOutputFrameSize(eye_model_input_width * eye_model_input_height * 3)
    face_crop_to_nn.initialConfig.setOutputSize(eye_model_input_width, eye_model_input_height)
    face_crop_disp.out.link(face_crop_to_nn.inputImage)

    stage2_nn: ParsingNeuralNetwork = pipeline.create(ParsingNeuralNetwork).build(face_crop_to_nn.out,
                                                                                  eye_model_archive)

    # gather for full-frame eye overlay
    gather = pipeline.create(GatherData).build(args.fps_limit)
    stage2_nn.out.link(gather.input_data)
    stage1_nn.out.link(gather.input_reference)

    gather_crops = pipeline.create(GatherData).build(args.fps_limit)
    face_crop_disp.out.link(gather_crops.input_data)
    stage1_nn.out.link(gather_crops.input_reference)

    # mosaic from crop stream
    mosaic_layout = pipeline.create(MosaicLayoutNode).build(
        crops_input=gather_crops.out,
        target_size=(eye_model_input_width, eye_model_input_height),
        frame_type=frame_type,
    )
    mosaic_layout.frame_type = frame_type

    # annotations
    eye_full = pipeline.create(AnnotationNode).build(gather.out)

    mosaic_eyes = pipeline.create(MosaicEyesAnnotationNode).build(
        gathered_pair_out=gather.out,
        mosaic_size=(eye_model_input_width, eye_model_input_height),
        crop_size=(eye_model_input_width, eye_model_input_height),
    )

    video_enc = pipeline.create(dai.node.VideoEncoder)
    video_enc.setDefaultProfilePreset(
        fps=args.fps_limit, profile=dai.VideoEncoderProperties.Profile.H264_MAIN
    )

    mosaic_enc = pipeline.create(dai.node.VideoEncoder)
    mosaic_enc.setDefaultProfilePreset(
        fps=args.fps_limit, profile=dai.VideoEncoderProperties.Profile.H264_MAIN
    )

    if args.media_path:
        out_NV12 = convert_to_nv12(replay.out, INPUT_WIDTH, INPUT_HEIGHT)
        out_NV12.out.link(video_enc.input)

    else:
        out_NV12.link(video_enc.input)

    mosaic_NV12 = convert_to_nv12(mosaic_layout.output, INPUT_WIDTH, INPUT_HEIGHT)
    mosaic_NV12.out.link(mosaic_enc.input)

    visualizer.addTopic("Video", video_enc.out, "images")
    visualizer.addTopic("Face Mosaic", mosaic_enc.out, "images")
    visualizer.addTopic("Face stage 1", eye_full.out, "annotations")
    visualizer.addTopic("Eyes (Crops)", mosaic_eyes.out, "annotations")

    pipeline.start()
    visualizer.registerPipeline(pipeline)

    while pipeline.isRunning():
        key = visualizer.waitKey(1)
        if key == ord("q"):
            print("[MAIN] Got q key. Exiting...")
            break
