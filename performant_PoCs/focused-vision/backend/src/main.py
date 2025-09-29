from pathlib import Path

import depthai as dai
from depthai_nodes.node import GatherData, ParsingNeuralNetwork, ImgDetectionsBridge
from depthai_nodes.node.utils import generate_script_content

from utils.arguments import initialize_argparser
from utils.mosaic_layout_node import MosaicLayoutNode
from utils.mosaic_stage_2_annotation_node import MosaicStage2AnnotationNode
from utils.safe_img_detections_bridge import SafeImgDetectionsBridge
from utils.stage2_to_full_annotation_node import Stage2CropToFullRemapNode
from utils.letterbox_unmap_crop_node import UnletterboxDetectionsNode

_, args = initialize_argparser()

HIGH_RES_WIDTH, HIGH_RES_HEIGHT = 1920*2, 1080*2
INPUT_WIDTH, INPUT_HEIGHT = 1920, 1080
STAGE_1_MODEL = "luxonis/yunet:320x240"
STAGE_2_MODEL = "luxonis/eye-detection:512x512"

visualizer = dai.RemoteConnection(httpPort=8082)
device = dai.Device(dai.DeviceInfo(args.device)) if args.device else dai.Device()
platform = device.getPlatform().name

frame_type = dai.ImgFrame.Type.BGR888i if platform == "RVC4" else dai.ImgFrame.Type.BGR888p

if not args.fps_limit:
    args.fps_limit = 8 if platform == "RVC2" else 15


def convert_to_nv12(original_video, width, height):
    video_NV12 = pipeline.create(dai.node.ImageManip)
    video_NV12.initialConfig.setOutputSize(width, height)
    video_NV12.initialConfig.setFrameType(dai.ImgFrame.Type.NV12)
    video_NV12.setMaxOutputFrameSize(int(width * height * 3))
    original_video.link(video_NV12.inputImage)

    return video_NV12


with dai.Pipeline(device) as pipeline:
    stage_1_model_desc = dai.NNModelDescription(STAGE_1_MODEL)
    stage_1_model_desc.platform = platform
    stage_1_model_archive = dai.NNArchive(dai.getModelFromZoo(stage_1_model_desc))

    stage_1_model_input_width = stage_1_model_archive.getInputWidth()
    stage_1_model_input_height = stage_1_model_archive.getInputHeight()

    stage_2_model_desc = dai.NNModelDescription(STAGE_2_MODEL)
    stage_2_model_desc.platform = platform
    stage_2_model_archive = dai.NNArchive(dai.getModelFromZoo(stage_2_model_desc))

    stage_2_model_input_width = stage_2_model_archive.getInputWidth()
    stage_2_model_input_height = stage_2_model_archive.getInputHeight()

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
        high_res_out = cam.requestOutput(size=(HIGH_RES_WIDTH, HIGH_RES_HEIGHT), type=frame_type, fps=args.fps_limit)
        input_node = cam_out

        out_NV12 = cam.requestOutput(size=(INPUT_WIDTH, INPUT_HEIGHT), type=dai.ImgFrame.Type.NV12, fps=args.fps_limit)

    resize_non_focused = pipeline.create(dai.node.ImageManip)
    resize_non_focused.setMaxOutputFrameSize(stage_2_model_input_width * stage_2_model_input_height * 3)
    resize_non_focused.initialConfig.setOutputSize(
        stage_2_model_input_width,
        stage_2_model_input_height,
        mode=dai.ImageManipConfig.ResizeMode.LETTERBOX,
    )
    resize_non_focused.initialConfig.setFrameType(frame_type)
    input_node.link(resize_non_focused.inputImage)

    non_focused_nn: ParsingNeuralNetwork = pipeline.create(ParsingNeuralNetwork).build(
        resize_non_focused.out, stage_2_model_archive
    )

    non_focused_bridge = pipeline.create(ImgDetectionsBridge).build(non_focused_nn.out)

    non_focused_remapped = pipeline.create(UnletterboxDetectionsNode).build(
        det_in=non_focused_nn.out,
        preview=input_node,
        model_size=(stage_2_model_input_width, stage_2_model_input_height),
    )

    non_focused_remapped_bridge = pipeline.create(SafeImgDetectionsBridge).build(
        non_focused_remapped.out
    )

    eye_script_non_focused = pipeline.create(dai.node.Script)

    eye_script_non_focused.setScript(generate_script_content(
        resize_width=512,
        resize_height=512,
    ))

    non_focused_remapped.out.link(eye_script_non_focused.inputs["det_in"])

    high_res_out.link(eye_script_non_focused.inputs["preview"])

    eye_crop_non_focused = pipeline.create(dai.node.ImageManip)
    eye_crop_non_focused.setMaxOutputFrameSize(512 * 512 * 3)
    eye_crop_non_focused.initialConfig.setOutputSize(512, 512)
    eye_crop_non_focused.initialConfig.setFrameType(frame_type)
    eye_crop_non_focused.inputConfig.setMaxSize(50)
    eye_crop_non_focused.inputImage.setMaxSize(50)
    eye_crop_non_focused.setNumFramesPool(50)
    eye_crop_non_focused.inputConfig.setWaitForMessage(True)

    eye_script_non_focused.outputs["manip_cfg"].link(eye_crop_non_focused.inputConfig)
    eye_script_non_focused.outputs["manip_img"].link(eye_crop_non_focused.inputImage)

    gather_eyes_non_focused = pipeline.create(GatherData).build(args.fps_limit)
    eye_crop_non_focused.out.link(gather_eyes_non_focused.input_data)

    non_focused_remapped.out.link(gather_eyes_non_focused.input_reference)

    eye_mosaic_non_focused = pipeline.create(MosaicLayoutNode).build(
        crops_input=gather_eyes_non_focused.out,
        target_size=(640, 640),
        frame_type=frame_type,
    )
    eye_mosaic_non_focused.frame_type = frame_type

    # stage1 resize + NN
    resize_stage1 = pipeline.create(dai.node.ImageManip)
    resize_stage1.setMaxOutputFrameSize(stage_1_model_input_width * stage_1_model_input_height * 3)
    resize_stage1.initialConfig.setOutputSize(
        stage_1_model_input_width,
        stage_1_model_input_height,
        mode=dai.ImageManipConfig.ResizeMode.STRETCH,
    )
    resize_stage1.initialConfig.setFrameType(frame_type)
    input_node.link(resize_stage1.inputImage)

    stage1_nn: ParsingNeuralNetwork = pipeline.create(ParsingNeuralNetwork).build(
        resize_stage1.out, stage_1_model_archive
    )

    stage1_detections_bridge = pipeline.create(ImgDetectionsBridge).build(
        stage1_nn.out
    )  # TODO: remove once we have it working with ImgDetectionsExtended
    script = pipeline.create(dai.node.Script)
    stage1_detections_bridge.out.link(script.inputs["det_in"])
    input_node.link(script.inputs["preview"])
    script_content = generate_script_content(
        resize_width=stage_2_model_input_width,
        resize_height=stage_2_model_input_height,
    )
    script.setScript(script_content)

    stage_1_detection_crop = pipeline.create(dai.node.ImageManip)
    stage_1_detection_crop.setMaxOutputFrameSize(stage_2_model_input_width * stage_2_model_input_height * 3)
    stage_1_detection_crop.initialConfig.setOutputSize(stage_2_model_input_width, stage_2_model_input_height)
    stage_1_detection_crop.initialConfig.setFrameType(frame_type)
    stage_1_detection_crop.inputConfig.setMaxSize(30)
    stage_1_detection_crop.inputImage.setMaxSize(30)
    stage_1_detection_crop.setNumFramesPool(30)
    stage_1_detection_crop.inputConfig.setWaitForMessage(True)

    script.outputs["manip_cfg"].link(stage_1_detection_crop.inputConfig)
    script.outputs["manip_img"].link(stage_1_detection_crop.inputImage)

    # stage-2 on crops
    stage_1_detection_crop_to_nn = pipeline.create(dai.node.ImageManip)
    stage_1_detection_crop_to_nn.setMaxOutputFrameSize(stage_2_model_input_width * stage_2_model_input_height * 3)
    stage_1_detection_crop_to_nn.initialConfig.setOutputSize(stage_2_model_input_width, stage_2_model_input_height)
    stage_1_detection_crop.out.link(stage_1_detection_crop_to_nn.inputImage)

    stage2_nn: ParsingNeuralNetwork = pipeline.create(ParsingNeuralNetwork).build(
        stage_1_detection_crop_to_nn.out, stage_2_model_archive
    )

    # gather for full-frame 2-nd stage annotation overlay
    stage2_detections_bridge = pipeline.create(SafeImgDetectionsBridge).build(
        stage2_nn.out
    )
    gather = pipeline.create(GatherData).build(args.fps_limit)
    stage2_detections_bridge.out.link(gather.input_data)
    stage1_detections_bridge.out.link(gather.input_reference)

    fullframe_remap = pipeline.create(Stage2CropToFullRemapNode).build(
        gathered_pair_out=gather.out
    )

    fullframe_remap_bridge = pipeline.create(ImgDetectionsBridge).build(fullframe_remap.out)

    gather_crops = pipeline.create(GatherData).build(args.fps_limit)
    stage_1_detection_crop.out.link(gather_crops.input_data)
    stage1_nn.out.link(gather_crops.input_reference)

    eye_script_focused = pipeline.create(dai.node.Script)

    eye_script_focused.setScript(generate_script_content(
        resize_width=512,
        resize_height=512,
    ))

    fullframe_remap_bridge.out.link(eye_script_focused.inputs["det_in"])

    high_res_out.link(eye_script_focused.inputs["preview"])

    eye_crop_stage_2 = pipeline.create(dai.node.ImageManip)
    eye_crop_stage_2.setMaxOutputFrameSize(512 * 512 * 3)
    eye_crop_stage_2.initialConfig.setOutputSize(512, 512)
    eye_crop_stage_2.initialConfig.setFrameType(frame_type)
    eye_crop_stage_2.inputConfig.setMaxSize(50)
    eye_crop_stage_2.inputImage.setMaxSize(50)
    eye_crop_stage_2.setNumFramesPool(50)
    eye_crop_stage_2.inputConfig.setWaitForMessage(True)

    eye_script_focused.outputs["manip_cfg"].link(eye_crop_stage_2.inputConfig)
    eye_script_focused.outputs["manip_img"].link(eye_crop_stage_2.inputImage)

    mosaic_layout = pipeline.create(MosaicLayoutNode).build(
        crops_input=gather_crops.out,
        target_size=(stage_2_model_input_width, stage_2_model_input_height),
        frame_type=frame_type,
    )
    mosaic_layout.frame_type = frame_type

    # annotations
    mosaic_annotation = pipeline.create(MosaicStage2AnnotationNode).build(
        gathered_pair_out=gather.out,
        mosaic_size=(stage_2_model_input_width, stage_2_model_input_height),
        crop_size=(stage_2_model_input_width, stage_2_model_input_height),
    )

    gather_eyes = pipeline.create(GatherData).build(args.fps_limit)
    eye_crop_stage_2.out.link(gather_eyes.input_data)

    fullframe_remap.out.link(gather_eyes.input_reference)

    eye_mosaic_stage_2 = pipeline.create(MosaicLayoutNode).build(
        crops_input=gather_eyes.out,
        target_size=(640, 640),
        frame_type=frame_type,
    )
    eye_mosaic_stage_2.frame_type = frame_type



    # encoders
    video_enc = pipeline.create(dai.node.VideoEncoder)
    video_enc.setDefaultProfilePreset(
        fps=args.fps_limit, profile=dai.VideoEncoderProperties.Profile.H264_MAIN
    )

    mosaic_enc = pipeline.create(dai.node.VideoEncoder)
    mosaic_enc.setDefaultProfilePreset(
        fps=args.fps_limit, profile=dai.VideoEncoderProperties.Profile.H264_HIGH
    )
    mosaic_enc.setRateControlMode(dai.VideoEncoderProperties.RateControlMode.VBR)

    non_focused_enc = pipeline.create(dai.node.VideoEncoder)
    non_focused_enc.setDefaultProfilePreset(
        fps=args.fps_limit, profile=dai.VideoEncoderProperties.Profile.H264_HIGH
    )

    eye_mosaic_stage_2_enc = pipeline.create(dai.node.VideoEncoder)
    eye_mosaic_stage_2_enc.setDefaultProfilePreset(
        fps=args.fps_limit, profile=dai.VideoEncoderProperties.Profile.H264_HIGH
    )
    eye_mosaic_stage_2_enc.setRateControlMode(dai.VideoEncoderProperties.RateControlMode.VBR)

    eye_mosaic_non_focused_enc = pipeline.create(dai.node.VideoEncoder)
    eye_mosaic_non_focused_enc.setDefaultProfilePreset(
        fps=args.fps_limit, profile=dai.VideoEncoderProperties.Profile.H264_HIGH
    )
    eye_mosaic_non_focused_enc.setRateControlMode(dai.VideoEncoderProperties.RateControlMode.VBR)

    if args.media_path:
        out_NV12 = convert_to_nv12(replay.out, INPUT_WIDTH, INPUT_HEIGHT)
        out_NV12.out.link(video_enc.input)

    else:
        out_NV12.link(video_enc.input)

    mosaic_NV12 = convert_to_nv12(mosaic_layout.output, stage_2_model_input_width, stage_2_model_input_height)
    mosaic_NV12.out.link(mosaic_enc.input)

    resize_non_focused_NV12 = convert_to_nv12(
        resize_non_focused.out, stage_2_model_input_width, stage_2_model_input_height
    )
    resize_non_focused_NV12.out.link(non_focused_enc.input)

    eye_mosaic_stage_2_NV12 = convert_to_nv12(eye_mosaic_stage_2.output, 640, 640)
    eye_mosaic_stage_2_NV12.out.link(eye_mosaic_stage_2_enc.input)

    eye_mosaic_non_focused_NV12 = convert_to_nv12(eye_mosaic_non_focused.output, 640, 640)
    eye_mosaic_non_focused_NV12.out.link(eye_mosaic_non_focused_enc.input)

    # visualizer topics
    visualizer.addTopic("Video", video_enc.out, "images")
    visualizer.addTopic("Detections Stage 1", stage1_nn.out, "annotations")

    visualizer.addTopic("Crops Mosaic", mosaic_enc.out, "images")

    visualizer.addTopic("Detections Stage 2 Crops", mosaic_annotation.out, "annotations")

    visualizer.addTopic("Eyes Mosaic", eye_mosaic_stage_2_enc.out, "images")

    visualizer.addTopic("Eyes Mosaic Non Focused", eye_mosaic_non_focused_enc.out, "images")

    visualizer.addTopic("Detections Non Focused Remapped", non_focused_remapped_bridge.out, "annotations")

    visualizer.addTopic("NN input Eye Detection", resize_non_focused.out, "images")

    visualizer.addTopic("Full Frame eyes detection", fullframe_remap.out, "annotations")

    visualizer.addTopic("NN input Face Detection", resize_stage1.out, "images")

    visualizer.addTopic("Detections NN Non Focused", non_focused_bridge.out, "annotations")

    pipeline.start()
    visualizer.registerPipeline(pipeline)

    while pipeline.isRunning():
        key = visualizer.waitKey(1)
        if key == ord("q"):
            print("[MAIN] Got q key. Exiting...")
            break
