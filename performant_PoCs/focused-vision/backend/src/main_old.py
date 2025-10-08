from pathlib import Path

import depthai as dai
from depthai_nodes.node import ParsingNeuralNetwork, ImgDetectionsBridge, GatherData

from utils.arguments import initialize_argparser
from utils.mosaic_stage_2_annotation_node import MosaicStage2AnnotationNode
from utils.safe_img_detections_bridge import SafeImgDetectionsBridge
from utils.stage2_to_full_annotation_node import Stage2CropToFullRemapNode
from utils.letterbox_unmap_crop_node import UnletterboxDetectionsNode
from utils.pipeline_build_helpers import (build_resizer, build_roi_cropper,
                                          build_h264_stream, build_mosaic_from_crops, load_model)

_, args = initialize_argparser()

INPUT_WIDTH, INPUT_HEIGHT = 3072, 1728
STAGE_1_MODEL = "luxonis/yunet:320x240"
STAGE_2_MODEL = "luxonis/eye-detection:512x512"

visualizer = dai.RemoteConnection(httpPort=8082)
device = dai.Device(dai.DeviceInfo(args.device)) if args.device else dai.Device()
platform = device.getPlatform().name

frame_type = dai.ImgFrame.Type.BGR888i if platform == "RVC4" else dai.ImgFrame.Type.BGR888p
if not args.fps_limit:
    args.fps_limit = 8 if platform == "RVC2" else 30

with dai.Pipeline(device) as pipeline:
    stage_1_model_input_width, stage_1_model_input_height, stage_1_archive = load_model(STAGE_1_MODEL, platform)

    stage_2_model_input_width, stage_2_model_input_height, stage_2_archive = load_model(STAGE_2_MODEL, platform)

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
        out_NV12_src = cam.requestOutput(size=(INPUT_WIDTH, INPUT_HEIGHT),
                                         type=dai.ImgFrame.Type.NV12, fps=args.fps_limit)

    # ------------------------------------------------------------------
    # Non-focused path: Detect eyes on input and make high-res eye crops
    # ------------------------------------------------------------------
    resize_non_focused = build_resizer(
        pipeline,
        input_stream=input_node,
        size=(stage_2_model_input_width, stage_2_model_input_height),
        frame_type=frame_type,
        mode=dai.ImageManipConfig.ResizeMode.LETTERBOX,
    )

    non_focused_nn: ParsingNeuralNetwork = pipeline.create(ParsingNeuralNetwork).build(
        resize_non_focused.out, stage_2_archive
    )
    non_focused_bridge = pipeline.create(ImgDetectionsBridge).build(non_focused_nn.out)

    non_focused_remapped = pipeline.create(UnletterboxDetectionsNode).build(
        det_in=non_focused_nn.out, preview=input_node, 
        model_size=(stage_2_model_input_width, stage_2_model_input_height)
    )
    
    non_focused_remapped_bridge = pipeline.create(SafeImgDetectionsBridge).build(non_focused_remapped.out)

    # Making high-res eyes crops
    eye_crops_non_focused = build_roi_cropper(
        pipeline,
        preview_stream=input_node,
        det_stream=non_focused_remapped.out,
        out_size=(512, 512),
        frame_type=frame_type,
    )
    eye_mosaic_non_focused_out = build_mosaic_from_crops(
        pipeline,
        fps_limit=args.fps_limit,
        crops_stream=eye_crops_non_focused,
        reference_stream=non_focused_remapped.out,
        target_size=(640, 640),
        frame_type=frame_type,
    )

    # ----------------------------------------------------------
    # Stage-1 Face Detection Path (Face Detection on full frame)
    # ----------------------------------------------------------
    resize_s1 = build_resizer(
        pipeline,
        input_stream=input_node,
        size=(stage_1_model_input_width, stage_1_model_input_height),
        frame_type=frame_type,
        mode=dai.ImageManipConfig.ResizeMode.STRETCH,
    )

    stage1_nn: ParsingNeuralNetwork = pipeline.create(ParsingNeuralNetwork).build(resize_s1.out, stage_1_archive)
    stage1_det_bridge = pipeline.create(ImgDetectionsBridge).build(stage1_nn.out)

    # Making high-res face crops for Stage-2 (eyes) detection
    s1_face_crops = build_roi_cropper(
        pipeline,
        preview_stream=input_node,
        det_stream=stage1_det_bridge.out,
        out_size=(stage_2_model_input_width, stage_2_model_input_height),
        frame_type=frame_type,
        pool_size=30,
        cfg_queue_size=30,
    )

    # ------------------------------------------------------------------
    # Stage-2 Face Detection Path (Eye detection on high-res face crops)
    # ------------------------------------------------------------------

    stage2_nn: ParsingNeuralNetwork = pipeline.create(ParsingNeuralNetwork).build(
        s1_face_crops, stage_2_archive
    )

    # Gather for full-frame Stage-2 overlay and then remap to full frame
    stage2_safe_bridge = pipeline.create(SafeImgDetectionsBridge).build(stage2_nn.out)

    gathered_s2_full_overlay = pipeline.create(GatherData).build(args.fps_limit)
    stage2_safe_bridge.out.link(gathered_s2_full_overlay.input_data)
    stage1_det_bridge.out.link(gathered_s2_full_overlay.input_reference)

    full_frame_remap = pipeline.create(Stage2CropToFullRemapNode).build(gathered_pair_out=gathered_s2_full_overlay.out)
    full_frame_remap_bridge = pipeline.create(ImgDetectionsBridge).build(full_frame_remap.out)

    face_mosaic_out = build_mosaic_from_crops(
        pipeline,
        fps_limit=args.fps_limit,
        crops_stream=s1_face_crops,
        reference_stream=stage1_nn.out,
        target_size=(stage_2_model_input_width, stage_2_model_input_height),
        frame_type=frame_type,
    )

    eye_crops_focused = build_roi_cropper(
        pipeline,
        preview_stream=input_node,
        det_stream=full_frame_remap_bridge.out,
        out_size=(512, 512),
        frame_type=frame_type,
    )
    eye_mosaic_focused_out = build_mosaic_from_crops(
        pipeline,
        fps_limit=args.fps_limit,
        crops_stream=eye_crops_focused,
        reference_stream=full_frame_remap.out,
        target_size=(640, 640),
        frame_type=frame_type,
    )

    mosaic_annotation = pipeline.create(MosaicStage2AnnotationNode).build(
        gathered_pair_out=gathered_s2_full_overlay.out,
        mosaic_size=(stage_2_model_input_width, stage_2_model_input_height),
        crop_size=(stage_2_model_input_width, stage_2_model_input_height),
    )

    # -------------------------------------------------
    # Encoders
    # -------------------------------------------------
    if args.media_path:
        video_enc_out = build_h264_stream(
            pipeline,
            src=input_node,
            size=(INPUT_WIDTH, INPUT_HEIGHT),
            fps=args.fps_limit,
            profile=dai.VideoEncoderProperties.Profile.H264_MAIN,
            assume_nv12=False,
        )
    else:
        video_enc_out = build_h264_stream(
            pipeline,
            src=out_NV12_src,
            size=(INPUT_WIDTH, INPUT_HEIGHT),
            fps=args.fps_limit,
            profile=dai.VideoEncoderProperties.Profile.H264_MAIN,
            assume_nv12=True,
        )

    # Face mosaic (Stage-1 crops) ->  stage_2_model_input_width x stage_2_model_input_height
    face_mosaic_enc_out = build_h264_stream(
        pipeline,
        src=face_mosaic_out,
        size=(stage_2_model_input_width, stage_2_model_input_height),
        fps=args.fps_limit,
        profile=dai.VideoEncoderProperties.Profile.H264_HIGH,
        vbr=True,
    )

    # Non-focused resized input (letterboxed) -> stage_2_model_input_width x stage_2_model_input_height
    nf_enc_out = build_h264_stream(
        pipeline,
        src=resize_non_focused.out,
        size=(stage_2_model_input_width, stage_2_model_input_height),
        fps=args.fps_limit,
        profile=dai.VideoEncoderProperties.Profile.H264_HIGH,
    )

    # Eyes mosaics (focused & non-focused) -> 640 x 640
    eyes_focused_enc_out = build_h264_stream(
        pipeline,
        src=eye_mosaic_focused_out,
        size=(640, 640),
        fps=args.fps_limit,
        profile=dai.VideoEncoderProperties.Profile.H264_HIGH,
        vbr=True,
    )
    eyes_non_focused_enc_out = build_h264_stream(
        pipeline,
        src=eye_mosaic_non_focused_out,
        size=(640, 640),
        fps=args.fps_limit,
        profile=dai.VideoEncoderProperties.Profile.H264_HIGH,
        vbr=True,
    )

    visualizer.addTopic("Video", video_enc_out, "images")
    visualizer.addTopic("Detections Stage 1", stage1_nn.out, "annotations")

    visualizer.addTopic("Crops Mosaic", face_mosaic_enc_out, "images")
    visualizer.addTopic("Detections Stage 2 Crops", mosaic_annotation.out, "annotations")

    visualizer.addTopic("Eyes Mosaic", eyes_focused_enc_out, "images")
    visualizer.addTopic("Eyes Mosaic Non Focused", eyes_non_focused_enc_out, "images")

    visualizer.addTopic("Detections Non Focused Remapped", non_focused_remapped_bridge.out, "annotations")
    visualizer.addTopic("NN input Eye Detection", resize_non_focused.out, "images")
    visualizer.addTopic("Full Frame eyes detection", full_frame_remap.out, "annotations")
    visualizer.addTopic("NN input Face Detection", resize_s1.out, "images")
    visualizer.addTopic("Detections NN Non Focused", non_focused_bridge.out, "annotations")
    pipeline.start()
    visualizer.registerPipeline(pipeline)

    while pipeline.isRunning():
        key = visualizer.waitKey(1)
        if key == ord("q"):
            print("[MAIN] Got q key. Exiting...")
            break
