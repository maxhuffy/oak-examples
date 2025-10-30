import os

from pathlib import Path
from dotenv import load_dotenv
from functools import partial

import depthai as dai
from depthai_nodes.node import (
    ParsingNeuralNetwork,
    ImgDetectionsFilter,
    SnapsProducer2Buffered,
    ImgDetectionsBridge,
)

from utils.arguments import initialize_argparser
from utils.annotation_node import AnnotationNode
from utils.frame_cache_node import FrameCacheNode
from utils.snaps.conditions_manager import ConditionsManager
from utils.snaps.custon_snap_process import custom_snap_process
from utils.snaps.tracklets import setup_tracker
from utils.services.class_update_service import ClassUpdateService
from utils.services.threshold_update_service import ThresholdUpdateService
from utils.services.image_upload_service import ImageUploadService
from utils.services.bbox_prompt_service import BBoxPromptService
from utils.services.snap_collection_service import SnapCollectionService
from utils.services.get_config_service import GetConfigService
from utils.core.tokenizer_manager import TokenizerManager
from utils.core.quantization import make_dummy_features
from utils.core.label_manager import LabelManager
import utils.constants as const

load_dotenv(override=True)

_, args = initialize_argparser()

if args.api_key:
    os.environ["DEPTHAI_HUB_API_KEY"] = args.api_key

IP = args.ip or "localhost"
PORT = args.port or 8080

visualizer = dai.RemoteConnection(serveFrontend=False)
device = dai.Device()
platform = device.getPlatformAsString()

if platform != "RVC4":
    raise ValueError("This example is supported only on RVC4 platform")

frame_type = dai.ImgFrame.Type.BGR888i

tokenizer = TokenizerManager(
    model_name=const.MODEL, precision=args.precision, max_classes=const.MAX_NUM_CLASSES
)

text_features = tokenizer.extract_text_embeddings(const.CLASS_NAMES)

image_prompt_features = make_dummy_features(
    const.MAX_NUM_CLASSES, model_name="yoloe", precision=args.precision
)

if args.fps_limit is None:
    args.fps_limit = 30
    print(
        f"\nFPS limit set to {args.fps_limit} for {platform} platform. If you want to set a custom FPS limit, use the --fps_limit flag.\n"
    )

with dai.Pipeline(device) as pipeline:
    print("Creating pipeline...")

    models_dir = Path(__file__).parent / "depthai_models"
    if args.precision != "fp16":
        raise SystemExit(
            f"Model YAML not found for YOLOE with precision {args.precision}. "
            f"YOLOE int8 YAML is not available; run with --precision fp16."
        )

    yaml_base = "yoloe_v8_l_fp16"
    yaml_filename = f"{yaml_base}.{platform}.yaml"
    yaml_path = models_dir / yaml_filename

    if not yaml_path.exists():
        raise SystemExit(f"Model YAML not found for YOLOE: {yaml_path}.")

    model_description = dai.NNModelDescription.fromYamlFile(str(yaml_path))
    model_description.platform = platform
    model_nn_archive = dai.NNArchive(dai.getModelFromZoo(model_description))
    model_w, model_h = model_nn_archive.getInputSize()

    if args.media_path:
        replay = pipeline.create(dai.node.ReplayVideo)
        replay.setReplayVideoFile(Path(args.media_path))
        replay.setOutFrameType(dai.ImgFrame.Type.NV12)
        replay.setLoop(True)
        if args.fps_limit:
            replay.setFps(args.fps_limit)
        replay.setSize(
            const.VISUALIZATION_RESOLUTION[0], const.VISUALIZATION_RESOLUTION[1]
        )
        video_src_out = replay.out

        image_manip = pipeline.create(dai.node.ImageManip)
        image_manip.setMaxOutputFrameSize(model_w * model_h * 3)
        image_manip.initialConfig.setOutputSize(model_w, model_h)
        image_manip.initialConfig.setFrameType(frame_type)
        video_src_out.link(image_manip.inputImage)

        input_node = image_manip.out
    else:
        cam = pipeline.create(dai.node.Camera).build(
            boardSocket=dai.CameraBoardSocket.CAM_A
        )
        video_src_out = cam.requestOutput(
            size=const.VISUALIZATION_RESOLUTION,
            type=dai.ImgFrame.Type.NV12,
            fps=args.fps_limit,
        )

        input_node = cam.requestOutput(
            size=(model_w, model_h), type=frame_type, fps=args.fps_limit
        )

    video_enc = pipeline.create(dai.node.VideoEncoder)
    video_enc.setDefaultProfilePreset(
        fps=args.fps_limit, profile=dai.VideoEncoderProperties.Profile.H264_MAIN
    )
    video_src_out.link(video_enc.input)

    nn_with_parser = pipeline.create(ParsingNeuralNetwork)
    nn_with_parser.setNNArchive(model_nn_archive)
    nn_with_parser.setBackend("snpe")
    nn_with_parser.setBackendProperties(
        {"runtime": "dsp", "performance_profile": "default"}
    )
    nn_with_parser.setNumInferenceThreads(1)
    nn_with_parser.getParser(0).setConfidenceThreshold(const.CONFIDENCE_THRESHOLD)

    input_node.link(nn_with_parser.inputs["images"])

    textInputQueue = nn_with_parser.inputs["texts"].createInputQueue()
    nn_with_parser.inputs["texts"].setReusePreviousMessage(True)

    imagePromptInputQueue = nn_with_parser.inputs["image_prompts"].createInputQueue()
    nn_with_parser.inputs["image_prompts"].setReusePreviousMessage(True)

    det_process_filter = pipeline.create(ImgDetectionsFilter).build(nn_with_parser.out)
    annotation_node = pipeline.create(AnnotationNode).build(
        det_process_filter.out,
        video_src_out,
    )

    filtered_bridge = pipeline.create(ImgDetectionsBridge).build(det_process_filter.out)

    frame_cache = pipeline.create(FrameCacheNode).build(video_src_out)

    cond_manager = ConditionsManager(default_cooldown_s=300.0, enabled=False)
    cond_manager.register_conditions(const.CONDITIONS)

    _runtime = {
        "conf_threshold": const.CONFIDENCE_THRESHOLD,
        "lost_mid_margin": 0.20,
        "snapping_running": False,
    }

    current_classes = const.CLASS_NAMES.copy()
    object_tracker = pipeline.create(dai.node.ObjectTracker)
    setup_tracker(object_tracker)

    input_node.link(object_tracker.inputTrackerFrame)
    input_node.link(object_tracker.inputDetectionFrame)
    filtered_bridge.out.link(object_tracker.inputDetections)

    snaps_producer = pipeline.create(SnapsProducer2Buffered).build(
        frame=video_src_out,
        msg=object_tracker.out,
        msg2=filtered_bridge.out,
        running=False,
        process_fn=partial(
            custom_snap_process,
            class_names=const.CLASS_NAMES,
            cond_manager=cond_manager,
            runtime=_runtime,
        ),
    )

    visualizer.addTopic("Video", video_enc.out, "images")
    visualizer.addTopic("Detections", annotation_node.out)

    ClassUpdateService(
        visualizer,
        textInputQueue,
        imagePromptInputQueue,
        args.precision,
        current_classes,
        det_process_filter,
        annotation_node,
    ).register()

    ThresholdUpdateService(visualizer, nn_with_parser, _runtime).register()

    ImageUploadService(
        visualizer,
        args.precision,
        imagePromptInputQueue,
        textInputQueue,
        det_process_filter,
        annotation_node,
    ).register()

    BBoxPromptService(visualizer, frame_cache).register()

    SnapCollectionService(visualizer, _runtime, cond_manager, snaps_producer).register()

    GetConfigService(visualizer, current_classes, _runtime, cond_manager).register()

    print("Pipeline created.")

    pipeline.start()
    visualizer.registerPipeline(pipeline)

    label_manager = LabelManager(det_process_filter, annotation_node)
    label_manager.update_labels(const.CLASS_NAMES, offset=const.CLASS_OFFSET)

    dataType = (
        dai.TensorInfo.DataType.FP16
        if args.precision == "fp16"
        else dai.TensorInfo.DataType.U8F
    )
    inputNNData = dai.NNData()
    inputNNData.addTensor(
        "texts",
        text_features,
        dataType=dataType,
    )
    textInputQueue.send(inputNNData)

    inputNNDataImg = dai.NNData()
    inputNNDataImg.addTensor(
        "image_prompts",
        image_prompt_features,
        dataType=dataType,
    )
    imagePromptInputQueue.send(inputNNDataImg)

    print("Press 'q' to stop")

    while pipeline.isRunning():
        pipeline.processTasks()
        key = visualizer.waitKey(1)
        if key == ord("q"):
            print("Got q key. Exiting...")
            break
