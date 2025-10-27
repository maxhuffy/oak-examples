from pathlib import Path
from dotenv import load_dotenv
import os
from functools import partial

os.environ["DEPTHAI_LEVEL"] = "info"
import depthai as dai
from depthai_nodes.node import (
    ParsingNeuralNetwork,
    ImgDetectionsFilter,
    ImgFrameOverlay,
    ApplyColormap,
    SnapsProducer2Buffered,
    ImgDetectionsBridge,
)

from utils.helper_functions import (
    extract_text_embeddings,
    extract_image_prompt_embeddings,
    base64_to_cv2_image,
    make_dummy_features,
)
from utils.arguments import initialize_argparser
from utils.annotation_node import AnnotationNode
from utils.frame_cache_node import FrameCacheNode
from utils.snap_utils.conditions_gate import ConditionsGate
from utils.snap_utils.custon_snap_process import custom_snap_process

load_dotenv(override=True)

_, args = initialize_argparser()

if args.api_key:
    os.environ["DEPTHAI_HUB_API_KEY"] = args.api_key

IP = args.ip or "localhost"
PORT = args.port or 8080

CLASS_NAMES = ["person", "chair", "TV"]
CLASS_OFFSET = 0
MAX_NUM_CLASSES = 80
CONFIDENCE_THRESHOLD = 0.1
VISUALIZATION_RESOLUTION = (1080, 1080)

visualizer = dai.RemoteConnection(serveFrontend=False)
device = dai.Device()
platform = device.getPlatformAsString()

if platform != "RVC4":
    raise ValueError("This example is supported only on RVC4 platform")

frame_type = dai.ImgFrame.Type.BGR888i

text_features = extract_text_embeddings(
    class_names=CLASS_NAMES,
    max_num_classes=MAX_NUM_CLASSES,
    model_name="yoloe",
    precision=args.precision,
)
image_prompt_features = make_dummy_features(
    MAX_NUM_CLASSES, model_name="yoloe", precision=args.precision
)

if args.fps_limit is None:
    args.fps_limit = 5
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
        replay.setSize(VISUALIZATION_RESOLUTION[0], VISUALIZATION_RESOLUTION[1])
        video_src_out = replay.out
    else:
        cam = pipeline.create(dai.node.Camera).build(
            boardSocket=dai.CameraBoardSocket.CAM_A
        )
        video_src_out = cam.requestOutput(
            size=VISUALIZATION_RESOLUTION,
            type=dai.ImgFrame.Type.NV12,
            fps=args.fps_limit,
        )

    image_manip = pipeline.create(dai.node.ImageManip)
    image_manip.setMaxOutputFrameSize(model_w * model_h * 3)
    image_manip.initialConfig.setOutputSize(model_w, model_h)
    image_manip.initialConfig.setFrameType(frame_type)
    video_src_out.link(image_manip.inputImage)

    video_enc = pipeline.create(dai.node.VideoEncoder)
    video_enc.setDefaultProfilePreset(
        fps=args.fps_limit, profile=dai.VideoEncoderProperties.Profile.H264_MAIN
    )
    video_src_out.link(video_enc.input)

    input_node = image_manip.out

    nn_with_parser = pipeline.create(ParsingNeuralNetwork)
    nn_with_parser.setNNArchive(model_nn_archive)
    nn_with_parser.setBackend("snpe")
    nn_with_parser.setBackendProperties(
        {"runtime": "dsp", "performance_profile": "default"}
    )
    nn_with_parser.setNumInferenceThreads(1)
    nn_with_parser.getParser(0).setConfidenceThreshold(CONFIDENCE_THRESHOLD)

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

    cond_gate = ConditionsGate(default_cooldown_s=0.0, enabled=True)
    cond_gate.set_key_enabled("no_detections", False)
    cond_gate.set_cooldown("no_detections", 15.0)

    cond_gate.set_key_enabled("timed", False)
    cond_gate.set_cooldown("timed", 15.0)

    cond_gate.set_key_enabled("low_conf", False)
    cond_gate.set_cooldown("low_conf", 15.0)

    cond_gate.set_key_enabled("lost_mid", False)
    cond_gate.set_cooldown("lost_mid", 15.0)

    _runtime = {
        "newdet_running": False,
        "conf_threshold": CONFIDENCE_THRESHOLD,
        "lost_mid_margin": 0.20,
    }
    object_tracker = pipeline.create(dai.node.ObjectTracker)

    object_tracker.setTrackerType(dai.TrackerType.SHORT_TERM_IMAGELESS)
    object_tracker.setTrackerIdAssignmentPolicy(dai.TrackerIdAssignmentPolicy.UNIQUE_ID)
    object_tracker.setTrackingPerClass(True)
    object_tracker.setTrackletBirthThreshold(3)
    object_tracker.setTrackletMaxLifespan(90)
    object_tracker.setOcclusionRatioThreshold(0.5)
    object_tracker.setTrackerThreshold(0.25)

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
            class_names=CLASS_NAMES,
            model="yoloe",
            cond_gate=cond_gate,
            runtime=_runtime,
        ),
    )

    def update_labels(label_names: list[str], offset: int = 0):
        det_process_filter.setLabels(
            labels=[i for i in range(offset, offset + len(label_names))], keep=True
        )
        annotation_node.setLabelEncoding(
            {offset + k: v for k, v in enumerate(label_names)}
        )

    apply_colormap_node = pipeline.create(ApplyColormap).build(nn_with_parser.out)
    overlay_frames_node = pipeline.create(ImgFrameOverlay).build(
        video_src_out,
        apply_colormap_node.out,
        preserve_background=True,
    )
    overlay_to_nv12 = pipeline.create(dai.node.ImageManip)
    overlay_to_nv12.setMaxOutputFrameSize(
        VISUALIZATION_RESOLUTION[0] * VISUALIZATION_RESOLUTION[1] * 3
    )
    overlay_to_nv12.initialConfig.setFrameType(dai.ImgFrame.Type.NV12)
    overlay_frames_node.out.link(overlay_to_nv12.inputImage)

    overlay_enc = pipeline.create(dai.node.VideoEncoder)
    overlay_enc.setDefaultProfilePreset(
        fps=args.fps_limit, profile=dai.VideoEncoderProperties.Profile.H264_MAIN
    )
    overlay_to_nv12.out.link(overlay_enc.input)

    visualizer.addTopic("Video", overlay_enc.out, "images")
    visualizer.addTopic("Detections", annotation_node.out)

    def class_update_service(new_classes: list[str]):
        if not new_classes:
            print("List of new classes empty, skipping.")
            return
        if len(new_classes) > MAX_NUM_CLASSES:
            print(
                f"Too many classes ({len(new_classes)}) > {MAX_NUM_CLASSES}, skipping."
            )
            return

        feats = extract_text_embeddings(
            class_names=new_classes,
            max_num_classes=MAX_NUM_CLASSES,
            model_name="yoloe",
            precision=args.precision,
        )
        nn_txt = dai.NNData()
        nn_txt.addTensor(
            "texts",
            feats,
            dataType=(
                dai.TensorInfo.DataType.FP16
                if args.precision == "fp16"
                else dai.TensorInfo.DataType.U8F
            ),
        )
        textInputQueue.send(nn_txt)

        dummy = make_dummy_features(
            MAX_NUM_CLASSES, model_name="yoloe", precision=args.precision
        )
        nn_img = dai.NNData()
        nn_img.addTensor(
            "image_prompts",
            dummy,
            dataType=(
                dai.TensorInfo.DataType.FP16
                if args.precision == "fp16"
                else dai.TensorInfo.DataType.U8F
            ),
        )
        imagePromptInputQueue.send(nn_img)

        update_labels(new_classes, offset=0)

        print(f"Classes updated (YOLOE text): {new_classes}")

    def conf_threshold_update_service(new_conf_threshold: float):
        """Changes confidence threshold based on the user input"""
        _runtime["conf_threshold"] = max(0.0, min(1.0, float(new_conf_threshold)))
        nn_with_parser.getParser(0).setConfidenceThreshold(_runtime["conf_threshold"])

    def image_upload_service(image_data):
        image = base64_to_cv2_image(image_data["data"])

        image_features = extract_image_prompt_embeddings(
            image, model_name="yoloe", precision=args.precision
        )

        input_NN_data_img = dai.NNData()
        input_NN_data_img.addTensor(
            "image_prompts",
            image_features,
            dataType=(
                dai.TensorInfo.DataType.FP16
                if args.precision == "fp16"
                else dai.TensorInfo.DataType.U8F
            ),
        )
        imagePromptInputQueue.send(input_NN_data_img)

        dummy = make_dummy_features(
            MAX_NUM_CLASSES, model_name="yoloe", precision=args.precision
        )
        inputNNDataTxt = dai.NNData()
        inputNNDataTxt.addTensor(
            "texts",
            dummy,
            dataType=(
                dai.TensorInfo.DataType.FP16
                if args.precision == "fp16"
                else dai.TensorInfo.DataType.U8F
            ),
        )
        textInputQueue.send(inputNNDataTxt)

        filename = image_data["filename"]
        class_names = [filename.split(".")[0]]
        update_labels(class_names, offset=80)
        return

    def bbox_prompt_service(payload):
        image = base64_to_cv2_image(payload["data"]) if payload.get("data") else None
        if image is None:
            image = frame_cache.get_last_frame()
            if image is None:
                return {"ok": False, "reason": "no_image"}
        if image is None:
            return {"ok": False, "reason": "decode_failed"}

        bbox = payload.get("bbox", {})
        bx = float(bbox.get("x", 0.0))
        by = float(bbox.get("y", 0.0))
        bw = float(bbox.get("width", 0.0))
        bh = float(bbox.get("height", 0.0))

        H, W = image.shape[:2]
        is_pixel = payload.get("bboxType", "normalized") == "pixel"
        if is_pixel:
            x0 = int(round(bx))
            y0 = int(round(by))
            x1 = int(round(bx + bw))
            y1 = int(round(by + bh))
        else:
            x0 = int(round(bx * W))
            y0 = int(round(by * H))
            x1 = int(round((bx + bw) * W))
            y1 = int(round((by + bh) * H))

        x0, x1 = sorted((x0, x1))
        y0, y1 = sorted((y0, y1))
        return {"ok": True}

    def snap_collection_service(payload):
        base_dt_seconds = 1

        if not isinstance(payload, dict):
            return {"ok": False, "reason": "payload_must_be_dict"}

        # timed
        tcfg = payload.get("timed")
        if isinstance(tcfg, dict):
            if "enabled" in tcfg:
                cond_gate.set_key_enabled("timed", bool(tcfg["enabled"]))
            if "interval" in tcfg:
                try:
                    cond_gate.set_cooldown("timed", float(tcfg["interval"]))
                except Exception:
                    pass
            if tcfg.get("reset"):
                cond_gate.reset(["timed"])

        # no detections
        ncfg = payload.get("noDetections")
        if isinstance(ncfg, dict):
            if "enabled" in ncfg:
                cond_gate.set_key_enabled("no_detections", bool(ncfg["enabled"]))
            if "cooldown" in ncfg:
                try:
                    cond_gate.set_cooldown("no_detections", float(ncfg["cooldown"]))
                except Exception:
                    pass
            if ncfg.get("reset"):
                cond_gate.reset(["no_detections"])

        # low confidence
        lcfg = payload.get("lowConfidence")
        if isinstance(lcfg, dict):
            enable = bool(lcfg.get("enabled", False))
            thr_raw = lcfg.get("threshold", None)

            if enable:
                if thr_raw is None:
                    return {"ok": False, "reason": "low_conf_threshold_required"}

                try:
                    thr = float(thr_raw)
                    if thr > 1.0:
                        thr = thr / 100.0
                    if not (0.0 <= thr <= 1.0):
                        raise ValueError
                except Exception:
                    return {"ok": False, "reason": "invalid_low_conf_threshold"}

                _runtime["low_conf_thresh"] = thr
                cond_gate.set_key_enabled("low_conf", True)
            else:
                cond_gate.set_key_enabled("low_conf", False)
                _runtime.pop("low_conf_thresh", None)

            if "cooldown" in lcfg:
                try:
                    cond_gate.set_cooldown("low_conf", float(lcfg["cooldown"]))
                except Exception:
                    pass

            if lcfg.get("reset"):
                cond_gate.reset(["low_conf"])

        # lost in middle
        lmcfg = payload.get("lostMid")
        if isinstance(lmcfg, dict):
            if "enabled" in lmcfg:
                cond_gate.set_key_enabled("lost_mid", bool(lmcfg["enabled"]))
            if "cooldown" in lmcfg:
                try:
                    cond_gate.set_cooldown("lost_mid", float(lmcfg["cooldown"]))
                except Exception:
                    pass
            if "margin" in lmcfg:
                try:
                    m = float(lmcfg["margin"])
                    _runtime["lost_mid_margin"] = max(0.0, min(0.49, m))
                except Exception:
                    pass
            if lmcfg.get("reset"):
                cond_gate.reset(["lost_mid"])

        any_active = cond_gate.enabled and (
            cond_gate.is_key_enabled("timed")
            or cond_gate.is_key_enabled("no_detections")
            or cond_gate.is_key_enabled("low_conf")
            or cond_gate.is_key_enabled("lost_mid")
        )
        snaps_producer.setRunning(any_active)
        if any_active:
            snaps_producer.setTimeInterval(base_dt_seconds)

        return {"ok": True}

    visualizer.registerService("Class Update Service", class_update_service)
    visualizer.registerService(
        "Threshold Update Service", conf_threshold_update_service
    )
    visualizer.registerService("Image Upload Service", image_upload_service)
    visualizer.registerService("BBox Prompt Service", bbox_prompt_service)
    visualizer.registerService("Snap Collection Service", snap_collection_service)

    print("Pipeline created.")

    pipeline.start()
    visualizer.registerPipeline(pipeline)

    update_labels(CLASS_NAMES, offset=CLASS_OFFSET)

    inputNNData = dai.NNData()
    inputNNData.addTensor(
        "texts",
        text_features,
        dataType=(
            dai.TensorInfo.DataType.FP16
            if args.precision == "fp16"
            else dai.TensorInfo.DataType.U8F
        ),
    )
    textInputQueue.send(inputNNData)

    inputNNDataImg = dai.NNData()
    inputNNDataImg.addTensor(
        "image_prompts",
        image_prompt_features,
        dataType=(
            dai.TensorInfo.DataType.FP16
            if args.precision == "fp16"
            else dai.TensorInfo.DataType.U8F
        ),
    )
    imagePromptInputQueue.send(inputNNDataImg)

    print("Press 'q' to stop")

    while pipeline.isRunning():
        pipeline.processTasks()
        key = visualizer.waitKey(1)
        if key == ord("q"):
            print("Got q key. Exiting...")
            break
