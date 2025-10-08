
import depthai as dai
import host_nodes
import pipeline_helpers as pipe
from depthai_nodes.node import ParsingNeuralNetwork, GatherData

from utils.coordinate_transformer import transform_ymin, transform_ymax
from utils.arguments import initialize_argparser
from utils.pipeline_build_helpers import build_h264_stream

_, args = initialize_argparser()

HIGH_RES_WIDTH, HIGH_RES_HEIGHT = 6000, 6000
LOW_RES_WIDTH, LOW_RES_HEIGHT = 640, 640
PEOPLE_DETECTION_MODEL = "luxonis/scrfd-person-detection:25g-640x640"
FACE_DETECTION_MODEL = "luxonis/yunet:320x240"

visualizer = dai.RemoteConnection(httpPort=8082)
device = dai.Device(dai.DeviceInfo(args.device)) if args.device else dai.Device()
platform = device.getPlatform().name

frame_type = dai.ImgFrame.Type.BGR888i if platform == "RVC4" else dai.ImgFrame.Type.BGR888p
if not args.fps_limit:
    args.fps_limit = 8 if platform == "RVC2" else 15

with dai.Pipeline(device) as pipeline:
    rgb_high_res_out, rgb_low_res_out = pipe.create_rgb(
        pipeline=pipeline,
        high_res=(HIGH_RES_WIDTH, HIGH_RES_HEIGHT),
        low_res=(LOW_RES_WIDTH, LOW_RES_HEIGHT),
        fps=args.fps_limit,
    )

    model_description = dai.NNModelDescription(PEOPLE_DETECTION_MODEL)
    model_description.platform = platform
    # model_archive = dai.NNArchive(dai.getModelFromZoo(model_description))
    people_detection_nn = pipeline.create(ParsingNeuralNetwork).build(
         rgb_low_res_out, model_description
    )
    largest_people_detection = host_nodes.PickLargestBbox().build(people_detection_nn.out)
    largest_people_detection_as_img_detection = host_nodes.SafeImgDetectionsExtendedBridge().build(largest_people_detection.out)
    largest_people_detection_cropped = host_nodes.CropPersonDetectionWaistDown(
        ymin_transformer=transform_ymin,
        ymax_transformer=transform_ymax,
    ).build(largest_people_detection_as_img_detection.out)
    people_crop_nn_manip_out = pipe.build_roi_cropper(
        pipeline=pipeline,
        preview_stream=rgb_high_res_out,
        det_stream=largest_people_detection_cropped.out,
        out_size=(320, 240),
        frame_type=frame_type,
        padding=0.,
        pool_size=5,
        cfg_queue_size=5,
    )

    # face detection model
    model_description = dai.NNModelDescription(FACE_DETECTION_MODEL)
    model_description.platform = platform
    face_detection_nn = pipeline.create(ParsingNeuralNetwork).build(
        people_crop_nn_manip_out, model_description
    )
    match_people_with_head_detections = GatherData().build(camera_fps=args.fps_limit)
    face_detection_nn.out.link(match_people_with_head_detections.input_data)
    largest_people_detection_cropped.out.link(match_people_with_head_detections.input_reference)

    map_dets_to_original_frame = host_nodes.MapDetectionsToOriginalFrame(
        face_detection_nn_width=320,
        face_detection_nn_height=240,
    ).build(match_people_with_head_detections.out)
    head_detection_as_img_frame = host_nodes.SafeImgDetectionsExtendedBridge().build(map_dets_to_original_frame.out)
    # head_dets_decreased_fps = host_nodes.DecreaseFps(send_every_x_seconds=0.5).build(head_detection_as_img_frame.out)
    head_cropper = pipe.build_roi_cropper(
        pipeline=pipeline,
        preview_stream=rgb_high_res_out,
        det_stream=head_detection_as_img_frame.out,
        out_size=(320, 320),
        frame_type=frame_type,
        padding=0.02,
        pool_size=5,
        cfg_queue_size=5,
    )
    head_cropper_low_res = pipe.build_roi_cropper(
        pipeline=pipeline,
        preview_stream=rgb_low_res_out,
        det_stream=head_detection_as_img_frame.out,
        out_size=(320, 320),
        frame_type=frame_type,
        padding=0.02,
        pool_size=5,
        cfg_queue_size=5,
    )
    rgb_low_res_encoder = build_h264_stream(
        pipeline,
        src=rgb_low_res_out,
        size=(LOW_RES_WIDTH, LOW_RES_HEIGHT),
        fps=args.fps_limit,
        profile=dai.VideoEncoderProperties.Profile.H264_MAIN,
        assume_nv12=False,
    )
    # rgb_high_res_encoder = build_h264_stream(
    #     pipeline,
    #     src=rgb_high_res_out,
    #     size=(HIGH_RES_WIDTH, HIGH_RES_HEIGHT),
    #     fps=args.fps_limit,
    #     profile=dai.VideoEncoderProperties.Profile.MJPEG,
    #     assume_nv12=False,
    # )
    visualizer.addTopic("People detections NN", largest_people_detection.out, "people_annotations")
    visualizer.addTopic("Face detection NN", map_dets_to_original_frame.out, "face_annotations")
    # visualizer.addTopic("Face detections original", face_detection_nn.out, "face_detections_original")
    # visualizer.addTopic(f"People crop manip", people_crop_nn_manip_out, "people_crop_manip")
    visualizer.addTopic("640x640 RGB", rgb_low_res_encoder, "low_res_image")
    # visualizer.addTopic("High Res RGB", rgb_high_res_encoder, "high_res_image")
    visualizer.addTopic("Focused Vision, crop from 6000x6000", head_cropper, "high_res_head")
    visualizer.addTopic("Non-Focused Vision, crop from 640x640", head_cropper_low_res, "low_res_head")
    pipeline.start()
    visualizer.registerPipeline(pipeline)

    while pipeline.isRunning():
        key = visualizer.waitKey(1)
        if key == ord("q"):
            print("[MAIN] Got q key. Exiting...")
            break