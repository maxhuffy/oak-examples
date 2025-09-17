from pathlib import Path

import depthai as dai
from depthai_nodes.node import ApplyColormap, ImgFrameOverlay

from utils.arguments import initialize_argparser
from utils.dino_patch_matcher import DINOPatchMatcher


REQ_WIDTH, REQ_HEIGHT = (
    1280,
    720,
)


def main() -> None:
    _, args = initialize_argparser()

    visualizer = dai.RemoteConnection(httpPort=8082)
    device = dai.Device(dai.DeviceInfo(args.device)) if args.device else dai.Device()
    platform = device.getPlatform().name
    print(f"Platform: {platform}")

    frame_type = (
        dai.ImgFrame.Type.BGR888p if platform == "RVC2" else dai.ImgFrame.Type.BGR888i
    )

    if args.fps_limit is None:
        args.fps_limit = 10 if platform == "RVC2" else 30
        print(
            f"\nFPS limit set to {args.fps_limit} for {platform} platform. If you want to set a custom FPS limit, use the --fps_limit flag.\n"
        )

    with dai.Pipeline(device) as pipeline:
        print("Creating pipeline...")

        # DINOv3 backbone model (produces patch features)
        model_description = dai.NNModelDescription.fromYamlFile(
            f"dinov3_backbone_convnext_base_352x480.{platform}.yaml"
        )
        # Allow overriding via slug if provided
        if args.model:
            try:
                # Accept HubAI slug like 'luxonis/dinov3-backbone:convnext-base-352x480'
                model_description = dai.NNModelDescription(args.model, platform=platform)
            except Exception as ex:
                print(f"Warning: failed to load model by slug '{args.model}': {ex}. Falling back to default for {platform}.")
        nn_archive = dai.NNArchive(dai.getModelFromZoo(model_description, useCached=False))    

        # media/camera input
        if args.media_path:
            replay = pipeline.create(dai.node.ReplayVideo)
            replay.setReplayVideoFile(Path(args.media_path))
            replay.setOutFrameType(frame_type)
            replay.setLoop(True)
            if args.fps_limit:
                replay.setFps(args.fps_limit)
            replay.setSize(REQ_WIDTH, REQ_HEIGHT)
            input_node_out = replay.out
        else:
            cam = pipeline.create(dai.node.Camera).build()
            cam_out = cam.requestOutput((REQ_WIDTH, REQ_HEIGHT), frame_type, fps=args.fps_limit)
            input_node_out = cam_out

        # resize to model input size
        resize = pipeline.create(dai.node.ImageManip)
        resize.initialConfig.setOutputSize(*nn_archive.getInputSize())
        resize.initialConfig.setFrameType(frame_type)
        resize.setMaxOutputFrameSize(
            nn_archive.getInputWidth() * nn_archive.getInputHeight() * 3
        )
        input_node_out.link(resize.inputImage)

        # raw NN (headless backbone)
        nn = pipeline.create(dai.node.NeuralNetwork).build(resize.out, nn_archive)

        # host-side feature matcher producing mask of similar patches
        matcher = pipeline.create(DINOPatchMatcher).build(nn=nn.out)
        matcher.set_nn_size(nn_archive.getInputSize())
        if args.select_x is not None and args.select_y is not None:
            matcher.set_selection(args.select_x, args.select_y, args.save_after_frames)
        matcher.set_similarity_threshold(args.similarity_thresh)
        if getattr(args, "save_path", None):
            matcher.set_save_path(args.save_path)

        # visualization: colormap the mask and overlay over the input video
        apply_colormap = pipeline.create(ApplyColormap).build(matcher.output)
        overlay = pipeline.create(ImgFrameOverlay).build(
            input_node_out, apply_colormap.out
        )
        visualizer.addTopic("Video", overlay.out, "images")

        print("Pipeline created.")

        pipeline.start()
        visualizer.registerPipeline(pipeline)

        while pipeline.isRunning():
            key = visualizer.waitKey(1)
            if key == ord("q"):
                print("Got q key. Exiting...")
                break
            if key == ord("s"):
                matcher.force_save_selection()


if __name__ == "__main__":
    main()

