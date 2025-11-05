import os
from dotenv import load_dotenv

import depthai as dai
from depthai_nodes.node import ParsingNeuralNetwork, ImgFrameOverlay, ApplyColormap

from utils.arguments import initialize_argparser
from utils.input import create_input_node

# Purpose: Keep behavior identical to generic-example/main.py, but reduce end-to-end latency by
# - encouraging drop-old on NN input (small queue, non-blocking)
# - reducing internal pool frames for the NN node
# No encoder is used in the base file, so no change there.

load_dotenv(override=True)

_, args = initialize_argparser()

if args.api_key:
    os.environ["DEPTHAI_HUB_API_KEY"] = args.api_key

visualizer = dai.RemoteConnection(httpPort=8082)
device = dai.Device(dai.DeviceInfo(args.device)) if args.device else dai.Device()
platform = device.getPlatformAsString()
print(f"Platform: {platform}")

with dai.Pipeline(device) as pipeline:
    print("Creating pipeline (reduced latency variant)...")

    # model
    model_description = dai.NNModelDescription(f"yolov6_nano_r2_coco.{platform}.yaml")
    if model_description.model != args.model:
        model_description = dai.NNModelDescription(args.model, platform=platform)
    nn_archive = dai.NNArchive(dai.getModelFromZoo(model_description))

    # media/camera input (same as main)
    input_node = create_input_node(
        pipeline,
        platform,
        args.media_path,
    )

    # NN with parser; keep FPS pacing from args
    nn_with_parser: ParsingNeuralNetwork = pipeline.create(ParsingNeuralNetwork).build(
        input_node, nn_archive, fps=args.fps_limit
    )

    # Low-latency tweaks: prefer dropping frames over building queues
    try:
        # Keep the queue tiny so new frames replace old ones
        nn_with_parser.input.setMaxSize(1)
        # Non-blocking input prevents upstream backpressure stalls
        nn_with_parser.input.setBlocking(False)
    except Exception:
        pass

    try:
        # Reduce internal pools to lower buffering latency
        nn_with_parser.setNumPoolFrames(4)
    except Exception:
        pass

    # annotation and visualization (unchanged)
    if args.overlay_mode:
        apply_colormap_node = pipeline.create(ApplyColormap).build(nn_with_parser.out)
        overlay_frames_node = pipeline.create(ImgFrameOverlay).build(
            nn_with_parser.passthrough,
            apply_colormap_node.out,
        )
        visualizer.addTopic("Video", overlay_frames_node.out, "images")
    else:
        visualizer.addTopic("Video", nn_with_parser.passthrough, "images")
    visualizer.addTopic("Detections", nn_with_parser.out, "detections")

    print("Pipeline created.")

    pipeline.start()
    visualizer.registerPipeline(pipeline)

    while pipeline.isRunning():
        key = visualizer.waitKey(1)
        if key == ord("q"):
            print("Got q key from the remote connection!")
            break
