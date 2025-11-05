import depthai as dai
from utils.arguments import initialize_argparser

# Parse CLI args (device selection, fps_limit, etc.)
_, args = initialize_argparser()

visualizer = dai.RemoteConnection(httpPort=8082)
device = dai.Device(dai.DeviceInfo(args.device)) if args.device else dai.Device()
platform = device.getPlatform()
print(f"Device Information: {device.getDeviceInfo()}")
print(f"Platform: {platform.name}")

# Apply a conservative default FPS if none was provided
if args.fps_limit is None:
    args.fps_limit = 15
    print("No --fps_limit provided. Using reduced default: 15 FPS")

# Choose encoder profile: prefer H265 on RVC4, H264 elsewhere
encoder_profile = (
    dai.VideoEncoderProperties.Profile.H265_MAIN
    if platform == dai.Platform.RVC4
    else dai.VideoEncoderProperties.Profile.H264_MAIN
)

with dai.Pipeline(device) as pipeline:
    print("Creating reduced-load pipeline (Option A)...")

    # Iterate all connected sensors and stream each with reduced resolution/FPS
    for sensor in device.getConnectedCameraFeatures():
        cam = pipeline.create(dai.node.Camera).build(sensor.socket)

        # Cap max output resolution to 1280x720 to reduce bandwidth/encode load
        req_w = 640 if sensor.width > 1280 else sensor.width
        req_h = 400 if sensor.height > 720 else sensor.height
        request_resolution = (req_w, req_h)

        cam_out = cam.requestOutput(
            size=request_resolution,
            type=dai.ImgFrame.Type.NV12,
            fps=args.fps_limit,
        )

        # Encode with non-blocking input and a tiny queue to drop instead of stall
        encoder = pipeline.create(dai.node.VideoEncoder)
        encoder.setDefaultProfilePreset(args.fps_limit, encoder_profile)
        encoder.input.setBlocking(False)
        encoder.input.setMaxSize(1)

        cam_out.link(encoder.input)

        # Visualizer topic per camera socket
        visualizer.addTopic(sensor.socket.name, encoder.out, "images")

    print("Pipeline created.")

    pipeline.start()
    visualizer.registerPipeline(pipeline)

    while pipeline.isRunning():
        key = visualizer.waitKey(1)
        if key == ord("q"):
            print("Got q key from the remote connection! Exiting...")
            break
