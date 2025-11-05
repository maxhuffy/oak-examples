import os
import depthai as dai
from utils.arguments import initialize_argparser

# Allow quick A/B via environment variables without touching repo utils
USE_ENCODED = os.getenv("OAK_USE_ENCODED", "0") == "1"  # default to RAW streaming
CODEC = os.getenv("OAK_CODEC", "auto").lower()  # auto|h264|h265|mjpeg

# Parse CLI args (device selection, fps_limit)
_, args = initialize_argparser()

visualizer = dai.RemoteConnection(httpPort=8082)
device = dai.Device(dai.DeviceInfo(args.device)) if args.device else dai.Device()
platform = device.getPlatform()
print(f"Device Information: {device.getDeviceInfo()}")
print(f"Platform: {platform.name}")

# Keep FPS conservative if not provided
if args.fps_limit is None:
    args.fps_limit = 15
    print("No --fps_limit provided. Using reduced default: 15 FPS")

# Decide encoder profile if encoding is enabled
if CODEC == "h264":
    encoder_profile = dai.VideoEncoderProperties.Profile.H264_MAIN
elif CODEC == "h265":
    encoder_profile = dai.VideoEncoderProperties.Profile.H265_MAIN
elif CODEC == "mjpeg":
    encoder_profile = dai.VideoEncoderProperties.Profile.MJPEG
else:  # auto
    encoder_profile = (
        dai.VideoEncoderProperties.Profile.H265_MAIN
        if platform == dai.Platform.RVC4
        else dai.VideoEncoderProperties.Profile.H264_MAIN
    )

with dai.Pipeline(device) as pipeline:
    print("Creating MEGA reduced-load pipeline (combined tips, all cameras)...")

    for sensor in device.getConnectedCameraFeatures():
        cam = pipeline.create(dai.node.Camera).build(sensor.socket)

        # Cap to 640x400 (do not lower further), but also don't upscale smaller sensors
        req_w = min(sensor.width, 640)
        req_h = min(sensor.height, 400)
        request_resolution = (req_w, req_h)

        cam_out = cam.requestOutput(
            size=request_resolution,
            type=dai.ImgFrame.Type.NV12,
            fps=args.fps_limit,
        )
        # Prefer small pools to avoid memory spikes and encourage dropping
        try:
            cam_out.setNumFramesPool(4)
        except Exception:
            pass

        if USE_ENCODED:
            # Non-blocking encoder path (Option A)
            encoder = pipeline.create(dai.node.VideoEncoder)
            encoder.setDefaultProfilePreset(args.fps_limit, encoder_profile)
            # Critical: drop instead of stall when host/browser is slow
            encoder.input.setBlocking(False)
            encoder.input.setMaxSize(1)

            cam_out.link(encoder.input)
            visualizer.addTopic(sensor.socket.name, encoder.out, "images")
        else:
            # RAW NV12 streaming path (Option B) at same resolution
            visualizer.addTopic(sensor.socket.name, cam_out, "images")

        print(
            f"Configured {sensor.socket.name}: {request_resolution[0]}x{request_resolution[1]} @ {args.fps_limit} FPS | "
            + ("ENCODED" if USE_ENCODED else "RAW")
        )

    print("Pipeline created.")

    pipeline.start()
    visualizer.registerPipeline(pipeline)

    while pipeline.isRunning():
        key = visualizer.waitKey(1)
        if key == ord("q"):
            print("Got q key from the remote connection! Exiting...")
            break
