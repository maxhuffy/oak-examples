import depthai as dai
from utils.arguments import initialize_argparser

# Keep this as close to main.py as possible, with ONLY the change that helped:
# - Stream RAW NV12 frames directly (no VideoEncoder) to avoid host decode bottlenecks.
# Everything else (res cap to 1080p, fps behavior, multi-camera) remains the same.

_, args = initialize_argparser()

visualizer = dai.RemoteConnection(httpPort=8082)
device = dai.Device(dai.DeviceInfo(args.device)) if args.device else dai.Device()
print("Device Information: ", device.getDeviceInfo())

# Optional: collect camera features (like main.py)
cam_features = {}
for cam in device.getConnectedCameraFeatures():
    cam_features[cam.socket] = (cam.width, cam.height)

with dai.Pipeline(device) as pipeline:
    print("Creating pipeline (main_v3: RAW streaming, no encoder)...")

    camera_sensors = device.getConnectedCameraFeatures()
    for sensor in camera_sensors:
        cam = pipeline.create(dai.node.Camera).build(sensor.socket)

        request_resolution = (
            (sensor.width, sensor.height)
            if sensor.width <= 1920 and sensor.height <= 1080
            else (1920, 1080)
        )  # same 1080p cap as main.py

        cam_out = cam.requestOutput(
            request_resolution, dai.ImgFrame.Type.NV12, fps=args.fps_limit
        )
        # Encourage drop over buffering spikes; ignore if unsupported
        try:
            cam_out.setNumFramesPool(4)
        except Exception:
            pass

        # Key change vs main.py: send RAW frames instead of encoded stream
        visualizer.addTopic(sensor.socket.name, cam_out, "images")

    print("Pipeline created.")

    pipeline.start()
    visualizer.registerPipeline(pipeline)

    while pipeline.isRunning():
        key = visualizer.waitKey(1)
        if key == ord("q"):
            print("Got q key from the remote connection!")
            break
