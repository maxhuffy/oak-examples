import depthai as dai
from utils.arguments import initialize_argparser
from depthai_nodes.node import ApplyColormap
import cv2

# Goal: mirror main.py but output ONLY depth with lower end-to-end latency.
# Tactics:
# - Do not create/send Color/Left/Right streams (reduce host load and queues).
# - Keep RAW processing (no encoder anywhere).
# - Use small frame pools on camera outputs to discourage buffering.
# - Keep StereoDepth algorithm flags the same as main.py to make comparison fair.

_, args = initialize_argparser()

visualizer = dai.RemoteConnection(httpPort=8082)
device = dai.Device(dai.DeviceInfo(args.device)) if args.device else dai.Device()
print("Device Information: ", device.getDeviceInfo())

with dai.Pipeline(device) as pipeline:
    print("Creating low-latency depth-only pipeline...")

    # Build mono cameras (same sockets as main.py)
    left = pipeline.create(dai.node.Camera).build(dai.CameraBoardSocket.CAM_B)
    right = pipeline.create(dai.node.Camera).build(dai.CameraBoardSocket.CAM_C)

    # Full-resolution mono outputs, paced by args.fps_limit
    left_output = left.requestFullResolutionOutput(
        dai.ImgFrame.Type.NV12, fps=args.fps_limit
    )
    right_output = right.requestFullResolutionOutput(
        dai.ImgFrame.Type.NV12, fps=args.fps_limit
    )

    # Small frame pools to reduce latency spikes (best-effort; safe if unsupported)
    try:
        left_output.setNumFramesPool(4)
        right_output.setNumFramesPool(4)
    except Exception:
        pass

    # Stereo depth (same core settings as main.py to keep output comparable)
    stereo = pipeline.create(dai.node.StereoDepth).build(
        left=left_output,
        right=right_output,
    )

    stereo.initialConfig.setMedianFilter(dai.MedianFilter.MEDIAN_OFF)
    stereo.setRectification(True)
    stereo.setExtendedDisparity(True)
    stereo.setLeftRightCheck(True)

    # Colorize disparity for visualization
    depth_colormap = pipeline.create(ApplyColormap).build(stereo.disparity)
    # depth_colormap.setMaxValue(int(stereo.initialConfig.getMaxDisparity()))  # enable when API bug is fixed
    depth_colormap.setColormap(cv2.COLORMAP_JET)

    # Only output the depth stream to minimize host-side work and queues
    visualizer.addTopic("Depth", depth_colormap.out, "images")

    print("Pipeline created.")

    pipeline.start()
    visualizer.registerPipeline(pipeline)

    while pipeline.isRunning():
        key = visualizer.waitKey(1)
        if key == ord("q"):
            print("Got q key from the remote connection!")
            break
