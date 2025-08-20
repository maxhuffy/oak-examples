# Extended 3D Measurement with YOLOE

This example showcases a possible approach for measuring objects in 3D using DepthAI. 
On the DepthAI backend, it runs **YOLOE** model on-device, with configurable class labels and confidence threshold — both controllable via the frontend.
The custom frontend allows a user to click on a detected object in the video stream, which triggers segmentation and projects the object’s mask onto the 3D point cloud. 
The frontend, built using the @luxonis/depthai-viewer-common package, provides a real-time video stream with overlayed detections and interactive object selection. It is combined with the [default oakapp docker image](https://hub.docker.com/r/luxonis/oakapp-base), which enables remote access via WebRTC.

> **Note:** This example works only on RVC4 in standalone mode.

## Usage

Running this example requires a **Luxonis device** connected to your computer. Refer to the [documentation](https://docs.luxonis.com/software-v3/) to setup your device if you haven't done it already.

Here is a list of all available parameters:

```
-d DEVICE, --device DEVICE
					Optional name, DeviceID or IP of the camera to connect to. (default: None)
-fps FPS_LIMIT, --fps-limit FPS_LIMIT
					FPS limit. (default: None)
-ip IP, --ip IP       IP address to serve the frontend on. (default: None)
-p PORT, --port PORT  Port to serve the frontend on. (default: None)
```

### Model Options

This example currently uses **YOLOE**: A fast and efficient object detection model, which provides both bounding boxes and segmentation masks.

### Prerequisites

Before running the example you’ll need to first build the frontend. Follow these steps:

1. Install FE dependencies: `cd frontend/ && npm i`
1. Build the FE: `npm run build`
1. Move back to origin directory: `cd ..`

## Standalone Mode (RVC4 only)

Running the example in the standalone mode, app runs entirely on the device.
To run the example in this mode, first install the `oakctl` tool using the installation instructions [here](https://docs.luxonis.com/software-v3/oak-apps/oakctl).

The app can then be run with:

```bash
oakctl connect <DEVICE_IP>
oakctl app run .
```

Once the app is built and running you can access the DepthAI Viewer locally by opening `https://<OAK4_IP>:9000/` in your browser (the exact URL will be shown in the terminal output).

### Remote access

1. You can upload oakapp to Luxonis Hub via oakctl
1. And then you can just remotly open App UI via App detail
