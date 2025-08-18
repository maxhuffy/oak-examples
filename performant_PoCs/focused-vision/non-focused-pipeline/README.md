# Non-Focused Eye Annotator

This example shows how to run a simple face-detection-based pipeline and view **three streams at the same time**:

1. **Video** – the plain camera feed.  
2. **NN Input** – the resized frame sent to the neural network (`320x240`).  
3. **Eyes (Non-Focused)** – the camera feed with detected eyes marked by squares as annotations.

## How it works
- The camera (or video file) provides frames.
- Frames are resized to the NN input size and passed to the [YuNet face detection model](https://models.luxonis.com/luxonis/yunet/5d635f3c-45c0-41d2-8800-7ca3681b1915?backTo=%2F).
- The model output is passed to a custom annotation node that draws eye bounding boxes.
- All three streams are sent to the DepthAI visualizer over WebSocket so they can be displayed side-by-side.

## How to run
1. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```
2. Connect your Luxonis device.  
3. Run:
   ```bash
   python3 main.py
   ```
4. Open [http://localhost:8082](http://localhost:8082) to view the streams.
