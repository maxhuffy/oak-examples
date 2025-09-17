# DINOv3 Patch Tracking (Minimal)

Runs the DINOv3 backbone (`luxonis/dinov3-backbone:convnext-base-352x480`) to get patch embeddings and highlights patches similar to a user-selected point. The mask is colorized and overlaid on the input stream.

## Run

```bash
python3 main.py --select_x 0.5 --select_y 0.5 --save_after_frames 15 --similarity_thresh 0.6
```

- Use `--media_path path/to/video.mp4` to run on a video instead of the camera.
- Press `s` to force-save the selection immediately; `q` to quit.

## Arguments

- `--model` (default: `luxonis/dinov3-backbone:convnext-base-352x480`)
- `--select_x`, `--select_y`: normalized point (0..1) to track
- `--save_after_frames`: delay before saving the selected patch feature
- `--similarity_thresh`: cosine similarity threshold for the mask
- `--save_path`: optional `.npy` path to save the selected feature
- `--device`, `--fps_limit`, `--media_path`

## Notes

- For private models, set `DEPTHAI_HUB_API_KEY` in the environment before running.
- Uses a raw `NeuralNetwork` node with a host node for similarity and a segmentation mask overlay via colormap.