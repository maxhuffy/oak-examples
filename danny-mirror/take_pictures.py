"""Preview the OAK-D Lite color stream and capture stills with the space bar."""

from __future__ import annotations

import argparse
from datetime import datetime
from pathlib import Path

import cv2
import depthai as dai


DEFAULT_OUTPUT_DIR = Path(__file__).resolve().parent / "try-on-clothes"
WINDOW_TITLE = "OAK-D Lite Preview"
CAPTURE_KEY_CODE = ord(" ")  # Space bar
AF_TRIGGER_KEY_CODE = ord("f")
AF_TRIGGER_KEY_CODE_UPPER = ord("F")
MISSING_HIGHGUI_MSG = (
	"OpenCV in this environment was built without GUI/highgui support. "
	"Install the full 'opencv-python' package (not the headless build) to enable preview windows, "
	"or run the script on a system with GUI-capable OpenCV."
)


def parse_args() -> argparse.Namespace:
	parser = argparse.ArgumentParser(
		description="Preview the OAK-D Lite color stream and capture stills with the space bar.",
		formatter_class=argparse.ArgumentDefaultsHelpFormatter,
	)
	parser.add_argument(
		"-d",
		"--device",
		type=str,
		default=None,
		help="Optional DeviceID or IP address to connect to a remote camera.",
	)
	parser.add_argument(
		"--width",
		type=int,
		default=1920,
		help="Width of the preview stream in pixels.",
	)
	parser.add_argument(
		"--height",
		type=int,
		default=1080,
		help="Height of the preview stream in pixels.",
	)
	parser.add_argument(
		"--focus-mode",
		choices=["continuous", "auto", "manual", "off"],
		default="continuous",
		help="Select autofocus behavior for the RGB camera.",
	)
	parser.add_argument(
		"--focus-position",
		type=int,
		default=None,
		help="Lens position (0-255) used when focus mode is manual.",
	)
	parser.add_argument(
		"--preview-width",
		type=int,
		default=1280,
		help="Width of the on-screen preview window in pixels.",
	)
	parser.add_argument(
		"--preview-height",
		type=int,
		default=720,
		help="Height of the on-screen preview window in pixels.",
	)
	parser.add_argument(
		"--fps",
		type=float,
		default=30.0,
		help="Desired frame rate for the preview stream.",
	)
	parser.add_argument(
		"--exit-key",
		type=str,
		default="q",
		help="Key used to close the preview window.",
	)
	parser.add_argument(
		"-o",
		"--output-dir",
		type=Path,
		default=DEFAULT_OUTPUT_DIR,
		help="Directory where captured frames will be saved.",
	)
	return parser.parse_args()


def ensure_output_dir(path: Path) -> Path:
	path.mkdir(parents=True, exist_ok=True)
	return path


def create_device(device_identifier: str | None) -> dai.Device:
	if device_identifier:
		return dai.Device(dai.DeviceInfo(device_identifier))
	return dai.Device()


def configure_focus(
	camera: dai.node.Camera,
	focus_mode: str,
	focus_position: int | None,
) -> dai.Node.InputQueue | None:
	focus_mode = focus_mode.lower()
	ctrl = camera.initialControl
	if focus_mode == "continuous":
		ctrl.setAutoFocusMode(dai.CameraControl.AutoFocusMode.CONTINUOUS_VIDEO)
		ctrl.setAutoFocusTrigger()
		try:
			return camera.inputControl.createInputQueue()
		except Exception:
			print("Warning: Unable to create autofocus control queue; trigger hotkey disabled.")
			return None
	if focus_mode == "auto":
		ctrl.setAutoFocusMode(dai.CameraControl.AutoFocusMode.AUTO)
		ctrl.setAutoFocusTrigger()
		try:
			return camera.inputControl.createInputQueue()
		except Exception:
			print("Warning: Unable to create autofocus control queue; trigger hotkey disabled.")
			return None
	if focus_mode == "manual":
		position = focus_position if focus_position is not None else 130
		position = max(0, min(255, position))
		ctrl.setManualFocus(position)
		return None
	# focus_mode == "off"; leave camera defaults untouched
	return None


def trigger_autofocus(
	control_queue: dai.Node.InputQueue | None,
	focus_mode: str,
) -> None:
	if control_queue is None:
		print("Autofocus control not available; ensure focus mode supports triggering.")
		return
	ctrl = dai.CameraControl()
	if focus_mode.lower() == "continuous":
		ctrl.setAutoFocusMode(dai.CameraControl.AutoFocusMode.CONTINUOUS_VIDEO)
	else:
		ctrl.setAutoFocusMode(dai.CameraControl.AutoFocusMode.AUTO)
	ctrl.setAutoFocusTrigger()
	try:
		control_queue.send(ctrl)
		print("Autofocus trigger sent.")
	except RuntimeError:
		print("Failed to send autofocus command; is the device still connected?")


def create_preview_queue(
	pipeline: dai.Pipeline,
	frame_size: tuple[int, int],
	fps: float,
	focus_mode: str,
	focus_position: int | None,
) -> tuple[dai.Node.OutputQueue, dai.Node.InputQueue | None]:
	camera = pipeline.create(dai.node.Camera).build(dai.CameraBoardSocket.CAM_A)
	control_queue = configure_focus(camera, focus_mode, focus_position)
	stream = camera.requestOutput(frame_size, dai.ImgFrame.Type.NV12, fps=fps)
	try:
		stream.setNumFramesPool(4)
	except Exception:
		pass
	return stream.createOutputQueue(blocking=False, maxSize=1), control_queue


def format_filename(output_dir: Path) -> Path:
	timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
	return output_dir / f"oak_capture_{timestamp}.png"


def ensure_highgui_support() -> None:
	try:
		cv2.namedWindow(WINDOW_TITLE, cv2.WINDOW_NORMAL)
	except cv2.error as exc:  # pragma: no cover - requires broken GUI build
		raise RuntimeError(MISSING_HIGHGUI_MSG) from exc
	else:
		try:
			cv2.destroyWindow(WINDOW_TITLE)
		except cv2.error:
			pass


def main() -> None:
	args = parse_args()
	output_dir = ensure_output_dir(args.output_dir.resolve())
	exit_key = (args.exit_key or "q")[0]
	exit_key_code = ord(exit_key.lower())
	exit_key_code_upper = ord(exit_key.upper())

	ensure_highgui_support()
	cv2.namedWindow(WINDOW_TITLE, cv2.WINDOW_NORMAL)
	if args.preview_width > 0 and args.preview_height > 0:
		try:
			cv2.resizeWindow(WINDOW_TITLE, args.preview_width, args.preview_height)
		except cv2.error:
			pass

	device = create_device(args.device)
	with dai.Pipeline(device) as pipeline:
		frame_queue, focus_control_queue = create_preview_queue(
			pipeline,
			(args.width, args.height),
			fps=args.fps,
			focus_mode=args.focus_mode,
			focus_position=args.focus_position,
		)

		message = "Press space to capture an image."
		if focus_control_queue is not None:
			message += " Press 'f' to re-trigger autofocus." 
		print(message + f" Press '{exit_key}' to exit.")

		pipeline.start()
		last_frame = None

		try:
			while pipeline.isRunning():
				msg = frame_queue.tryGet()
				if msg is not None:
					frame = msg.getCvFrame()
					if frame is not None:
						last_frame = frame
						display_frame = frame
						if (
							args.preview_width > 0
							and args.preview_height > 0
							and (frame.shape[1], frame.shape[0])
							!= (args.preview_width, args.preview_height)
						):
							display_frame = cv2.resize(
								frame, (args.preview_width, args.preview_height)
							)
						try:
							cv2.imshow(WINDOW_TITLE, display_frame)
						except cv2.error as exc:  # pragma: no cover - depends on GUI libs
							raise RuntimeError(MISSING_HIGHGUI_MSG) from exc

				try:
					key = cv2.waitKey(1)
				except cv2.error as exc:  # pragma: no cover - depends on GUI libs
					raise RuntimeError(MISSING_HIGHGUI_MSG) from exc
				if key == -1:
					continue
				if key == CAPTURE_KEY_CODE:
					if last_frame is None:
						print("No frame available yet; skipping capture.")
						continue
					file_path = format_filename(output_dir)
					cv2.imwrite(str(file_path), last_frame)
					print(f"Saved capture to {file_path}")
				elif key == AF_TRIGGER_KEY_CODE or key == AF_TRIGGER_KEY_CODE_UPPER:
					trigger_autofocus(focus_control_queue, args.focus_mode)
				elif key == exit_key_code or key == exit_key_code_upper:
					break
		finally:
			try:
				cv2.destroyAllWindows()
			except cv2.error:
				pass


if __name__ == "__main__":
	try:
		main()
	except KeyboardInterrupt:
		cv2.destroyAllWindows()
