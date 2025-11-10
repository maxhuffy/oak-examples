import argparse


def initialize_argparser():
    """Initialize the argument parser for the script."""
    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.description = (
        "Run face detection (e.g., yunet) on the RGB stream and compute spatial coordinates of the detected face using stereo depth on host."
    )

    parser.add_argument(
        "-m",
        "--model",
        help="HubAI model reference.",
        default="luxonis/yunet:640x480",
        type=str,
    )

    parser.add_argument(
        "-d",
        "--device",
        help="Optional name, DeviceID or IP of the camera to connect to.",
        required=False,
        default=None,
        type=str,
    )

    parser.add_argument(
        "-fps",
        "--fps_limit",
        help="FPS limit for the model runtime.",
        required=False,
        default=None,
        type=int,
    )

    parser.add_argument(
        "-media",
        "--media_path",
        help="Path to a media file to run the model on (otherwise uses the camera).",
        required=False,
        default=None,
        type=str,
    )

    parser.add_argument(
        "-api",
        "--api_key",
        help="HubAI API key to access private model. Alternatively use DEPTHAI_HUB_API_KEY env var.",
        required=False,
        default="",
        type=str,
    )

    parser.add_argument(
        "-overlay",
        "--overlay_mode",
        help="Overlay model output on the input image when output is an array (depth maps, segmentation).",
        required=False,
        action="store_true",
    )

    parser.add_argument(
        "--eye_roi",
        help="Use an ROI computed from the two eye keypoints instead of the full face rectangle.",
        required=False,
        action="store_true",
    )

    parser.add_argument(
        "--show_roi",
        help="Overlay the ROI rectangle on the color view (works alone or together with --eye_roi).",
        required=False,
        action="store_true",
    )

    # Display/monitor selection options
    parser.add_argument(
        "--display-index",
        help="0-based monitor index from screeninfo.get_monitors().",
        required=False,
        default=None,
        type=int,
    )

    parser.add_argument(
        "--monitor-name",
        help="Substring to match monitor name/device when choosing display.",
        required=False,
        default=None,
        type=str,
    )

    parser.add_argument(
        "--prefer-portrait",
        help="Prefer portrait monitors (height > width) when choosing display.",
        required=False,
        action="store_true",
    )

    parser.add_argument(
        "--windowed",
        help="Do not force fullscreen; keep windowed mode.",
        required=False,
        action="store_true",
    )

    parser.add_argument(
        "--distance-scale-weight",
        help="Weight of distance-based scaling (0 disables, 1 full).",
        required=False,
        default=0.3,
        type=float,
    )

    parser.add_argument(
        "--parallax-weight-x",
        help="Weight for horizontal parallax based on eye X (mm→px).",
        required=False,
        default=1.0,
        type=float,
    )

    parser.add_argument(
        "--parallax-weight-y",
        help="Weight for vertical parallax based on eye Y (mm→px).",
        required=False,
        default=1.0,
        type=float,
    )

    parser.add_argument(
        "--parallax-distance-scale",
        help="Exponent for distance-based parallax scaling (1.0 increases parallax when closer).",
        required=False,
        default=1.0,
        type=float,
    )

    args = parser.parse_args()

    return parser, args
