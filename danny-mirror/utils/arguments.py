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

    args = parser.parse_args()

    return parser, args
