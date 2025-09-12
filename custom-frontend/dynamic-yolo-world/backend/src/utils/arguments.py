import argparse


def initialize_argparser():
    """Initialize the argument parser for the script."""
    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
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
        help="Path to the media file you aim to run the model on. If not set, the model will run on the camera input.",
        required=False,
        default=None,
        type=str,
    )

    parser.add_argument(
        "-ip",
        "--ip",
        help="IP address to serve the frontend on.",
        required=False,
        type=str,
    )
    parser.add_argument(
        "-p",
        "--port",
        help="Port to serve the frontend on.",
        required=False,
        type=int,
    )
    parser.add_argument(
        "-m",
        "--model",
        help="Name of the model to use: yolo-world, yoloe or yoloe-image",
        required=False,
        default="yolo-world",
        type=str,
        choices=["yolo-world", "yoloe", "yoloe-image"],
    )
    parser.add_argument(
        "--precision",
        help="Model precision for YOLOE models: int8 (default) or fp16. fp16 disables input quantization.",
        required=False,
        default="int8",
        type=str,
        choices=["int8", "fp16"],
    )

    args = parser.parse_args()

    return parser, args
