import argparse


def initialize_argparser():
    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.description = (
        "Minimal PoC for DINOv3 patch feature tracking. Select a point and the app "
        "will save its patch features and highlight similar patches across frames."
    )

    parser.add_argument(
        "-m",
        "--model",
        help="HubAI model reference.",
        default="luxonis/dinov3-backbone:convnext-base-352x480",
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
        help="FPS limit for the runtime.",
        required=False,
        default=None,
        type=int,
    )

    parser.add_argument(
        "-media",
        "--media_path",
        help="Path to a media file. If not set, the camera is used.",
        required=False,
        default=None,
        type=str,
    )

    parser.add_argument(
        "--select_x",
        help="Normalized X coordinate (0..1) of the selected point.",
        required=False,
        default=None,
        type=float,
    )

    parser.add_argument(
        "--select_y",
        help="Normalized Y coordinate (0..1) of the selected point.",
        required=False,
        default=None,
        type=float,
    )

    parser.add_argument(
        "--save_after_frames",
        help="Save selected patch features after this many frames.",
        required=False,
        default=15,
        type=int,
    )

    parser.add_argument(
        "--similarity_thresh",
        help="Cosine similarity threshold for highlighting patches.",
        required=False,
        default=0.5,
        type=float,
    )

    parser.add_argument(
        "--save_path",
        help="Optional path to save the selected patch feature as .npy.",
        required=False,
        default=None,
        type=str,
    )

    args = parser.parse_args()
    return parser, args

