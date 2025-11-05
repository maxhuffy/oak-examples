import time
from typing import Optional
from depthai_nodes import ImgDetectionsExtended, ImgDetectionExtended


class KeypointsPrinter:
    """Throttled console printer for detections and their keypoints.

    Usage:
        printer = KeypointsPrinter(interval_seconds=0.5)
        printer.maybe_print(det_msg)
    """

    def __init__(self, interval_seconds: float = 0.5) -> None:
        self.interval_seconds = interval_seconds
        self._last_print_time: float = 0.0

    def maybe_print(self, det_msg: object) -> None:
        now = time.time()
        if now - self._last_print_time < self.interval_seconds:
            return

        if not isinstance(det_msg, ImgDetectionsExtended):
            return

        for idx, det in enumerate(det_msg.detections):
            # Rotated rect summary
            try:
                rr = det.rotated_rect
                print(
                    f"[Det {idx}] center=({rr.center.x:.3f},{rr.center.y:.3f}) "
                    f"size=({rr.size.width:.3f},{rr.size.height:.3f}) ang={rr.angle:.1f}"
                )
            except Exception:
                pass

            # Keypoints (normalized coordinates)
            try:
                if hasattr(det, "keypoints") and det.keypoints:
                    kps_norm = ", ".join(
                        [f"({kp.x:.3f},{kp.y:.3f})" for kp in det.keypoints]
                    )
                    print(f"         keypoints_norm: [{kps_norm}]")
            except Exception:
                pass

        print("#" * 10)
        self._last_print_time = now
