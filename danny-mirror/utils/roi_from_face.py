import math
import depthai as dai
from depthai_nodes import ImgDetectionsExtended, ImgDetectionExtended
from .measure_distance import RegionOfInterest


class ROIFromFace(dai.node.HostNode):
    """
    Takes NN detections and the disparity frame to compute a pixel ROI matching the face
    and sends it to MeasureDistance via output_roi.
    """

    def __init__(self) -> None:
        super().__init__()
        self.output_roi = self.createOutput()
        self._use_eye_roi = False

    def build(
        self,
        disparity_frames: dai.Node.Output,
        parser_output: dai.Node.Output,
        use_eye_roi: bool = False,
    ) -> "ROIFromFace":
        self.link_args(disparity_frames, parser_output)
        self._use_eye_roi = use_eye_roi
        return self

    def process(self, disparity: dai.ImgFrame, detections: dai.Buffer) -> None:
        assert isinstance(detections, ImgDetectionsExtended)
        width = disparity.getWidth()
        height = disparity.getHeight()

        if len(detections.detections) == 0:
            return

        # Pick the first detection (yunet is face-only). If multiple, pick the largest by area.
        def rect_area(det: ImgDetectionExtended) -> float:
            return det.rotated_rect.size.width * det.rotated_rect.size.height

        best_det = max(detections.detections, key=rect_area)

        # Default to rotated-rect AABB in pixel space
        def rect_from_rotated_rect():
            rr = best_det.rotated_rect
            xmin_n = rr.center.x - rr.size.width / 2.0
            xmax_n = rr.center.x + rr.size.width / 2.0
            ymin_n = rr.center.y - rr.size.height / 2.0
            ymax_n = rr.center.y + rr.size.height / 2.0
            # Clamp to [0,1]
            xmin_n_cl = max(0.0, xmin_n)
            ymin_n_cl = max(0.0, ymin_n)
            xmax_n_cl = min(1.0, xmax_n)
            ymax_n_cl = min(1.0, ymax_n)
            # Convert to pixels
            return (
                int(xmin_n_cl * width),
                int(ymin_n_cl * height),
                int(xmax_n_cl * width),
                int(ymax_n_cl * height),
            )

        def rect_from_eye_keypoints():
            try:
                kps = getattr(best_det, "keypoints", None)
                if not kps or len(kps) < 2:
                    return None
                # Pick two top-most keypoints by y (normalized), then order by x for left/right
                top_two = sorted(kps, key=lambda kp: kp.y)[:2]
                left, right = sorted(top_two, key=lambda kp: kp.x)
                # Convert to pixels
                lx, ly = int(left.x * width), int(left.y * height)
                rx, ry = int(right.x * width), int(right.y * height)
                # Border of at least 10 pixels around both points
                border = 10
                xmin = max(0, min(lx, rx) - border)
                xmax = min(width, max(lx, rx) + border)
                ymin = max(0, min(ly, ry) - border)
                ymax = min(height, max(ly, ry) + border)
                # Ensure non-empty rectangle (at least 1px)
                if xmax <= xmin:
                    xmax = min(width, xmin + 1)
                if ymax <= ymin:
                    ymax = min(height, ymin + 1)
                return (xmin, ymin, xmax, ymax)
            except Exception:
                return None

        if self._use_eye_roi:
            rect = rect_from_eye_keypoints()
            if rect is None:
                rect = rect_from_rotated_rect()
        else:
            rect = rect_from_rotated_rect()

        xmin, ymin, xmax, ymax = rect

        roi = RegionOfInterest(xmin, ymin, xmax, ymax)
        self.output_roi.send(roi)
