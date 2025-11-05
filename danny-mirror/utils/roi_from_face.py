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

    def build(
        self,
        disparity_frames: dai.Node.Output,
        parser_output: dai.Node.Output,
    ) -> "ROIFromFace":
        self.link_args(disparity_frames, parser_output)
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
        rr = best_det.rotated_rect

        # RotatedRect is expressed in normalized coordinates [0,1] for center and size
        xmin_n = rr.center.x - rr.size.width / 2.0
        xmax_n = rr.center.x + rr.size.width / 2.0
        ymin_n = rr.center.y - rr.size.height / 2.0
        ymax_n = rr.center.y + rr.size.height / 2.0

        # Clamp to [0,1]
        xmin_n = max(0.0, xmin_n)
        ymin_n = max(0.0, ymin_n)
        xmax_n = min(1.0, xmax_n)
        ymax_n = min(1.0, ymax_n)

        # Convert to pixel coordinates in disparity/depth resolution
        xmin = int(xmin_n * width)
        ymin = int(ymin_n * height)
        xmax = int(xmax_n * width)
        ymax = int(ymax_n * height)

        roi = RegionOfInterest(xmin, ymin, xmax, ymax)
        self.output_roi.send(roi)
