import math
import depthai as dai
from depthai_nodes.utils import AnnotationHelper
from .measure_distance import RegionOfInterest, SpatialDistance


class ROIAnnotator(dai.node.HostNode):
    """Draw the current ROI as an annotation over the incoming disparity image.
    Exposes:
      - roi_input: receives RegionOfInterest updates (pixel coordinates)
      - passthrough: forwards the disparity ImgFrame
      - annotation_output: ImgAnnotations with a rectangle overlay
    """

    def __init__(self) -> None:
        super().__init__()
        self.roi_input = self.createInput()
        self.distance_input = self.createInput()
        self.passthrough = self.createOutput(
            possibleDatatypes=[
                dai.Node.DatatypeHierarchy(dai.DatatypeEnum.ImgFrame, True)
            ]
        )
        self.annotation_output = self.createOutput(
            possibleDatatypes=[
                dai.Node.DatatypeHierarchy(dai.DatatypeEnum.ImgAnnotations, True)
            ]
        )
        # Fallback ROI if none received yet
        self._roi = RegionOfInterest(300, 150, 340, 190)
        self._last_spatial: SpatialDistance | None = None

    def build(self, disparity_frames: dai.Node.Output) -> "ROIAnnotator":
        self.link_args(disparity_frames)
        return self

    def process(self, disparity: dai.ImgFrame) -> None:
        # Update ROI if there are new ones
        rois = self.roi_input.tryGetAll()
        if rois:
            self._roi = rois[-1]

        # Update last spatial measurement if available
        measurements = self.distance_input.tryGetAll()
        if measurements:
            last = measurements[-1]
            if isinstance(last, SpatialDistance):
                self._last_spatial = last

        width = disparity.getWidth()
        height = disparity.getHeight()

        rel_xmin = self._roi.xmin / width
        rel_ymin = self._roi.ymin / height
        rel_xmax = self._roi.xmax / width
        rel_ymax = self._roi.ymax / height

        ann = AnnotationHelper()
        ann.draw_rectangle(
            top_left=(rel_xmin, rel_ymin),
            bottom_right=(rel_xmax, rel_ymax),
            outline_color=(1, 1, 1, 1),
            thickness=2,
        )

        # Draw XYZ text if we have a measurement
        if self._last_spatial is not None:
            x = self._roi.xmax
            y = self._roi.ymin

            sx = self._last_spatial.spatials.x
            sy = self._last_spatial.spatials.y
            sz = self._last_spatial.spatials.z

            # Convert mm -> cm with one decimal place (preserves mm precision)
            text_x = f"X: {sx / 10.0:.1f}cm" if not math.isnan(sx) else "X: --"
            text_y = f"Y: {sy / 10.0:.1f}cm" if not math.isnan(sy) else "Y: --"
            text_z = f"Z: {sz / 10.0:.1f}cm" if not math.isnan(sz) else "Z: --"

            # Use a larger, legible font size and scale line spacing accordingly
            text_size = 28
            text_offset_x = 6
            line_step = int(text_size * 1.2)
            text_offset_y_1 = line_step
            text_offset_y_2 = 2 * line_step
            text_offset_y_3 = 3 * line_step

            ann.draw_text(
                text=text_x,
                position=((x + text_offset_x) / width, (y + text_offset_y_1) / height),
                color=(1, 1, 1, 1),
                size=text_size,
                background_color=(0, 0, 0, 0.6),
            )
            ann.draw_text(
                text=text_y,
                position=((x + text_offset_x) / width, (y + text_offset_y_2) / height),
                color=(1, 1, 1, 1),
                size=text_size,
                background_color=(0, 0, 0, 0.6),
            )
            ann.draw_text(
                text=text_z,
                position=((x + text_offset_x) / width, (y + text_offset_y_3) / height),
                color=(1, 1, 1, 1),
                size=text_size,
                background_color=(0, 0, 0, 0.6),
            )

        annotations = ann.build(disparity.getTimestamp(), disparity.getSequenceNum())
        self.annotation_output.send(annotations)
        self.passthrough.send(disparity)
