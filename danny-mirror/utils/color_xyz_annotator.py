import math
import depthai as dai
from depthai_nodes.utils import AnnotationHelper
from .measure_distance import SpatialDistance, RegionOfInterest


class ColorXYZAnnotator(dai.node.HostNode):
    """Overlay XYZ text (from SpatialDistance) onto a color frame.

    Inputs:
      - video_input: ImgFrame (color)
      - distance_input: SpatialDistance (with spatials in mm)
      - roi_input (optional): RegionOfInterest to position text near the ROI

    Outputs:
      - passthrough: ImgFrame (for the video)
      - annotation_output: ImgAnnotations (text overlay)
    """

    def __init__(self) -> None:
        super().__init__()
        self.video_input = self.createInput()
        self.distance_input = self.createInput()
        self.roi_input = self.createInput()

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
        self._last_spatial = None
        self._roi = None
        # Config
        self._show_roi = False
        # If set, ROI coords are in this pixel space (w,h) and will be scaled to the output frame
        self._roi_src_size = None

    def build(
        self,
        video_frames: dai.Node.Output,
        show_roi: bool = False,
        roi_src_size = None,
    ) -> "ColorXYZAnnotator":
        self._show_roi = show_roi
        self._roi_src_size = roi_src_size
        self.link_args(video_frames)
        return self

    def process(self, video_frame: dai.ImgFrame) -> None:
        # Update ROI and spatial if any
        rois = self.roi_input.tryGetAll()
        if rois:
            self._roi = rois[-1]

        measurements = self.distance_input.tryGetAll()
        if measurements:
            last = measurements[-1]
            if isinstance(last, SpatialDistance):
                self._last_spatial = last

        width = video_frame.getWidth()
        height = video_frame.getHeight()

        ann = AnnotationHelper()

        # Draw ROI rectangle if requested
        if self._show_roi and self._roi is not None:
            # Map ROI to output frame: if roi_src_size provided, scale to current frame
            if self._roi_src_size is not None:
                src_w, src_h = self._roi_src_size
            else:
                src_w, src_h = width, height

            rel_xmin = self._roi.xmin / float(src_w)
            rel_ymin = self._roi.ymin / float(src_h)
            rel_xmax = self._roi.xmax / float(src_w)
            rel_ymax = self._roi.ymax / float(src_h)

            ann.draw_rectangle(
                top_left=(rel_xmin, rel_ymin),
                bottom_right=(rel_xmax, rel_ymax),
                outline_color=(1, 1, 1, 1),
                thickness=2,
            )

        # Draw XYZ text
        if self._last_spatial is not None:
            # Position text near ROI if available; otherwise top-left corner
            if self._roi is not None:
                # If ROI is in a different pixel space than the frame, scale the anchor.
                if self._roi_src_size is not None:
                    sx = width / float(self._roi_src_size[0])
                    sy = height / float(self._roi_src_size[1])
                else:
                    sx = sy = 1.0
                x_px = int(self._roi.xmax * sx) + 6
                y_px = int(self._roi.ymin * sy) + 6
            else:
                x_px = int(0.02 * width)
                y_px = int(0.06 * height)

            sx = self._last_spatial.spatials.x
            sy = self._last_spatial.spatials.y
            sz = self._last_spatial.spatials.z

            # mm -> cm with one decimal place
            text_x = f"X: {sx / 10.0:.1f}cm" if not math.isnan(sx) else "X: --"
            text_y = f"Y: {sy / 10.0:.1f}cm" if not math.isnan(sy) else "Y: --"
            text_z = f"Z: {sz / 10.0:.1f}cm" if not math.isnan(sz) else "Z: --"

            text_size = 28
            line_step = int(text_size * 1.2)

            ann.draw_text(
                text=text_x,
                position=(x_px / width, y_px / height),
                color=(1, 1, 1, 1),
                size=text_size,
                background_color=(0, 0, 0, 0.6),
            )
            ann.draw_text(
                text=text_y,
                position=(x_px / width, (y_px + line_step) / height),
                color=(1, 1, 1, 1),
                size=text_size,
                background_color=(0, 0, 0, 0.6),
            )
            ann.draw_text(
                text=text_z,
                position=(x_px / width, (y_px + 2 * line_step) / height),
                color=(1, 1, 1, 1),
                size=text_size,
                background_color=(0, 0, 0, 0.6),
            )

        annotations = ann.build(video_frame.getTimestamp(), video_frame.getSequenceNum())
        self.annotation_output.send(annotations)
        self.passthrough.send(video_frame)
