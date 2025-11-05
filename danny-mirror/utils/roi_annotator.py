import depthai as dai
from depthai_nodes.utils import AnnotationHelper
from .measure_distance import RegionOfInterest


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

    def build(self, disparity_frames: dai.Node.Output) -> "ROIAnnotator":
        self.link_args(disparity_frames)
        return self

    def process(self, disparity: dai.ImgFrame) -> None:
        # Update ROI if there are new ones
        rois = self.roi_input.tryGetAll()
        if rois:
            self._roi = rois[-1]

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

        annotations = ann.build(disparity.getTimestamp(), disparity.getSequenceNum())
        self.annotation_output.send(annotations)
        self.passthrough.send(disparity)
