import depthai as dai
from depthai_nodes import ImgDetectionsExtended
from depthai_nodes.utils import AnnotationHelper


class FaceAnnotationNode(dai.node.HostNode):
    def __init__(self) -> None:
        super().__init__()

    def build(self, gathered_pair_out: dai.Node.Output) -> "FaceAnnotationNode":
        self.link_args(gathered_pair_out)
        return self

    def process(self, gather_msg) -> None:
        dets_msg: ImgDetectionsExtended = gather_msg.reference_data
        ann = AnnotationHelper()

        if not dets_msg or not dets_msg.detections:
            out = ann.build(timestamp=dets_msg.getTimestamp(), sequence_num=dets_msg.getSequenceNum())
            self.out.send(out)
            return

        for detection in dets_msg.detections:
            face_bbox = detection.rotated_rect.getPoints()

            ann.draw_rectangle(
                [face_bbox[0].x, face_bbox[0].y], [face_bbox[2].x, face_bbox[2].y]
            )

        out = ann.build(timestamp=dets_msg.getTimestamp(), sequence_num=dets_msg.getSequenceNum())
        self.out.send(out)
