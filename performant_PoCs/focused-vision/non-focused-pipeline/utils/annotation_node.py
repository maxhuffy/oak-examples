import depthai as dai

from depthai_nodes import ImgDetectionsExtended
from depthai_nodes.utils import AnnotationHelper


class AnnotationNode(dai.node.HostNode):
    def __init__(self) -> None:
        super().__init__()
        self._src_w = None
        self._src_h = None

    def build(self, detections_out: dai.Node.Output) -> "AnnotationNode":
        self.link_args(detections_out)
        return self

    def process(self, msg) -> None:
        dets_msg = msg
        assert isinstance(dets_msg, ImgDetectionsExtended)

        # Source size of the frame the NN saw
        src_w, src_h = dets_msg.transformation.getSize()
        annotations = AnnotationHelper()

        EYE_SCALE = 0.25  # fraction of min(face_w, face_h)

        for det in dets_msg.detections:
            face_w = det.rotated_rect.size.width * src_w
            face_h = det.rotated_rect.size.height * src_h
            side_px = max(6, int(min(face_w, face_h) * EYE_SCALE))
            half_w_norm = (side_px / src_w) / 2.0
            half_h_norm = (side_px / src_h) / 2.0

            left_eye = det.keypoints[0]
            right_eye = det.keypoints[1]

            # Left eye square (normalized corners)
            lx1, ly1 = left_eye.x - half_w_norm, left_eye.y - half_h_norm
            lx2, ly2 = left_eye.x + half_w_norm, left_eye.y + half_h_norm
            annotations.draw_rectangle([lx1, ly1], [lx2, ly2])

            # Right eye square
            rx1, ry1 = right_eye.x - half_w_norm, right_eye.y - half_h_norm
            rx2, ry2 = right_eye.x + half_w_norm, right_eye.y + half_h_norm
            annotations.draw_rectangle([rx1, ry1], [rx2, ry2])

        annotations_msg = annotations.build(
            timestamp=dets_msg.getTimestamp(),
            sequence_num=dets_msg.getSequenceNum(),
        )
        self.out.send(annotations_msg)
