from typing import List
import depthai as dai

from depthai_nodes import ImgDetectionsExtended
from depthai_nodes.utils import AnnotationHelper


class AnnotationNode(dai.node.HostNode):
    def __init__(self) -> None:
        super().__init__()
        self._mode = "focused"
        self._src_w = None
        self._src_h = None

    def build(self, gather_data_msg) -> "AnnotationNode":
        """Focused mode : original build form gaze-estimation"""
        self._mode = "focused"
        self.link_args(gather_data_msg)
        return self

    def build_non_focused(self, detections_out: dai.Node.Output) -> "AnnotationNode":
        """Non-focused mode : link directly to YuNet detections"""
        self._mode = "non-focused"
        self.link_args(detections_out)
        return self

    def process(self, msg) -> None:
        if self._mode == "focused":
            detections_msg = msg.reference_data
            assert isinstance(detections_msg, ImgDetectionsExtended)
            src_w, src_h = detections_msg.transformation.getSize()

            gaze_msg_list: List[dai.NNData] = msg.gathered
            assert isinstance(gaze_msg_list, list)
            assert all(isinstance(rec_msg, dai.NNData) for rec_msg in gaze_msg_list)
            assert len(gaze_msg_list) == len(detections_msg.detections)

            annotations = AnnotationHelper()

            for detection, gaze in zip(detections_msg.detections, gaze_msg_list):
                face_bbox = detection.rotated_rect.getPoints()
                keypoints = detection.keypoints

                # Draw bbox
                annotations.draw_rectangle(
                    [face_bbox[0].x, face_bbox[0].y], [face_bbox[2].x, face_bbox[2].y]
                )

                # Draw gaze
                gaze_tensor = gaze.getFirstTensor(dequantize=True)
                gaze_tensor = gaze_tensor.flatten()

                left_eye = keypoints[0]
                annotations.draw_line(
                    [left_eye.x, left_eye.y],
                    self._get_end_point(left_eye, gaze_tensor, src_w, src_h),
                )

                right_eye = keypoints[1]
                annotations.draw_line(
                    [right_eye.x, right_eye.y],
                    self._get_end_point(right_eye, gaze_tensor, src_w, src_h),
                )

            annotations_msg = annotations.build(
                timestamp=detections_msg.getTimestamp(),
                sequence_num=detections_msg.getSequenceNum(),
            )
            self.out.send(annotations_msg)

        else:
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

    def _get_end_point(
        self, start_point: dai.Point2f, vector: list, src_w: int, src_h: int
    ) -> dai.PointsAnnotation:
        gaze_vector = (vector * 640)[:2]
        gaze_vector_x = gaze_vector[0] / src_w
        gaze_vector_y = gaze_vector[1] / src_h
        end_point = [
            start_point.x + gaze_vector_x.item(),
            start_point.y - gaze_vector_y.item(),
        ]
        return end_point

