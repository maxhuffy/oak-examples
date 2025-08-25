import depthai as dai
from depthai_nodes import ImgDetectionsExtended, ImgDetectionExtended
from depthai_nodes.utils import AnnotationHelper


class AnnotationNode(dai.node.HostNode):
    def __init__(self) -> None:
        super().__init__()
        self._mode = "full"
        self._padding = 0.1
        self.EYE_SCALE = 0.25

    def build(self, gathered_pair_out: dai.Node.Output, padding: float) -> "AnnotationNode":
        self._mode = "full"
        self._padding = padding
        self.link_args(gathered_pair_out)
        return self

    def _draw_eye_squares(self, detections, src_w, src_h, annotation_helper):
        """Common method to draw eye squares"""
        for det in detections:
            rect = det.rotated_rect
            # Calculate eye square size based on face size
            face_px_w = rect.size.width * src_w
            face_px_h = rect.size.height * src_h
            side_px = max(6, int(min(face_px_w, face_px_h) * self.EYE_SCALE))
            half_w_n = (side_px / src_w) / 2.0
            half_h_n = (side_px / src_h) / 2.0

            # Draw squares for left and right eyes (YuNet: 0=left eye, 1=right eye)
            for kp in det.keypoints[:2]:
                annotation_helper.draw_rectangle(
                    [kp.x - half_w_n, kp.y - half_h_n],
                    [kp.x + half_w_n, kp.y + half_h_n]
                )

    def process(self, gather_msg) -> None:
        dets_msg: ImgDetectionsExtended = gather_msg.reference_data
        assert isinstance(dets_msg, ImgDetectionsExtended)
        src_w, src_h = dets_msg.transformation.getSize()
        ann = AnnotationHelper()

        stage2_list = gather_msg.gathered or []

        for i, face_det in enumerate(dets_msg.detections):
            if i >= len(stage2_list):
                continue
                
            crop_msg = stage2_list[i]
            assert isinstance(crop_msg, ImgDetectionsExtended)
            if not crop_msg.detections:
                continue

            # Get face bounding box with padding
            rect = face_det.rotated_rect
            pad = self._padding
            new_w = rect.size.width + pad * 2.0
            new_h = rect.size.height + pad * 2.0
            left = rect.center.x - new_w / 2.0
            top = rect.center.y - new_h / 2.0

            crop_det = crop_msg.detections[0]
            transformed_detections = []
            
            for kp in crop_det.keypoints[:2]:
                # Transform from crop space to full frame space
                # Use original face dimensions, not padded dimensions
                fx = left + kp.x * new_w
                fy = top + kp.y * new_h
                transformed_detections.append(type('Keypoint', (), {'x': fx, 'y': fy})())

            # Create a temporary detection object with transformed coordinates
            # Use original face size for eye square calculation, not padded size
            temp_det = type('Detection', (), {
                'rotated_rect': type('Rect', (), {
                    'size': type('Size', (), {'width': rect.size.width, 'height': rect.size.height})()
                })(),
                'keypoints': transformed_detections
            })()

            self._draw_eye_squares([temp_det], src_w, src_h, ann)

        out = ann.build(timestamp=dets_msg.getTimestamp(), sequence_num=dets_msg.getSequenceNum())
        self.out.send(out)