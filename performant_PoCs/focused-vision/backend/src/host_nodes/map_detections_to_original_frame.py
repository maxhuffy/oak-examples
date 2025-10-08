
import depthai as dai

from depthai_nodes import ImgDetectionExtended, ImgDetectionsExtended
from depthai_nodes.message import GatheredData


class MapDetectionsToOriginalFrame(dai.node.HostNode):
    """"""
    def __init__(
            self,
            face_detection_nn_width: int,
            face_detection_nn_height: int,
    ) -> None:
        super().__init__()
        self._face_detection_nn_width = face_detection_nn_width
        self._face_detection_nn_height = face_detection_nn_height

    def build(self, nn_output: dai.Node.Output) -> "MapDetectionsToOriginalFrame":
        self.link_args(nn_output)
        self.sendProcessingToPipeline(False)
        return self

    def process(self, matched_detections: dai.Buffer) -> None:
        assert (matched_detections, GatheredData)
        matched_detections: GatheredData
        people_detections: dai.ImgDetections = matched_detections.reference_data
        face_detections: list[dai.ImgDetections] = matched_detections.gathered
        face_detections_output = ImgDetectionsExtended()
        transformed_detections = []
        for people_detection, face_dets in zip(people_detections.detections, face_detections):
            face_dets: ImgDetectionsExtended
            best_face_detection = None
            best_confidence = -0.1
            for face_detection in face_dets.detections:
                # print(f"Confidence: {face_detection.confidence}")
                if face_detection.confidence > best_confidence:
                    best_face_detection = face_detection
                    best_confidence = face_detection.confidence
            if best_face_detection is None:
                continue
            people_xmin = people_detection.xmin
            people_xmax = people_detection.xmax
            people_ymin = people_detection.ymin
            people_ymax = people_detection.ymax
            people_det_width = min(1., people_xmax) - max(0., people_xmin)
            people_det_height = min(1., people_ymax) - max(0., people_ymin)
            people_det_width_abs = round(people_det_width * 640)
            people_det_height_abs = round(people_det_height * 640)
            """
            now we need: 
              _face_detection_nn_width / _face_detection_nn_height = people_det_width_abs / people_det_height_abs * C
              C = (_face_detection_nn_width * people_det_height_abs) / (_face_detection_nn_height * people_det_width_abs)
            If C < 1 -> height had to be padded
            If C > 1 -> width had to be padded
            Calculate the padding in pixels to be able to disregard the padded area as it is not present in the original frame to which we
            convert the coordinates of the face detection.
            """
            C = (self._face_detection_nn_width * people_det_height_abs) / (self._face_detection_nn_height * people_det_width_abs)
            height_padding = 0
            width_padding = 0
            height_padding_pixels = 0
            width_padding_pixels = 0
            if C < 1:
                height_padding_pixels = (self._face_detection_nn_height - (self._face_detection_nn_width / people_det_width_abs * people_det_height_abs)) // 2
                height_padding = height_padding_pixels / self._face_detection_nn_height
            elif C > 1:
                width_padding_pixels = (self._face_detection_nn_width - (self._face_detection_nn_height / people_det_height_abs * people_det_width_abs)) // 2
                width_padding = width_padding_pixels / self._face_detection_nn_width
            # print(f"Face NN width: {self._face_detection_nn_width}, height: {self._face_detection_nn_height}")
            # print(f"xmin={people_detection.xmin},ymin={people_detection.ymin},xmax={people_detection.xmax},ymax={people_detection.ymax}, width={people_det_width_abs}, height={people_det_height_abs} {C=}")
            # print(f"Padding: {width_padding=} {height_padding=} {height_padding_pixels=} {width_padding_pixels=}")
            x_center = people_xmin + (best_face_detection.rotated_rect.center.x - width_padding) * people_det_width
            y_center = people_ymin + (best_face_detection.rotated_rect.center.y - height_padding) * people_det_height
            width = (best_face_detection.rotated_rect.size.width + (2 * width_padding)) * people_det_width
            height = (best_face_detection.rotated_rect.size.height + (2 * height_padding)) * people_det_height
            face_det_extended = ImgDetectionExtended()
            face_det_extended.label = best_face_detection.label
            face_det_extended.label_name = best_face_detection.label_name
            face_det_extended.confidence = best_face_detection.confidence
            face_det_extended.rotated_rect = (
                x_center,
                y_center,
                width,
                height,
                0,  # dai.ImgDetections has no angle info
            )
            # print(f"Transformed: {x_center=} {y_center=} {width=} {height=}")
            transformed_detections.append(face_det_extended)
        face_detections_output.detections = transformed_detections
        face_detections_output.setTimestamp(people_detections.getTimestamp())
        face_detections_output.setSequenceNum(people_detections.getSequenceNum())
        face_detections_output.setTransformation(people_detections.getTransformation())
        self.out.send(face_detections_output)
