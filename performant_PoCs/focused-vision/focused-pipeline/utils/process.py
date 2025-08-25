import depthai as dai
from depthai_nodes import ImgDetectionsExtended
from typing import Tuple


class ProcessDetections(dai.node.HostNode):
    """
    For each detector frame:
      - emits a COUNT Buffer with seq = gid (= base_seq * GROUP_STRIDE)
      - emits N crop configs, each with seq = gid + i (i=0..N-1)
      - reusePreviousImage = False (Script sends a matching frame for each config)
    """

    def __init__(self):
        super().__init__()
        self.detections_input = self.createInput()
        self.config_output = self.createOutput()
        self.num_configs_output = self.createOutput()
        self.padding = 0.0
        self._target_w = 0
        self._target_h = 0

    def build(self, detections_input: dai.Node.Output, padding: float,
              target_size: Tuple[int, int]) -> "ProcessDetections":
        self.padding = float(padding or 0.0)
        self._target_w, self._target_h = map(int, target_size)
        self.link_args(detections_input)
        return self

    def process(self, img_detections: dai.Buffer) -> None:
        assert isinstance(img_detections, ImgDetectionsExtended)
        dets = img_detections.detections
        base_seq = int(img_detections.getSequenceNum())
        ts = img_detections.getTimestamp()

        num_cfgs = len(dets)

        # COUNT Buffer: seq = gid; data length == number of crops
        count_msg = dai.Buffer()
        count_msg.setData(b"\x00" * num_cfgs)
        count_msg.setTimestamp(ts)
        self.num_configs_output.send(count_msg)

        # One config per detection
        for i, det in enumerate(dets):
            rect = det.rotated_rect

            new_rect = dai.RotatedRect()
            new_rect.center.x = rect.center.x
            new_rect.center.y = rect.center.y
            new_rect.size.width = rect.size.width + 2.0 * self.padding
            new_rect.size.height = rect.size.height + 2.0 * self.padding
            new_rect.angle = 0

            cfg = dai.ImageManipConfig()
            cfg.addCropRotatedRect(new_rect, normalizedCoords=True)
            cfg.setOutputSize(self._target_w, self._target_h, dai.ImageManipConfig.ResizeMode.STRETCH)

            cfg.setTimestamp(ts)
            self.config_output.send(cfg)
