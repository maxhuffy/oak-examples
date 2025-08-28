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
        self._target_w = 0
        self._target_h = 0

    def build(self, detections_input: dai.Node.Output,
              target_size: Tuple[int, int]) -> "ProcessDetections":
        self._target_w, self._target_h = map(int, target_size)
        self.link_args(detections_input)
        return self

    def process(self, img_detections: dai.Buffer) -> None:
        assert isinstance(img_detections, ImgDetectionsExtended)
        dets = img_detections.detections
        ts = img_detections.getTimestamp()

        num_cfgs = len(dets)

        # COUNT Buffer: seq = gid; data length == number of crops
        count_msg = dai.Buffer()
        count_msg.setData(b"\x00" * num_cfgs)
        count_msg.setTimestamp(ts)
        self.num_configs_output.send(count_msg)

        # One config per detection
        for i, det in enumerate(dets):

            cfg = dai.ImageManipConfig()
            cfg.addCropRotatedRect(det.rotated_rect, normalizedCoords=True)
            cfg.setOutputSize(self._target_w, self._target_h, dai.ImageManipConfig.ResizeMode.STRETCH)

            cfg.setTimestamp(ts)
            self.config_output.send(cfg)
