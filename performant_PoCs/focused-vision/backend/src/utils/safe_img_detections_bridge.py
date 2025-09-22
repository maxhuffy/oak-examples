import depthai as dai
from depthai_nodes.node import ImgDetectionsBridge


class SafeImgDetectionsBridge(ImgDetectionsBridge):
    """
    Type-aware ImgDetectionsBridge: only converts when input is ImgDetections.
    Otherwise, forwards the message as-is.
    """
    def process(self, msg: dai.Buffer) -> None:
        if isinstance(msg, dai.ImgDetections):
            super().process(msg)
            return
        else:
            self.out.send(msg)
