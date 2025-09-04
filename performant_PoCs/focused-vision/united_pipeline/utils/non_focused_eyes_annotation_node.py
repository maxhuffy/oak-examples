import depthai as dai
from depthai_nodes.utils import AnnotationHelper


class NonFocusedEyesAnnotationNode(dai.node.HostNode):
    def __init__(self) -> None:
        super().__init__()

    def build(self, detections_out: dai.Node.Output) -> "NonFocusedEyesAnnotationNode":
        self.link_args(detections_out)
        return self

    def process(self, msg) -> None:
        if not hasattr(msg, "detections"):
            return

        ann = AnnotationHelper()
        for d in msg.detections:
            ann.draw_rectangle([d.xmin, d.ymin], [d.xmax, d.ymax])

        self.out.send(ann.build(
            timestamp=msg.getTimestamp(),
            sequence_num=msg.getSequenceNum()
        ))
