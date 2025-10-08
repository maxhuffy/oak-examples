
import depthai as dai
import time


class DecreaseFps(dai.node.HostNode):
    """"""

    def __init__(self, send_every_x_seconds: float = 1.) -> None:
        super().__init__()
        self._last_sent = time.monotonic()
        self._send_every_x_seconds = send_every_x_seconds

    def build(self, node_out: dai.Node.Output) -> "DecreaseFps":
        self.link_args(node_out)
        self.sendProcessingToPipeline(False)
        return self

    def process(self, _any: dai.Buffer) -> None:
        now = time.monotonic()
        if now - self._last_sent >= self._send_every_x_seconds:
            self._last_sent = now
            self.out.send(_any)
