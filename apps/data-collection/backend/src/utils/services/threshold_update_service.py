from .base_service import BaseService


class ThresholdUpdateService(BaseService):
    """
    Updates and synchronizes the neural network confidence threshold.

    Receives a new threshold value from the frontend, clamps it to
    a valid range [0.0, 1.0], and updates both the runtime state and
    the active neural network parser configuration.
    """

    def __init__(self, visualizer, nn, runtime):
        super().__init__(visualizer, "Threshold Update Service")
        self.nn = nn
        self.runtime = runtime

    def handle(self, payload: float | None = None):
        self.runtime["conf_threshold"] = max(0.0, min(1.0, payload))
        self.nn.getParser(0).setConfidenceThreshold(self.runtime["conf_threshold"])
        return {"ok": True, "threshold": self.runtime["conf_threshold"]}
