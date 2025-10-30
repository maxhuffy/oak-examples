from .base_service import BaseService


class GetConfigService(BaseService):
    """
    Returns the current configuration state to the frontend.

    Reports runtime parameters such as confidence threshold,
    active snap conditions, cooldown timers, and margin/threshold
    values for each configured condition.
    """

    def __init__(self, visualizer, current_classes, runtime, cond_manager):
        super().__init__(visualizer, "Get Config Service")
        self.current_classes = current_classes
        self.runtime = runtime
        self.cond_manager = cond_manager

    def handle(self, payload=None):
        def s_to_min(seconds: float) -> float:
            try:
                return float(seconds) / 60.0 if float(seconds) > 0 else 0.0
            except Exception:
                return 0.0

        config = {
            "classes": self.current_classes.copy(),
            "confidence_threshold": self.runtime["conf_threshold"],
            "snapping": {
                "running": self.runtime.get("snapping_running", False),
                "timed": {
                    "enabled": self.cond_manager.is_key_enabled("timed"),
                    "interval": s_to_min(self.cond_manager.get_cooldown("timed")),
                },
                "noDetections": {
                    "enabled": self.cond_manager.is_key_enabled("no_detections"),
                    "cooldown": s_to_min(
                        self.cond_manager.get_cooldown("no_detections")
                    ),
                },
                "lowConfidence": {
                    "enabled": self.cond_manager.is_key_enabled("low_conf"),
                    "threshold": self.runtime.get("low_conf_thresh", 0.30),
                    "cooldown": s_to_min(self.cond_manager.get_cooldown("low_conf")),
                },
                "lostMid": {
                    "enabled": self.cond_manager.is_key_enabled("lost_mid"),
                    "cooldown": s_to_min(self.cond_manager.get_cooldown("lost_mid")),
                    "margin": self.runtime.get("lost_mid_margin", 0.20),
                },
            },
        }
        return config
