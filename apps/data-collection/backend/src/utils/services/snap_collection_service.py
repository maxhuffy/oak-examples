from .base_service import BaseService
import depthai as dai
from ..snaps.conditions_manager import ConditionsManager
from depthai_nodes.node import SnapsProducer


class SnapCollectionService(BaseService):
    """
    Handles frontend updates to snap collection configuration:
    enabling/disabling conditions, adjusting cooldowns, and
    controlling the SnapsProducer runtime state.
    """

    def __init__(
        self,
        visualizer: dai.RemoteConnection,
        runtime: dict,
        cond_manager: ConditionsManager,
        snaps_producer: SnapsProducer,
    ):
        super().__init__(visualizer, "Snap Collection Service")
        self.runtime = runtime
        self.cond_manager = cond_manager
        self.snaps_producer = snaps_producer

    def handle(self, payload: dict | None = None):
        if not isinstance(payload, dict):
            return {"ok": False, "reason": "payload_must_be_dict"}

        base_dt_seconds = 1
        prev_running = bool(self.runtime.get("snapping_running", False))

        for key, conf in payload.items():
            if key not in self.cond_manager.conditions:
                continue
            if not isinstance(conf, dict):
                continue

            if "enabled" in conf:
                self.cond_manager.set_key_enabled(key, bool(conf["enabled"]))

            if "cooldown" in conf or "interval" in conf:
                try:
                    cooldown = float(conf.get("cooldown", conf.get("interval", 0)))
                    self.cond_manager.set_cooldown(key, cooldown)
                except Exception:
                    pass

            if key == "lowConfidence" and "threshold" in conf:
                try:
                    thr = float(conf["threshold"])
                    if thr > 1.0:
                        thr /= 100.0
                    self.runtime["low_conf_thresh"] = max(0.0, min(1.0, thr))
                except Exception:
                    return {"ok": False, "reason": "invalid_low_conf_threshold"}

            if key == "lostMid" and "margin" in conf:
                try:
                    m = float(conf["margin"])
                    self.runtime["lost_mid_margin"] = max(0.0, min(0.49, m))
                except Exception:
                    pass

        any_active = any(
            self.cond_manager.is_key_enabled(k)
            for k in self.cond_manager.conditions.keys()
        )

        self.cond_manager.set_enabled(any_active)

        if any_active and not prev_running:
            self.cond_manager.reset_cooldowns()

        self.snaps_producer.setRunning(any_active)
        self.runtime["snapping_running"] = any_active
        if any_active:
            self.snaps_producer.setTimeInterval(base_dt_seconds)

        return {"ok": True, "active": any_active}
