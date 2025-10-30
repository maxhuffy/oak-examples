from .base_condition import ConditionBase


class LowConfidenceCondition(ConditionBase):
    """
    Triggers a snap when detections fall below a defined confidence threshold.

    Compares detection confidences against the current runtime threshold
    and activates when any detection is lower than that value.
    """

    def __init__(self, key="lowConfidence", name="Low Confidence"):
        super().__init__(key, name)

    def should_trigger(self, cond_manager, det_data=None, runtime=None, **_) -> bool:
        if det_data is None or not cond_manager.is_key_enabled(self.key):
            return False

        thr = runtime.get("low_conf_thresh")
        if thr is None:
            return False
        dets = getattr(det_data, "detections", [])
        has_low = any(float(getattr(d, "confidence", 1.0)) < float(thr) for d in dets)
        return has_low and cond_manager.hit(self.key)

    def handle(self, det_data=None, runtime=None, **_):
        dets = getattr(det_data, "detections", [])
        min_conf = min(
            (float(getattr(d, "confidence", 1.0)) for d in dets), default=1.0
        )
        thr = (runtime or {}).get("low_conf_thresh", 0.3)
        return {
            "reason": "low_confidence",
            "min_conf": f"{min_conf:.3f}",
            "threshold": f"{float(thr):.3f}",
        }
