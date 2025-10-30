from .base_condition import ConditionBase
from ..tracklets import (
    _track_id,
    _status_is_lost,
    _status_is_tracked,
    _roi_center_area_norm,
    _label_idx_name,
)
from collections import defaultdict


class LostMidCondition(ConditionBase):
    """
    Triggers when a tracked object becomes lost while still
    positioned near the center of the frame.

    Detects transition from 'tracked' to 'lost' state within a
    user-defined margin, and records affected tracklets for reporting.
    """

    def __init__(self, key="lostMid", name="Lost in Middle", default_margin=0.20):
        super().__init__(key, name)
        self._last_fired = None
        self.prev_tracked = defaultdict(bool)
        self.default_margin = default_margin

    def should_trigger(self, cond_manager, tracklets=None, runtime=None, **_):
        if not cond_manager.is_key_enabled(self.key) or tracklets is None:
            return False

        margin = float(runtime.get("lost_mid_margin", self.default_margin))
        margin = max(0.0, min(0.49, margin))
        fired = False
        fired_tracklets = []

        tks = getattr(tracklets, "tracklets", None) or []
        for t in tks:
            tid = _track_id(t)
            if tid is None:
                continue

            was_tracked = self.prev_tracked[tid]
            is_lost = _status_is_lost(t)
            is_tracked = _status_is_tracked(t)

            if is_lost and was_tracked:
                rc = _roi_center_area_norm(t)
                if rc is not None:
                    cx, cy, area = rc
                    if margin <= cx <= 1.0 - margin and margin <= cy <= 1.0 - margin:
                        if cond_manager.hit(self.key):
                            fired = True
                            fired_tracklets.append((tid, t, area, margin))

            self.prev_tracked[tid] = is_tracked

        self._last_fired = fired_tracklets
        return fired

    def handle(self, tracklets=None, class_names=None, model=None, **_):
        results = []
        for tid, t, area, margin in getattr(self, "_last_fired", []):
            lbl_idx, lbl_name = _label_idx_name(t, class_names)
            results.append(
                {
                    "reason": "lost_in_middle",
                    "model": str(model),
                    "track_id": str(int(tid)),
                    "label_idx": str(int(lbl_idx)),
                    "label": str(lbl_name),
                    "area_frac": f"{area:.6f}",
                    "margin": f"{margin:.3f}",
                }
            )

        return results[0] if len(results) == 1 else results
