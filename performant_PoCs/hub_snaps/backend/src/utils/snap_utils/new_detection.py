from __future__ import annotations
from typing import List
import time
import depthai as dai
from depthai_nodes.node import SnapsProducer

from .helpers import (
    _frame_ts_seconds,
    _status_is_tracked,
    _track_id,
    _label_idx_name,
    _send_snap,
)


class _NewDetectionsState:
    def __init__(self) -> None:
        self.seen_ids: set[int] = set()
        self.last_label_ts: dict[str, float] = {}


_NEWDET_STATE = _NewDetectionsState()


def reset_new_detections_state() -> None:
    _NEWDET_STATE.seen_ids.clear()
    _NEWDET_STATE.last_label_ts.clear()
    print("[NewDet] State reset")


def tracklet_new_detection_process(
    producer: SnapsProducer,
    frame: dai.ImgFrame,
    tracklets_msg: dai.Tracklets,
    det_data: dai.ImgDetections,
    class_names: List[str],
    model: str,
    min_label_cooldown_s: float = 2.0,
):
    """
    Send a snap when a tracklet becomes TRACKED for the first time (ID not seen).
    One snap per unique track ID; per-label cooldown to avoid bursts.
    """
    tks = getattr(tracklets_msg, "tracklets", []) or []
    now_s = _frame_ts_seconds(frame) or time.time()

    for t in tks:
        if not _status_is_tracked(t):
            continue

        tid = _track_id(t)
        if tid is None or tid in _NEWDET_STATE.seen_ids:
            continue

        label_idx, label_str = _label_idx_name(t, class_names)
        last_ts = _NEWDET_STATE.last_label_ts.get(label_str, -1.0)
        if label_str and last_ts >= 0 and (now_s - last_ts) < float(min_label_cooldown_s):
            continue

        extras = {
            "model": model,
            "reason": "new_detection",
            "track_id": str(int(tid)),
            "label_idx": str(int(label_idx)),
            "label": label_str,
            "ts_s": f"{now_s:.3f}",
        }
        try:
            tl = t.roi.topLeft(); br = t.roi.bottomRight()
            extras["roi_xyxy_norm"] = f"{tl.x:.3f},{tl.y:.3f},{br.x:.3f},{br.y:.3f}"
        except Exception:
            pass

        if _send_snap("New Detection", producer, frame, ["new_detection"], extras, det_data):
            _NEWDET_STATE.seen_ids.add(tid)
            if label_str:
                _NEWDET_STATE.last_label_ts[label_str] = now_s
            print(f"[NewDet] Snap sent: id={tid}, label='{label_str}'")
