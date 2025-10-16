from typing import List, Optional
import time
import depthai as dai

from depthai_nodes.node import SnapsProducer
from depthai_nodes import ImgDetectionsExtended


class NoDetectionsGate:
    """
    Minimal state machine:
    - enabled: externally toggled
    - in_run: True while we are in a zero-detection streak
    """
    def __init__(self) -> None:
        self.enabled: bool = False
        self.in_run: bool = False

    def set_enabled(self, enabled: bool) -> None:
        self.enabled = bool(enabled)
        self.in_run = False  # reset streak whenever toggled

    def on_frame(self, num_detections: int) -> bool:
        """
        Returns True exactly on the first zero-detection frame of a streak (when enabled).
        Resets when detections resume (>0).
        """
        if not self.enabled:
            self.in_run = False
            return False
        if num_detections == 0:
            if not self.in_run:
                self.in_run = True
                return True
            return False
        # detections present -> exit streak
        if self.in_run:
            self.in_run = False
        return False


def _build_extras(
    model: str,
    det_data: dai.ImgDetections | ImgDetectionsExtended,
    class_names: List[str],
):
    detections = det_data.detections or []
    dets_labels = [det.label for det in detections]
    dets_confs = [det.confidence for det in detections]

    extras = {
        "model": model,
        "detection_label": str(dets_labels),
        "detection_confidence": str(dets_confs),
    }

    if isinstance(det_data, dai.ImgDetections):
        dets_labels_str = [class_names[det.label] for det in detections] if detections else []
        dets_xyxy = [(det.xmin, det.ymin, det.xmax, det.ymax) for det in detections]
        extras["detection_xyxy"] = str(dets_xyxy)
        extras["detection_label_str"] = str(dets_labels_str)
    elif isinstance(det_data, ImgDetectionsExtended):
        dets_labels_str = [det.label_name for det in detections]
        dets_cxcywh = [
            (
                det.rotated_rect.center.x,
                det.rotated_rect.center.y,
                det.rotated_rect.size.width,
                det.rotated_rect.size.height,
            )
            for det in detections
        ]
        extras["detection_cxcywh"] = str(dets_cxcywh)
        extras["detection_label_str"] = str(dets_labels_str)
    else:
        raise NotImplementedError

    return extras


def custom_snap_process(
    producer: SnapsProducer,
    frame: dai.ImgFrame,
    det_data: dai.ImgDetections | ImgDetectionsExtended,
    class_names: List[str],
    model: str,
    no_det_gate: Optional[NoDetectionsGate] = None,
    timed_state: Optional[dict] = None,
):
    """
    Called frequently by SnapsProducer; we decide what to send and with which tags.
    - TIMED snaps: respect user interval using timed_state["interval"] and ["last_sent_s"].
      Always tagged "Timing_snap", regardless of detections.
    - NO-DETECTIONS snaps: fire once at the start of a zero-detection streak, independent
      of timing interval. Tagged "no_detections".
    """
    detections = det_data.detections or []
    num = len(detections)

    # Current time from frame (seconds)
    now_s = None
    if timed_state is not None:
        try:
            ts = frame.getTimestamp()
            now_s = float(ts.total_seconds())
        except Exception:
            now_s = None

    # 1) TIMED (priority path)
    if timed_state and timed_state.get("timed_enabled", False):
        interval = float(timed_state.get("interval", 0)) or 0.0
        last_sent = float(timed_state.get("last_sent_s", -1.0))
        due = (now_s is not None) and (interval > 0.0) and ((last_sent < 0) or (now_s - last_sent >= interval))
        if due:
            file_group = [dai.FileData(frame, "rgb")]
            extras = _build_extras(model, det_data, class_names)
            if producer.sendSnap(name="rgb", file_group=file_group, tags=["Timing_snap"], extras=extras):
                timed_state["last_sent_s"] = now_s
                if no_det_gate:
                    no_det_gate.on_frame(num)  # keep gate in sync
                print("[Timed] Snap sent!")
                return  # don't also send no-det on the same tick

    # 2) NO-DETECTIONS one-shot at streak start (independent cadence)
    if num == 0:
        if no_det_gate and no_det_gate.on_frame(0):
            file_group = [dai.FileData(frame, "rgb")]
            extras = {"model": model, "reason": "no_detections_start"}
            if producer.sendSnap(name="rgb", file_group=file_group, tags=["no_detections"], extras=extras):
                print("[NoDet] Snap sent (start of empty streak)")
        return

    # 3) detections present -> keep gate in sync
    if no_det_gate:
        no_det_gate.on_frame(num)
    return


# ------------------ New-detection snap (TRACKED gating + cooldown) ------------------

class _NewDetectionsState:
    """Plain dicts (no defaultdict) + a set for seen IDs."""
    def __init__(self) -> None:
        self.seen_ids: set[int] = set()
        self.last_label_ts: dict[str, float] = {}  # label -> last snap epoch seconds

_NEWDET_STATE = _NewDetectionsState()


def tracklet_new_detection_process(
    producer: SnapsProducer,
    frame: dai.ImgFrame,
    tracklets_msg: dai.Tracklets,
    class_names: List[str],
    model: str,
    min_label_cooldown_s: float = 2.0,
):
    """
    Send a snap when a tracklet becomes TRACKED for the first time (ID not seen).
    - Waits for TRACKED (stable after birth threshold) to reduce flicker re-IDs.
    - One snap per unique track ID.
    - Per-label cooldown to avoid bursts (uses classic dict .get()).
    """
    tks = getattr(tracklets_msg, "tracklets", []) or []
    now = time.time()
    fired_any = False

    for t in tks:
        # Robustly read status (enum or int)
        status_val = getattr(t, "status", None)
        try:
            is_tracked = (status_val == dai.Tracklet.TrackingStatus.TRACKED)
        except Exception:
            # Fallback: enum may not be available; typical mapping NEW=0, TRACKED=1, LOST=2
            is_tracked = int(status_val) == 1

        if not is_tracked:
            continue

        # Only once per ID
        try:
            tid = int(getattr(t, "id", -1))
        except Exception:
            continue
        if tid < 0 or tid in _NEWDET_STATE.seen_ids:
            continue

        # Optional per-label cooldown
        label_idx = int(getattr(t, "label", -1))
        label_str = class_names[label_idx] if 0 <= label_idx < len(class_names) else ""
        last_ts = _NEWDET_STATE.last_label_ts.get(label_str, 0.0)
        if label_str and (now - last_ts) < float(min_label_cooldown_s):
            # still in cooldown window for this label
            continue

        # Build extras + send
        file_group = [dai.FileData(frame, "rgb")]
        extras = {
            "model": model,
            "tracker_id": str(tid),
            "label_idx": str(label_idx),
            "label_str": label_str,
            "status": "TRACKED",
        }
        if producer.sendSnap(name="rgb", file_group=file_group, tags=["new_detection"], extras=extras):
            _NEWDET_STATE.seen_ids.add(tid)
            if label_str:
                _NEWDET_STATE.last_label_ts[label_str] = now
            fired_any = True

    return
