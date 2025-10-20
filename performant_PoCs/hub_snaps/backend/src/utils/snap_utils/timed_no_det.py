from __future__ import annotations
from typing import Optional, List
import depthai as dai
from depthai_nodes.node import SnapsProducer

from .helpers import _frame_ts_seconds, _send_snap
from .extras import build_extras
from .no_detections_gate import NoDetectionsGate


def custom_snap_process(
    producer: SnapsProducer,
    frame: dai.ImgFrame,
    det_data: dai.ImgDetections | dai.ImgDetections,
    class_names: List[str],
    model: str,
    no_det_gate: Optional[NoDetectionsGate] = None,
    timed_state: Optional[dict] = None,
):
    dets = getattr(det_data, "detections", None) or []
    num = len(dets)

    now_s = _frame_ts_seconds(frame) if timed_state is not None else None
    if timed_state and timed_state.get("timed_enabled", False):
        interval = float(timed_state.get("interval", 0.0)) or 0.0
        last = float(timed_state.get("last_sent_s", -1.0))
        due = (now_s is not None) and (interval > 0.0) and ((last < 0.0) or (now_s - last >= interval))
        if due:
            extras = build_extras(model, det_data, class_names)
            if _send_snap(producer, frame, ["Timing_snap"], extras):
                timed_state["last_sent_s"] = now_s
                if no_det_gate:
                    no_det_gate.on_frame(num)
                print("[Timed] Snap sent!")
                return

    if num == 0:
        if no_det_gate and no_det_gate.on_frame(0):
            extras = {"model": model, "reason": "no_detections_start"}
            if _send_snap(producer, frame, ["no_detections"], extras):
                print("[NoDet] Snap sent (start of empty streak)")
        return

    if no_det_gate:
        no_det_gate.on_frame(num)
