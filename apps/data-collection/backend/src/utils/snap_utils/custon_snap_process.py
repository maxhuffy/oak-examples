from typing import List, Optional
import depthai as dai
from depthai_nodes.node import SnapsProducer
from .conditions_gate import ConditionsGate
from .helpers import (
    _send_snap,
    _track_id,
    _label_idx_name,
    _status_is_lost,
    _roi_center_area_norm,
    _LOSTMID_STATE,
    _status_is_tracked,
)
from .extras import build_extras


def custom_snap_process(
    producer: SnapsProducer,
    frame: dai.ImgFrame,
    tracklets_msg: Optional[dai.Tracklets],
    det_data: Optional[dai.ImgDetections],
    class_names: List[str],
    model: str,
    cond_gate: ConditionsGate,
    runtime: dict | None = None,
):
    # timed
    if cond_gate.is_key_enabled("timed") and cond_gate.hit("timed"):
        extras = (
            build_extras(model, det_data, class_names)
            if det_data is not None
            else {"model": model}
        )
        if _send_snap(
            "Timing Snap", producer, frame, ["timing_snap"], extras, det_data
        ):
            print("[Timed] Snap sent!")

    dets = getattr(det_data, "detections", None) or []
    num = len(dets)

    # no detections
    if det_data is not None and num == 0:
        if cond_gate.is_key_enabled("no_detections") and cond_gate.hit("no_detections"):
            if _send_snap(
                "No Detections",
                producer,
                frame,
                ["no_detections"],
                {"model": model, "reason": "no_detections"},
            ):
                print("[NoDet] Snap sent")

    # low confidence
    if det_data is not None and cond_gate.is_key_enabled("low_conf"):
        thr = None if runtime is None else runtime.get("low_conf_thresh")
        if thr is not None:
            has_low = any(
                float(getattr(d, "confidence", 1.0)) < float(thr) for d in dets
            )
            if has_low and cond_gate.hit("low_conf"):
                min_conf = (
                    min(float(getattr(d, "confidence", 1.0)) for d in dets)
                    if dets
                    else 1.0
                )
                extras = {
                    "model": model,
                    "reason": "low_confidence",
                    "min_conf": f"{min_conf:.3f}",
                    "threshold": f"{float(thr):.3f}",
                }
                if _send_snap(
                    "Low Confidence",
                    producer,
                    frame,
                    ["low_confidence"],
                    extras,
                    det_data,
                ):
                    print(
                        f"[LowConf] Snap sent (min={min_conf:.3f} < thr={float(thr):.3f})"
                    )

    # lost detection
    if tracklets_msg is not None and cond_gate.is_key_enabled("lost_mid"):
        margin = float((runtime or {}).get("lost_mid_margin", 0.20))
        margin = max(0.0, min(0.49, margin))

        fired = False
        tks = getattr(tracklets_msg, "tracklets", None) or []
        for t in tks:
            tid = _track_id(t)
            if tid is None:
                continue

            was_tracked_prev = _LOSTMID_STATE.prev_tracked.get(tid, False)
            is_lost_now = _status_is_lost(t)
            is_tracked_now = _status_is_tracked(t)

            if is_lost_now and was_tracked_prev:
                rc = _roi_center_area_norm(t)
                if rc is not None:
                    cx, cy, area = rc
                    if (margin <= cx <= 1.0 - margin) and (
                        margin <= cy <= 1.0 - margin
                    ):
                        if cond_gate.hit("lost_mid"):
                            lbl_idx, lbl_name = _label_idx_name(t, class_names)
                            extras = {
                                "model": str(model),
                                "reason": "lost_in_middle",
                                "track_id": str(int(tid)),
                                "label_idx": str(int(lbl_idx)),
                                "label": str(lbl_name),
                                "area_frac": f"{area:.6f}",
                                "margin": f"{margin:.3f}",
                            }
                            if _send_snap(
                                "Lost in middle",
                                producer,
                                frame,
                                ["lost_in_middle"],
                                extras,
                                det_data,
                            ):
                                fired = True

            _LOSTMID_STATE.prev_tracked[tid] = bool(is_tracked_now)

        if fired:
            print("[LostMid] Snap(s) sent")
