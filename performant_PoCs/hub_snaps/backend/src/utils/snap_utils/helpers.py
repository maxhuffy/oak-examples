from __future__ import annotations
from typing import List, Optional, Tuple
import depthai as dai
from depthai_nodes.node import SnapsProducer


def _frame_ts_seconds(frame: dai.ImgFrame) -> Optional[float]:
    try:
        return float(frame.getTimestamp().total_seconds())
    except Exception:
        return None


def _status_is_tracked(tracklet) -> bool:
    val = getattr(tracklet, "status", None)
    try:
        return val == dai.Tracklet.TrackingStatus.TRACKED
    except Exception:
        try:
            return int(val) == 1
        except Exception:
            return False


def _track_id(tracklet) -> Optional[int]:
    try:
        tid = int(getattr(tracklet, "id", -1))
        return tid if tid >= 0 else None
    except Exception:
        return None


def _label_idx_name(tracklet, class_names: List[str]) -> Tuple[int, str]:
    try:
        idx = int(getattr(tracklet, "label", -1))
    except Exception:
        idx = -1
    name = class_names[idx] if 0 <= idx < len(class_names) else ""
    return idx, name


def _sanitize_extras(extras: dict, max_key: int = 40, max_val: int = 100) -> dict[str, str]:
    out: dict[str, str] = {}
    for k, v in (extras or {}).items():
        ks = str(k)[:max_key] or "k"
        vs = str(v)
        if len(vs) > max_val:
            vs = vs[: max_val - 3] + "..."
        out[ks] = vs
    return out


def _send_snap(name: str, producer: SnapsProducer, frame: dai.ImgFrame, tags: List[str], extras: dict,
               detections: dai.ImgDetections = None) -> bool:
    ex = _sanitize_extras(extras)
    return producer.sendSnap(name, frame, detections, tags, ex)
