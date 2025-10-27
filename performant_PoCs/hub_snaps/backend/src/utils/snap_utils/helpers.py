from __future__ import annotations
from typing import List, Optional, Tuple
import depthai as dai
from depthai_nodes.node import SnapsProducer


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


def _send_snap(
    name: str,
    producer: SnapsProducer,
    frame: dai.ImgFrame,
    tags: List[str],
    extras: dict,
    detections: dai.ImgDetections = None,
) -> bool:
    return producer.sendSnap(name, frame, detections, tags, extras)


class _LostMidState:
    def __init__(self) -> None:
        self.prev_tracked: dict[int, bool] = {}


_LOSTMID_STATE = _LostMidState()


def _status_is_lost(t) -> bool:
    try:
        return t.status == dai.Tracklet.TrackingStatus.LOST
    except Exception:
        try:
            return int(getattr(t, "status", -1)) == 2
        except Exception:
            return False


def _roi_center_area_norm(t) -> Optional[Tuple[float, float, float]]:
    roi = getattr(t, "roi", None)
    if roi is not None:
        try:
            tl = roi.topLeft()
            br = roi.bottomRight()
            x0, y0, x1, y1 = float(tl.x), float(tl.y), float(br.x), float(br.y)
            cx, cy = 0.5 * (x0 + x1), 0.5 * (y0 + y1)
            return cx, cy, max(0.0, (x1 - x0) * (y1 - y0))
        except Exception:
            pass
    d = getattr(t, "srcImgDetection", None)
    if d is not None:
        x = float(getattr(d, "x", getattr(d, "xmin", 0.0)))
        y = float(getattr(d, "y", getattr(d, "ymin", 0.0)))
        w = float(getattr(d, "width", 0.0))
        h = float(getattr(d, "height", 0.0))
        return x + 0.5 * w, y + 0.5 * h, max(0.0, w * h)
    return None
