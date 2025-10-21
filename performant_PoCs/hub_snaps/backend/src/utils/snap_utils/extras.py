from __future__ import annotations
from typing import List
import depthai as dai
from depthai_nodes import ImgDetectionsExtended


def build_extras(
    model: str,
    det_data: dai.ImgDetections | ImgDetectionsExtended,
    class_names: List[str],
) -> dict[str, str]:
    dets = getattr(det_data, "detections", None) or []
    labels = [getattr(d, "label", -1) for d in dets[:20]]
    confs  = [getattr(d, "confidence", 0.0) for d in dets[:20]]

    extras: dict = {
        "model": model,
        "detection_count": str(len(dets)),
        "detection_label": ",".join(map(str, labels)),
        "detection_conf": ",".join(f"{c:.2f}" for c in confs),
    }

    if isinstance(det_data, dai.ImgDetections):
        xyxy = [
            (getattr(d, "xmin", 0.0), getattr(d, "ymin", 0.0),
             getattr(d, "xmax", 0.0), getattr(d, "ymax", 0.0))
            for d in dets[:10]
        ]
        extras["detection_xyxy"] = ";".join(f"{x1:.3f},{y1:.3f},{x2:.3f},{y2:.3f}" for x1, y1, x2, y2 in xyxy)
        names = []
        if dets:
            for d in dets[:20]:
                li = getattr(d, "label", -1)
                names.append(class_names[li] if 0 <= li < len(class_names) else "")
        extras["detection_label_str"] = ",".join(names) if names else ""
    elif isinstance(det_data, ImgDetectionsExtended):
        cxcywh = []
        for d in dets[:10]:
            try:
                c = d.rotated_rect.center
                s = d.rotated_rect.size
                cxcywh.append((c.x, c.y, s.width, s.height))
            except Exception:
                pass
        extras["detection_cxcywh"] = ";".join(f"{cx:.3f},{cy:.3f},{w:.3f},{h:.3f}" for cx, cy, w, h in cxcywh)
        names = []
        for d in dets[:20]:
            try:
                names.append(str(d.label_name))
            except Exception:
                names.append("")
        extras["detection_label_str"] = ",".join(names)
    else:
        extras["dtype"] = type(det_data).__name__

    return extras
