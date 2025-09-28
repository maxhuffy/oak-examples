from typing import List
import depthai as dai

from depthai_nodes.node import SnapsProducer
from depthai_nodes import ImgDetectionsExtended


def custom_snap_process(
    producer: SnapsProducer,
    frame: dai.ImgFrame,
    det_data: dai.ImgDetections | ImgDetectionsExtended,
    class_names: List[str],
    model: str,
):
    detections = det_data.detections
    if len(detections) == 0:
        return

    file_group = [dai.FileData(frame, "rgb")]

    dets_labels = [det.label for det in detections]
    dets_confs = [det.confidence for det in detections]

    extras = {
        "model": model,
        "detection_label": str(dets_labels),
        "detection_confidence": str(dets_confs),
    }

    if isinstance(det_data, dai.ImgDetections):
        dets_labels_str = [class_names[det.label] for det in detections]
        dets_xyxy = [(det.xmin, det.ymin, det.xmax, det.ymax) for det in detections]
        extras["detection_xyxy"] = str(dets_xyxy)

        file_group.append(dai.FileData(det_data, "detections"))

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
    else:
        raise NotImplementedError

    extras["detection_label_str"] = str(dets_labels_str)

    # if producer.sendSnap(
    #     name="rgb", file_group=file_group, tags=["demo"], extras=extras
    # ):
    #     print("Snap sent!")

    if producer.sendSnap(name="rgb", file_group=file_group, tags=["demo"]):
        print("Snap sent!")
