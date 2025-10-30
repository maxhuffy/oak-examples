from depthai_nodes.node import SnapsProducer
from ..constants import MODEL
import depthai as dai
from .conditions_manager import ConditionsManager


def custom_snap_process(
    producer: SnapsProducer,
    frame: dai.ImgFrame,
    tracklets_msg: dai.Tracklets | None,
    det_data: dai.ImgDetections | None,
    class_names: list[str],
    cond_manager: ConditionsManager,
    runtime: dict | None = None,
):
    """
    Evaluates all registered snap conditions and sends triggered snaps.

    Iterates through active conditions managed by `cond_manager`, checks
    whether each should fire based on the current detections and
    tracklets, and sends the corresponding frame via the SnapsProducer.
    """

    for condition, extras in cond_manager.evaluate(
        det_data=det_data,
        tracklets=tracklets_msg,
        frame=frame,
        runtime=runtime,
        class_names=class_names,
        model=MODEL,
    ):
        if producer.sendSnap(condition.name, frame, det_data, [condition.key], extras):
            print(f"[{condition.key}] Snap sent!")
