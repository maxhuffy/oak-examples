import depthai as dai
from depthai_nodes.node import SnapsProducer
from core.snapping.conditions_engine import ConditionsEngine


def process_snaps(
    producer: SnapsProducer,
    frame: dai.ImgFrame,
    tracklets: dai.Tracklets | None,
    det_data: dai.ImgDetections | None,
    engine: ConditionsEngine,
):
    """
    Evaluates all registered snap conditions and sends triggered snaps.
    """

    for condition in engine.evaluate(
        detections=det_data.detections,
        tracklets=tracklets,
    ):
        sent = producer.sendSnap(
            condition.name,
            frame,
            det_data,
            condition.tags,
            condition.make_extras(),
        )
        if sent:
            condition.mark_triggered()
            print(f"[{condition.get_key().value}] Snap sent!")
