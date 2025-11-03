from functools import partial
from depthai_nodes.node import SnapsProducer2Buffered, SnapsProducer
from core.snapping.custom_snap_process import process_snaps
import depthai as dai
from depthai_nodes.node import ImgDetectionsBridge


class SnapsProducerFactory:
    @staticmethod
    def create(
        pipeline,
        video_source,
        tracker: dai.node.ObjectTracker,
        detections: ImgDetectionsBridge,
        engine,
    ) -> SnapsProducer:
        return pipeline.create(SnapsProducer2Buffered).build(
            frame=video_source.video_src_out,
            msg=tracker.out,
            msg2=detections.out,
            running=False,
            process_fn=partial(process_snaps, engine=engine),
        )
