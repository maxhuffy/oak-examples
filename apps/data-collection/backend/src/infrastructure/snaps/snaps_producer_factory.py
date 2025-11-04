from functools import partial
from depthai_nodes.node import SnapsProducer2Buffered, SnapsProducer

from core.snapping.conditions_engine import ConditionsEngine
from core.snapping.custom_snap_process import process_snaps
import depthai as dai
from depthai_nodes.node import ImgDetectionsBridge

from infrastructure.video_source_manager import VideoSourceManager


class SnapsProducerFactory:
    @staticmethod
    def create(
        pipeline: dai.Pipeline,
        video_source: VideoSourceManager,
        tracker: dai.node.ObjectTracker,
        detections: ImgDetectionsBridge,
        engine: ConditionsEngine,
    ) -> SnapsProducer:
        return pipeline.create(SnapsProducer2Buffered).build(
            frame=video_source.get_video_source_output(),
            msg=tracker.out,
            msg2=detections.out,
            running=False,
            process_fn=partial(process_snaps, engine=engine),
        )
