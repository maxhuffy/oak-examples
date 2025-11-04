import depthai as dai
from depthai_nodes.node import SnapsProducer

from config.yaml_loader import YamlLoader
from core.snapping.conditions_engine import ConditionsEngine
from infrastructure.neural_network.neural_network_manager import NeuralNetworkManager
from infrastructure.snaps.conditions_factory import ConditionsFactory
from infrastructure.snaps.snaps_producer_factory import SnapsProducerFactory
from infrastructure.video_source_manager import VideoSourceManager
from services.snap_collection.snap_collection_service import SnapCollectionService


class SnapsManager:
    """
    Facade for the snapping subsystem.
    """

    def __init__(
        self,
        pipeline: dai.Pipeline,
        video_source: VideoSourceManager,
        nn_manager: NeuralNetworkManager,
        conditions_config: YamlLoader,
    ):
        self._pipeline = pipeline
        self._video_source = video_source
        self._nn_manager = nn_manager
        self._conditions_config = conditions_config
        self._producer: SnapsProducer = None

        self._engine: ConditionsEngine = None

        self._build()

    def _build(self) -> "SnapsManager":
        cond_manager = ConditionsFactory(self._conditions_config)
        self._engine = cond_manager.get_engine()

        snaps_producer = SnapsProducerFactory.create(
            self._pipeline,
            self._video_source,
            self._nn_manager.get_tracker(),
            self._nn_manager.get_detections(),
            self._engine,
        )
        self._producer = snaps_producer
        self._snap_service = SnapCollectionService(self._engine, self._producer)
        return self

    def get_service(self) -> SnapCollectionService:
        return self._snap_service

    def get_engine(self):
        return self._engine
