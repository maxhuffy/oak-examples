import depthai as dai
from config.system_configuration import SystemConfiguration
from core.snapping.conditions_engine import ConditionsEngine
from infrastructure.neural_network.neural_network_manager import NeuralNetworkManager
from services.snap_collection.snap_collection_service import SnapCollectionService
from infrastructure.conditions_factory import ConditionsFactory
from infrastructure.snaps.snaps_producer_factory import SnapsProducerFactory
from infrastructure.video_source_manager import VideoSourceManager
from depthai_nodes.node import SnapsProducer


class SnapsManager:
    """
    Facade for the snapping subsystem.
    """

    def __init__(
        self,
        pipeline: dai.Pipeline,
        video_source: VideoSourceManager,
        nn_manager: NeuralNetworkManager,
        config: SystemConfiguration,
    ):
        self._pipeline = pipeline
        self._video_source = video_source
        self._nn_manager = nn_manager
        self._config = config
        self._producer: SnapsProducer = None

        self._engine: ConditionsEngine = None

        self._build()

    def _build(self) -> "SnapsManager":
        cond_manager = ConditionsFactory(self._config.conditions)
        self._engine = cond_manager.get_engine()

        snaps_producer = SnapsProducerFactory.create(
            self._pipeline,
            self._video_source,
            self._nn_manager.get_tracker(),
            self._nn_manager.get_detections(),
            self._engine,
        )
        self._producer = snaps_producer
        snap_service = SnapCollectionService(self._engine, self._producer)
        self._register_service(snap_service)
        return self

    def _register_service(self, snap_service: SnapCollectionService) -> None:
        self._config.visualizer.registerService(
            snap_service.get_name(), snap_service.handle
        )

    def get_engine(self):
        return self._engine
