from typing import List
import depthai as dai

from depthai_nodes.node import ImgDetectionsBridge

from config.config_data_classes import NeuralNetworkConfig
from core.controllers.nn_controller import YOLONNController
from core.model_state import ModelState
from infrastructure.neural_network.annotation_node import AnnotationNode
from infrastructure.neural_network.nn_service_factory import NNServiceFactory
from infrastructure.video_source_manager import VideoSourceManager
from infrastructure.neural_network.nn_pipeline_setup import NNPipelineSetup
from infrastructure.neural_network.encoders_manager import EncodersManager
from infrastructure.neural_network.handlers_manager import HandlersManager
from services.base_service import BaseService


class NeuralNetworkManager:
    """
    Facade for the neural-network subsystem.
    """

    def __init__(
        self,
        pipeline: dai.Pipeline,
        video_source: VideoSourceManager,
        config: NeuralNetworkConfig,
        model_state: ModelState,
    ):
        self._pipeline = pipeline
        self._video_source = video_source
        self._config = config
        self._model_state = model_state
        self._tracker: dai.node.ObjectTracker = None
        self._filtered_detections: ImgDetectionsBridge = None
        self._controller: YOLONNController = None
        self._annotations: AnnotationNode = None
        self._services: List[BaseService] = []
        self.build()

    def build(self) -> "NeuralNetworkManager":
        pipeline_builder = NNPipelineSetup(
            self._pipeline,
            self._video_source,
            self._config,
            self._model_state,
        )
        self._controller = pipeline_builder.get_controller()

        encoders = EncodersManager(self._config.nn_yaml.model, self._config.constants)

        handlers = HandlersManager(
            encoders,
            pipeline_builder.get_filter(),
            pipeline_builder.get_annotation_node(),
        )
        handlers.label_manager.update_labels(
            self._config.constants.class_names, self._config.constants.class_offset
        )

        service_manager = NNServiceFactory(
            self._controller, handlers, self._video_source.get_frame_cache()
        )

        self._services = service_manager.services

        self._controller.send_embeddings_pair(
            encoders.image_prompt,
            encoders.text_prompt,
            self._config.constants.class_names,
        )

        self._annotations = pipeline_builder.get_annotation_node()

        self._tracker = pipeline_builder.get_tracker()
        self._filtered_detections = pipeline_builder.get_detections()

        return self

    def get_tracker(self):
        return self._tracker

    def get_detections(self):
        return self._filtered_detections

    def get_services(self) -> list[BaseService]:
        return self._services

    def get_annotations(self) -> dai.Node.Output:
        return self._annotations.out
