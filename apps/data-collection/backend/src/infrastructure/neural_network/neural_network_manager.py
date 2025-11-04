from typing import List
import depthai as dai

from depthai_nodes.node import ImgDetectionsBridge

from config.system_configuration import SystemConfiguration
from core.controllers.nn_controller import YOLONNController
from core.model_state import ModelState
from infrastructure.neural_network.annotation_node import AnnotationNode
from infrastructure.neural_network.nn_service_manager import NNServiceManager
from infrastructure.video_source_manager import VideoSourceManager
from infrastructure.neural_network.pipeline_builder import NNPipelineSetup
from infrastructure.neural_network.encoders_manager import EncodersManager
from infrastructure.neural_network.handlers_manager import HandlersManager


class NeuralNetworkManager:
    """
    Facade for the neural-network subsystem.
    """

    def __init__(
        self,
        pipeline: dai.Pipeline,
        video_source: VideoSourceManager,
        runtime,
        config: SystemConfiguration,
        model_state: ModelState,
    ):
        self._pipeline = pipeline
        self._video_source = video_source
        self._runtime = runtime
        self._config = config
        self._model_state = model_state
        self._tracker: dai.node.ObjectTracker = None
        self._filtered_detections: ImgDetectionsBridge = None
        self._controller: YOLONNController = None
        self.build()

    def build(self) -> "NeuralNetworkManager":
        pipeline_builder = NNPipelineSetup(
            self._pipeline,
            self._video_source,
            self._runtime,
            self._config.nn_config,
            self._model_state,
        )
        self._controller = pipeline_builder.get_controller()

        encoders = EncodersManager(self._config, self._runtime)

        handlers = HandlersManager(
            encoders,
            pipeline_builder.get_filter(),
            pipeline_builder.get_annotation_node(),
        )
        handlers.label_manager.update_labels(
            self._config.constants.class_names, self._config.constants.class_offset
        )

        service_manager = NNServiceManager(
            self._controller, handlers, self._video_source.get_frame_cache()
        )

        self._register_services(service_manager.services)
        self._controller.send_embeddings_pair(
            encoders.image_prompt,
            encoders.text_prompt,
            self._config.constants.class_names,
        )
        self._register_annotations(pipeline_builder.get_annotation_node())
        self._tracker = pipeline_builder.get_tracker()
        self._filtered_detections = pipeline_builder.get_detections()

        return self

    def _register_services(self, services: List):
        for service in services:
            self._config.visualizer.registerService(service.get_name(), service.handle)

    def _register_annotations(self, annotation_node: AnnotationNode):
        self._config.visualizer.addTopic("Detections", annotation_node.out)

    def get_tracker(self):
        return self._tracker

    def get_detections(self):
        return self._filtered_detections
