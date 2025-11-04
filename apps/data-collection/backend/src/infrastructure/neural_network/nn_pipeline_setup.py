import depthai as dai
from depthai_nodes.node import (
    ParsingNeuralNetwork,
    ImgDetectionsFilter,
    ImgDetectionsBridge,
)

from config.config_data_classes import NeuralNetworkConfig
from infrastructure.neural_network.annotation_node import AnnotationNode
from core.controllers.nn_controller import YOLONNController
from core.model_state import ModelState
from infrastructure.video_source_manager import VideoSourceManager


class NNPipelineSetup:
    def __init__(
        self,
        pipeline: dai.Pipeline,
        video_source: VideoSourceManager,
        nn_config: NeuralNetworkConfig,
        model_state: ModelState,
    ):
        self._pipeline = pipeline
        self._video_source = video_source
        self._nn_config = nn_config
        self._model_state = model_state

        self._det_filter = None
        self._annotation_node = None
        self._filtered_bridge = None
        self._object_tracker = None
        self._controller = None
        self._nn = None

        self.build()

    def build(self):
        nn = self._build_nn()
        self._controller = self._build_controller(nn)
        self._filtered_bridge = self._build_filters(nn)
        self._object_tracker = self._build_tracker()

    def _build_nn(self):
        nn = self._pipeline.create(ParsingNeuralNetwork)
        nn.setNNArchive(self._nn_config.model.archive)
        nn.setBackend(self._nn_config.nn_yaml.nn_backend.type)
        nn.setBackendProperties(
            {
                "runtime": self._nn_config.nn_yaml.nn_backend.runtime,
                "performance_profile": self._nn_config.nn_yaml.nn_backend.performance_profile,
            }
        )
        nn.setNumInferenceThreads(self._nn_config.nn_yaml.nn_backend.inference_threads)
        nn.getParser(0).setConfidenceThreshold(0.1)
        self._video_source.get_input_node().link(nn.inputs["images"])
        self._nn = nn
        return nn

    def _build_filters(self, nn) -> ImgDetectionsBridge:
        self._det_filter = self._pipeline.create(ImgDetectionsFilter).build(nn.out)
        self._annotation_node = self._pipeline.create(AnnotationNode).build(
            self._det_filter.out, self._video_source.get_video_source_output()
        )
        return self._pipeline.create(ImgDetectionsBridge).build(self._det_filter.out)

    def _build_controller(self, nn) -> YOLONNController:
        text_q = nn.inputs["texts"].createInputQueue()
        img_q = nn.inputs["image_prompts"].createInputQueue()
        nn.inputs["texts"].setReusePreviousMessage(True)
        nn.inputs["image_prompts"].setReusePreviousMessage(True)
        parser = nn.getParser(0)
        return YOLONNController(
            img_q, text_q, self._nn_config.model.precision, parser, self._model_state
        )

    def _build_tracker(self) -> dai.node.ObjectTracker:
        tracker = self._pipeline.create(dai.node.ObjectTracker)
        tracker.setTrackerType(dai.TrackerType.SHORT_TERM_IMAGELESS)
        tracker.setTrackerIdAssignmentPolicy(dai.TrackerIdAssignmentPolicy.UNIQUE_ID)
        tracker.setTrackingPerClass(self._nn_config.nn_yaml.tracker.track_per_class)
        tracker.setTrackletBirthThreshold(
            self._nn_config.nn_yaml.tracker.birth_threshold
        )
        tracker.setTrackletMaxLifespan(self._nn_config.nn_yaml.tracker.max_lifespan)
        tracker.setOcclusionRatioThreshold(
            self._nn_config.nn_yaml.tracker.occlusion_ratio_threshold
        )
        tracker.setTrackerThreshold(self._nn_config.nn_yaml.tracker.tracker_threshold)
        self._video_source.get_input_node().link(tracker.inputTrackerFrame)
        self._video_source.get_input_node().link(tracker.inputDetectionFrame)
        self._filtered_bridge.out.link(tracker.inputDetections)
        return tracker

    def get_tracker(self) -> dai.node.ObjectTracker:
        return self._object_tracker

    def get_detections(self) -> ImgDetectionsBridge:
        return self._filtered_bridge

    def get_filter(self) -> ImgDetectionsFilter:
        return self._det_filter

    def get_annotation_node(self) -> AnnotationNode:
        return self._annotation_node

    def get_controller(self) -> YOLONNController:
        return self._controller
