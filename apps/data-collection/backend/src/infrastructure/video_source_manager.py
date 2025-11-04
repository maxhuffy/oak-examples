from pathlib import Path
import depthai as dai
from infrastructure.frame_cache_node import FrameCacheNode
from config.config_data_classes import RuntimeConfig
from config.system_configuration import SystemConfiguration


class VideoSourceManager:
    """
    Creates and manages video input nodes (camera or replay) and encodes video streams.
    """

    def __init__(
        self,
        pipeline: dai.Pipeline,
        config: SystemConfiguration,
        runtime: RuntimeConfig,
    ):
        self._pipeline = pipeline
        self._config = config
        self._runtime = runtime

        self._video_src_out: dai.Node.Output = None
        self._input_node: dai.Node.Output = None
        self._video_enc: dai.node.VideoEncoder = None
        self._frame_cache: FrameCacheNode = None

        self._build()

    def _build(self):
        """Decide source type and build full video setup."""
        if self._config.args.media_path:
            self._create_replay_source()
        else:
            self._create_camera_source()
        self._create_encoder()
        self._frame_cache = self._pipeline.create(FrameCacheNode).build(
            self._video_src_out
        )
        self._add_video_topic()

    def _create_replay_source(self):
        """Replay from a video file."""
        model_w, model_h = (
            self._runtime.model_info.width,
            self._runtime.model_info.height,
        )

        replay = self._pipeline.create(dai.node.ReplayVideo)
        replay.setReplayVideoFile(Path(self._config.args.media_path))
        replay.setOutFrameType(dai.ImgFrame.Type.NV12)
        replay.setLoop(True)
        if self._runtime.fps_limit:
            replay.setFps(self._runtime.fps_limit)
        replay.setSize(self._config.constants.visualization_resolution)
        self._video_src_out = replay.out

        manip = self._pipeline.create(dai.node.ImageManip)
        manip.setMaxOutputFrameSize(model_w * model_h * 3)
        manip.initialConfig.setOutputSize(model_w, model_h)
        manip.initialConfig.setFrameType(dai.ImgFrame.Type.BGR888i)
        self._video_src_out.link(manip.inputImage)
        self._input_node = manip.out

    def _create_camera_source(self):
        """Live camera source."""
        model_w, model_h = (
            self._runtime.model_info.width,
            self._runtime.model_info.height,
        )

        cam = self._pipeline.create(dai.node.Camera).build(
            boardSocket=dai.CameraBoardSocket.CAM_A
        )
        self._video_src_out = cam.requestOutput(
            size=self._config.constants.visualization_resolution,
            type=dai.ImgFrame.Type.NV12,
            fps=self._runtime.fps_limit,
        )
        self._input_node = cam.requestOutput(
            size=(model_w, model_h),
            type=dai.ImgFrame.Type.BGR888i,
            fps=self._runtime.fps_limit,
        )

    def _create_encoder(self):
        """Attach encoder to video source."""
        self._video_enc = self._pipeline.create(dai.node.VideoEncoder)
        self._video_enc.setDefaultProfilePreset(
            fps=self._runtime.fps_limit,
            profile=dai.VideoEncoderProperties.Profile.H264_MAIN,
        )
        self._video_src_out.link(self._video_enc.input)

    def _add_video_topic(self):
        """Register video stream with visualizer."""
        self._config.visualizer.addTopic("Video", self._video_enc.out)

    def get_input_node(self) -> dai.Node.Output:
        """Get the input node for model processing."""
        return self._input_node

    def get_frame_cache(self) -> FrameCacheNode:
        """Get the frame cache node."""
        return self._frame_cache

    def get_video_source_output(self) -> dai.Node.Output:
        """Get the video source output node."""
        return self._video_src_out
