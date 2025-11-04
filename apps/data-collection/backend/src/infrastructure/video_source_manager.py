from pathlib import Path
import depthai as dai
from infrastructure.frame_cache_node import FrameCacheNode
from config.config_data_classes import VideoConfig


class VideoSourceManager:
    """
    Creates and manages video input nodes (camera or replay) and encodes video streams.
    """

    def __init__(
        self,
        pipeline: dai.Pipeline,
        video_config: VideoConfig,
    ):
        self._pipeline = pipeline
        self._video_config = video_config

        self._video_src_out: dai.Node.Output = None
        self._input_node: dai.Node.Output = None
        self._video_enc: dai.node.VideoEncoder = None
        self._frame_cache: FrameCacheNode = None

        self._build()

    def _build(self):
        """Decide source type and build full video setup."""
        if self._video_config.media_path:
            self._create_replay_source()
        else:
            self._create_camera_source()
        self._create_encoder()
        self._frame_cache = self._pipeline.create(FrameCacheNode).build(
            self._video_src_out
        )

    def _create_replay_source(self):
        """Replay from a video file."""

        replay = self._pipeline.create(dai.node.ReplayVideo)
        replay.setReplayVideoFile(Path(self._video_config.media_path))
        replay.setOutFrameType(dai.ImgFrame.Type.NV12)
        replay.setLoop(True)
        if self._video_config.fps:
            replay.setFps(self._video_config.fps)
        replay.setSize(self._video_config.resolution)
        self._video_src_out = replay.out

        manip = self._pipeline.create(dai.node.ImageManip)
        manip.setMaxOutputFrameSize(
            self._video_config.width * self._video_config.height * 3
        )
        manip.initialConfig.setOutputSize(
            self._video_config.width, self._video_config.height
        )
        manip.initialConfig.setFrameType(dai.ImgFrame.Type.BGR888i)

        self._video_src_out.link(manip.inputImage)
        self._input_node = manip.out

    def _create_camera_source(self):
        """Live camera source."""

        cam = self._pipeline.create(dai.node.Camera).build(
            boardSocket=dai.CameraBoardSocket.CAM_A
        )
        self._video_src_out = cam.requestOutput(
            size=self._video_config.resolution,
            type=dai.ImgFrame.Type.NV12,
            fps=self._video_config.fps,
        )
        self._input_node = cam.requestOutput(
            size=(self._video_config.width, self._video_config.height),
            type=dai.ImgFrame.Type.BGR888i,
            fps=self._video_config.fps,
        )

    def _create_encoder(self):
        """Attach encoder to video source."""
        self._video_enc = self._pipeline.create(dai.node.VideoEncoder)
        self._video_enc.setDefaultProfilePreset(
            fps=self._video_config.fps,
            profile=dai.VideoEncoderProperties.Profile.H264_MAIN,
        )
        self._video_src_out.link(self._video_enc.input)

    def get_input_node(self) -> dai.Node.Output:
        """Get the input node for model processing."""
        return self._input_node

    def get_frame_cache(self) -> FrameCacheNode:
        """Get the frame cache node."""
        return self._frame_cache

    def get_video_source_output(self) -> dai.Node.Output:
        """Get the video source output node."""
        return self._video_src_out

    def get_video_topic(self) -> dai.Node.Output:
        """Get the video encoder output node."""
        return self._video_enc.out
