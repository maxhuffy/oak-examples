from __future__ import annotations
import os
from pathlib import Path
from dotenv import load_dotenv
import depthai as dai

from config.yaml_loader import YamlLoader
from config.config_data_classes import (
    ModelInfo,
    VideoConfig,
    NeuralNetworkConfig,
)
from config.arguments import initialize_argparser


class SystemConfiguration:
    """
    Loads and aggregates static configuration for all subsystems.
    Handles:
      • YAML loading
      • CLI + dotenv parsing
      • Model info resolution
      • Provides ready-to-use config slices
    """

    def __init__(self, platform: str):
        load_dotenv(override=True)
        self._platform: str = platform
        # Allow explicit args (for tests) or parse CLI when None
        _, self.args = initialize_argparser()

        base = Path(__file__).parent / "yaml_configs"
        self._nn_yaml = YamlLoader(base / "nn_config.yaml")
        self._conditions_yaml = YamlLoader(base / "conditions.yaml")
        self._detection_yaml = YamlLoader(base / "detection_constants.yaml")
        self._video_yaml = YamlLoader(base / "visual_constants.yaml")

        self._model_info: ModelInfo | None = None
        self._initialize_for_platform()

    def _initialize_for_platform(self) -> None:
        """
        Perform initialization that depends on runtime platform.
        """

        if self.args.api_key:
            os.environ["DEPTHAI_HUB_API_KEY"] = self.args.api_key

        if self._platform != "RVC4":
            raise ValueError(
                f"This application currently supports only RVC4, got {self._platform}"
            )

        # Use default FPS if not specified
        if self.args.fps_limit is None:
            self.args.fps_limit = self._video_yaml.default_fps
            print(f"\nFPS limit set to {self.args.fps_limit} for {self._platform}\n")

        self._model_info = self._load_model_info()

    def _load_model_info(self) -> ModelInfo:
        models_dir = Path(__file__).parent.parent / "depthai_models"
        model_name = self._nn_yaml.model.name
        yaml_file = f"{model_name}_v8_l_fp16.{self._platform}.yaml"
        yaml_path = models_dir / yaml_file

        if not yaml_path.exists():
            raise SystemExit(f"Model YAML not found for {model_name}: {yaml_path}")

        if self.args.precision != self._nn_yaml.model.default_precision:
            raise SystemExit(f"{model_name} int8 YAML not available; use fp16.")

        desc = dai.NNModelDescription.fromYamlFile(str(yaml_path))
        desc.platform = self._platform
        archive = dai.NNArchive(dai.getModelFromZoo(desc))
        w, h = archive.getInputSize()
        return ModelInfo(
            yaml_path=yaml_path,
            width=w,
            height=h,
            description=desc,
            archive=archive,
            precision=self.args.precision,
        )

    def get_video_config(self) -> VideoConfig:
        if not self._model_info:
            raise RuntimeError(
                "Model info not loaded yet — call initialize_for_platform() first."
            )

        return VideoConfig(
            resolution=self._video_yaml.video_resolution,
            fps=self.args.fps_limit,
            media_path=self.args.media_path,
            width=self._model_info.width,
            height=self._model_info.height,
        )

    def get_neural_network_config(self) -> NeuralNetworkConfig:
        if not self._model_info:
            raise RuntimeError(
                "Model info not loaded yet — call initialize_for_platform() first."
            )

        return NeuralNetworkConfig(
            nn_yaml=self._nn_yaml,
            constants=self._detection_yaml,
            model=self._model_info,
        )

    def get_snaps_config(self) -> YamlLoader:
        return self._conditions_yaml
