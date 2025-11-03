from config.yaml_loader import YamlLoader
from config.config_data_classes import ModelInfo, RuntimeConfig
from config.arguments import initialize_argparser
from dotenv import load_dotenv
from pathlib import Path
import os
import depthai as dai


class SystemConfiguration:
    """Central system config. Loads each YAML file once and exposes it as an attribute."""

    def __init__(self):
        load_dotenv(override=True)
        _, args = initialize_argparser()
        self.args = args

        self.visualizer = dai.RemoteConnection(serveFrontend=False)
        self.device = dai.Device()
        self.platform = self.device.getPlatformAsString()

        base = Path(__file__).parent / "yaml_configs"
        self.constants = YamlLoader(base / "constants.yaml")
        self.model = YamlLoader(base / "model.yaml")
        self.nn_config = YamlLoader(base / "nn_config.yaml")
        self.conditions = YamlLoader(base / "conditions.yaml")

        self._initialize()

    def _initialize(self):
        if self.args.api_key:
            os.environ["DEPTHAI_HUB_API_KEY"] = self.args.api_key

        if self.platform != "RVC4":
            raise ValueError("This example is supported only on RVC4 platform")

        if self.args.fps_limit is None:
            self.args.fps_limit = self.constants.default_fps
            print(f"\nFPS limit set to {self.args.fps_limit} for {self.platform}\n")

        if self.args.precision != self.model.default_precision:
            raise SystemExit(f"{self.model.name} int8 YAML not available; use fp16.")

        self.model_info = self._load_model_info()

    def _load_model_info(self):
        models_dir = Path(__file__).parent.parent / "depthai_models"
        yaml_file = f"{self.model.name}_v8_l_fp16.{self.platform}.yaml"
        yaml_path = models_dir / yaml_file
        if not yaml_path.exists():
            raise SystemExit(f"Model YAML not found for {self.model.name}: {yaml_path}")

        desc = dai.NNModelDescription.fromYamlFile(str(yaml_path))
        desc.platform = self.platform
        archive = dai.NNArchive(dai.getModelFromZoo(desc))
        w, h = archive.getInputSize()
        return ModelInfo(yaml_path, w, h, desc, archive)

    def build_runtime_config(self):
        return RuntimeConfig(
            precision=self.args.precision,
            fps_limit=self.args.fps_limit,
            platform=self.platform,
            model_info=self.model_info,
            ip=self.args.ip or "localhost",
            port=self.args.port or 8080,
        )
