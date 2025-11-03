from pathlib import Path
from dataclasses import dataclass
import depthai as dai


@dataclass
class ModelInfo:
    """Stores paths and dimensions of the detection model."""

    yaml_path: Path
    width: int
    height: int
    description: dai.NNModelDescription
    archive: dai.NNArchive


@dataclass
class RuntimeConfig:
    """Stores runtime configuration (precision, FPS, platform, etc.)."""

    precision: str
    fps_limit: int
    platform: str
    model_info: ModelInfo
    ip: str
    port: int
