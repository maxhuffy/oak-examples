import numpy as np

from config.config_data_classes import RuntimeConfig
from config.system_configuration import SystemConfiguration
from core.encoders.textual_encoder import TextualEncoder
from core.encoders.visual_encoder import VisualEncoder


class EncodersManager:
    """
    Central manager for initializing and caching encoder components.
    """

    def __init__(self, config: SystemConfiguration, runtime: RuntimeConfig):
        self._config = config
        self._runtime = runtime

        self.textual_encoder = self._init_textual_encoder()
        self.visual_encoder = self._init_visual_encoder()

        self.text_prompt, self.image_prompt = self._prepare_initial_prompts()

    def _init_textual_encoder(self) -> TextualEncoder:
        model_cfg = self._config.model
        constants = self._config.constants

        return TextualEncoder(
            config=model_cfg,
            precision=self._runtime.precision,
            max_classes=constants.max_num_classes,
        )

    def _init_visual_encoder(self) -> VisualEncoder:
        model_cfg = self._config.model
        constants = self._config.constants

        return VisualEncoder(
            config=model_cfg,
            precision=self._runtime.precision,
            max_classes=constants.max_num_classes,
        )

    def _prepare_initial_prompts(self) -> tuple[np.ndarray, np.ndarray]:
        text_prompt = self.textual_encoder.extract_embeddings(
            self._config.constants.class_names
        )
        image_prompt = self.textual_encoder.make_dummy_features()
        return text_prompt, image_prompt
