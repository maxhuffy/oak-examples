import numpy as np

from config.yaml_loader import YamlLoader
from core.encoders.textual_encoder import TextualEncoder
from core.encoders.visual_encoder import VisualEncoder


class EncodersManager:
    """
    Central manager for initializing and caching encoder components.
    """

    def __init__(self, config: YamlLoader, constants: YamlLoader):
        self._model_config = config
        self._constants = constants

        self.textual_encoder = self._init_textual_encoder()
        self.visual_encoder = self._init_visual_encoder()

        self.text_prompt, self.image_prompt = self._prepare_initial_prompts()

    def _init_textual_encoder(self) -> TextualEncoder:
        return TextualEncoder(
            config=self._model_config,
            precision=self._model_config.default_precision,
            max_classes=self._constants.max_num_classes,
        )

    def _init_visual_encoder(self) -> VisualEncoder:
        return VisualEncoder(
            config=self._model_config,
            precision=self._model_config.default_precision,
            max_classes=self._constants.max_num_classes,
        )

    def _prepare_initial_prompts(self) -> tuple[np.ndarray, np.ndarray]:
        text_prompt = self.textual_encoder.extract_embeddings(
            self._constants.class_names
        )
        image_prompt = self.textual_encoder.make_dummy_features()
        return text_prompt, image_prompt
