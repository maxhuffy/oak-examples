import numpy as np
from ..constants import QUANT_VALUES


def pad_and_quantize_features(
    features, max_num_classes: int, model_name: str, precision="int8"
):
    """
    Pad features to (1, 512, max_num_classes) and quantize if precision is int8.
    For FP16, return padded float16 features (no quantization).
    """
    num_padding = max_num_classes - features.shape[0]
    padded = np.pad(features, ((0, num_padding), (0, 0)), "constant").T.reshape(
        1, 512, max_num_classes
    )

    if precision == "fp16":
        return padded.astype(np.float16)

    quant = QUANT_VALUES[model_name]
    out = (padded / quant["quant_scale"]) + quant["quant_zero_point"]
    return out.astype(np.uint8)


def make_dummy_features(max_num_classes: int, model_name: str, precision: str):
    """
    Create a dummy tensor of shape (1, 512, max_num_classes) for model input.
    For FP16, return zeros; for INT8, fill with the model's quantization zero point.
    """
    if precision == "fp16":
        return np.zeros((1, 512, max_num_classes), dtype=np.float16)
    qzp = int(round(QUANT_VALUES.get(model_name, {}).get("quant_zero_point", 0)))
    return np.full((1, 512, max_num_classes), qzp, dtype=np.uint8)
