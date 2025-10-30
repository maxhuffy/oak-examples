from .snaps.conditions.timed_condition import TimedCondition
from .snaps.conditions.no_detections_condition import NoDetectionsCondition
from .snaps.conditions.low_conf_condition import LowConfidenceCondition
from .snaps.conditions.lost_mid_condition import LostMidCondition


CLASS_NAMES = ["person", "chair", "TV"]
CLASS_OFFSET = 0
MAX_NUM_CLASSES = 80
CONFIDENCE_THRESHOLD = 0.1
VISUALIZATION_RESOLUTION = (1080, 1080)
MODEL = "yoloe"

QUANT_VALUES = {
    "yoloe": {
        "quant_zero_point": 174.0,
        "quant_scale": 0.003328413470,
    },
    "yoloe-image": {
        "quant_zero_point": 137.0,
        "quant_scale": 0.002327915514,
    },
}

CONDITIONS = [
    TimedCondition(),
    NoDetectionsCondition(),
    LowConfidenceCondition(),
    LostMidCondition(),
]
