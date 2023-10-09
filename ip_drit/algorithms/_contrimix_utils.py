from enum import auto
from enum import Enum


class ContrimixTrainingMode(Enum):
    """The training mode of Contrimix."""

    ENCODERS = auto()
    BACKBONE = auto()
    JOINTLY = auto()


class ContriMixMixingType(Enum):
    """An enum class that defines the type of mixings."""

    RANDOM = auto()
    WITHIN_CHUNK = auto()
