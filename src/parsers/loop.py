from typing import List, Optional
from simple_parsing.helpers import list_field
from dataclasses import dataclass

from src.utils.paths import CONFIG

@dataclass
class LoopParameters:
    """General Parameters for Graph"""

    # Name for saving artefacts.
    name: Optional[str] = None # "16-04-2022"
    # # Whether the training labels correspond to the original labels.
    # path_to_config: str = CONFIG / 'graph' / "empty.yaml"
    train: bool = None
    test: bool = None
    infer: bool = None