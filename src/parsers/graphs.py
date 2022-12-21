from typing import List, Optional
from simple_parsing.helpers import list_field
from dataclasses import dataclass

@dataclass
class GraphGeneralParameters:
    """General Parameters for Graph"""

    # Name for saving artefacts.
    arg_name: Optional[str] = None # "16-04-2022"
    # Whether the training labels correspond to the original labels.
    converted: bool = True
    # Whether I want to load elements (model, optimizer, writer) from checkpoint
    from_checkpoint: bool = False

@dataclass
class GraphPreprocessParameters:
    """Preprocess Parameters for Graph"""

    # Max pixels and edge can be
    max_dist: int = 500
    # Number of neighbors if knn is used
    k:int = 5
    # How far a node can be from other classes. The lower the number, the more the nodes will be removed.
    range_island:int = 4
    # Random seed for reproducibility
    seed:int = 42
    # Whether to pad with 0 or not scale input dimension
    padding: bool = False
    # Graph - edges - between -> "knn" / "visibility"
    mode: str = None
    # choose features to load in graph nodes -> "bbox" / "repr" / "spacy" / "scibert"
    features: List[str] = list_field([])
    # choose edge features to load in graph edges -> "distances"
    edge_features: bool = None
    # make edges bidirectional
    bidirectional: bool = None

@dataclass
class DataLoaderParameters:
    """DataLoader Parameters for Graph, both train and test"""

    # use only the pages with a table within
    only_tables: bool = None
    # remove from graphs those nodes too far from other different classes
    remove_islands: bool = None
    # train split, range 0 -> 1
    rate: Optional[float] = None

@dataclass
class GraphTrainingConfig:
    """Settings related to Training for Graph"""

    # number of training graphs
    num_graphs: Optional[int] = None
    # adding nodes self loops
    self_loop: bool= False
    # gpu to use
    gpu: int = 0
    # how many features to drop 0 -> 100 %
    dropout: int =0
    # number of epochs
    n_epochs: int =2000
    # optimizer settings
    lr: float=0.01
    weight_decay: float= 0.0005 # 5e-4
    # k-Fold cross validation
    n_splits: int =10
    #* add class weights for data imbalance
    class_weights: bool=False
    # auto or default if class_weights is true
    class_weights_method: str = None
    # number of pages at runtime
    batch_size: int = 100
    # early stopper max epoch before quit training
    es_patience: int = 50
    # informations about the model mode -> mode for calculating hidden_layed_dimensions ["fixed", "scaled", "half", "padding"]
    mode_params: str = None
    # number of layers in the network
    n_layers: int = None

@dataclass
class LabelConfig:
    """Settings related to Label choice"""    
    
    # labels parameters
    to_remove: List[int] = list_field(4, 9, 11, 12)

@dataclass
class FixedConfig:
    """Settings for Fixed modes choice"""

    # number of parameter to have in each hidden layer
    h_layer_dim: Optional[int] = None

@dataclass
class ScaledConfig:
    """Settings for Fixed modes choice"""

    # number of parameter to have in each hidden layer
    params_no: Optional[int] = None

@dataclass
class GraphModesConfig:
    """Settings related to Models choice for Graph"""    
    
    # if it"s fixed, no additional parameters are required
    fixed: FixedConfig = FixedConfig()
    # number of parameters of the network; if None -> h_layer_dim = in_feats
    scaled: ScaledConfig = ScaledConfig()
