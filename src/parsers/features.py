from typing import List, Optional
from simple_parsing.helpers import list_field
from dataclasses import dataclass

@dataclass
class FeaturesGraphConfig:
    """Feature node Config for chosing the Graph collection"""

    # Which collection to use: ["train", "test"]
    split: List[str] = list_field()
    # Number of papers [number or null]
    num_papers: Optional[int] = None

@dataclass
class FeaturesChoice:
    """Features choice for the creation procedure"""
    choices: List[str] = list_field()

@dataclass
class FeatureBaseSpecifics:
    """Base specifications"""
    model_path: Optional[str] = None
    is_cuda: bool = None
    return_tensor: bool = None

@dataclass
class FeaturesBBOX:
    """Configuration for BBOX feature"""
    specifics: FeatureBaseSpecifics = FeatureBaseSpecifics(
        model_path=None, 
        is_cuda=False, 
        return_tensor=False
    )
    input: any = None

@dataclass
class FeatureREPRSpecifics(FeatureBaseSpecifics):
    """REPR specifics"""
    use_words: bool = None
    top_k: int = None

@dataclass
class FeatureREPRInput:
    """REPR inputs"""
    run_name: str = None
    epoch_no: int = None
    centroids_no: int = None
    component_no: int = None
    alpha_no: float = None

@dataclass
class FeaturesREPR:
    """Configuration for REPR feature"""
    specifics: FeatureREPRSpecifics = FeatureREPRSpecifics(
        is_cuda=False, 
        return_tensor=False,
        use_words = False,
        top_k = 2000
    )
    input: FeatureREPRInput = FeatureREPRInput(
        run_name = '2022-03-08 15:43:29',
        epoch_no = 10,
        centroids_no = 47,
        component_no = 3,
        alpha_no = 1.0
    )


@dataclass
class FeatureNLPInput:
    """NLP spacy or scibert inputs"""
    type: str = None
    reference: Optional[str] = None
    
@dataclass
class FeaturesSPACY:
    """Configuration for SPACY feature"""
    specifics: FeatureBaseSpecifics = FeatureBaseSpecifics(
        model_path='en_core_web_lg',
        is_cuda=False,
        return_tensor=False
    )
    input: FeatureNLPInput = FeatureNLPInput(
        type='singular'
    )

@dataclass
class FeaturesSCIBERTSpecifics(FeatureBaseSpecifics):
    """SCIBERT specifics"""
    per_chunk: int = None
    pooling: str = None

@dataclass
class FeaturesSCIBERT:
    """Configuration for SCIBERT feature"""
    specifics: FeaturesSCIBERTSpecifics = FeaturesSCIBERTSpecifics(
        model_path='en_core_web_lg',
        is_cuda=False,
        return_tensor=False,
        per_chunk=6000,
        pooling='mean'
    )
    input: FeatureNLPInput = FeatureNLPInput(
        type='singular'
    )