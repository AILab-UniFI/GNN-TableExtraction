import os
from attrdict import AttrDict
import yaml
import json
import spacy


from enum import Enum

from src.components.nlp.bbox import Bbox
from src.components.nlp.repr import Repr
from src.components.nlp.spacy import Spacy
from src.components.nlp.scibert import SciBERT

from src.utils.paths import CONFIG

class Embedder_classes(Enum):
    BBOX= Bbox
    REPR= Repr
    SPACY= Spacy
    SCIBERT= SciBERT


def get_embedder(EMBEDDER_NAME):
    
    if EMBEDDER_NAME == Embedder_classes.REPR.name:
        return Embedder_classes.REPR.value

    elif EMBEDDER_NAME == Embedder_classes.SPACY.name:
        return Embedder_classes.SPACY.value
        
    elif EMBEDDER_NAME == Embedder_classes.SCIBERT.name:
        return Embedder_classes.SCIBERT.value

    elif EMBEDDER_NAME == Embedder_classes.BBOX.name:
        return Embedder_classes.BBOX.value

    raise ValueError(f'{EMBEDDER_NAME} is not supported!')        

# def save_embeddings(data, config_path = CONFIG):
    
#     """Embed texts of data and save them

#     Args:
#         data (Papers2Graphs): collection of graphs
#         config_path (configuration ini file, optional): load configurations. Defaults to CONFIG path.
#     """
    
#     if not os.path.isdir(REPR):
#         os.mkdir(REPR)
    
#     with open(config_path) as fileobj:
#         config = AttrDict(yaml.safe_load(fileobj))

#     name = data.collection
#     out = REPR / config.PREPROCESS.embs_file
    
#     if os.path.isfile(out):
#         print(f"Embeddings already exist: adding to them. \nOpening {out} ...")
#         embeddings = json.load(open(out))
#     else:
#         embeddings = {}
    
#     before = len(embeddings.keys())
#     spacy_embedder = spacy.load(config.PREPROCESS.spacy_model)
#     repr_embedder = Repr()
    
#     print(f"Saving {name} embeddings:")
    
#     for p, page in enumerate(data.pages):
#         print(f" - {p + 1}/{data.__len__()} {page['page']}")
#         for text in page['texts']:
#             if text not in embeddings.keys():
#                 try:
#                     repr = repr_embedder(text).tolist()
#                 except:
#                     repr = "ERR0R!"
#                 try:
#                     token = spacy_embedder(text).vector.tolist()
#                 except:
#                     token = "ERR0R!"
#                 embeddings[text] = [repr, token]
    
#     then = len(embeddings.keys())
#     with open(out, 'w') as f:
#         json.dump(embeddings, f)
    
#     print(f"Done! \nWe had {before} embeddings, now {then}")
    
#     return
