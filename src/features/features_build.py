import numpy as np

import yaml
from attrdict import AttrDict

from tqdm import tqdm

from dgl.data.utils import load_info, save_info
from src.components.features.utils import parse_args_Features 
# from dgl import load_graph

from src.components.nlp.utils import get_embedder
from src.utils.fs import create_folder_if_not_exists
from src.utils.paths import CONFIG, GRAPHS, FEATURES

def create_features(pages, config, split):

    features_choices = list(config.FEATURES.choices)
    
    texts = [page['texts'] for page in pages]
    bboxs = [page['bboxs'] for page in pages]
    titles = [page['page'] for page in pages]
        
    success = []

    for feature in features_choices:
        
        specifics = config[feature.upper()]['specifics']
        input = config[feature.upper()]['input']
        
        EMBEDDER_CLASS = get_embedder(feature.upper())
        embedder = EMBEDDER_CLASS(specifics, input)
        
        result = embedder(bboxs, texts, titles, split=split)

        success.append(result)
    
    return all(success)


def create_models(config):

    features_choices = list(config.PREPROCESS.features)
           
    models = []

    for feature in features_choices:
        
        specifics = config.FEATURES[feature.upper()]['specifics']
        input = config.FEATURES[feature.upper()]['input']
        
        EMBEDDER_CLASS = get_embedder(feature.upper())
        embedder = EMBEDDER_CLASS(specifics, input)
        
        models.append(embedder)
    
    return models


if __name__ == '__main__':
    
        # loading data
    with open(CONFIG / "features.yaml") as fileobj:
        config = AttrDict(yaml.safe_load(fileobj))
        config = AttrDict(parse_args_Features(config))
    
    print(config)

    #* start assert
    assert len(set(config.GRAPH.split).difference(set(['train', 'test']))) == 0, ValueError('config.GRAPH.split must be "train" and/or "test"')
    assert len(set(config.GRAPH.choice).difference(set(['BBOX', 'REPR', 'SPACY', 'SCIBERT']))) == 0, ValueError('config.GRAPH.choice Error')
    #* end assert

    create_folder_if_not_exists(FEATURES)

    for split in config.GRAPH.split:

        print(f' -> Preparing {split.upper()} split')

        # todo -> load bboxs and texts
        if config.GRAPH.num_papers == None:
            info_path = GRAPHS / f"{split}" / "INFO.pkl"
            # graph_path = GRAPHS / f"{split}_all.bin"
        else:
            info_path = GRAPHS / f"{split}_n{config.GRAPH.num_papers}_scibert_info.pkl"
            # graph_path = GRAPHS / f"{split}_n{config.GRAPH.num_papers}_{feature_name}.bin"
        
        raw_features = load_info(info_path)
        # raw_graph = load_graph(graph_path)
            
        num_classes = raw_features['num_classes']
        stats = raw_features['stats']
        # pages = [ 
        #       {'page': page_1, 'bboxs' : bboxs_2, 'texts' : texts_2},
        #       {'page': page_2, 'bboxs' : bboxs_2, 'texts' : texts_2},
        #       ...
        #   ]
        pages = raw_features['pages']

        create_folder_if_not_exists(FEATURES / split)
        
        # todo -> create features
        create_features(pages, config, split)
