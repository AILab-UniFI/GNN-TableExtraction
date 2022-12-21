from math import inf, sqrt
import math
from typing import List
import simple_parsing

import torch

from src.parsers.features import FeaturesBBOX, FeaturesChoice, FeaturesGraphConfig, FeaturesREPR, FeaturesSCIBERT, FeaturesSPACY

def get_idxs(mode, padding):
    
    # if not padding:
    #     return None
    
    if mode == 'only_scibert':
        idxs = [-50, -1]
    elif mode == 'only_spacy':
        idxs = [313, -1]
    elif mode == 'only_repr':
        idxs = [13, 313]
    elif mode == 'only_bbox':
        idxs = [13, -1]
    else: # using all
        idxs = None

    return idxs


def split_features(batch_features, idxs):
    first = batch_features[:,:idxs[0]]
    if idxs[1]==-1:
        batch_features = first
    else:
        second = batch_features[:,idxs[1]:]
        batch_features = torch.cat((first, second), axis = 1)
    return batch_features


def get_in_feats(mode, padding):
    
    feats_dim = dict({
        'bbox': 13,
        'repr': 50,
        'scibert': 768,
        'spacy': 300,
    })

    if padding:
        return feats_dim['bbox'] + feats_dim['repr'] + feats_dim['scibert']

    if mode == 'only_scibert':
        return feats_dim['bbox'] + feats_dim['scibert']

    elif mode == 'only_spacy':
        return feats_dim['bbox'] + feats_dim['spacy']

    elif mode == 'only_repr':
        return feats_dim['bbox'] + feats_dim['repr']

    elif mode == 'only_bbox':
        return feats_dim['bbox']

    elif mode == 'all_scibert':
        return feats_dim['bbox'] + feats_dim['repr'] + feats_dim['scibert']

    elif mode == 'all_spacy':
        return feats_dim['bbox'] + feats_dim['repr'] + feats_dim['spacy']
    
    return None

def get_in_feats_(config):
    
    feats_dim = dict({
        'BBOX': 13,
        'REPR': 50,
        'SPACY': 300,
        'SCIBERT': 768,
    })

    if config.PREPROCESS.padding:
        # if paffing, we take the maximum size (BBOX-REPR-SCIBERT) 
        # todo -> why not SPACY too ?
        features = ['BBOX', 'REPR', 'SCIBERT']
    else:
        # if not padding, we sum return the real features dimension
        features = config.PREPROCESS.features

    return sum([feats_dim[feat] for feat in features])

def calculate_hidden(input_dim, classes_no, params_no, layer_no):

    hidden_layer = layer_no - 1

    def calculate_delta(a, b, c):
        return b**2 - 4*a*c

    delta = calculate_delta(hidden_layer, classes_no + input_dim, - params_no)
    x1 = (- (classes_no + input_dim) - math.sqrt(delta) )/(2*hidden_layer)
    x2 = (- (classes_no + input_dim) + math.sqrt(delta) )/(2*hidden_layer)
    results = max( x1, x2)
    return results

def parse_args_Features(dictionary):
    # create a parser,
    parser = simple_parsing.ArgumentParser()

    # automatically add arguments for all the fields of the classes above:
    parser.add_arguments(FeaturesGraphConfig, "GRAPH")
    parser.add_arguments(FeaturesChoice, "FEATURES")
    parser.add_arguments(FeaturesBBOX, "BBOX")
    parser.add_arguments(FeaturesREPR, "REPR")
    parser.add_arguments(FeaturesSPACY, "SPACY")
    parser.add_arguments(FeaturesSCIBERT, "SCIBERT")

    args = parser.parse_args()

    return {
        **dictionary,
        **{
            'GRAPH': {
                **dictionary.GRAPH, 
                **{key:value for key, value in args.GRAPH.__dict__.items() if value != None}
            },
            'FEATURES': {
                **dictionary.FEATURES, 
                **{key:value for key, value in args.FEATURES.__dict__.items() if value != None}
            },
            'BBOX': {
                **dictionary.BBOX,
                'specifics': {
                    **dictionary.BBOX.specifics,
                    **{key:value for key, value in args.BBOX.specifics.__dict__.items() if value != None}
                },
                'input': {
                    **dictionary.BBOX.input,
                    **{key:value for key, value in args.BBOX.input.__dict__.items() if value != None}
                }
            },
            'REPR': {
                **dictionary.REPR,
                'specifics': {
                    **dictionary.REPR.specifics,
                    **{key:value for key, value in args.REPR.specifics.__dict__.items() if value != None}
                },
                'input': {
                    **dictionary.REPR.input,
                    **{key:value for key, value in args.REPR.input.__dict__.items() if value != None}
                }
            },
            'SPACY': {
                **dictionary.SPACY,
                'specifics': {
                    **dictionary.SPACY.specifics,
                    **{key:value for key, value in args.SPACY.specifics.__dict__.items() if value != None}
                },
                'input': {
                    **dictionary.SPACY.input,
                    **{key:value for key, value in args.SPACY.input.__dict__.items() if value != None}
                }
            },
            'SCIBERT': {
                **dictionary.SCIBERT,
                'specifics': {
                    **dictionary.SCIBERT.specifics,
                    **{key:value for key, value in args.SCIBERT.specifics.__dict__.items() if value != None}
                },
                'input': {
                    **dictionary.SCIBERT.input,
                    **{key:value for key, value in args.SCIBERT.input.__dict__.items() if value != None}
                }
            },
        }
    }    

if __name__ == '__main__':
    input_dim = 10000
    classes_no = 8
    params_no = 100000
    layer_no = 3

    hidden_dim = calculate_hidden(input_dim, classes_no, params_no, layer_no)

    print(hidden_dim)


