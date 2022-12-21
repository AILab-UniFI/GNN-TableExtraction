from math import inf, sqrt
import os
import torch
import simple_parsing
from src.parsers.features import FeaturesREPR
from src.parsers.graphs import *
from src.parsers.loop import LoopParameters

def _generate_features(bboxs, texts, titles, MODELS_list):

    # print(f" -> Adding features")

    results = None
    
    for model in MODELS_list:
                
        result = model._online_batch_(bboxs, texts, titles)
        if not results:
            results = result
        else:
            for i, page in enumerate(result):
                lenght_ = results[i].shape[0]
                results[i] = torch.cat((results[i], page[:lenght_]), dim=1)
        
    return results

def intersect(rectA, rectB):
    x1 = max(min(rectA[0], rectA[2]), min(rectB[0], rectB[2]))
    y1 = max(min(rectA[1], rectA[3]), min(rectB[1], rectB[3]))
    x2 = min(max(rectA[0], rectA[2]), max(rectB[0], rectB[2]))
    y2 = min(max(rectA[1], rectA[3]), max(rectB[1], rectB[3]))
    if x1<x2 and y1<y2:
        return True
    else: return False

def get_statistics(labels, num_classes):
    """How many samples per class we got from data preprocessing
    """
    stats = []
    percentages = []
    total = len(labels)
    for i in range(num_classes):
        num_samples = labels.count(i)
        stats.append(num_samples)
        percentages.append(round(num_samples/total, 2))
    return stats, percentages

def fast_get_statistics(total_count, total_dict, labels_dicts):
    
    num_classes = len(total_dict.keys())
    stats = [value for value in total_dict.values()]
    percentages = [round(value/total_count, 2) for value in total_dict.values()]
    
    return num_classes, stats, percentages

def distance(rectA, rectB):
    """Compute distance from two given bounding boxes
    """
    
    # check relative position
    left = (rectB[2] - rectA[0]) <= 0
    bottom = (rectA[3] - rectB[1]) <= 0
    right = (rectA[2] - rectB[0]) <= 0
    top = (rectB[3] - rectA[1]) <= 0
    
    vp_intersect = (rectA[0] <= rectB[2] and rectB[0] <= rectA[2]) # True if two rects "see" each other vertically, above or under
    hp_intersect = (rectA[1] <= rectB[3] and rectB[1] <= rectA[3]) # True if two rects "see" each other horizontally, right or left
    rect_intersect = vp_intersect and hp_intersect 
    
    if rect_intersect:
        return 0
    elif top and left:
        return int(sqrt((rectB[2] - rectA[0])**2 + (rectB[3] - rectA[1])**2))
    elif left and bottom:
        return int(sqrt((rectB[2] - rectA[0])**2 + (rectB[1] - rectA[3])**2))
    elif bottom and right:
        return int(sqrt((rectB[0] - rectA[2])**2 + (rectB[1] - rectA[3])**2))
    elif right and top:
        return int(sqrt((rectB[0] - rectA[2])**2 + (rectB[3] - rectA[1])**2))
    elif left:
        return (rectA[0] - rectB[2])
    elif right:
        return (rectB[0] - rectA[2])
    elif bottom:
        return (rectB[1] - rectA[3])
    elif top:
        return (rectA[1] - rectB[3])
    else: return inf

def normalize(features, size, maxw, maxh):
    """Normalize bounding boxes given the pdf size
    """
    max_area = maxw*maxh
    for feat in features:
        feat[0] = feat[0]/maxw
        feat[1] = feat[1]/maxh
        feat[2] = feat[2]/size[0]
        feat[3] = feat[3]/size[1]
        feat[4] = feat[4]/max_area
        feat[5] = feat[5]/size[0]
        feat[6] = feat[6]/size[1]
        feat[7] = feat[7]/size[0]
        feat[8] = feat[8]/size[1]
        
    return features

def center(rect):
    return [int(rect[2]-(rect[2]-rect[0])/2), int(rect[3]-(rect[3]-rect[1])/2)]


def parse_args_ModelTrain(dictionary):
    # create a parser,
    parser = simple_parsing.ArgumentParser()

    # automatically add arguments for all the fields of the classes above:
    parser.add_arguments(GraphGeneralParameters, "GENERAL")
    parser.add_arguments(GraphPreprocessParameters, "PREPROCESS")
    parser.add_arguments(DataLoaderParameters(only_tables=True, rate=0.95), "DLTRAIN")
    parser.add_arguments(DataLoaderParameters(only_tables=False, remove_islands=False), "DLTEST")
    parser.add_arguments(GraphTrainingConfig, "TRAINING")
    parser.add_arguments(LabelConfig, "LABELS")
    parser.add_arguments(GraphModesConfig, "MODES")

    # NOTE: `ArgumentParser` is just a subclass of `argparse.ArgumentParser`,
    # so we could add some other arguments as usual:
    # parser.add_argument(...)
    # parser.add_argument(...)
    # (...)
    # parser.add_argument(...)
    # parser.add_argument(...)

    args = parser.parse_args()

    general_config: GraphGeneralParameters = args.GENERAL
    # Retrieve the objects from the parsed args!
    preprocess_params: GraphPreprocessParameters = args.PREPROCESS

    train_config: GraphTrainingConfig = args.TRAINING
    
    dltrain_config: DataLoaderParameters = args.DLTRAIN
    
    modes_config: GraphModesConfig = args.MODES

    print(preprocess_params, train_config, modes_config, sep="\n")

    return {
        **dictionary,
        **{
            'GENERAL': {
                **dictionary.GENERAL,
                 **{key:value for key, value in general_config.__dict__.items() if value != None}
            },
            'PREPROCESS': {
                **dictionary.PREPROCESS, 
                **{key:value for key, value in preprocess_params.__dict__.items() if value != None}
            },
            'TRAINING': {
                **dictionary.TRAINING, 
                **{key:value for key, value in train_config.__dict__.items() if value != None}
            },
            'MODES': {
                **dictionary.MODES, 
                'fixed': {
                    **dictionary.MODES.fixed,
                    **{key:value for key, value in modes_config.fixed.__dict__.items() if value != None}
                },
                'scaled': {
                    **dictionary.MODES.scaled,
                    **{key:value for key, value in modes_config.scaled.__dict__.items() if value != None}
                }
            },
            'DLTRAIN': {
                **dictionary.DLTRAIN, 
                **{key:value for key, value in dltrain_config.__dict__.items() if value != None}
            }
        }
    }    

def parse_args_ModelPredict(dictionary):
    # create a parser,
    parser = simple_parsing.ArgumentParser()

    parser.add_arguments(GraphGeneralParameters, "GENERAL")
    parser.add_arguments(GraphPreprocessParameters, "PREPROCESS")
    parser.add_arguments(DataLoaderParameters(only_tables=True, remove_islands=True, rate=0.95), "DLTRAIN")
    parser.add_arguments(DataLoaderParameters(only_tables=False, remove_islands=False), "DLTEST")
    parser.add_arguments(GraphTrainingConfig, "TRAINING")
    parser.add_arguments(LabelConfig, "LABELS")
    parser.add_arguments(GraphModesConfig, "MODES")
    parser.add_arguments(FeaturesREPR, "REPR")

    args = parser.parse_args()

    return {
        **dictionary,
        **{
            'PREPROCESS': {
                **dictionary.PREPROCESS, 
                **{key:value for key, value in args.PREPROCESS.__dict__.items() if value != None}
            },
            'TRAINING': {
                **dictionary.TRAINING, 
                **{key:value for key, value in args.TRAINING.__dict__.items() if value != None}
            },
            'MODES': {
                **dictionary.MODES, 
                'fixed': {
                    **dictionary.MODES.fixed,
                    **{key:value for key, value in args.MODES.fixed.__dict__.items() if value != None}
                },
                'scaled': {
                    **dictionary.MODES.scaled,
                    **{key:value for key, value in args.MODES.scaled.__dict__.items() if value != None}
                }
            },
            'REPR': {
                **dictionary.get('REPR', {}),
                'specifics': {
                    **dictionary.get('REPR', {}).get('specifics', {}),
                    **{key:value for key, value in args.REPR.specifics.__dict__.items() if value != None}
                },
                'input': {
                    **dictionary.get('REPR', {}).get('input', {}),
                    **{key:value for key, value in args.REPR.input.__dict__.items() if value != None}
                }
            }
        }
    }    

def parse_args_ModelLoop(dictionary):
    # create a parser,
    parser = simple_parsing.ArgumentParser()

    parser.add_arguments(GraphGeneralParameters, "GENERAL")
    parser.add_arguments(GraphPreprocessParameters, "PREPROCESS")
    parser.add_arguments(DataLoaderParameters(only_tables=True, remove_islands=True, rate=0.95), "DLTRAIN")
    parser.add_arguments(DataLoaderParameters(only_tables=False, remove_islands=False), "DLTEST")
    parser.add_arguments(GraphTrainingConfig, "TRAINING")
    parser.add_arguments(LabelConfig, "LABELS")
    parser.add_arguments(GraphModesConfig, "MODES")
    parser.add_arguments(FeaturesREPR, "REPR")
    parser.add_arguments(LoopParameters, "BASE")

    args = parser.parse_args()

    return {
        **dictionary,
        **{
            'BASE': {
                **dictionary.get('BASE', {}), 
                **{key:value for key, value in args.BASE.__dict__.items() if value != None}
            },
            'PREPROCESS': {
                **dictionary.PREPROCESS, 
                **{key:value for key, value in args.PREPROCESS.__dict__.items() if value != None}
            },
            'TRAINING': {
                **dictionary.TRAINING, 
                **{key:value for key, value in args.TRAINING.__dict__.items() if value != None}
            },
            'MODES': {
                **dictionary.MODES, 
                'fixed': {
                    **dictionary.MODES.fixed,
                    **{key:value for key, value in args.MODES.fixed.__dict__.items() if value != None}
                },
                'scaled': {
                    **dictionary.MODES.scaled,
                    **{key:value for key, value in args.MODES.scaled.__dict__.items() if value != None}
                }
            },
            'REPR': {
                **dictionary.get('REPR', {}),
                'specifics': {
                    **dictionary.get('REPR', {}).get('specifics', {}),
                    **{key:value for key, value in args.REPR.specifics.__dict__.items() if value != None}
                },
                'input': {
                    **dictionary.get('REPR', {}).get('input', {}),
                    **{key:value for key, value in args.REPR.input.__dict__.items() if value != None}
                }
            }
        }
    }    


def logs_from_config(config):
    logs = str(config.TRAINING.num_graphs) + '-' if config.TRAINING.num_graphs != None else 'all'
    if config.TRAINING.class_weights:
        logs += 'cw-'
    logs += config.PREPROCESS.mode + '-'
    logs += f'nfeat_{"_".join(config.PREPROCESS.features)}-'
    if config.PREPROCESS.edge_features:
        logs += 'efeat-'
    if config.PREPROCESS.bidirectional:
        logs += 'dibi-'
    logs += f'bt_{config.TRAINING.batch_size}-'
    logs += f'nlay_{config.TRAINING.n_layers}-'
    logs += f'rhop_{config.PREPROCESS.range_island}-'
    logs += f'pmode_{config.TRAINING.mode_params}-'
    if config.TRAINING.mode_params == 'fixed':
        logs += f'hdim_{config.MODES.fixed.h_layer_dim}'
    elif config.TRAINING.mode_params == 'scaled':
        logs += f'pno_{config.MODES.scaled.params_no}'

    return logs