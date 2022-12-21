import pickle
from attrdict import AttrDict
import numpy
import json
import yaml
import os
import dgl
from types import SimpleNamespace
from collections import Counter 
import torch
from datetime import datetime

from src.components.features.utils import calculate_hidden, get_in_feats_
from src.components.graphs.loader import Papers2Graphs
from src.components.graphs.utils import logs_from_config, parse_args_ModelPredict
from src.components.nlp.repr import Repr
from src.components.graphs.models import GcnSAGE
from src.features.features_build import create_models
from src.components.graphs.utils import _generate_features
from src.utils.const import Categories_names
from src.utils.fs import create_folder_if_not_exists

from src.utils.training import cm, evaluate_test, new_cm
from src.utils.paths import CMS, CONFIG, IMAGES, INFERENCE, OUTPUT, WEIGHTS

import torch.nn.functional as F
from sklearn.metrics import precision_recall_fscore_support

def read_json(path):
    with open(path, 'r') as f:
        page_tokens = json.load(f, object_hook=lambda d: SimpleNamespace(**d))
    return page_tokens


def test(data:Papers2Graphs, config, arg_name=None):
        
    n_classes = data.num_classes 
    stats = data.stats

    print()
    print(f"MODE: using {' '.join(config.PREPROCESS.features)} features", end='\n')
    print("DATA classes stats:\n - n [{}]\n - # {}\n - % {}".format(n_classes, stats['numbers'], stats['percentages']))
    
    if config.TRAINING.gpu >= 0 and torch.cuda.is_available():
        device = 'cuda:{}'.format(config.TRAINING.gpu)
        print("TRAINING:\n -> USING CUDA")
    else:
        device = 'cpu'
        print("TRAINING:\n -> USING CUDA")
        
    # padding = config.PREPROCESS.padding
    all = dgl.batch(data.graphs)
    # all_labels = all.ndata['label']
    in_feats = get_in_feats_(config)

    #! calculate hidden layers dimensions
    # 3 options:
    mode_params = config.TRAINING.mode_params
    if mode_params not in ['fixed', 'scaled', 'half']: #, 'padding']:
        raise ValueError(f'Mode {mode_params} not in list: check config file. Exit.')

    if mode_params == 'fixed':
        # -> h_layer_dim is specified in config #! we use it
        h_layer_dim = config.MODES.fixed.h_layer_dim
    
    elif mode_params == 'scaled':
        # -> params_no is fixed #! we calculate h_layer_dim consequently
        params_no = config.MODES.scaled.params_no
        layer_no = config.TRAINING.n_layers
        assert params_no != None, ValueError('params_no must exists')
        assert layer_no != None, ValueError('layer_no must exists')
        h_layer_dim = calculate_hidden(in_feats, n_classes, params_no, layer_no)

    elif mode_params == 'half':
        # -> h_layer_dim and params_no are not specified #! we use in_feats/2
        h_layer_dim = in_feats/2

    # elif mode_params == 'padding':
    #     # -> h_layer_dim and params_no are not specified #! we use in_feats/2
    #     h_layer_dim = in_feats/2

    config.TRAINING.h_layer_dim = h_layer_dim

    print(f" -> f [{in_feats}]")
    
    # with open(data.annotations_path, 'r') as f:
    #     ann = json.load(f)

    MODELS_list = create_models(config)
    
    pages = data.pages

    n_classes = data.num_classes
    

    texts = [page['texts'] for page in pages]
    bboxs = [page['bboxs'] for page in pages]

    test_feat = _generate_features(bboxs, texts, None, MODELS_list)

    def convert(el):
        el = int(el)
        return data.label_tranformer.origin_to_conv.get(el)

    if not arg_name:
        arg_name = config.GENERAL.arg_name
    assert arg_name is not None, 'arg_name can\'t be None'
    
    # todo -> logs must have:
    # todo -> {mode}-nfeat_{BBOX_}-{efeat}-{bidi}-nlay_{}-pmode_{}-hlaydim_{}
    logs = logs_from_config(config)
    
    model = GcnSAGE(in_feats,
                    int(h_layer_dim),
                    n_classes,
                    config.TRAINING.n_layers,
                    F.relu,
                    config.TRAINING.dropout)

    model.load_state_dict(torch.load(f"{WEIGHTS}/{logs}.pt"))
    model.to(device)
    model.eval()

    #! start new
    mean_test_acc = 0.
    all_true = []
    all_pred = []
    #! end new
    
    for i, g in enumerate(data.graphs):
        g.ndata['feat'] = test_feat[i].float()
        
        g = g.to(device)

        labels = g.ndata['label']
        # print(f"SINGLE labels set: {set(g.ndata['label'].tolist())}")
        # labels = [convert(label) for label in labels]
        # labels = [label if label != None else 0 for label in labels]
        labels = torch.tensor(labels)

        # page = data.pages[i]['page']
        

        with torch.no_grad():
            logits = model(g)
            preds = logits.argmax(dim=1)
            
            acc = torch.sum(preds.to(device) == labels.to(device)).item() / g.num_nodes()

        # print("{} Test Accuracy {:.4f}".format(page, acc))

        mean_test_acc += acc
        all_true.extend(labels.tolist())
        all_pred.extend(preds.tolist())

    cms_folder = CMS
    create_folder_if_not_exists(cms_folder)

    new_cm(all_pred, all_true, logs, label_converted=data.label_tranformer, converted=config.GENERAL.converted)
    print("Mean Test Accuracy {:.4f}".format(mean_test_acc / data.__len__()))
    p_vect, r_vect, f1_vect, _ = precision_recall_fscore_support(all_true, all_pred, labels=range(n_classes))
    # todo -> change indexes
    print(f"\n\
            -> cell precision {p_vect[convert(Categories_names.TABLE_TCELL.value)]:.4f}\n\
            -> cell recall {r_vect[convert(Categories_names.TABLE_TCELL.value)]:.4f}\n\
            -> cell f1 {f1_vect[convert(Categories_names.TABLE_TCELL.value)]:.4f}\n\
            -> header precision {p_vect[convert(Categories_names.TABLE_COLH.value)]:.4f}\n\
            -> header recall {r_vect[convert(Categories_names.TABLE_COLH.value)]:.4f}\n\
            -> header f1 {f1_vect[convert(Categories_names.TABLE_COLH.value)]:.4f}\n")

    # todo -> save into file - all_pred
    all_pred_path = OUTPUT / 'all_pred'
    create_folder_if_not_exists(all_pred_path)
    pickle.dump(all_pred, open(all_pred_path / logs, 'wb'))

    # todo ->     
    # if not os.path.isfile(results_path):
    #     file = open(results_path, 'a')
    #     json.dump({}, file, indent=4)
    #     file.close()
    
    # with open(results_path, 'r+') as f:
    #     results = json.load(f)
    #     try:
    #         results[]["test_accuracy"] = metrics_threshold['acc'] / metrics_threshold['count']
    #     except:
    #         results[new_logs] = {}
    #         results[new_logs]["test_accuracy"] = metrics_threshold['acc'] / metrics_threshold['count']
        
    #     results[new_logs]["test_cell_precision"] = p[5]
    #     results[new_logs]["test_cell_recall"] = r[5]
    #     results[new_logs]["test_cell_f1"] = f1[5]
    #     results[new_logs]["test_header_precision"] = p[3]
    #     results[new_logs]["test_header_recall"] = r[3]
    #     results[new_logs]["test_header_f1"] = f1[3]
    #     results[new_logs]["test_classes_f1"] = all_f1
    #     f.seek(0)
    #     json.dump(results, f, indent=4)
    #     f.truncate() 

    # create_folder_if_not_exists(OUTPUT / 'thresholds')
    # pickle.dump(metrics_thresholds, open(OUTPUT / 'thresholds' / 'metrics_thresholds.dat', 'wb'))


if __name__ == '__main__':

    # loading data
    with open(CONFIG / 'graph' / "empty.yaml") as fileobj:
        config = AttrDict(yaml.safe_load(fileobj))
        config = AttrDict(parse_args_ModelPredict(config))
    
    print(config)

    #* start assert
    assert config.PREPROCESS.mode in ['knn', 'visibility'], ValueError('config.PREPROCESS.edge_features must be "knn" or "visibility"')
    assert len(set(config.PREPROCESS.features).difference(set(['BBOX', 'REPR', 'SPACY', 'SCIBERT']))) == 0, ValueError('config.PREPROCESS.features Error')
    assert config.PREPROCESS.edge_features != None, ValueError('config.PREPROCESS.edge_features must be != None')
    assert config.PREPROCESS.bidirectional != None, ValueError('config.PREPROCESS.bidirectional must be != None')
    
    assert config.TRAINING.mode_params in ['fixed', 'scaled'], ValueError('config.TRAINING.mode_params must exists')
    assert config.TRAINING.n_layers != None, ValueError('config.TRAINING.n_layers must exists')
    
    if config.TRAINING.mode_params == 'fixed':
        assert config.MODES.fixed.h_layer_dim != None, ValueError('config.MODES.fixed.h_layer_dim must exists')
    elif config.TRAINING.mode_params == 'scaled':
        assert config.MODES.scaled.params_no != None, ValueError('config.MODES.scaled.params_no must exists')
    #* end assert
    
    # loading data
    data = Papers2Graphs(config= config, test=True)
    
    #! (start) for get_infos -> saving images
    now = datetime.now()
    name_time = now.strftime("%d-%m-%y_%H-%M-%S")
    pre_path = IMAGES / 'test'/ f'pre_{name_time}'
    create_folder_if_not_exists(pre_path)
    post_path = IMAGES / 'test'/ f'post_{name_time}'
    create_folder_if_not_exists(post_path)
    #! (end)
    
    data.modify_graphs(num_graphs=config.TRAINING.num_graphs)
    data.get_infos(False, folder= post_path, converted=config.GENERAL.converted)

    # testing
    test(data, config)
