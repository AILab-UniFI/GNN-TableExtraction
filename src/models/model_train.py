import argparse
import simple_parsing

import time
from dgl.data.utils import load_info
from random import randint

from typing import List
from xmlrpc.client import boolean
from sklearn import preprocessing
from tqdm import tqdm
import yaml
import json
from math import inf
import os
from attrdict import AttrDict
import dgl
import numpy as np
from datetime import datetime
import torch
import torch.nn as nn
import torch.nn.functional as F

from torch.utils.tensorboard import SummaryWriter
from sklearn.utils import class_weight, shuffle
from sklearn.metrics import precision_recall_fscore_support
from src.components.graphs.utils import _generate_features, logs_from_config, parse_args_ModelTrain
from src.components.nlp.scibert import SciBERT
from src.features.features_build import create_models
from src.utils.const import Categories_names

from src.utils.paths import CONFIG, FEATURES, GRAPHS, IMAGES, OUTPUT, RAW, WEIGHTS

from src.utils.training import EarlyStopping
from src.components.graphs.loader import Papers2Graphs
from src.components.graphs.models import GcnSAGE
from src.components.features.utils import calculate_hidden, get_in_feats_
from src.utils.fs import create_folder_if_not_exists

# from rich.console import Console
# from rich.table import Table
# from rich.live import Live

def train(data:Papers2Graphs, config, name_time=None):
    # console = Console()
    # table = Table(show_footer=False)
    # console.clear()

    #! half has became misteriously -> padding
    #! they are different!
    # Step 1: Train Settings =================================================================== #
    
    # TODO NUM_CLASSES, NUMBERS AND PERCENTAGES HAS TO BE UPDATED BEFORE TRAINING
    # TODO CLASS/FUNCTION TO MAP CONST TO NEW LABELS
    
    n_classes = data.num_classes
    stats = data.stats
    batch_size = config.TRAINING.batch_size
    # mode = config.TRAINING.mode
    
    # if mode not in ['all_scibert', 'all_spacy', 'only_scibert', 'only_spacy', 'only_repr', 'only_bbox']:
    #     raise ValueError(f'Mode {mode} not in list: check config file. Exit.')

    print()
    print(f"MODE: using {' '.join(config.PREPROCESS.features)} features", end='\n')
    print("DATA: classes stats:\n -> n [{}]\n -> # {}\n -> % {}".format(n_classes, stats['numbers'], stats['percentages']), end='\n')
    
    all = dgl.batch(data.graphs)
    all_labels = all.ndata['label']
    
    in_feats = get_in_feats_(config)
    # # set range of features to 0.0 for testing purposes
    # idxs = get_idxs(mode, padding)
    
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

    if config.TRAINING.class_weights:
        if config.TRAINING.class_weights_method == 'auto':
            mask = all_labels != 4
            wl = all_labels[mask]
            class_weights=class_weight.compute_class_weight(class_weight='balanced', 
                                                            classes=np.unique(wl), 
                                                            y=wl.numpy())
            class_weights = np.insert(class_weights, 4, 0.1)
        elif config.TRAINING.class_weights_method == 'default':
            class_weights = np.asarray([1.]*8)
            class_weights = np.insert(class_weights, 6, 2.)
            print(class_weights)
        else:
            raise ValueError('please specify the "class_weights_method" attribute')

        print(" -> class weights {}".format(class_weights.tolist()))
    else:
        class_weights = None
    
    if config.TRAINING.gpu >= 0 and torch.cuda.is_available():
        device = 'cuda:{}'.format(config.TRAINING.gpu)
        print("TRAINING:\n -> USING CUDA")
        if class_weights is not None: class_weights = torch.tensor(class_weights, dtype=torch.float32).to(device)
    else:
        device = 'cpu'
        print("TRAINING:\n -> USING CPU")
        
    #class_weights = torch.tensor(class_weights, dtype=torch.float).to(device)
    # if not arg_name:
    #     arg_name = config.GENERAL.arg_name
    # assert arg_name is not None, 'arg_name can\'t be None'
    
    # todo -> logs must have:
    # todo -> {mode}-nfeat_{BBOX_}-{efeat}-{bidi}-nlay_{}-pmode_{}-hlaydim_{}
    logs = logs_from_config(config)
    
    writer = SummaryWriter(f'output/runs/{logs}')
    metrics = AttrDict({
        'train': AttrDict({
            'loss': inf, 
            'acc': 0.0
        }),
        'val': AttrDict({
            'loss': inf, 
            'acc': 0.0
        }),
        'f1_vect': [0.0 for _ in range(data.num_classes)]
    })
    
    # Step 2: Create model =================================================================== #
    
    model = GcnSAGE(in_feats,
                    int(h_layer_dim),
                    n_classes,
                    config.TRAINING.n_layers,
                    F.relu,
                    config.TRAINING.dropout)
    
    model = model.to(device)
    print(model)
    
    # Step 3: Create training components ===================================================== #
        
    optimizer = torch.optim.Adam(model.parameters(), lr=config.TRAINING.lr,
                                weight_decay=config.TRAINING.weight_decay)
    # TODO adaptive lr
    loss_fn = nn.CrossEntropyLoss(weight=class_weights)
    create_folder_if_not_exists(WEIGHTS)
    
    stopper = EarlyStopping(weights=WEIGHTS, name=logs, patience=config.TRAINING.es_patience)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', factor=0.5, verbose=True)

    # Step 3.5: Load model if needed ======================================================== #

    start_epoch = 0

    def load_checkpoint(model, optimizer, writer, metrics, filename='checkpoint.pth.tar'):
        # Note: Input model & optimizer should be pre-defined.  This routine only updates their states.
        start_epoch = 0
        if os.path.isfile(filename):
            print("=> loading checkpoint '{}'".format(filename))
            checkpoint = torch.load(filename)
            start_epoch = checkpoint['epoch']
            model.load_state_dict(checkpoint['state_dict'])
            optimizer.load_state_dict(checkpoint['optimizer'])
            # writer = checkpoint['writer']
            metrics = checkpoint['metrics']
            print("=> loaded checkpoint '{}' (epoch {})"
                    .format(filename, checkpoint['epoch']))
        else:
            print("=> no checkpoint found at '{}'".format(filename))

        return model, optimizer, start_epoch, writer, metrics

    RESULT_check = OUTPUT / 'checkpoints'
    create_folder_if_not_exists(RESULT_check)

    if config.GENERAL.from_checkpoint:
        model, optimizer, start_epoch, writer, metrics = load_checkpoint(model, optimizer, writer, metrics, filename= RESULT_check / logs)
        model = model.to(device)
        # now individually transfer the optimizer parts...
        for state in optimizer.state.values():
            for k, v in state.items():
                if isinstance(v, torch.Tensor):
                    state[k] = v.to(device)
    
    # Step 4: training epochs =============================================================== #

    train_graphs, val_graphs, (train_idx, val_idx) = data.split(data.__len__())
    
    MODELS_list = create_models(config)

    def convert(el):
        return data.label_tranformer.origin_to_conv.get(el)

    def revert(el):
        return data.label_tranformer.conv_to_origin.get(el)
            

    # raw_features = load_info(GRAPHS / "train" / "INFO.pkl")
    
    # pages = raw_features['pages']
    pages = data.pages

    texts = [page['texts'] for page in pages]
    bboxs = [page['bboxs'] for page in pages]
    titles = [page['page'] for page in pages]

    train_texts = [texts[i] for i in train_idx]
    train_bboxs = [bboxs[i] for i in train_idx]
    train_titles = [titles[i]for i in train_idx]

    val_texts = [texts[i] for i in val_idx]
    val_bboxs = [bboxs[i] for i in val_idx]
    val_titles = [titles[i]for i in val_idx]

    # val_feat = _generate_features(val_bboxs, val_texts, val_titles, MODELS_list)
    val_feat = _generate_features(val_bboxs, val_texts, val_titles, MODELS_list)
    for graph_idx in range(len(val_graphs)):
        val_graphs[graph_idx].ndata['feat'] = val_feat[graph_idx].float()

    val_graph = dgl.batch(val_graphs).to(device)
    val_features = val_graph.ndata['feat'].to(device)
    print(f"VAL labels set: {set(val_graph.ndata['label'].tolist())}")
    val_labels = val_graph.ndata['label'].to(device)
    #! convert label : origin -> filtered
    # val_labels = [convert(label) for label in val_labels.tolist()]
    
    # if idxs != None:
    #     val_features = split_features(val_features, idxs)     
    #     val_graph.ndata['feat'] = val_features

    print()
    print("### START TRAINING ###", end="\n\n")
    # table = Table(show_header=True)
    # table.add_column("Epoch", header_style="bold cyan", style="cyan")
    # table.add_column("TRAIN loss", justify="right", header_style="bold blue", style="blue")
    # table.add_column("TRAIN acc", justify="right", header_style="bold blue", style="blue")
    # table.add_column("VAL loss", justify="right", header_style="bold cyan", style="cyan")
    # table.add_column("VAL acc", justify="right", header_style="bold cyan", style="cyan")
    # table.add_column("F1 cell", justify="right", header_style="bold blue", style="blue")
    # table.add_column("F1 header", justify="right", header_style="bold blue", style="blue")
    # table.add_column("count", justify="right", header_style="bold magenta", style="magenta")

    # with Live(table, refresh_per_second=4):
    
    for epoch in tqdm(range(start_epoch, config.TRAINING.n_epochs), desc='epochs'):
    # for epoch in range(config.TRAINING.n_epochs):
        
        # training the model with mini-batches
        model.train()
        # print()
        # print("Epoch {}/{}".format(epoch + 1, config.TRAINING.n_epochs), end="\n")
        
        train_graphs, train_bboxs, train_texts, train_titles = shuffle(train_graphs, train_bboxs, train_texts, train_titles)
        #train_graphs, train_bboxs, train_texts = shuffle(train_graphs, train_bboxs, train_texts)

        # tqdm_batch = tqdm(range(int(len(train_graphs) / batch_size)), desc='batches')
        for b in range(int(len(train_graphs) / batch_size)):
        # for b in range(int(len(train_graphs) / batch_size)):
    
            train_batch = train_graphs[b * batch_size: min((b+1)*batch_size, len(train_graphs))]

            batch_bboxs = train_bboxs[b * batch_size: min((b+1)*batch_size, len(train_graphs))]
            batch_texts = train_texts[b * batch_size: min((b+1)*batch_size, len(train_graphs))]
            batch_titles = train_titles[b * batch_size: min((b+1)*batch_size, len(train_graphs))]

            # batch_feat = _generate_features(batch_bboxs, batch_texts, batch_titles, MODELS_list)
            batch_feat = _generate_features(batch_bboxs, batch_texts, None, MODELS_list)
            for graph_idx in range(len(train_batch)):
                train_batch[graph_idx].ndata['feat'] = batch_feat[graph_idx].float()

            batch_graph = dgl.batch(train_batch).to(device)
            batch_features = batch_graph.ndata['feat'].to(device)

            # texts = [page['texts'] for page in pages]
            # bboxs = [page['bboxs'] for page in pages]
            # titles = [page['page'] for page in pages]
            
            # if idxs != None:
            #     batch_features = split_features(batch_features, idxs)
            #     batch_graph.ndata['feat'] = batch_features

            # forward
            
            # rand_ids = [randint(0, batch_size) for i in range(3)]
            # for ri in rand_ids:
            #     g = dgl.unbatch(batch_graph)[ri]
            #     bboxs = batch_bboxs[ri]
            #     texts = batch_texts[ri]
            #     title = batch_titles[ri]
            #     print(g.edata['feat'].tolist()[:3])

            #     data.get_gb().print_graph(g, [], bboxs,  RAW / 'train' / title, f'{title.split(".")[0]}.png', [revert(l) for l in g.ndata['label'].tolist()])
            
            logits = model(batch_graph)
            # print(f"BATCH labels set: {set(batch_graph.ndata['label'].tolist())}")
            batch_labels = batch_graph.ndata['label'].to(device)
            #! convert label : origin -> filtered
            # batch_labels = [convert(label) for label in batch_labels.tolist()]

            # https://github.com/pytorch/pytorch/issues/40388#issuecomment-647781334
            train_loss = loss_fn(logits, batch_labels.type(torch.long))
            train_acc = torch.sum(logits.argmax(dim=1) == batch_labels).item() / batch_features.shape[0]

            optimizer.zero_grad()
            train_loss.backward()
            optimizer.step()
            
            # tqdm_batch.set_description(f'TRAIN loss {train_loss}, acc {train_acc}')
            print(" -> Batch {}/{} : Training: Loss {:.4f} | Accuracy {:.4f}".format(
                    b + 1, 
                    int(len(train_graphs)/batch_size), 
                    train_loss, 
                    train_acc), 
            end="\r")
        
        # evaluate the model on validation split
        print(" -> Batch {}/{} : Training: Loss {:.4f} | Accuracy {:.4f}".format(
                b + 1, 
                int(len(train_graphs)/batch_size), 
                train_loss, 
                train_acc)
        )
        model.eval()
        
        with torch.no_grad():
            logits = model(val_graph)
            val_acc = torch.sum(logits.argmax(dim=1) == val_labels).item() / val_features.shape[0]
            val_loss = loss_fn(logits, val_labels.type(torch.long))
            
                
            #! check this magic labels -> [3, 5]
            _, _, f1, _ = precision_recall_fscore_support(
                val_labels.cpu(), 
                logits.argmax(dim=1).cpu(), 
                average=None, 
                labels=[convert(element.value) for element in Categories_names if convert(element.value) != None],
                zero_division=0
            )
            
            #! chack this magic indexes -> table_header_f1, cell_f1 = f1[0], f1[1]
            f1_vect = f1
            
        scheduler.step(val_loss)
        early_stop, counter = stopper.step(val_loss, model)

        print(" -> Validation: Loss {:.4f} | Accuracy {:.4f} | Cell F1 {:.4f} | Table Header F1 {:.4f}".format(
            val_loss.item(), 
            val_acc, 
            f1_vect[convert(Categories_names.TABLE_TCELL.value)], # cell
            f1_vect[convert(Categories_names.TABLE_COLH.value)] # table_header
        ))

        # table.add_row(
        #     f'{epoch + 1}/{config.TRAINING.n_epochs}',
        #     f'{train_loss.item():.4f}',
        #     f'{train_acc:.4f}',
        #     f'{val_loss.item():.4f}', 
        #     f'{val_acc:.4f}', 
        #     f'{f1_vect[convert(Categories_names.TABLE_TCELL.value)]:.4f}',
        #     f'{f1_vect[convert(Categories_names.TABLE_COLH.value)]:.4f}',
        #     f'{counter}/{config.TRAINING.es_patience}',
        # )
        
        # console.clear()
        # console.print(table)

        writer.add_scalar('Loss/train', train_loss, epoch+1)
        writer.add_scalar('Accuracy/train', train_acc, epoch+1)
        writer.add_scalar('Loss/val', val_loss, epoch+1)
        writer.add_scalar('Accuracy/val', val_acc, epoch+1)
        writer.add_scalar('f1/t-cell', f1_vect[convert(Categories_names.TABLE_TCELL.value)], epoch+1 )
        writer.add_scalar('f1/h-cell', f1_vect[convert(Categories_names.TABLE_COLH.value)], epoch+1 )
        writer.add_scalar('Accuracy/counter', counter, epoch+1 )
        
        if early_stop:
            break
        
        if val_loss < metrics.val.loss:
            metrics['train']['loss'] = train_loss.item()
            metrics['train']['acc'] =train_acc
            metrics['val']['loss'] = val_loss.item()
            metrics['val']['acc'] = val_acc
            metrics['f1_vect'] = f1_vect

        state = {
            'epoch': epoch+1, 
            'state_dict': model.state_dict(),
            'optimizer': optimizer.state_dict(), 
            # 'writer': writer, 
            'metrics': metrics
        }
        
        torch.save(state, RESULT_check / logs)
    
    # Step 5: save results =============================================================== #
    
    print("")
    print("### TRAINING ENDED ###", end="\n\n")
    print("Best Results:\n\n -> TrainLoss {}\n -> TrainAcc {}\n -> ValLoss {}\n -> ValAcc {}\n -> CellF1 {}\n -> HeaderF1 {}\n".format(
        metrics.train.loss, 
        metrics.train.acc, 
        metrics.val.loss, 
        metrics.val.acc,
        metrics.f1_vect[convert(Categories_names.TABLE_TCELL.value)], # cell
        metrics.f1_vect[convert(Categories_names.TABLE_COLH.value)] # table_header
    ))
    
    RESULT_ = OUTPUT / 'results'
    create_folder_if_not_exists(RESULT_)

    if not os.path.isfile( RESULT_ / f'{logs}.json'):
        file = open(RESULT_ / f'{logs}.json', 'a')
        json.dump({}, file, indent=4)
        file.close()
    
    with open(RESULT_ / f'{logs}.json', 'r+') as f:
        results = json.load(f)
        try:
            results[logs]["train_loss"] = metrics.train.loss
        except:
            results[logs] = {}
            results[logs]["train_loss"] = metrics.train.loss
            
        results[logs]["train_acc"] = metrics.train.acc
        results[logs]["val_loss"] = metrics.val.loss
        results[logs]["val_acc"] = metrics.val.acc
        results[logs]["cell_f1"] = metrics.f1_vect[convert(Categories_names.TABLE_TCELL.value)]
        results[logs]["header_f1"] = metrics.f1_vect[convert(Categories_names.TABLE_COLH.value)]
        f.seek(0)
        json.dump(results, f, indent=4)
        f.truncate()     

if __name__ == '__main__':
    
    # loading data
    with open(CONFIG / 'graph' / "empty.yaml") as fileobj:
        config = AttrDict(yaml.safe_load(fileobj))
        config = AttrDict(parse_args_ModelTrain(config))
    
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

    data = Papers2Graphs(config= config)
    
    #! (start) for get_infos -> saving images
    now = datetime.now()
    name_time = now.strftime("%y-%m-%d_%H-%M-%S")
    pre_path = IMAGES / 'train'/ f'pre_{name_time}'
    create_folder_if_not_exists(pre_path)
    post_path = IMAGES / 'train'/ f'post_{name_time}'
    create_folder_if_not_exists(post_path)
    #! (end)

    data.get_infos(False, folder= pre_path) #? it's in images/train/pre
    data.modify_graphs(num_graphs=config.TRAINING.num_graphs)
    data.get_infos(False, folder=post_path, converted=config.GENERAL.converted) #! it's going to be in images/train/post
    
    # training
    train(data, config, name_time)