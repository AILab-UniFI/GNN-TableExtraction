import os
from attrdict import AttrDict
import yaml
import torch
import pickle
import argparse
import numpy as np
from torch.utils.tensorboard import SummaryWriter

from src.utils.paths import CONFIG, REPR_LOGS, REPR_PTS, TABLES_PREPROCESS


def parse_args(dictionary):
    parser = argparse.ArgumentParser()
    subparsers = parser.add_subparsers(help='specifications for saving embeddings')

    vis_parser = subparsers.add_parser("VISUALIZATION")
    vis_parser.add_argument('--top_k', type=int, help="scatter top-k words")
    vis_parser.add_argument('--run_name', type=str, help="run folder name")
    vis_parser.add_argument('--epoch_no', type=str, help="epoch number")
    vis_parser.add_argument('--centroids_no', type=str, help="centroid number")
    vis_parser.add_argument('--alpha_no', type=str, help="alpha number")
    vis_dict = {k:v for k,v in vis_parser.parse_args().__dict__.items() if v is not None}
    
    return {
        **dictionary,
        **{
            'VISUALIZATION': {
                **dictionary.VISUALIZATION, 
                **vis_dict
            }
        }
    }    

if __name__ == '__main__':

    with open(CONFIG / 'tables.yaml') as fileobj:
        config = AttrDict(yaml.safe_load(fileobj))
        args = AttrDict(parse_args(config))

    print(args)

    writer = SummaryWriter(REPR_LOGS)

    ## Read in WORDS
    wc = pickle.load(open(TABLES_PREPROCESS /  'wc.dat', 'rb'))
    word2idx = pickle.load(open(TABLES_PREPROCESS / 'word2idx.dat', 'rb'))
    idx2vec = pickle.load(open(REPR_PTS / args.VISUALIZATION.run_name / f'idx2vec_i_epoch{args.VISUALIZATION.epoch_no}.dat', 'rb'))
    embed_repr = pickle.load(open(TABLES_PREPROCESS / f'{args.TRAINING.repr_name}.dat', 'rb'))
    centroid_words = embed_repr['words'][embed_repr['centers']]
    
    words = sorted(wc, key=wc.get, reverse=True)[:args.VISUALIZATION.top_k-1]
    words = ['<UNK_W>'] + words
    words_embeddings = [idx2vec[word2idx[word]] for word in words]

    ## Read in NUMERALS
    # prototypes_loaded = {
    #     'i_embedding': iprototype_embeddings,
    #     'o_embedding': oprototype_embeddings,
    #     'prototypes': prototypes # pickle.load(open(os.path.join(args.som, args.prototypes_name), 'rb'))
    # }
    prototypes_loaded = pickle.load(open(REPR_PTS / args.VISUALIZATION.run_name / f'trained_prototypes_epoch{args.VISUALIZATION.epoch_no}_{args.VISUALIZATION.centroids_no}_{args.VISUALIZATION.alpha_no}.dat', 'rb'))
    
    all_embeddings = torch.Tensor(np.concatenate((words_embeddings, prototypes_loaded['i_embedding'])))
    writer.add_embedding(all_embeddings, metadata = 
        np.concatenate( 
            (words, centroid_words)
        ), tag='words_numbers_i_embedding')

    ## Stack into tensors
    all_embeddings = torch.Tensor(np.array(words_embeddings))
    writer.add_embedding(all_embeddings, metadata = words, tag='idx2vec_i_epoch100')

    ## Stack into tensors
    all_embeddings = torch.Tensor(prototypes_loaded['i_embedding'])
    writer.add_embedding(all_embeddings, metadata = prototypes_loaded['prototypes'], tag='proto_i_epoch100')
    
    
    all_embeddings = torch.Tensor(prototypes_loaded['o_embedding'])
    writer.add_embedding(all_embeddings, metadata = prototypes_loaded['prototypes'], tag='proto_o_epoch100')

