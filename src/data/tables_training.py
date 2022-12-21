# -*- coding: utf-8 -*-
import os
import time
import yaml
import pickle
import argparse
from attrdict import AttrDict
import torch as t
import numpy as np
from tqdm import tqdm
from glob import glob
from datetime import datetime

from torch.optim import Adam
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

from tblemb.classes.types import DotDict
from tblemb.settings.config import Config
from tblemb.classes.word2vec import Word2VecRepr
from tblemb.classes.sgns import SGNSRepr
from tblemb.classes.dataloader import PermutedSubsampledCorpus, NanLossError
from tblemb.utils.numerals import to_numeral
from src.utils.fs import create_folder_if_not_exists

from src.utils.paths import CONFIG, REPR_LOGS, REPR_MODELS, TABLES_PREPROCESS, REPR_PTS, TABLES_TRAINS

def parse_args(dictionary):
    parser = argparse.ArgumentParser()
    subparsers = parser.add_subparsers(help='specifications for training')

    train_parser = subparsers.add_parser("TRAINING")
    train_parser.add_argument('--repr_name', type=str, 
        # default='embed-repr-30-3-2000', 
        help="representation name")
    train_parser.add_argument('--model_name', type=str, 
        # default='sgns', 
        help="model name")
    train_parser.add_argument('--e_dim', type=int, 
        # default=50, 
        help="embedding dimension")
    train_parser.add_argument('--n_negs', type=int, 
        # default=10, 
        help="number of negative samples")
    train_parser.add_argument('--numeral_pow', type=float, 
        # default=0.75, 
        help="power of numerals")
    train_parser.add_argument('--epoch', type=int, 
        # default=100, 
        help="number of epochs")
    train_parser.add_argument('--mb', type=int, 
        # default=2048, 
        help="mini-batch size")
    train_parser.add_argument('--ss_t', type=float, 
        # default=1e-5, 
        help="subsample threshold for words")
    train_parser.add_argument('--ss_r', type=float, 
        # default=1e-5, 
        help="subsample threshold for representations")
    train_parser.add_argument('--scheme', type=str, 
        # default='prototype', 
        help="scheme for handling numbers, options [prototype, none, RNN, LSTM, fixed]")
    train_parser.add_argument('--lr', type=float, 
        # default=1e-3, 
        help="learning rate for adam optimizer")
    train_parser.add_argument('--gmms_name', type=str, 
        # default='gmm.dat', 
        help="filename of gmm instance in the preprocessed path")
    train_parser.add_argument('--alpha', type=float, 
        # default=1.0, 
        help='exponential power factor of the prototype interpolation')
    train_parser.add_argument('--clip', type=float, 
        # default=0.02, 
        help="clipping value for gradient")
    #! store true
    train_parser.add_argument('--conti', action='store_true', help="continue learning")
    train_parser.add_argument('--weights', action='store_true',help="use weights for negative sampling")
    train_parser.add_argument('--no_subsample', action='store_true', help="do not use subsample for tokens")
    train_parser.add_argument('--cuda', action='store_true', help="use CUDA")
    train_parser.add_argument('--log_space', action='store_true', help='log space mode for GMM and SOM')
    
    # app = dictionary.get("TRAINING", {})
    train_dict = train_parser.parse_args()
    
    return {
        **dictionary, 
        **{
            'TRAINING': train_dict.__dict__,
        }
    }

# constants
max_token_len = 20 # should be equal to

# collate functions
numeral2idx = None

def custom_collate_prototype(batch):

    iword = []
    owords = []
    iword_id = []
    iword_numerals = []
    owords_id = []
    owords_numerals = []
    
    for i in batch:
        iword.append(i[0])
        owords.append(i[1])
        iword_id.append(i[2])
        if i[3] != None:
            iword_numerals.append(i[3])

        owords_id.append(i[4])

        if i[5] != []:
            owords_numerals += i[5]

    return ( t.tensor(iword),
             t.tensor(owords),
             t.tensor(iword_id, dtype=t.uint8),
             t.tensor(iword_numerals, dtype=t.float),
             t.tensor(owords_id, dtype=t.uint8),
             t.tensor(owords_numerals, dtype=t.float) )

if __name__ == '__main__':

    with open(CONFIG / 'tables.yaml') as fileobj:
        config = AttrDict(yaml.safe_load(fileobj))
        args = AttrDict(parse_args(config))

    print(args)

    writer = SummaryWriter(REPR_LOGS)

    idx2word = pickle.load(open(TABLES_PREPROCESS / 'idx2word.dat', 'rb'))
    wc = pickle.load(open(TABLES_PREPROCESS /  'wc.dat', 'rb'))
    nc = pickle.load(open(TABLES_PREPROCESS /  'nc.dat', 'rb'))
    idx2repr = pickle.load(open(TABLES_PREPROCESS /  'idx2repr.dat', 'rb'))
    rc = pickle.load(open(TABLES_PREPROCESS /  'rc.dat', 'rb'))
    embed_repr = pickle.load(open(TABLES_PREPROCESS / f'{args.TRAINING.repr_name}.dat', 'rb'))

    # filter nc
    for k, v in nc.copy().items():
        f = np.float32(k) # caution need to be float32 cause we use float32 in further caculation
        if np.isnan(f) or np.isinf(f):
            nc.pop(k)
            print(f)

    numeral2idx = {to_numeral(numeral):idx for idx, numeral in enumerate(list(nc.keys()))}

    # ! word frequency
    wf = np.array([wc[word] for word in idx2word])
    w_sum = wf.sum()
    wf = wf / w_sum
    # word subsampling
    ws = 1 - np.sqrt(args.TRAINING.ss_t / wf) # word with freq < args.ss_t get removed with np.clip
    ws = np.clip(ws, 0, 1)
    vocab_size = args.GENERAL.max_vocab #! len(idx2word)
    token_weights = wf if args.TRAINING.weights else None

    # ! repr frequency
    rf = np.array([rc[rep_] for rep_ in idx2repr])
    r_sum = rf.sum()
    rf = rf / r_sum
    represals = np.array(list(embed_repr['embeddings'])) # embedding of all 
    represal_weights = rf
    
    centroids = embed_repr['embeddings'][embed_repr['centers']]
    centroids = t.from_numpy(centroids.astype(np.float))

    n_rate = r_sum / (r_sum + w_sum)

    create_folder_if_not_exists(REPR_PTS)
    
    now = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    
    create_folder_if_not_exists(REPR_PTS / now)
    create_folder_if_not_exists(REPR_LOGS)
    create_folder_if_not_exists(REPR_MODELS)

    # check for ags.scheme, config the custom_collate_fn
    assert args.TRAINING.scheme in ['prototype'], f'args.scheme {args.TRAINING.scheme} not valid !'

    if args.TRAINING.cuda:
        print('centroids to cuda')
        centroids = centroids.cuda()

    if args.TRAINING.scheme == 'prototype':
        custom_collate_fn = custom_collate_prototype
        model = Word2VecRepr(prototypes=centroids, alpha=args.TRAINING.alpha, vocab_size=vocab_size, embedding_size=args.TRAINING.e_dim,
                         is_cuda=args.TRAINING.cuda, log_space=args.TRAINING.log_space)

    if args.TRAINING.cuda:
        print('model to cuda')
        model = model.cuda()

    modelpath = REPR_MODELS / '{}.pt'.format(args.TRAINING.model_name)

    sgns = SGNSRepr(token_weights=token_weights, numeral_weights=represal_weights, embedding=model, vocab_size=vocab_size, n_negs=args.TRAINING.n_negs, n_rate=n_rate, numerals=represals, scheme=args.TRAINING.scheme, numeral_pow=args.TRAINING.numeral_pow)

    if os.path.isfile(modelpath) and args.TRAINING.conti:
        sgns.load_state_dict(t.load(modelpath))

    optim = Adam(sgns.parameters(), lr=args.TRAINING.lr)
    optimpath = REPR_MODELS / '{}.optim.pt'.format(args.TRAINING.model_name)
    if os.path.isfile(optimpath) and args.TRAINING.conti:
        optim.load_state_dict(t.load(optimpath))

    path_list = glob(str(TABLES_TRAINS) + '/train-batch-*.dat')
    if args.GENERAL.debug: path_list = [path_list[0]]
    file_numbers = [os.path.split(path)[1].split('.')[0].split('-')[-1] for path in path_list]
    print(len(file_numbers), file_numbers)

    # Serialized Training
    for epoch in range(1, args.TRAINING.epoch + 1):
        start_time = time.time()
        total_loss = 0

        path_list = glob(str(TABLES_TRAINS) + '/train-batch-*.dat')
        if args.GENERAL.debug: path_list = [TABLES_TRAINS / 'train-batch-1.dat'] # 221 # [path_list[0]]
        file_numbers = [os.path.split(path)[1].split('.')[0].split('-')[-1] for path in path_list]
        print(len(file_numbers)) #, file_numbers)

        for idx, train_file in enumerate(path_list):
            
            # ! remove
            if args.GENERAL.debug and idx > 0: break

            dataset = PermutedSubsampledCorpus(train_file, None) if args.TRAINING.no_subsample else PermutedSubsampledCorpus(train_file, ws)
    
            if dataset.data == None or len(dataset.data) < 10:
                continue

            dataloader = DataLoader(dataset, batch_size=args.TRAINING.mb, shuffle=True, collate_fn=custom_collate_fn)
            total_batches = int(np.ceil(len(dataset) / args.TRAINING.mb))

            pbar = tqdm(dataloader)
            pbar.set_description("[Epoch {}, File {}]".format(epoch, train_file))

            for iword, owords, iword_indicator, iword_numerals, owords_indicator, owords_numerals in pbar:

                # todo: remove fix for warning
                iword_indicator = iword_indicator.clone().detach().type(t.bool)
                owords_indicator = owords_indicator.clone().detach().type(t.bool)
                # todo

                loss = sgns(iword,
                            owords,
                            iword_indicator,
                            iword_numerals,
                            owords_indicator,
                            owords_numerals)

                if t.isnan(loss):
                    pickle.dump([iword, owords, iword_numerals, owords_numerals],
                                open(REPR_LOGS / 'nan_info.dat', 'wb'))

                    raise NanLossError()

                optim.zero_grad()
                loss.backward()
                
                norms = []
                for p in sgns.parameters():
                    if p.grad is not None:
                        norm = p.grad.data.norm(2)
                        norms.append(norm.cpu())
                max_norm = np.max(norms)

                if np.isnan(max_norm):
                    continue

                t.nn.utils.clip_grad_norm_(sgns.parameters(), max_norm=args.TRAINING.clip, norm_type=2)

                optim.step()
                pbar.set_postfix(loss=loss.item(), total_loss=total_loss, max_norm=max_norm)
                total_loss += loss.item()

        writer.add_scalar('data/loss', total_loss, epoch)
        for name, param in model.named_parameters():
            try:
                writer.add_histogram(name, param.clone().cpu().data.numpy(), epoch)
            except:
                print("Error when adding histogram for param {}".format(name))

        idx2vec_i = model.ivectors.weight.data.clone().cpu().numpy()
        idx2vec_o = model.ovectors.weight.data.clone().cpu().numpy()

        # save the trained embeddings
        if args.TRAINING.scheme == 'prototype':
            iprototype_embeddings = model.iprototypes_embeddings.clone().cpu().detach().numpy()
            oprototype_embeddings = model.oprototypes_embeddings.clone().cpu().detach().numpy()
            if type(centroids).__module__ != np.__name__:
                centroids = centroids.clone().cpu().detach().numpy()
            pickle.dump({
                'i_embedding': iprototype_embeddings,
                'o_embedding': oprototype_embeddings,
                'prototypes': centroids,
            }, open(REPR_PTS / now / f'trained_prototypes_epoch{epoch}_{len(centroids)}_{args.TRAINING.alpha}.dat', 'wb'))

        # save the word vectors and the model/optims
        pickle.dump(idx2vec_i, open(REPR_PTS / now / f'idx2vec_i_epoch{epoch}.dat', 'wb'))
        pickle.dump(idx2vec_o, open(REPR_PTS / now / f'idx2vec_o_epoch{epoch}.dat', 'wb'))
        t.save(sgns.state_dict(), REPR_PTS / now / f'{args.TRAINING.model_name}_epoch{epoch}.pt')
        t.save(optim.state_dict(), REPR_PTS / now / f'{args.TRAINING.model_name}_epoch{epoch}.optim.pt')

        print("--- %s seconds ---" % (time.time() - start_time))

    # save the arguments
    with open(REPR_PTS / now / 'args.txt', 'w') as f:
        f.write(str(args))

