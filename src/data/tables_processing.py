# -*- coding: utf-8 -*-
import yaml
import argparse

from attrdict import AttrDict
from src.utils.paths import CONFIG

from src.utils.seeds import set_seeds
from src.components.tables.extractor import RhombusExtractor
from src.components.tables.preprocessor import Preprocessor
from src.components.tables.vocabulator import PyMuPDFVocabulator

set_seeds(42)

# %%

def parse_args(dictionary):

    parser = argparse.ArgumentParser()
    # parser.add_argument('--save_dir', type=str, default='./preprocess/', help="saving data directory path")
    subparsers = parser.add_subparsers(help='specifications for SOM and GMM')

    nlp_parser = subparsers.add_parser("NLP")
    nlp_parser.add_argument('--corpus', type=str, default='./data/corpus.txt', help="corpus path / directory for the corpus need preprocess")
    nlp_parser.add_argument('--filtered', type=str, default='filtered.txt', help="filename for saving filtered corpus")
    nlp_parser.add_argument('--unk_w', type=str, default='<UNK_W>', help="UNK token word")
    nlp_parser.add_argument('--unk_n', type=str, default='<UNK_N>', help="UNK numeral")
    nlp_parser.add_argument('--scheme', type=str, default='numeral_as_numeral', help="scheme should be in ['numeral_as_numeral','numeral_as_unk_numeral', 'numeral_as_token, numeral_as_token_unk_numeral']")
    nlp_dict = nlp_parser.parse_args(dictionary.get("NLP", {}))
    
    # * table_processing
    table_processing = subparsers.add_parser("TABLE_PROC")
    table_processing.add_argument('--window', type=int, default=5, help="window size")
    table_processing.add_argument('--mode', type=str, default='build', help='build, train_gmm, train_som, convert, all')
    table_processing.add_argument('--max_vocab', type=int, default=2000, help="maximum number of vocab for tokens (not numerals)")
    table_processing.add_argument('--maxdata', type=int, default=5000000, help='Max datapair for batch, 16GB ram set to 5000000, HPC can set larger')
    # if log space?
    table_processing.add_argument('--log_space', action='store_true', help='if do SOM and GMM in log space')
    tab_dict = table_processing.parse_args(dictionary.get("TABLE_PROC", {}))
    
    # * train som part
    som_parser = subparsers.add_parser("SOM")
    som_parser.add_argument('--num_prototypes', type=int, default=100, help='number of prototypes')
    som_parser.add_argument('--lr', type=float, default=0.3, help='learning rate of som')
    som_parser.add_argument('--sigma', type=float, default=3, help='sigma of gaussian neighbouring function of som')
    som_parser.add_argument('--num_iters', type=int, default=10000, help='number of iterations')
    som_dict = som_parser.parse_args(dictionary.get("SOM", {}))

    # * train gmm part
    gmm_parser = subparsers.add_parser("GMM")
    gmm_parser.add_argument('--num_components', type=int, default=30, help='number of gmm components')
    gmm_parser.add_argument('--gmm_iters', type=int, default=1000, help='number of gmm iterations')
    gmm_parser.add_argument('--prototype_path', type=str, default=None, help='if given, not none, initialize it from prototype')
    gmm_parser.add_argument('--gmm_init_mode', type=str, default='rd', help='init mode of gmm prototypes, should be one of [rd, km, fp]')
    gmm_parser.add_argument('--gmm_type', type=str, default='soft', help='gmm type, soft EM GMM and hard EM GMM, should be in [soft, hard]')
    gmm_dict = gmm_parser.parse_args(dictionary.get("GMM", {}))

    # * train repr part
    reps_parser = subparsers.add_parser("REPR")
    reps_parser.add_argument('--num_prototypes', type=int, default=30, help='number of gmm components')
    reps_parser.add_argument('--n_components', type=int, default=3, help='number of gmm components')
    reps_parser.add_argument('--limit', type=int, default=2000, help='number of gmm components')
    reps_dict = reps_parser.parse_args(dictionary.get("REPR", {}))
    
    # saving dir name
    # parser.add_argument('--saved_dir_name', type=str, default=None, help='saved dir name, eg. NumeralAsNumeral')
    # parser.add_argument('--data_dir', type=str, default='/home/vivoli/TableEmbedding/data', help='saved dir name, eg. NumeralAsNumeral')
    # parser.add_argument('--tables_dir', type=str, default='tables', help='partial directory name (from .../data) where Tables are saved, eg. tables')
    # parser.add_argument('--save_dir', type=str, default='preprocess', help='partial directory name (from .../data) where to save dictionaries and vocabularies, eg. preprocess')

    return {
        **dictionary, 
        **{
            'NLP': nlp_dict.__dict__, 
            'TABLE_PROC': tab_dict.__dict__, 
            'SOM': som_dict.__dict__, 
            'GMM': gmm_dict.__dict__, 
            'REPR': reps_dict.__dict__
        }
    }


if __name__ == '__main__':
    
    with open(CONFIG / 'tables.yaml') as fileobj:
        config = AttrDict(yaml.safe_load(fileobj))
        args = AttrDict(parse_args(config))

    # args = AttrDict({ **parse_args({}).__dict__, **Config.preprocess_dictionary() , **Config.dataset_table_outputs()})
    # args.dataset_name = 'tables-1m/repr'
    print(args)

    assert args.NLP.scheme in ['numeral_as_numeral']
    assert args.TABLE_PROC.mode in ['train_som', 'train_gmm', 'train_repr', 'build', 'convert', 'all']

    preprocessor = Preprocessor(
        unk=args.NLP.unk_w,
        args=args
    )

    if args.PREPROCESS.build:
        vocabulator = preprocessor.create_Vocabulator(PyMuPDFVocabulator, 'first_attempt')
        # try:
        vocabulator.filter_and_count(args)
        # except:
            # print('filter_and_count crashed !')
        vocabulator.build_word_vocab(args.GENERAL.max_vocab)
        vocabulator.build_repr_vocab(args.GENERAL.max_vocab)
        vocabulator.dump_built_files()
        del(vocabulator)
    
    if args.PREPROCESS.convert:
        extractor = preprocessor.create_Extractor(RhombusExtractor, 'first_attempt')
        extractor.load_files()
        # try:
        extractor.convert_items()
        # except:
        #     print('convert_items crashed !')
        del(extractor)
        
    if args.PREPROCESS.train_som:
        preprocessor.train_som(
            prototypes=args.SOM.num_prototypes, 
            sigma=args.SOM.sigma, 
            lr=args.SOM.lr, 
            iters=args.SOM.num_iters,
            log_space=args.GENERAL.log_space
        )

    if args.PREPROCESS.train_gmm:
        preprocessor.train_gmm(
            components=args.GMM.num_components, 
            iters=args.GMM.gmm_iters, 
            gmm_init_mode=args.GMM.gmm_init_mode, 
            gmm_type=args.GMM.gmm_type, 
            prototype_path=args.GMM.prototype_path, 
            log_space=args.GENERAL.log_space
        )

    if args.PREPROCESS.train_repr:
        preprocessor.train_repr(
            prototypes=args.REPR.num_prototypes, 
            n_components=args.REPR.n_components, 
            limit=args.REPR.limit
        )
    
