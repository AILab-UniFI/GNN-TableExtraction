import os
from pydoc import describe
from tqdm import tqdm
import re
import pickle
import numpy as np
from copy import copy
from glob import glob
from typing import List
from bs4 import UnicodeDammit
from attrdict import AttrDict
from src.utils.fs import create_folder_if_not_exists
from src.utils.strings import custom_tokenizer, pymu_custom_tokenizer
from src.components.tables.manager import Manager
from src.components.tables.types import CountDict, Divider
from src.utils.nums import is_numeral, number_handler, to_numeral
from src.utils.paths import TABLES_PREPROCESS, TABLES_REPR

class Vocabulator:
    name: str

    wc: dict # words count
    nc: dict # numbers count
    rc: dict # numbers count

    vocab: set
    idx2word: List[str]
    word2idx: dict

    repr_vocab: set
    idx2repr: List[str]
    repr2idx: dict

    def __init__(self, name=None, unk=None):
        assert name != None and unk != None, ValueError('can\' instantiate this class!')
        self.name = name
        self.unk = unk
        self.wc = CountDict( {self.unk: 1} )
        self.nc = CountDict({})
        self.rc = CountDict( {self.unk: 1} )

   

    def filter_and_count(self, args=AttrDict({})):
        # from list of tables it creates
        # wc and nc and filtered tables
        # based on * with punctuation * or not
        pass

    def build_word_vocab(self, max_vocab=20000):
        # from wc it creates idx2word, word2idx, vocab
        # from nc it can create * something * based on # todo
        pass

    def build_repr_vocab(self, max_vocab=20000):
        # from wc it creates idx2word, word2idx, vocab
        # from nc it can create * something * based on # todo
        pass

    def dump_built_files(self):
        # words
        create_folder_if_not_exists(TABLES_PREPROCESS)
        pickle.dump(self.wc, open(TABLES_PREPROCESS / 'wc.dat', 'wb'))
        pickle.dump(self.vocab, open(TABLES_PREPROCESS / 'vocab.dat', 'wb'))
        pickle.dump(self.idx2word, open(TABLES_PREPROCESS / 'idx2word.dat', 'wb'))
        pickle.dump(self.word2idx, open(TABLES_PREPROCESS / 'word2idx.dat', 'wb'))
        # numbers
        pickle.dump(self.nc, open(TABLES_PREPROCESS / 'nc.dat', 'wb'))
        # representations
        pickle.dump(self.rc, open(TABLES_PREPROCESS / 'rc.dat', 'wb'))
        pickle.dump(self.repr_vocab, open(TABLES_PREPROCESS / 'repr_vocab.dat', 'wb'))
        pickle.dump(self.idx2repr, open(TABLES_PREPROCESS / 'idx2repr.dat', 'wb'))
        pickle.dump(self.repr2idx, open(TABLES_PREPROCESS / 'repr2idx.dat', 'wb'))
        print("Dump done")

    def load_files(self):
        self.wc = pickle.load(open(TABLES_PREPROCESS / 'wc.dat', 'rb'))
        self.vocab = pickle.load(open(TABLES_PREPROCESS / 'vocab.dat', 'rb'))
        self.idx2word = pickle.load(open(TABLES_PREPROCESS / 'idx2word.dat', 'rb'))
        self.word2idx = pickle.load(open(TABLES_PREPROCESS / 'word2idx.dat', 'rb'))

        self.nc = pickle.load(open(TABLES_PREPROCESS / 'nc.dat', 'rb'))

        self.rc = pickle.load(open(TABLES_PREPROCESS / 'rc.dat', 'rb'))
        self.repr_vocab = pickle.load(open(TABLES_PREPROCESS / 'repr_vocab.dat', 'rb'))
        self.idx2repr = pickle.load(open(TABLES_PREPROCESS / 'idx2repr.dat', 'rb'))
        self.repr2idx = pickle.load(open(TABLES_PREPROCESS / 'repr2idx.dat', 'rb'))

class RangesSGVocabulator(Vocabulator):
    pass


class PyMuPDFVocabulator(Vocabulator):
    
    def __init__(self, name='PyMuPDF', unk=None) -> None:
        super(PyMuPDFVocabulator, self).__init__(name, unk)

    def filter_and_count(self, args=AttrDict({})):
        print("Filtering numbers ...")

        # TODO creation of:
        # TODO - wc
        # TODO - nc
        # TODO - rc

        step = 0

        # the re for all possible token number
        RE_NUM = re.compile(r"(((-?\d+(,\d{3})*(\.\d+)?)\/(-?\d+(,\d{3})*(\.\d+)?))|(-?\d+(,\d{3})*(\.\d+)?))", re.UNICODE)

        output = open(TABLES_PREPROCESS / 'filtered', 'w', encoding='utf-8')
        
        path_list = glob(str(TABLES_REPR) + '/tables-batch-*.dat')
        length = len(path_list)
        tqdm_list = tqdm(path_list, desc=f'{step // 1000}kth')

        for idx_, tables_file in enumerate(tqdm_list):
            
            #! if idx_ > 1: break

            # print()
            # print(f'[ { int(idx_ / length * 100)} % ] - kth', end=' ')
            
            tables_pairs = pickle.load(open(tables_file, 'rb'))
            
            for tables in tables_pairs:
                
                _, value_table = tables
                step += 1
                # if not step % 1000:
                #     print(f'{step // 1000}kth', end=' ')

                sent_filtered = []
                # for line in np.concatenate(value_table.values):
                
                for i in range(value_table.shape[0]):
                    for j in range(value_table.shape[1]):
                        cell = value_table.iloc[i,j]
                        cell = UnicodeDammit(cell).unicode_markup
                        if not cell:
                            # continue
                            cell = ''
                        manager = Manager(cell)\
                            .set_new_content()\
                            .set_replace_content()\
                            .remove_nextline()\
                            .remove_unicode()
                        
                        replace_content = manager\
                            .replace_chars_and_digits() \
                            .remove_number_sign() \
                            .replace_content

                        # saving "representations" to dictionary
                        self.rc.add_count(replace_content.split(' '))
                        
                        # calculates "numbers" and "words" dictionaries
                        new_content = manager.new_content.strip()
                        sent = pymu_custom_tokenizer(new_content)
                        sent_filtered = []
                        for token in sent:
                            
                            # we treat word and numerals differently
                            # match numerals
                            res = re.findall(RE_NUM, token)
                            if res != []:
                                
                                if args.GENERAL.divider == Divider.strictly_words.value:

                                    if is_numeral(token):
                                        number = str(to_numeral(token))
                                        self.nc.add_count([number])
                                        sent_filtered.append(number)
                                    else:
                                        self.wc.add_count([token])
                                        sent_filtered.append(token)
                                
                                else:

                                    target = number_handler(token)
                                    # we do not want nc to record ''
                                    if target == '':
                                        continue

                                    if type(target) is list:
                                        # ['u-32'] to ['u','-','32']
                                        # [1997/07] to ['1997','/','7']

                                        for k in target:
                                            if is_numeral(k):
                                                number = str(to_numeral(k))
                                                self.nc.add_count([number])
                                                sent_filtered.append(number)
                                            else:
                                                self.wc.add_count([k])
                                                sent_filtered.append(k)

                                    elif is_numeral(target):
                                        # ['-32.000'] to ['-32']
                                        # prevent '-haha' like token, double check
                                        number = str(to_numeral(target))
                                        self.nc.add_count([number])
                                        sent_filtered.append(number)

                            else:
                                self.wc.add_count([token])
                                sent_filtered.append(token)

                output.write(bytes(' '.join(sent_filtered), 'utf-8').decode('utf-8') + '\n')

        output.close()
        print("filtering corpus done")

    def build_word_vocab(self, max_vocab=2000):
        print("start building WORDS vocab")
        wc_nounk = copy(self.wc)
        wc_nounk.__delitem__(self.unk)
        self.idx2word = [self.unk]  + sorted(wc_nounk, key=wc_nounk.get, reverse=True)[:max_vocab - 1]
        self.word2idx = {self.idx2word[idx]: idx for idx, _ in enumerate(self.idx2word)}
        self.vocab = set([word for word in self.word2idx])
        print("building vocab WORDS done")

    def build_repr_vocab(self, max_vocab=2000):
        """ It just sort the Repr_counter dictionary by value """
        print("start building REPR vocab")
        rc_nounk = copy(self.rc)
        rc_nounk.__delitem__(self.unk)
        self.idx2repr = [self.unk]  + sorted(rc_nounk, key=rc_nounk.get, reverse=True)[:max_vocab - 1]
        self.repr2idx = {self.idx2repr[idx]: idx for idx, _ in enumerate(self.idx2repr)}
        self.repr_vocab = set([re for re in self.repr2idx])
        print("building vocab REPR done")

class StdSGVocabulator(Vocabulator):

    def __init__(self, name='stdSG', unk=None) -> None:
        super(StdSGVocabulator, self).__init__(name, unk)

    def filter_and_count(self, args=AttrDict({})):
        print("Filtering numbers ...")

        # TODO creation of:
        # TODO - wc 
        # TODO - nc
        
        step = 0

        # the re for all possible token number
        RE_NUM = re.compile(r"(((-?\d+(,\d{3})*(\.\d+)?)\/(-?\d+(,\d{3})*(\.\d+)?))|(-?\d+(,\d{3})*(\.\d+)?))", re.UNICODE)

        output = open(TABLES_PREPROCESS / 'filtered', 'w', encoding='utf-8')
        
        path_list = glob(str(TABLES_REPR) + '/tables-batch-*.dat')
        length = len(path_list)

        for idx_, tables_file in enumerate(path_list):
            
            # if idx_ > 1: break

            print()
            print(f'[ { int(idx_ / length * 100)} % ] - 0kth', end=' ')
            
            tables_pairs = pickle.load(open(tables_file, 'rb'))
            
            for tables in tables_pairs:
                
                _, value_table = tables
                step += 1
                if not step % 1000:
                    print(f'{step // 1000}kth', end=' ')

                sent_filtered = []
                # for line in np.concatenate(value_table.values):
                
                for i in range(value_table.shape[0]):
                    for j in range(value_table.shape[1]):
                        line = value_table.iloc[i,j]
                        line = UnicodeDammit(line).unicode_markup
                        if not line:
                            # continue
                            line = ''
                        line = line.strip()
                        sent = custom_tokenizer(line)
                        sent_filtered = []
                        for token in sent:
                            
                            # we treat word and numerals differently
                            # match numerals
                            res = re.findall(RE_NUM, token)
                            if res != []:
                                target = number_handler(token)
                                # we do not want nc to record ''
                                if target == '':
                                    continue

                                if type(target) is list:
                                    # ['u-32'] to ['u','-','32']
                                    # [1997/07] to ['1997','/','7']

                                    for k in target:
                                        if is_numeral(k):
                                            number = str(to_numeral(k))
                                            self.nc[number] = self.nc.get(number, 0) + 1
                                            sent_filtered.append(number)
                                        else:
                                            self.wc[k] = self.wc.get(k, 0) + 1
                                            sent_filtered.append(k)

                                elif is_numeral(target):
                                    # ['-32.000'] to ['-32']
                                    # prevent '-haha' like token, double check
                                    number = str(to_numeral(target))
                                    self.nc[number] = self.nc.get(number, 0) + 1
                                    sent_filtered.append(number)

                            else:
                                self.wc[token] = self.wc.get(token, 0) + 1
                                sent_filtered.append(token)

                output.write(bytes(' '.join(sent_filtered), 'utf-8').decode('utf-8') + '\n')

        output.close()
        print("filtering corpus done")

    def build_word_vocab(self, max_vocab=20000):
        print("start building vocab")
        wc_nounk = copy(self.wc)
        wc_nounk.__delitem__(self.unk)
        self.idx2word = [self.unk]  + sorted(wc_nounk, key=wc_nounk.get, reverse=True)[:max_vocab - 1]
        self.word2idx = {self.idx2word[idx]: idx for idx, _ in enumerate(self.idx2word)}
        self.vocab = set([word for word in self.word2idx])
        print("building vocab done")
