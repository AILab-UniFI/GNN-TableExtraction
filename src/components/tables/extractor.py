import os
import pickle
import numpy as np
import pandas as pd
from glob import glob
from tqdm import tqdm
from typing import List, Union
from bs4 import UnicodeDammit
from src.utils.fs import create_folder_if_not_exists
from src.utils.nums import is_numeral, to_numeral, to_numeral_if_possible
from src.utils.paths import TABLES_PREPROCESS, TABLES_REPR, TABLES_TRAINS
from src.utils.strings import pymu_custom_tokenizer, to_representation

class DataItem:
    iword: int
    oword: List[int]
    iword_indicator: int
    iword_numeral: Union[float, None]
    oword_indicators: List[int]
    all_numerals: List[float]

class SkipGramData:
    iword: any
    left: List[any]
    right: List[any]

    def __init__(self, *args, **dicts):
        self.iword = dicts['iword']
        self.left = dicts['left']
        self.right = dicts['right']

    @property
    def data(self):
        return self.iword, self.left + self.right


class Extractor:
    iword: any
    left: List[any]
    right: List[any]
    window: int

    def __init__(self, name='Extractor', window=None, unk=None):
        assert window != None, ValueError('window can\'t be None !')
        assert unk != None, ValueError('unk can\'t be None !')
        self.name = name
        self.window = window
        self.unk = unk
        print(self.name)


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

    def extract(self, table_df, i, j):
        pass # raise ValueError("Method not implemented")

    def convert_items(self):
        pass # raise ValueError("Method not implemented")

    def get_items(self):
        pass # raise ValueError("Method not implemented")

    @property
    def data(self):
        return SkipGramData(self).data

class RhombusExtractor(Extractor):
    """
    Extractor that generates skipgram data as following:
    > df
    i\j➡️ 0  1  2  3  4
    ⬇   ------------------
    0  | /  a  b  c  d  |
    1  | A  0  1  2  3  |
    2  | B  4  5  6  7  |
    3  | C  8  9  10 11 |
    4  | D  12 13 14 15 |
        -----------------
    
    If the coordinates of the central element are (2,3) we have:
    > iword = 9
    > left = 8, 5
    > right= 13, 10
    """

    def extract(self, table_df, i, j):
        self.iword = table_df[i, j]
        self.left = table_df.iloc[[i, i-1], [j-1, j]]
        self.right = table_df.iloc[[i+1, i], [j, j+1]]
        return self.iword, self.left + self.right

    def extract_linear(self, table_df, i, j):
        max_i = table_df.shape[0]
        max_j = table_df.shape[1]
        return np.diag(table_df.iloc[[i, i-1, i, (i+1)%max_i, i], [j-1, j, j, j, (j+1)%max_j]]).T

    def get_item_linear(self, words):
        """
        :param words:
        :return:
        """
        words = ' '.join([UnicodeDammit(el).unicode_markup if el!=b'' else '' for el in words])
        words = words.strip()
        words = pymu_custom_tokenizer(words, avoid=self.unk)
         
        central = (self.window)
        steps = len(words)-(self.window*2)
        if steps<=0:
            return []

        assert steps>0, ValueError("Window too large for this Extractor!")
        items = [[None, [], 0, None, [0] * 2 * self.window, []] for _ in range(0, steps)]
        
        #! ATTENTION -> we go only with representations
        
        for i in range(0, steps):
            iword = words[central+i]
            owords = words[0+i:central+i] + words[central+i+1:central+i+1+self.window]
            
            items[i][0] = self.repr2idx.get(to_representation(iword), 0)
            items[i][2] = 1

            for j in range(len(owords)):
                oword = owords[j]
                items[i][1].append(self.repr2idx.get(to_representation(oword), 0))
                items[i][4][j] = 1

        return items


    def convert_items(self, MAXDATA=200000):
        print("[MUST] loading built information")
        assert self.wc != None, ValueError('ERROR wc is None')
        assert self.vocab != None, ValueError('ERROR vocab is None')
        assert self.idx2word != None, ValueError('ERROR idx2word is None')
        assert self.word2idx != None, ValueError('ERROR word2idx is None')
        
        assert self.nc != None, ValueError('ERROR nc is None')
        
        assert self.rc != None, ValueError('ERROR rc is None')
        assert self.repr_vocab != None, ValueError('ERROR rc is None')
        assert self.idx2repr != None, ValueError('ERROR rc is None')
        assert self.repr2idx != None, ValueError('ERROR rc is None')

        create_folder_if_not_exists(TABLES_TRAINS)

        print("converting corpus...")
        step = 0
        data = []
        batches = 0
        # Very Important Arguments
        # todo if works -> '/tables-batch-*.dat'
        path_list = glob(str(TABLES_REPR) + '/tables-batch-*.dat')
        length = len(path_list)
        tqdm_list = tqdm(path_list, desc=f'{step / 1000}kth')
        
        for idx_, tables_file in enumerate(tqdm_list):
            #! to remove 
            # if idx_ > 1: break

            # print()
            # print(f'[ { int(idx_ / length * 100)} % ] - 0kth', end=' ')
        
            tables_pairs = pickle.load(open(tables_file, 'rb'))
        
            for tables in tables_pairs:
                _, value_table = tables

                step += 1
                # if not step % 1000:
                #     print(f'{step // 1000}kth', end=' ')

                # add row of <UNK_CELL> to pandas
                value_table = value_table.append(pd.Series([self.unk.encode('utf-8')]*value_table.shape[1]).T, ignore_index=True)
                # add column of <UNK_CELL> to pandas
                value_table = value_table.T.append(pd.Series([self.unk.encode('utf-8')]*value_table.shape[0]).T, ignore_index=True).T


                for i in range(value_table.shape[0]):
                    for j in range(value_table.shape[1]):
                        # iword, owords = self.extract(value_table, i, j)
                        words = self.extract_linear(value_table, i, j)
                        # item = self.get_item(iword, owords)
                        items = self.get_item_linear(words)
                        # data.append(item)    
                        data.extend(items)

                if len(data) > MAXDATA:
                    batches += 1
                    pickle.dump(data, open(TABLES_TRAINS / f'train-batch-{batches}.dat', 'wb'))
                    print(f'Saving train-batch-{batches}.dat')
                    data = [] # reset the data
                    #! to remove 
                    # break

            batches += 1
            pickle.dump(tables, open(TABLES_TRAINS / f'train-batch-{batches}.dat', 'wb'))

        print("")
        print("conversion done")


class HalfRhombusExtractor(Extractor):
    """
    Extractor that generates skipgram data as following:
    > df
    i\j➡️ 0  1  2  3  4
    ⬇ ------------------
    0  | /  a  b  c  d  |
    1  | A  0  1  2  3  |
    2  | B  4  5  6  7  |
    3  | C  8  9  10 11 |
    4  | D  12 13 14 15 |
        -----------------

    If the coordinates of the central element are (2,3) we have:
    > iword = 9
    > left = C, 8
    > right= 5, b
    """
    
    def extract(self, table_df, i, j):
        self.iword = table_df[i, j]
        self.left = table_df.iloc[i, [0, j]]
        self.right = table_df.iloc[[i-1, 0], j]
        return self.iword, self.left + self.right

class LinearExtractor(Extractor):
    """
    Extractor that generates skipgram data as following:
    > df
    i\j➡️ 0  1  2  3  4
    ⬇ ------------------
    0  | /  a  b  c  d  |
    1  | A  0  1  2  3  |
    2  | B  4  5  6  7  |
    3  | C  8  9  10 11 |
    4  | D  12 13 14 15 |
        -----------------

    If the coordinates of the central element are (2,3) we have:
    > iword = 8
    > left = -, C
    > right= 9, 10
    """
    
    def extract(self, table_df, i, j):
        self.iword = table_df[i, j]
        self.left = table_df.iloc[i, [j-2, j-1]]
        self.right = table_df.iloc[i, [j+1, j+2]]
        return self.iword, self.left + self.right

class ProfExtractor(Extractor):
    """
    Extractor that generates skipgram data as following:
    > df
    i\j➡️ 0  1  2  3  4
    ⬇ ------------------
    0  | /  a  b  c  d  |
    1  | A  0  1  2  3  |
    2  | B  4  5  6  7  |
    3  | C  8  9  10 11 |
    4  | D  12 13 14 15 |
        -----------------

    If the coordinates of the central element are (2,3) we have:
    > iword = 9
    > left = B, 5, b, C
    > right= b, D, 13, b
    """
    def __init__(self, name='stdSG', window=None, unk=None) -> None:
        super(ProfExtractor, self).__init__(name, window, unk)
    
    def extract(self, table_df, i, j):
        self.iword = table_df.iloc[i, j]
        self.left = np.diag(table_df.iloc[[i-1, i-1, 0, i], [0, j, j, 0]])
        self.right = np.diag(table_df.iloc[[0, i+1, i+1, 0], [j, 0, j, j]])
        return self.iword, np.concatenate((self.left, self.right)).T

    def extract_linear(self, table_df, i, j):
        max_i = table_df.shape[0]
        return np.diag(table_df.iloc[[i-1, i-1, 0, i, i, 0, (i+1)%max_i, (i+1)%max_i, 0], [0, j, j, 0, j, j, 0, j, j]]).T

    def get_item(self, iword, owords):
        # TODO: manage words with numerals
        # TODO: manage numerals with expression
        # TODO: manage words and numerals
        # ? USE get_item from ``Learning Numeral Embeddigs''
        # raise Exception('Non implemented method!')

        """
        form a proper data structure
        :param iword:
        :param owords:
        :return:
        """
        item = [None, [], 0, None, [0] * 2 * self.window, []]
        # [
        #   iword,
        #   [list of owords],
        #   0 or 1, indicator of iwords,
        #   None if iword is a token, numeral float if iword is a numeral,
        #   [one-hot indicator of owords],
        #   [list of numerals]
        # ]
        #
        # For example: if She is the center word and the window size is 2
        # oh , (She) is 1.67 m
        # [12, [99, 4, 5, 0], 0, None, [0,0,0,1], [1.67]]

        if is_numeral(iword):
            item[0] = self.word2idx[self.unk]
            item[2] = 1
            item[3] = to_numeral(iword)

        else:
            item[0] = self.word2idx[iword]

        for j in range(len(owords)):
            flag, oword = to_numeral_if_possible(owords[j])

            if flag:
                item[1].append(self.word2idx[self.unk])
                item[4][j] = 1
                item[5].append(oword)
            else:
                item[1].append(self.word2idx.get(oword, 0))

        return item


    def get_item_linear(self, words):
        """
        :param words:
        :return:
        """
        words = ' '.join([UnicodeDammit(el).unicode_markup if el!=b'' else '' for el in words])
        words = words.strip()
        words = pymu_custom_tokenizer(words, avoid=self.unk)
         
        central = (self.window)
        steps = len(words)-(self.window*2)
        if steps<=0:
            return []

        assert steps>0, ValueError("Window too large for this Extractor!")
        items = [[None, [], 0, None, [0] * 2 * self.window, []] for _ in range(0, steps)]
        # [
        #   iword, #! None for number, otherwise  
        #   [list of owords], #! None for numbers
        #   0 or 1 or 2, indicator of iwords, #! 0 wc, 1 nc, 2 rc
        #   None if iword is a token and a repr, numeral float if iword is a numeral #! None if repr
        #   [one-hot indicator of owords], #! 0 wc, 1 nc, 2 rc
        #   [list of numerals],
        # ]
        #
        # For example: if She is the center word and the window size is 2
        # oh , (She) is 1.67 m
        # [12, [99, 4, 5, 0], 0, None, [0,0,0,1], [1.67]]

        for i in range(0, steps):
            iword = words[central+i]
            owords = words[0+i:central+i] + words[central+i+1:central+i+1+self.window]
            #! NO NUMERALS
            # if is_numeral(iword):
            #     items[i][0] = self.repr2idx[self.unk]
            #     items[i][2] = 1
            #     items[i][3] = to_numeral(iword)

            # else:
            if self.word2idx.get(iword, None) != None:
                items[i][0] = self.word2idx[iword]
                # items[i][2] = 0
                # items[i][3] = None
            else:
                items[i][0] = self.repr2idx.get(to_representation(iword), 0)
                items[i][2] = 1
                # items[i][3] = None

            for j in range(len(owords)):
                #! NO NUMERALS
                # flag, oword = to_numeral_if_possible(owords[j])
                # if flag:
                #     items[i][1].append(self.repr2idx[self.unk])
                #     items[i][4][j] = 1
                #     items[i][5].append(oword)
                # else:
                oword = owords[j]
                if self.word2idx.get(oword, None) != None:
                    items[i][1].append(self.word2idx[oword])
                    # items[i][4][j] = 0
                else:
                    items[i][1].append(self.repr2idx.get(to_representation(oword), 0))
                    items[i][4][j] = 1

        return items


    def convert_items(self, MAXDATA=200000):
        print("[MUST] loading built information")
        assert self.wc != None, ValueError('ERROR wc is None')
        assert self.vocab != None, ValueError('ERROR vocab is None')
        assert self.idx2word != None, ValueError('ERROR idx2word is None')
        assert self.word2idx != None, ValueError('ERROR word2idx is None')
        
        assert self.nc != None, ValueError('ERROR nc is None')
        
        assert self.rc != None, ValueError('ERROR rc is None')
        assert self.repr_vocab != None, ValueError('ERROR rc is None')
        assert self.idx2repr != None, ValueError('ERROR rc is None')
        assert self.repr2idx != None, ValueError('ERROR rc is None')


        print("converting corpus...")
        step = 0
        data = []
        batches = 0
        # Very Important Arguments
        # todo check if works -> path_list = glob(join(self.tables_dir)+'/tables-batch-*.dat')
        path_list = glob(str(TABLES_REPR) + '/tables-batch-*.dat')
        length = len(path_list)
        
        for idx_, tables_file in enumerate(path_list):
            #! to remove 
            # if idx_ > 1: break

            print()
            print(f'[ { int(idx_ / length * 100)} % ] - 0kth', end=' ')
        
            tables_pairs = pickle.load(open(tables_file, 'rb'))
        
            for tables in tables_pairs:
                _, value_table = tables

                step += 1
                if not step % 1000:
                    print(f'{step // 1000}kth', end=' ')

                # add row of <UNK_CELL> to pandas
                value_table = value_table.append(pd.Series([self.unk.encode('utf-8')]*value_table.shape[1]).T, ignore_index=True)
                # add column of <UNK_CELL> to pandas
                value_table = value_table.T.append(pd.Series([self.unk.encode('utf-8')]*value_table.shape[0]).T, ignore_index=True).T


                for i in range(value_table.shape[0]):
                    for j in range(value_table.shape[1]):
                        # iword, owords = self.extract(value_table, i, j)
                        words = self.extract_linear(value_table, i, j)
                        # item = self.get_item(iword, owords)
                        items = self.get_item_linear(words)
                        # data.append(item)    
                        data.extend(items)

                if len(data) > MAXDATA:
                    batches += 1
                    pickle.dump(data, open(TABLES_TRAINS / f'train-batch-{batches}.dat', 'wb'))
                    print(f'Saving train-batch-{batches}.dat')
                    data = [] # reset the data
                    #! to remove 
                    # break

            batches += 1
            pickle.dump(tables, open(TABLES_TRAINS / f'train-batch-{batches}.dat', 'wb'))

        print("")
        print("conversion done")

    def convert_skipgram(self, MAXDATA=200000):
        if None:
            print("[SKIPP] loading built information")
            print("converting corpus...")
            step = 0
            data = []
            batches = 0
            # Very Important Arguments
            path_list = glob(os.path.join(self.tables_dir)+'/tables-batch-*.dat')
            for tables_file in path_list:
                tables_pairs = pickle.load(open(tables_file, 'rb'))
                for tables in tables_pairs:
                    _, value_table = tables
                    step += 1
                    if not step % 1000:
                        print(f"working on {step // 1000}kth line")

                    for i in value_table.shape[0]:
                        for j in value_table.shape[1]:
                            iword, owords = self.extractor.extract(value_table, i, j).data
                            item = (iword, owords)
                            data.append(item)    

                    if len(data) > MAXDATA:
                        batches += 1
                        pickle.dump(data, open(os.path.join(self.skipgram_dir, f'skipgram-batch-{batches}.dat'), 'wb'))
                        print(f'Saving train-batch-{batches}.dat')
                        data = [] # reset the data

                batches += 1
                pickle.dump(tables, open(os.path.join(self.skipgram_dir, f'skipgram-batch-{batches}.dat'), 'wb'))

            print("")
            print("conversion done")
        else:
            return None

    

