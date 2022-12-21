#!/usr/bin/env python
# coding: utf-8

import yaml
import os
import pickle
import argparse
from typing import List, Union
from attrdict import AttrDict
from src.components.tables.types import CountDict
from src.utils.fs import create_folder, create_folder_if_not_exists, read_json
from src.components.tables.tables import Table
from src.utils.dicts import append
from src.utils.nums import truncate
from src.utils.paths import CONFIG, PUBTABLES1M, TABLES_DICT, TABLES_REPR


def create_tables(all_files: List[str], config: AttrDict) -> None:
    """Creates multiple tables files from Pub-Tables-1M dataset

    Args:
        all_files (List[str]): list of files' names contained in 'annotation' path
        config (AttrDict): configuration file
    """
    
    create_folder_if_not_exists(TABLES_REPR)
    assert os.path.exists(TABLES_REPR)

    batches = 0
    tables = []
    dicts = CountDict({})

    try:
        for tables_idx, tables_json in enumerate(all_files):
            if not tables_idx % 1000:
                print(f"working on {tables_idx // 1000}kth table")

            if config.GENERAL.debug:
                if tables_idx > 10: break
                print()
                print(f"[ { truncate(tables_idx / tables_length, 6)} % ]", tables_json, end=" - ")

            if tables_json[-4:] != 'json': continue
            page_tokens = read_json(PUBTABLES1M / tables_json)

            if config.GENERAL.debug: print(len(page_tokens), end=" - ")  

            for index, table_json in enumerate(page_tokens):

                table = Table(table_json)
                if config.GENERAL.debug:print(table.table_number, end=" ")
                
                table.table_extraction(config)
                tables.append(table.get_matrices()) # .get_skipgram(extactor, window=None)
                dicts.append(table._keys_dict)
                
                if len(tables) > config.GENERAL.max_data:
                    batches += 1

                    pickle.dump(tables, open(TABLES_REPR / f'tables-batch-{batches}.dat', 'wb'))
                    print(f'Saving tables-batch-{batches}.dat')
                    tables = [] # reset the tables        

    except:
        print("error occured!")

    finally:
        batches += 1
        
        pickle.dump(tables, open(TABLES_REPR / f'tables-batch-{batches}.dat', 'wb'))

        create_folder_if_not_exists(TABLES_DICT)
        assert os.path.exists(TABLES_DICT)

        pickle.dump(dicts, open(TABLES_DICT / f'dict-repr-vocab.dat', 'wb'))


# %%

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--iter', default=100, type=Union[int, float], help="number for limit iterations") # default = float('inf')

    with open(CONFIG / 'tables.yaml') as fileobj:
        config = AttrDict({ **yaml.safe_load(fileobj), **parser.parse_args({}).__dict__ })

    assert os.path.exists(PUBTABLES1M)

    all_files = os.listdir(PUBTABLES1M)
    tables_length = len(all_files)
    print(tables_length)

    create_tables(all_files, config)





        