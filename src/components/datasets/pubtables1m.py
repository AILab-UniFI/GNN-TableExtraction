import os
from tqdm import tqdm
from src.components.datasets.utils import get_number, read_json, tables_to_pages


####################
#! PUBTAB-1M Dataset
####################

def pt1m_preprocess(papers, JSON_PATH):

    filenames = papers.keys()
    dict_presence = dict.fromkeys(filenames, True)

    dict_ = dict()
    for t in tqdm(dict_presence, desc=" - getting table infos"):

        # print(f" - getting table infos: {int((t_idx+1)/len(dict_presence)*100)}%", end="\r")

        tables_json = t + "_tables.json"
        page_tokens = read_json(JSON_PATH / tables_json)

        names = papers[t]
        names = names["pages"]
        pages_idxs = []
        for n in names:
            new_n = get_number(n)
            pages_idxs.append(new_n)

        pages = tables_to_pages(page_tokens, pages_idxs)
        dict_[t] = pages

    return dict_