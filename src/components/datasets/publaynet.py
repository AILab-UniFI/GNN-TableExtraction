import os
import json
from tqdm import tqdm
from src.utils.const import Categories_names
from src.utils.paths import PUBTABLES1M

####################
#! PUBLAYNET Dataset
####################

def pln_preprocess(annotation_path):
    pln_papers = dict()
    idx_pages = dict()

    pt1m_filenames = {
        f.split("_")[0]: True for f in os.listdir(PUBTABLES1M) if os.path.isfile(PUBTABLES1M / f)
    }
    
    assert pt1m_filenames, "PubTables1M folder / path is empty."

    with open(annotation_path, "r") as fp:
        annotations = json.load(fp)

        for img in tqdm(annotations["images"], desc=" - getting indices"):
            paper = img["file_name"].split("_")[0]
            if pt1m_filenames.get(paper, False):
                app = pln_papers.get(paper, {"pages": [], "annotations": []})
                app["pages"].append(img["file_name"])
                pln_papers[paper] = app
                idx_pages[img["id"]] = img["file_name"]

        pages_idx = dict(sorted(idx_pages.items()))
        pages_idx_keys = {id: True for id in pages_idx.keys()} # for fast control later

        for ann in tqdm(annotations["annotations"], desc=" - getting annotations"):
            if pages_idx_keys.get(ann["image_id"], False):
                page_name = pages_idx[ann["image_id"]]
                paper_name = page_name.split("_")[0]
                pln_papers[paper_name]["annotations"].append(ann)

    # invert key / values in idx_pages dict
    pages_idx = {v: k for k, v in idx_pages.items()}

    return pln_papers, pages_idx

def pln_filter_tables(papers, pages_idx):
    num_tables = dict()

    for k, v in tqdm(papers.copy().items(), desc=" - filtering papers"):
        # remove papers from dictionary with no tables annotations
        if 4 not in [ann["category_id"] for ann in v["annotations"]]:
            del papers[k]
        else:
            # count num_tables per each remained page
            # {..., pag_1: #tab, pag_2: #tab, pag_3: #tab, ...}
            for page in v["pages"]:
                # PMC5928050_00004
                file_name = page.split(".")[0]
                # PMC5928050, 4
                file_name, page_number = file_name.split("_")
                page_number = int(page_number)

                page_tbl_count = 0
                page_id = pages_idx[page]
                for ann in v["annotations"]:
                    if ann["image_id"] == page_id and ann["category_id"] == 4:
                        page_tbl_count += 1

                num_tables[file_name] = num_tables.get(file_name, {})
                num_tables[file_name][page_number] = page_tbl_count

    return num_tables