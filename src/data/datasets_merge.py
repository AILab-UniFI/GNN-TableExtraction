from distutils.log import debug
import os
import json

from src.components.datasets import \
    diff_pln_pt1m, \
    get_not_annotated_tables, \
    merge_annotations, \
    pt1m_preprocess, \
    pln_filter_tables, \
    pln_preprocess

from src.utils.paths import EXTERNAL, RAW
from src.utils.const import Categories_names, categories_colors
from src.components.datasets.utils import print_annotations


"""
        The resulting structure of data folder must be like follow

*       data/
*           external/
?               annotations/            <- previously done (pubtab-1m)
?               publaynet-test.json     <- previously done (publaynet)
?               publaynet-train.json    <- previously done (publaynet)
*           raw/
?               publaynet-train/        <- previously done (publaynet)
?               publaynet-test/         <- previously done (publaynet)
!               test.json       <- We'll create this here  (pubtab-1m + publaynet)
!               train.json      <- We'll create this here  (pubtab-1m + publaynet)
*           processed/
*               .
*           interim/
*               .
""" 

####################
#* MAIN
####################

if __name__ == "__main__":
    
    SPLITS = ["test", "train"]
    debug = True # set to False to do not print annotation examples

    for phase, split in enumerate(SPLITS):

        print("")
        print(f"Processing {split}:")

        ### PATHS ###

        publaynet_path = EXTERNAL / f"publaynet-{split}.json"  # PubLayNet Annotations
        annotation_path = EXTERNAL / "annotations"          # PubTable-1M Annotations
        output_path = RAW / f"{split}.json"             # final json to be used

        ### PubLayNet processing ###

        papers, pages_idx = pln_preprocess(publaynet_path)
        num_tables = pln_filter_tables(papers, pages_idx)

        ### PubTable-1M processing ###

        split_dict = pt1m_preprocess(papers, annotation_path)

        ### MERGE processing ###
        
        differences = diff_pln_pt1m(num_tables, split_dict)
        
        todiscard = get_not_annotated_tables(differences)

        refactored_papers = merge_annotations(papers, pages_idx, split_dict, todiscard)

        ### End file -> saving ###
        # {"id": 0, "name": "OTHER", "color": (127, 127, 127)},
        refactored_dict = {
            "categories": [{'id':category.value, 'name': category.name, 'color': categories_colors[category.value] } for category in Categories_names],
            "papers": refactored_papers
        }

        with open(output_path, "w") as f:
            f.write(json.dumps(refactored_dict))
        
        if debug:
            with open(output_path, "r") as ann:
                annotations = json.load(ann)
                print_annotations(annotations, split)
