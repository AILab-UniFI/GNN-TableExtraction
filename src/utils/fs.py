import os
import json
from types import SimpleNamespace

############
## FOLDER
## MANAGER
############


def create_folder(folder_path, debug=False):
    try:
        os.makedirs(folder_path)
    except OSError as e:
        print(f"Creation of the directory {folder_path} failed: -> {e}")
    else:
        if debug:
            print("Successfully created the directory %s " % folder_path)

def create_folder_if_not_exists(folder_path, debug=False):
    if not os.path.exists(folder_path):
        create_folder(folder_path, debug)
    else:
        if debug: print(f"Folder {folder_path} already exists")

def get_files_name(path, extentions=None, group=False, reverse=False):
    files_name = sorted([
        file for file in os.listdir(path) 
            if os.path.isfile(os.path.join(path, file))
    ], reverse=reverse)
    
    if extentions:
        # filter for extentions
        return [file for file in files_name if files_name.split('.')[-1] in extentions]

    if group:
        # group by name
        from itertools import groupby
        files_name = {key: [name for _, name in val] for key, val in groupby([(file.split('_')[0], file) for file in files_name], lambda x: x[0])}
    
    return files_name

############
## FILES
## READERS
############

def read_json(path):
    with open(path, "r", encoding="utf-8") as f:
        page_tokens = json.load(f, object_hook=lambda d: SimpleNamespace(**d))
    return page_tokens