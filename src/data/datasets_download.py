import os
from attrdict import AttrDict
import wget
import yaml


from dotenv import dotenv_values
from src.utils.fs import create_folder
from src.utils.paths import EXTERNAL, CONFIG

if __name__ == "__main__":

    with open(CONFIG / 'db.yaml') as fileobj:
        config = AttrDict(yaml.safe_load(fileobj))

    config.DATABASE.publaynet.pdfs
    
    secret = dotenv_values("secret.env")
    config['secret_token'] = secret["SAS_TOKEN_HERE"]

    create_folder(EXTERNAL)
    
    #####################
    #! DOWNLOAD datasets  
    #####################
    
    # PubLayNet PDFs
    wget.download(
        config.DATABASE.publaynet.pdfs, 
        out=EXTERNAL)
        
    # PubLayNet Annotations
    wget.download(
        config.DATABASE.publaynet.annotations,
        out=EXTERNAL)

    # PubTables1Million
    os.system(
        f'azcopy copy {config.DATABASE.pubtables1m}?{config.secret_token} "{EXTERNAL}" --recursive'
        )
        
    #####################
    #! ORGANISE folders
    #####################
    
    # Please follow the following files structure for data folder
    
    # data/
    #   external/
    #       annotations/            <- pubtables-1m
    #           PMC2376063_tables.json
    #           PMC2370909_tables.json
    #           ...
    #       publaynet-test.json     <- publaynet-test
    #       publaynet-train.json    <- publaynet-train
    #   raw/
    #       publaynet-train/        <- publaynet-train
    #           PMC2376063_00000.pdf
    #           ...
    #       publaynet-test/         <- publaynet-test
    #           PMC2370909_00000.pdf
    #           ...
    #   processed/
    #       .
    #   interim/
    #       .