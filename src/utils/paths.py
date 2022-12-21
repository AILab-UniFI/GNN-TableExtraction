import os
from pathlib import Path
from dotenv import dotenv_values

config = dotenv_values(".env")

# Data folder
root = ''
if not os.path.exists(root):
    raise Exception("Define a valid ROOT path for your project -> /path/to/GNN-TableExtraction")
config['ROOT'] = root
DATA = Path(config['ROOT']) / 'data'
    
OUTPUT = Path(config['ROOT']) / 'output'
IMAGES = Path(config['ROOT']) / 'images'
# REPR = Path('/home/vivoli/TableEmbedding/outputs/tables-1m/repr_old')
SRC = Path(config['ROOT']) / 'src'
CONFIG = Path(config['ROOT']) / 'configs'

# -> Notebooks
NOTEBOOKS = Path(config['ROOT']) / 'notebooks'

# -> Data manipulation
EXTERNAL    = DATA / 'external'
RAW         = DATA / 'raw' 
PROCESSED   = DATA / 'processed'
INTERIM     = DATA / 'interim'
IMGS        = DATA / 'imgs'

# -> Output
CMS = OUTPUT / "confusion_matrices"

# -> Data -> External data get from url
PUBTABLES1M = EXTERNAL / "annotations"

# -> Data -> Raw data get from url and elaborated
PUBLAYNET_TEST  = RAW / "test"     # <- "publaynet-test"
PUBLAYNET_TRAIN = RAW / "train"   # <- "publaynet-train"

# -> Data -> interim (is intermediate data)
TABLES_REPR     = INTERIM / "tables"
TABLES_DICT     = INTERIM / 'dicts'
TABLES_TRAINS   = INTERIM / 'trains'
TABLES_PREPROCESS = INTERIM / 'preprocess'
REPR_FOLDER     = INTERIM / 'repr'

# -> Data -> interim -> tables
REPR_SKIPGRAM   = REPR_FOLDER / 'skipgrams'
# REPR_GMM        = TABLES_REPR / 'gmm'
# REPR_GMM_LOG    = TABLES_REPR / 'gmm_log'
REPR_LOGS       = REPR_FOLDER / 'logs'
REPR_MODELS     = REPR_FOLDER / 'models'
REPR_PTS        = REPR_FOLDER / 'pts'
# REPR_SOM        = TABLES_REPR / 'som'
# REPR_SOM_LOG    = TABLES_REPR / 'som_log'
REPR_TB_WRITER  = REPR_FOLDER / 'tb_writer'


# -> Data -> processed 
GRAPHS      = PROCESSED / 'graphs'
FEATURES    = PROCESSED / "features"

# -> Output
EXAMPLES    = OUTPUT / "examples"
WEIGHTS     = OUTPUT / "weights"
INFERENCE   = OUTPUT / "inference"

