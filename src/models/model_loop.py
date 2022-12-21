from datetime import datetime
import argparse
import yaml

from attrdict import AttrDict
from src.components.graphs.utils import parse_args_ModelLoop

from src.utils.paths import CONFIG
from src.components.graphs.loader import Papers2Graphs
from src.models.model_train import train
from src.models.model_predict import test

def main(config):
    

    if not config.BASE.name:
        config.BASE.name = datetime.now().strftime('%y%m%d.%H%M%S')
    
    if config.BASE.train:
        # loading data
        train_data = Papers2Graphs(config = config)
        
        # modifying data
        config = train_data.get_config()
        train_data.modify_graphs(num_graphs=config.TRAINING.num_graphs)
        train_data.get_infos()
        
        # training
        train(train_data, config, config.BASE.name)
    
    if config.BASE.test:
        # loading data
        test_data = Papers2Graphs(config = config, test=True)
        
        # modifying data
        config = test_data.get_config()
        test_data.modify_graphs()
        test_data.get_infos()
        
        # testing
        test(test_data, config, config.BASE.name, n_show=10)
    
    if config.BASE.infer:
        print("COMING SOON!")

if __name__ == "__main__":

    # loading data
    with open(CONFIG / 'graph' / "empty.yaml") as fileobj:
        config = AttrDict(yaml.safe_load(fileobj))
        config = AttrDict(parse_args_ModelLoop(config))
    
    print(config)
    
    main(config)