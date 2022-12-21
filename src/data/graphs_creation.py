####################
#* MAIN
####################

# g = dgl.remove_self_loop(g)
# g = dgl.to_simple(g)
# g = dgl.to_bidirected(g)

import datetime
from src.components.graphs.loader import Papers2Graphs
from src.utils.fs import create_folder_if_not_exists
from src.utils.paths import IMAGES

if __name__ == "__main__":
    
    SPLITS = ["test", "train"]
    debug = True # set to False to do not print annotation examples

    now = datetime.now()
    name_time = now.strftime("%D-%M-%Y_%H-%M-%S")

    

    for phase, split in enumerate(SPLITS):
        
        creation_path = IMAGES / split/ 'creation'+name_time
        create_folder_if_not_exists(creation_path)

        data = Papers2Graphs(test = 'test' == split)
        data.get_infos(show_sample=debug, folder=creation_path)
        
        # todo add modify_graphs for remnove_islands here
        # modify_graph for remnove_islands
        
        # todo add edge_features calculation here
        # edge_feature_calculus