
from src.components.graphs.loader import Papers2Graphs
from src.utils.paths import CONFIG


if __name__ == '__main__':
    
    # loading data
    data = Papers2Graphs(path_to_config= CONFIG / 'graph' / "graph.yaml")
    
    config = data.get_config()
    # create islands indexes for graphs
    data.calculate_islands(num_graphs=config.TRAINING.num_graphs)