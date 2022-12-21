import random
import time
import sys
import dgl
import os
import json
from dgl.data import DGLDataset
from numpy import True_
from pdf2image.pdf2image import convert_from_path
import torch
from tqdm import tqdm
from dgl import save_graphs, load_graphs
from dgl.data.utils import save_info, load_info
from random import randint
from src.components.graphs.labels import LableModification
from src.utils.decorators import timeit

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.dirname(SCRIPT_DIR))
from src.components.graphs.utils import center, distance, normalize
from src.utils.paths import CONFIG, FEATURES, IMAGES, PUBLAYNET_TEST, PUBLAYNET_TRAIN, RAW, GRAPHS
from src.components.graphs.utils import fast_get_statistics
from src.components.graphs.builder import GraphBuilder

class Papers2Graphs(DGLDataset):
    """ Template for customizing graph datasets in DGL.
    """
    def __init__(self,
                 num_papers = None,
                 config = None,
                 path_to_config = None,
                 raw_dir = RAW,
                 save_dir = GRAPHS,
                 test = False):
        
        self.num_papers = num_papers
        self.test = test

        # assert config != None, 'ERROR -> config file can\'t be NONE'
        
        self.gb = GraphBuilder(config=config, path_to_config=path_to_config)
        self.label_tranformer = LableModification(config=self.gb.get_config())
        self.stats = {}
        self.graphs = []
        self.num_classes = 0
        self.pages = []
        self.seed = self.get_config().PREPROCESS.seed
        random.seed(self.seed)
        
        if self.test: 
            self.collection = 'test'
            self.data_path = PUBLAYNET_TEST
            self.annotations_path = RAW / "test.json"
            self.only_tables = self.get_config().DLTEST.only_tables
        else:
            self.collection = 'train'
            self.data_path = PUBLAYNET_TRAIN
            self.annotations_path = RAW / "train.json"
            self.only_tables = self.get_config().DLTRAIN.only_tables
        
        super(Papers2Graphs, self).__init__(name = '', raw_dir = raw_dir, save_dir = save_dir, verbose = False)

    def process(self):
        # processing raw data from raw_dir, and saving graph/s to save_dir
        all_labels = []
            
        with open(self.annotations_path, 'r') as ann:
            data = json.load(ann)
            j = 0
            for paper in tqdm(list(data['papers'].keys())[:self.num_papers], desc = f"processing {self.collection} data"):
                pages = data['papers'][paper]['pages']
                annotations = data['papers'][paper]['annotations']
                
                for i, page in enumerate(pages):
                    
                    if self.only_tables:
                        if "TABLE" not in [ann[2] for ann in annotations[i]]:
                            continue
                        
                    g, bboxs, texts = self.gb.get_graph(self.data_path / page, annotations[i], self.get_config().PREPROCESS.mode)
                    if g is not None:
                        self.graphs.append(g)
                        self.pages.append({'page': page, 'bboxs' : bboxs, 'texts' : texts})
                        all_labels.extend(g.ndata['label'].tolist())
        
        # todo -> change and take lenght of classes
        self.num_classes = max(all_labels) + 1
        numbers, percentages = fast_get_statistics(all_labels, self.num_classes)
        self.stats['numbers'] = numbers
        self.stats['percentages'] = percentages

    def __getitem__(self, idx):
        return self.graphs[idx]

    def __len__(self):
        return len(self.graphs)

    def save(self):
        # save processed data to directory `self.save_path`
        if not os.path.exists(self.save_dir): 
            os.mkdir(self.save_dir)
            os.mkdir(os.path.join(self.save_dir, 'train'))
            os.mkdir(os.path.join(self.save_dir, 'test'))
        graph_path = f"{self.save_path}{self.collection}/"+f"{self.get_config().PREPROCESS.mode}".upper()+f"_n{self.num_papers}.bin"
        info_path = f"{self.save_path}{self.collection}/n{self.num_papers}_INFO.pkl"
        if self.num_papers == None:
            graph_path = f"{self.save_path}{self.collection}/"+f"{self.get_config().PREPROCESS.mode}".upper()+".bin"
            info_path = f"{self.save_path}{self.collection}/INFO.pkl"
        save_graphs(graph_path, self.graphs)
        # save other information in python dict
        save_info(info_path, {'num_classes': self.num_classes, 
                              'stats': self.stats,
                              'pages': self.pages})

    def load(self):
        # load processed data from directory `self.save_path`
        print(f"\nLoading {self.collection} data ... ", end='')
        start_time = time.time()
        graph_path = f"{self.save_path}{self.collection}/"+f"{self.get_config().PREPROCESS.mode}".upper()+f"_n{self.num_papers}.bin"
        info_path = f"{self.save_path}{self.collection}/n{self.num_papers}_INFO.pkl"
        if self.num_papers == None:
            graph_path = f"{self.save_path}{self.collection}/"+f"{self.get_config().PREPROCESS.mode}".upper()+".bin"
            info_path = f"{self.save_path}{self.collection}/INFO.pkl"
            
        self.graphs, _ = load_graphs(graph_path)
        self.num_classes = load_info(info_path)['num_classes']
        self.stats = load_info(info_path)['stats']
        self.pages = load_info(info_path)['pages']
        print("took %ss" % (round(time.time() - start_time, 2)))

    def has_cache(self):
        # check whether there are processed data in `self.save_path`
        graph_path = f"{self.save_path}{self.collection}/"+f"{self.get_config().PREPROCESS.mode}".upper()+f"_n{self.num_papers}.bin"
        info_path = f"{self.save_path}{self.collection}/n{self.num_papers}_INFO.pkl"
        if self.num_papers == None:
            graph_path = f"{self.save_path}{self.collection}/"+f"{self.get_config().PREPROCESS.mode}".upper()+".bin"
            info_path = f"{self.save_path}{self.collection}/INFO.pkl"
        return os.path.exists(graph_path) and os.path.exists(info_path)
    
    def get_infos(self, show_sample=False, folder=None, converted=False):

        assert folder != None, 'ERROR: folder can\'t be None'
        
        # print information about data
        print(" -> {} data: {} pages".format(self.collection, self.__len__()))
        print(f" -> only tables: {self.only_tables}")
        if self.test:
            print(" -> remove islands: {}".format(self.get_config().DLTEST.remove_islands))
        else:
            print(" -> remove islands: {}".format(self.get_config().DLTRAIN.remove_islands))
        print(" -> classes %:", self.stats['percentages'])
        print(" -> classes #:", self.stats['numbers'])
        
        if show_sample:
            rand_ids = [randint(0, self.__len__() - 1) for _ in range(10)]
            # rand_id = randint(0, self.__len__() - 1)
            for rand_id in rand_ids:
                graph = self.__getitem__(rand_id)
                #graph = self.__getitem__(17)
                graph = dgl.to_simple(graph)
                page = self.pages[rand_id]['page']
                # page = 'PMC2570569_00004.pdf'
                paper = page.split("_")[0]
                bboxs = self.pages[rand_id]['bboxs']
                # bboxs = self.pages[17]['bboxs']
                with open(self.annotations_path, 'r') as ann:
                    data = json.load(ann)
                    page_idx = data['papers'][paper]['pages'].index(page)
                    annotations = data['papers'][paper]['annotations'][page_idx]
                self.gb.print_graph(graph, annotations, bboxs, self.data_path / page, folder / f"{self.collection}_{rand_id}.jpg", label_converted=self.label_tranformer, converted=converted)

    def calculate_islands(self, num_graphs = None, debug = False):
        print(f"\nCalculating islands (to remove) ... ", end='\n')
        start_time = time.time()

        tqdm_idx_graph = tqdm(range(0, num_graphs), desc=f'- looping into graphs file')

        split = "TEST" if self.test else "TRAIN"
        remove_islands = self.gb.get_config()[f'DL{split}']['remove_islands']
        
        # todo -> consider the path as: graphs/ train/ khop/ REMOVE_IDXS.pkl
        mode = self.get_config().PREPROCESS.mode
        khop = self.get_config().PREPROCESS.range_island
        remove_path = os.path.join(GRAPHS, self.collection,  f'{str(mode).upper()}_REMOVE_IDXS_{str(khop)}.pkl')

        all_to_remove = []
        to_remove = []
        
        if remove_islands and not os.path.isfile(remove_path): 

            for g in tqdm_idx_graph:
            
                print(f" -> Modify edges")
                #? calculate or load node idxs to remove
                to_remove = self.gb.fast_remove_islands(self.graphs[g]) # evaluate
                # graph.remove_nodes(to_remove)
                all_to_remove.append(to_remove)

        if not os.path.isfile(remove_path):
            save_info(remove_path, all_to_remove)
                
        print()
        print("took %ss" % (round(time.time() - start_time, 2)))
        return        

    def modify_graphs(self, num_graphs = None, debug = False):
        
        print(f"\nModifying graphs ... ", end='\n')
        
        if num_graphs == None or num_graphs > self.__len__():
            print(" -> Using all available graphs : {}".format(self.__len__()))
            num_graphs = self.__len__()
        
        print(f" -> SKIPPING - Adding features - LEFT FOR RUNTIME")
        # print(f" -> Adding features")
        start_time = time.time()
        split = "TEST" if self.test else "TRAIN"

        # for f, feature_name in enumerate(self.get_config().PREPROCESS.features):
        #     print(" -> Loading {} | {}/{} | {} seconds".format(feature_name, f+1, len(self.get_config().PREPROCESS.features), round(time.time() - start_time, 2)), end='\r')
            
        #     if feature_name == 'SCIBERT':
        #         feature_chunk_names = []
        #         for chunk in os.listdir(FEATURES / self.collection):
        #             if 'SCIBERT' in chunk:
        #                 feature_chunk_names.append(chunk)
        #         previous_start = 0
        #         tqdm_sorted = tqdm(sorted(feature_chunk_names), desc=f'- looping over {len(feature_chunk_names)} chunks')
        #         for feature_chunk_name in tqdm_sorted:
        #             feature_list = load_info(os.path.join(FEATURES, self.collection, feature_chunk_name))
        #             feature_list = feature_list['chunk']
        #             current_end = (previous_start+len(feature_list)) if (previous_start+len(feature_list)) < num_graphs else num_graphs
        #             tqdm_inside_chunk = tqdm(range(previous_start, current_end), desc=f'- looping into {feature_chunk_name} chunk file')
        #             for g in tqdm_inside_chunk:
        #                 list_idx = g-previous_start
        #                 try:
        #                     feat = self.graphs[g].ndata['feat']
        #                     self.graphs[g].ndata['feat'] = torch.cat((feat, feature_list[list_idx]), 1)
        #                 except:
        #                     self.graphs[g].ndata['feat'] = torch.tensor(feature_list[list_idx], dtype=torch.float32)
        #             previous_start += len(feature_list)

                
                
        #     else:
        #         feature_list = load_info(os.path.join(FEATURES, self.collection, feature_name))
        #         tqdm_idx_graph = tqdm(range(0, num_graphs), desc=f'- looping into {feature_name} file')
        #         for g in tqdm_idx_graph:
        #             if feature_name == 'REPR':
        #                 node_no = self.graphs[g].num_nodes()
        #                 features_to_use = feature_list[g][:node_no]
        #             else:
        #                 features_to_use = feature_list[g][:]
        
        #             try:
        #                 feat = self.graphs[g].ndata['feat']
        #                 self.graphs[g].ndata['feat'] = torch.cat((feat, features_to_use), 1)
        #             except:
        #                 self.graphs[g].ndata['feat'] = torch.tensor(features_to_use, dtype=torch.float32)

                    
                
        # if True, remove "islands" - nodes too far form different classes
        remove_islands = self.gb.get_config()[f'DL{split}']['remove_islands']
        bidirectional = self.gb.get_config()['PREPROCESS']['bidirectional']
        
        remove_path = os.path.join(GRAPHS, self.collection, f'{str(self.get_config().PREPROCESS.mode).upper()}_REMOVE_IDXS_{str(self.get_config().PREPROCESS.range_island)}.pkl')
        all_to_remove = []
        all_remove = []
        
        def remove_nodes_infos(indeces):
            bboxs = self.pages[g]['bboxs']
            texts = self.pages[g]['texts']
            for idx in sorted(indeces, reverse=True):
                del bboxs[idx]
                del texts[idx]
            return bboxs, texts

        #! labels for statistics
        labels_dicts = []
        total_dict = dict()
        total_count = 0
        
        if remove_islands:
            print(f" -> Remove Islands")
            if os.path.isfile(remove_path):
                print(" -- Loading remove_islands indexes")
                all_remove = load_info(remove_path) # load
            else:
                print(" -- Calculating FAST remove_islands indexes")

        tqdm_idx_graph = tqdm(range(0, num_graphs), desc=f' -- remove_islands + bidirectional + edges_feat graph loop')
        for g in tqdm_idx_graph:

            classes_dict = dict()         

            # print(f" -> Modify edges")
            #? calculate or load node idxs to remove
            if remove_islands: 
                if not all_remove:
                    to_remove = self.gb.fast_remove_islands(self.graphs[g]) # evaluate
                    self.graphs[g].remove_nodes(to_remove)
                    all_to_remove.append(to_remove)
                else:
                    to_remove = all_remove[g]
                    self.graphs[g].remove_nodes(to_remove)
                    if debug: remove_nodes_infos(to_remove)
            else:
                to_remove = []

            # otherwise doesn't work
            #? modify edges from directed to bidirectional
            if bidirectional:
                # as https://docs.dgl.ai/en/0.6.x/_modules/dgl/transform.html#to_bidirected
                # copy_ndata
                #  -> If True, the node features of the bidirected graph are copied from the original graph. 
                #  -> If False, the bidirected graph will not have any node features. (Default: False)
                #g = dgl.remove_self_loop(g)
                self.graphs[g] = dgl.to_simple(self.graphs[g])
                self.graphs[g] = dgl.to_bidirected(self.graphs[g], copy_ndata=True)

            # todo (if simply 'to bidirected' does not work)
            # g = dgl.remove_self_loop(g)
            # g = dgl.to_simple(g)

            # todo -> assure you do this only one time
            bboxs, _ = remove_nodes_infos(to_remove)

            #! start
            # todo calculate or load edge features 
            #? some type could be ['distances']
            if self.get_config().PREPROCESS.edge_features:
                #print(" -> adding edge features")
                u, v = self.graphs[g].edges()

                srcs, dsts =  u.tolist(), v.tolist()
                distances = []
                for i, src in enumerate(srcs):
                    distances.append(distance(bboxs[src], bboxs[dsts[i]]))

                m = max(distances)
                distances = [(1 - d/m) for d in distances]
                    
                self.graphs[g].edata['feat'] = torch.tensor(distances, dtype=torch.float32)
            #! end

            # print(f" -> Modify Lables")
            if self.get_config().GENERAL.converted:

                self.graphs[g].ndata['label'] =  torch.tensor( 
                    list(map(
                        lambda x: self.label_tranformer.origin_to_conv[x], 
                        self.graphs[g].ndata['label'].tolist()
                    )), dtype=torch.float32)

            #! do statistics
            # print(f" -> Calculate statistics")
            labels = self.graphs[g].ndata['label'].numpy()
            for label in labels:
                int_ = classes_dict.get(int(label), 0)
                int_ += 1
                classes_dict[int(label)] = int_
            
            labels_dicts.append(classes_dict)
            for key, val in classes_dict.items():
                # total_dict[key] += val
                int_ = total_dict.get(int(key), 0)
                int_ += val
                total_dict[key] = int_
            total_count += len(labels)
        
        self.graphs = self.graphs[:num_graphs]
        
        @timeit
        def save_wrap(remove_path, all_to_remove):
            save_info(remove_path, all_to_remove)

        if not os.path.isfile(remove_path):
            print(f" -> Save Remove idxs")
            save_wrap(remove_path, all_to_remove)

        #! do statistics
        print(f" -> Do Statistics")
        self.num_classes, numbers, percentages = fast_get_statistics(total_count, total_dict, labels_dicts)
        self.stats['numbers'] = numbers
        self.stats['percentages'] = percentages        

        # todo -> change the calculated self.num_classes
        # self.num_classes = ?
            
        print()
        print("took %ss" % (round(time.time() - start_time, 2)))
        return
    
    def split(self, num_graphs):
        
        if self.collection == "train":
            rate = self.get_config().DLTRAIN.rate
            if num_graphs > self.__len__():
                print("WARNING: You have selected a larger number of graphs (available {}). \
                    \n The function is using just all the available.".format(self.__len__()))
                num_graphs = self.__len__()
                
            train_amount = int(num_graphs * rate)
            validation_amount = num_graphs - train_amount
            print(" -> Split {} pages : Train {} | Val {}.".format(num_graphs, train_amount, validation_amount))
            
            train_indexes = random.sample(range(0, num_graphs), train_amount)
            val_indexes = list(set(range(0, num_graphs)) - set(train_indexes))
            
            train_graphs = []
            val_graphs = []
            
            for i in train_indexes:
                train_graphs.append(self.graphs[i])
                
            for i in val_indexes:
                val_graphs.append(self.graphs[i])
                    
            return train_graphs, val_graphs, (train_indexes, val_indexes)
        else:
            print("WARNING: split in train/val not available for testing purposes.")
            return

    def get_config(self):
        return self.gb.get_config()
    
    def get_gb(self):
        return self.gb

class GenericPapers2Graphs(DGLDataset):
    """ Template for customizing graph datasets in DGL.
    """
    def __init__(self,
                 num_papers = None,
                 config = None,
                 path_to_config = None,
                 raw_dir = RAW,
                 save_dir = GRAPHS):
        
        self.num_papers = num_papers

        # assert config != None, 'ERROR -> config file can\'t be NONE'

        self.gb = GraphBuilder(config=config, path_to_config=path_to_config)
        self.graphs = []
        self.pages = []
        self.seed = self.get_config().PREPROCESS.seed
        random.seed(self.seed)
        super(GenericPapers2Graphs, self).__init__(name = '', raw_dir = raw_dir, save_dir = save_dir, verbose = False)

    def process(self):
        # processing raw data from raw_dir, and saving graph/s to save_dir
            
        for page in tqdm(os.listdir(self.raw_dir)):
            page_path = os.path.join(self.raw_dir, page_path)  
            g, bboxs, texts = self.gb.get_graph(self.data_path / page, None, self.get_config().PREPROCESS.mode, set_labels=False)
            if g is not None:
                self.graphs.append(g)
                self.pages.append({'page': page, 'bboxs' : bboxs, 'texts' : texts})

    def __getitem__(self, idx):
        return self.graphs[idx]

    def __len__(self):
        return len(self.graphs)

    def save(self):
        # save processed data to directory `self.save_path`
        if not os.path.exists(self.save_dir): os.mkdir(self.save_dir)
        graph_path = f"{self.save_path}graphs.bin"
        info_path = f"{self.save_path}info.pkl"
        save_graphs(graph_path, self.graphs)
        save_info(info_path, {'pages': self.pages})

    def load(self):
        # load processed data from directory `self.save_path`
        print(f"\nLoading {self.collection} data ... ", end='')
        start_time = time.time()
        graph_path = f"{self.save_path}graphs.bin"
        info_path = f"{self.save_path}info.pkl"
        self.graphs, _ = load_graphs(graph_path)
        self.pages = load_info(info_path)['pages']
        print("took %ss" % (round(time.time() - start_time, 2)))

    def has_cache(self):
        # check whether there are processed data in `self.save_path`
        graph_path = f"{self.save_path}graphs.bin"
        info_path = f"{self.save_path}info.pkl"
        return os.path.exists(graph_path) and os.path.exists(info_path)
    
    def modify_graphs(self):
        
        print(f"\nModifying graphs ... ", end='\n')
        
        num_graphs = self.__len__()
        start_time = time.time()
        for g in range(0, num_graphs):
            print(" -> Progress {}/{} | {}".format(g+1, num_graphs, round(time.time() - start_time, 2)), end='\r')
            
            # lists
            bboxs = self.pages[g]['bboxs']
            texts = self.pages[g]['texts']
            
            # set node and edge features
            page = self.pages[g]['page']
            size = convert_from_path(self.data_path / page)[0].size
            self.gb.set_features(self.graphs[g], bboxs, texts, size)
            
            # TODO get rid of all nodes that are not 'table'
            #? load tables bboxs from annotations
            #? check which bboxs is not in those table annotations and save theirs 'ids' (list indices) -> node_ids_toremove
            #? graph.remove_node(node_ids_toremove)
            #? graph.label = table_type
            
        print()
        print("took %ss" % (round(time.time() - start_time, 2)))
        
        # TODO handle new graph savings
        
        """graph_path = f"{self.save_path}graphs_modified.bin"
        info_path = f"{self.save_path}info_modified.pkl"
        
        if os.path.exists(graph_path) and os.path.exists(info_path):
            print("\n -> Loading already modified graphs ... ", end='')
            start_time = time.time()
            self.graphs, _ = load_graphs(graph_path)
            self.pages = load_info(info_path)['pages']
            print("took %ss" % (round(time.time() - start_time, 2)))
            return
        
        if save:
            print("Saving modified graphs.")
            save_graphs(graph_path, self.graphs[:num_graphs])
            save_info(info_path, {'pages': self.pages[:num_graphs]})"""
                
        return
    
    def split(self, num_graphs):
        
        if self.collection == "train":
            rate = self.get_config().DLTRAIN.rate
            if num_graphs > self.__len__():
                print("WARNING: You have selected a larger number of graphs (available {}). \
                    \n The function is using just all the available.".format(self.__len__()))
                num_graphs = self.__len__()
                
            train_amount = int(num_graphs * rate)
            validation_amount = num_graphs - train_amount
            print(" -> Split {} pages : Train {} | Val {}.".format(num_graphs, train_amount, validation_amount))
            
            train_indexes = random.sample(range(0, num_graphs), train_amount)
            val_indexes = list(set(range(0, num_graphs)) - set(train_indexes))
            
            train_graphs = []
            val_graphs = []
            
            for i in train_indexes:
                train_graphs.append(self.graphs[i])
                
            for i in val_indexes:
                val_graphs.append(self.graphs[i])
                    
            return train_graphs, val_graphs
        else:
            print("WARNING: split in train/val not available for testing purposes.")
            return

    def get_config(self):
        return self.gb.get_config()
    
    def get_gb(self):
        return self.gb
    
if __name__ == "__main__":
    
    start_time = time.time()
    data = Papers2Graphs(path_to_config='/home/gemelli/projects/publaynet_preprocess/configs/graph/graphs2.yaml')
    print("%ss" % (round(time.time() - start_time, 2)))
    data.modify_graphs()
    print(data[0])
    data.get_infos(show_sample=True, folder=RAW, converted=True)
