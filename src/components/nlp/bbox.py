import yaml
import torch
from tqdm import tqdm
import numpy as np
from attrdict import AttrDict
from dgl.data.utils import save_info, load_info
from src.components.graphs.utils import distance, normalize
from pdf2image import convert_from_path
from src.components.nlp.embedder import Embedder
from src.utils.paths import CONFIG, FEATURES, GRAPHS


class Bbox(Embedder):

    def __init__(self,
                name='BBOX',
                specifics=None,
                input=None):
        super(Bbox, self).__init__(specifics, input, name)

        print(f"allert-> implementation with input: {self.input}")

    def inizialize(self):
        print('ready to set pages!')

    def set_pages(self, pages):
        self.page2idx = {page['page']: idx for idx, page in enumerate(pages)}
        self.idx2bboxes = np.array([np.array(page['bboxs']) for page in pages])
        return self.page2idx, self.idx2bboxes

    def _online_batch_(self, bboxs, texts, titles):
        return self.__call__(bboxs, texts, titles)

    def __call__(self, bboxs, texts, titles, split=None):

        # assert split != None, 'split must exists'
        
        """ get BBOX from info_file and calculates some additional surrogates feature

        Args:
            page_name (_type_, optional): _description_. Defaults to None.
            page_id (_type_, optional): _description_. Defaults to None.
            bbox_idx (_type_, optional): _description_. Defaults to None.

        Returns:
            _type_: _description_
        """
                
        def get_shape(bbox):
            width_bbox = bbox[2] - bbox[0]
            height_bbox = bbox[3] - bbox[1]
            center_bbox = [bbox[2] - int(width_bbox/2), bbox[3] - int(height_bbox/2)]
            return [width_bbox, height_bbox, center_bbox[0], center_bbox[1], 
                    width_bbox * height_bbox, bbox[0], bbox[1], bbox[2], bbox[3]]
        
        def get_histogram(text):
                """
                Function
                ----------
                Create histogram of content given a text.

                Parameters
                ----------
                text : string
                    word/s

                Returns
                ----------
                [x, y, z] - 3-dimension list with float values summing up to 1 where:
                            - x is the % of literals inside the text
                            - y is the % of numbers inside the text
                            - z is the % of other symbols i.e. @, #, .., inside the text
                """
                

                
                
                num_symbols = 0
                num_literals = 0
                num_numbers = 0
                num_others = 0
                
                histogram = [0.0000, 0.0000, 0.0000, 0.0000]
                for symbol in text.replace(" ", ""):
                    if symbol.isalpha():
                        num_literals += 1
                    elif symbol.isdigit():
                        num_numbers += 1
                    else:
                        num_others += 1
                    num_symbols += 1

                if num_symbols != 0: 
                    histogram[0] = num_literals / num_symbols
                    histogram[1] = num_numbers / num_symbols
                    histogram[2] = num_others / num_symbols
                    
                    # keep sum 1 after truncate
                    if sum(histogram) != 1.0:
                        diff = 1.0 - sum(histogram)
                        m = max(histogram) + diff
                        histogram[histogram.index(max(histogram))] = m
                    
                if histogram[:3] == [0.0,0.0,0.0]:
                    histogram[3]=1.0

                return histogram

        
        features = []
        
        # tqdm_zip = tqdm(, desc='zipped bboxs and texts')
        for page_id in range(len(bboxs)):
            page_bbox, page_text = bboxs[page_id], texts[page_id]
                        
            #* bbox features like shape, position and content
            emb_shape = list(map(get_shape, page_bbox)) #! return 9-lenght list
            
            #* histogram
            emb_hist = list(map(get_histogram, page_text)) #! return 9-lenght list
            
            features.append(torch.tensor(np.append(emb_shape, emb_hist, 1)))
        
        return features
        
        # todo -> add extention pickle
        feature_path = FEATURES / split / 'BBOX'
        
        save_info(feature_path, features)

        return True


if __name__ == '__main__':
    print('Bbox test start')
    
    # todo -> load config
    with open(CONFIG / 'features.yaml') as fileobj:
        config = AttrDict(yaml.safe_load(fileobj))


    bbox_specifics = config.BBOX.specifics
    bbox_input = config.BBOX.input
    
    #! GENERAL
    # todo -> load strings
    collection = config.GRAPH.collection
    num_papers = config.GRAPH.num_papers
    
    if num_papers == None:
        info_path = GRAPHS / f"{collection}_all_info.pkl"
        # graph_path = GRAPHS / f"{collection}_all.bin"
    else:
        info_path = GRAPHS / f"{collection}_n{num_papers}_scibert_info.pkl"
        # graph_path = GRAPHS / f"{collection}_n{num_papers}_{feature_name}.bin"
    
    # pages = [ 
    #       {'page': page_1, 'bboxs' : bboxs_2, 'texts' : texts_2},
    #       {'page': page_2, 'bboxs' : bboxs_2, 'texts' : texts_2},
    #       ...
    #   ]
    pages = load_info(info_path)['pages']


    # create the embedder
    embedder = Bbox( specifics=bbox_specifics, input=bbox_input )
    embedder.set_pages(pages)
    
    #! in order to test it, we need a real page element with bbox
    page_to_test_id = 5
    page_to_test = pages[page_to_test_id]
    bbox_to_test = page_to_test['bboxs']

    assert bbox_to_test == embedder(page_id=page_to_test_id), 'Error'