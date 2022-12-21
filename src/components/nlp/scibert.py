from transformers import * # pip install transformers
import torch
from dgl.data.utils import save_info 
import torch.nn.functional as F
from tqdm import tqdm
from transformers import logging

from src.components.nlp.embedder import Embedder
from src.utils.paths import FEATURES

logging.set_verbosity_error()

class SciBERT(Embedder):
    """
        #! ATTENTION
        THIS CLASS IS DIFFERENT FROM OTHER EMBEDDER INHERITS
    """

    def __init__(self,
                specifics=None,
                input=None,
                name='SCIBERT'):
        super(SciBERT, self).__init__(specifics, input, name)

        print(f"allert-> implementation with input: {self.input}")

    def inizialize(self):
        
        if self.specifics.pooling == 'mean':
            self.pooling = self.mean_pooling

        elif self.specifics.pooling == 'max':
            self.pooling = self.max_pooling

        else:
            raise ValueError('POOLING method not yet implemented !')

        #Load AutoModel from huggingface model repository
        self.tokenizer = AutoTokenizer.from_pretrained(self.specifics.model_path, local_files_only=True)
        model = AutoModel.from_pretrained(self.specifics.model_path, local_files_only=True)

        #Compute token embeddings
        with torch.no_grad():
            matrix = model.get_input_embeddings().weight.clone()
            weight = torch.FloatTensor(matrix)
            weight = F.normalize(weight)
            self.embeddings = torch.nn.Embedding.from_pretrained(weight)

        if self.specifics.is_cuda:
            self.embeddings = self.embeddings.cuda()

    #Mean Pooling - Take attention mask into account for correct averaging
    def mean_pooling(self, model_output, attention_mask):
        token_embeddings = model_output # [0] #First element of model_output contains all token embeddings
        input_mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
        sum_embeddings = torch.sum(token_embeddings * input_mask_expanded, 1)
        sum_mask = torch.clamp(input_mask_expanded.sum(1), min=1e-9)
        return sum_embeddings / sum_mask

    #Max Pooling - Take attention mask into account for correct averaging
    def max_pooling(self, model_output, attention_mask):
        token_embeddings = model_output # [0] #First element of model_output contains all token embeddings
        input_mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
        max_embeddings = torch.max(token_embeddings * input_mask_expanded, 1)
        return max_embeddings.values

    def _online_batch_(self, bboxs, texts, titles):
        # print('SCIBERT', end='\n')
        sentences = texts if type(texts) == list else [texts]

        max_length=16
        texts = [[''.join(bbox_texts.split(' ')) for bbox_texts in page_texts] for page_texts in sentences] # tqdm(sentences, desc='- single texts join [paper->bbox]')]
        
        # texts = [' '.join(page_texts) for page_texts in tqdm(texts, desc='- page texts join [paper]')] 
        # results = self.list_of_sentence(texts, max_length=max_length)
        # return results

        chunk = []
        texts_len = len(texts)
        for page_id in range(texts_len): # tqdm(range(texts_len), desc='- single texts embedding'):
            vect = self.list_of_sentence(texts[page_id], max_length=max_length)
            chunk.append(vect)

        return chunk

    def __call__(self, bboxs, texts, titles, split=None, by_word=True, save=True, return_vectors=True):
        assert split != None
        print('SCIBERT', end='\n')

        sentences = texts if type(texts) == list else [texts]
        
        chunk_id = 1
        counter = 0
        
        ids2chunk = dict()
        chunk = []

        if by_word:
            max_length=16
            texts = [[''.join(bbox_texts.split(' ')) for bbox_texts in page_texts] for page_texts in sentences] # tqdm(sentences, desc='- single texts join [paper->bbox]')]
            texts_len = len(texts)
            for page_id in range(texts_len): # tqdm(range(texts_len), desc='- single texts embedding'):
                vect = self.list_of_sentence(texts[page_id], max_length=max_length)
                chunk.append(vect)
                ids2chunk[titles[page_id]] = [chunk_id-1, counter ,page_id]
                counter += 1

                if page_id > 0 and page_id % self.specifics.per_chunk == 0:

                    feature_path = FEATURES / split / f'SCIBERT_{chunk_id-1}'
                    print(f'-start saving chunk {chunk_id-1}')
                    save_info(feature_path, {'chunk': chunk, 'ids2chunk': ids2chunk})
                    print(f'---end saving chunk {chunk_id-1}')

                    chunk_id += 1
                    counter = 0

                    ids2chunk = dict()
                    chunk = []
            
            if save:
                feature_path = FEATURES / split / f'SCIBERT_{chunk_id-1}'
                print(f'-start saving chunk {chunk_id-1}')
                save_info(feature_path, chunk)
                print(f'---end saving chunk {chunk_id-1}')
        else:
            max_length = max([len(e) for e in sentences])
            texts = [[''.join(bbox_texts.split(' ')) for bbox_texts in page_texts] for page_texts in sentences] # tqdm(sentences, desc='- single texts join [paper->bbox]')]
            texts = [' '.join(page_texts) for page_texts in texts] # tqdm(texts, desc='- page texts join [paper]')] 

            # todo -> unffeseable -> need to do it in batch (otherwise full size = 120 GB)
            results = self.list_of_sentence(texts, max_length=max_length)
            ids2chunk = dict()

            if save:
                feature_path = FEATURES / split / f'SCIBERT_{chunk_id-1}'
                print(f'-start saving results')
                save_info(feature_path, results)
                print(f'---end saving results')
            
        return True

    def list_of_sentence(self, texts, max_length):
        encoded_input = self.tokenizer(texts, padding=True, truncation=True, max_length=max_length, return_tensors='pt')
        
        input_ids = encoded_input['input_ids']
        attention_mask = encoded_input['attention_mask']
        
        attention_mask[:,0] = 0
        attention_mask[:,-1] = 0

        if self.specifics.is_cuda:
            input_ids = input_ids.cuda()
            attention_mask = attention_mask.cuda()

        model_output = self.embeddings.forward(input_ids)
        vect = self.pooling(model_output, attention_mask)
        # vect = F.normalize(vect)
        vect = vect if self.specifics.return_tensor else vect.numpy()
        return vect
    
if __name__ == '__main__':
    print('Repr test start')
    # create the embedder
    embedder = SciBERT()
    sentences = ['banana33', 'p-value', '33', '1.1', '(1.1,']
    
    #? using singularly
    for w in sentences:
        result = embedder(w)
        print(w, len(result), result)
    
    #? using with batches
    result = embedder(sentences)
    print(sentences, len(result), result)
