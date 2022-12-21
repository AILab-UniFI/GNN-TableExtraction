import os
import re
from attrdict import AttrDict
import yaml
import pickle
import torch as t
import torch.nn.functional as F

from tqdm import tqdm
# from transformers import * # pip install transformers
from tokenizers import Tokenizer, Regex, NormalizedString
from tokenizers.normalizers import Normalizer
from tokenizers.pre_tokenizers import WhitespaceSplit
from tokenizers.models import WordLevel

from src.components.nlp.embedder import Embedder
from src.utils.paths import CONFIG, TABLES_PREPROCESS, REPR_PTS

os.environ["TOKENIZERS_PARALLELISM"] = "false"

class CustomNormalizer:
    def normalize(self, normalized: NormalizedString):
        normalized.nfkc()
        normalized.map(lambda char: 'x' if char.isdigit() else 'w' if char.isalpha() else char)
        normalized.replace(Regex('x+'), 'x')
        normalized.replace(Regex('w+'), 'w')
        normalized.replace(Regex('\w*(?<!(w|\+|x))-x'), 'x')
        normalized.lowercase()

class Repr(Embedder):
    
    def __init__(self,
                specifics=None,
                input=None,
                name='REPR'):
        super(Repr, self).__init__(specifics, input, name)

        print(f"allert-> implementation with input: {self.input}")

    def inizialize(self):

        if self.specifics.use_words:
            wc = pickle.load(open(TABLES_PREPROCESS / 'wc.dat', 'rb'))
            words = sorted(wc, key=wc.get, reverse=True)[:self.specifics.top_k]
            del wc
            self.word2idx = pickle.load(open(TABLES_PREPROCESS / 'word2idx.dat', 'rb'))
            self.word2idx = {k: v for k,v in self.word2idx.items() if k in words}
            del words
            self.idx2vec = pickle.load(open(REPR_PTS / self.input.run_name /  f'idx2vec_i_epoch{self.input.epoch_no}.dat', 'rb'))
        else:
            self.word2idx = {}
            
        train_repr = pickle.load(open(REPR_PTS / self.input.run_name / f'trained_prototypes_epoch{self.input.epoch_no}_{self.input.centroids_no}_{self.input.alpha_no}.dat', 'rb'))
        self.centers = t.from_numpy(train_repr['prototypes']).double()
        self.i_prototypes = t.from_numpy(train_repr['i_embedding']).double()
        del train_repr

        embed_repr = pickle.load(open(TABLES_PREPROCESS / f'embed-repr-{self.input.centroids_no}-{self.input.component_no}-{self.specifics.top_k}.dat', 'rb'))
        self.embeddings = t.from_numpy(embed_repr['embeddings']).double()
                
        self.repr2idx = pickle.load(open(TABLES_PREPROCESS / 'repr2idx.dat', 'rb'))

        tok = Tokenizer(WordLevel(self.repr2idx, unk_token="w"))
        tok.normalizer = Normalizer.custom(CustomNormalizer())
        tok.pre_tokenizer = WhitespaceSplit()
        tok.model = WordLevel(self.repr2idx, unk_token="<UNK_W>")
        tok.pad_token = ""
        self.tokenizer = tok

        if self.specifics.is_cuda:
            self.centers = self.centers.cuda()
            self.i_prototypes = self.i_prototypes.cuda()
            self.embeddings = self.embeddings.cuda()
    
    def get_similarity(self, query, vectors, alpha=1.0):
        min_margin =  t.tensor(0.0001, dtype=t.float64)
        min_margin = min_margin.cuda() if self.specifics.is_cuda else min_margin
        query = t.unsqueeze(query, dim=1)
        query = query.expand(query.shape[0], self.i_prototypes.shape[0], -1)
        vectors = t.norm(t.abs(vectors - query), dim=2)
        two = t.max(vectors, min_margin)
        three = 1 / two
        prototype_weights = t.pow(three, alpha)
        divid = t.unsqueeze(t.sum(prototype_weights, 1), dim=1)
        divid = divid.expand(divid.shape[0], self.i_prototypes.shape[0])
        prototype_weights /= divid
        return prototype_weights

    def _online_batch_(self, bboxs, texts, titles):
        return self.__call__(bboxs, texts, titles)

    def __call__(self, bboxs, texts, titles, split=None, combined=False):

        # assert split != None, 'split must exists'
        # print('REPR', end='\n')

        # if self.specifics.use_words and self.word2idx.get(texts_compatted, None) != None:
        #     vect = self.idx2vec[self.word2idx[texts_compatted]]
        #     vect = t.tensor(vect).cuda() if self.specifics.is_cuda else t.tensor(vect)
        #     vect = vect if self.specifics.return_tensor else vect.numpy()
        #     return vect
        
        texts = [[''.join(bbox_texts.split(' ')) for bbox_texts in page_texts] for page_texts in texts] # tqdm(texts, desc='- single texts join [paper->repr]')]
        texts_compatted = [' '.join(page_texts) for page_texts in texts] # tqdm(texts, desc='- page texts join [paper]')] 

        encodings = [self.tokenizer.encode(word) for word in texts_compatted] #tqdm(texts_compatted, desc='- tokenizer page texts [paper]')] #(sentences, padding=True, truncation=True, max_length=16, return_tensors='pt')

        results = t.empty((len(encodings), max([len(e) for e in encodings]), self.i_prototypes.shape[1]), device = 'cuda' if self.specifics.is_cuda else None)

        # tqdm_iter = tqdm(encodings, desc="- encodings clustering and get centroids")
        for i, encoding in enumerate(encodings):
            
            idxs = encoding.ids
            
            if combined:
                emb = self.embeddings[idxs]
                coefficients = self.get_similarity(emb, self.centers)
            else:
                coefficients = t.zeros((len(idxs), self.i_prototypes.shape[0]), device = 'cuda' if self.specifics.is_cuda else None)
                try:
                    emb = self.embeddings[idxs]
                except:
                    emb = self.embeddings[0]
                vectors = self.centers
                vectors = vectors.expand(emb.shape[0], vectors.shape[0], -1)
                w_coefficients = self.get_similarity(emb, vectors)
                w_idx = t.argmax(w_coefficients, 1)
                for j in range(len(w_idx)):
                    coefficients[j][w_idx[j]] = 1

            try:
                result = t.matmul(coefficients.float(), self.i_prototypes.float())
            except:
                raise Exception('coefficents: {} - i_prototypes: {}'.format(type(coefficients), type(self.i_prototypes)))
            
            if list(result.shape) != [max([len(e) for e in encodings]), self.i_prototypes.shape[1]]:
                app = t.zeros((max([len(e) for e in encodings]) - result.shape[0]), self.i_prototypes.shape[1])
                result = t.cat((result, app), 0)
            results[i] = result
        
        # print(results.shape)
        results = results if self.specifics.return_tensor else results.numpy()
        return results        
    
if __name__ == '__main__':
    print('Repr test start')
    # create the embedder
    with open(CONFIG / 'features.yaml') as fileobj:
        config = AttrDict(yaml.safe_load(fileobj))

    repr_specifics = config.REPR.specifics
    repr_input = config.REPR.input

    embedder = Repr( is_cuda = False, return_tensor = False )
    
    #! N page -> N row (all cell content ''.join() together for a page)
    # or
    #! 1 page -> M row
    sentences = [   'banana33 p-value 33 1.1 (1.1,',
                    'ciao', 
                    ' mamma guarda 1.1 (1.1,' ]
    
    #! using singularly
    # for w in sentences:
    #     result = embedder(w)
    #     print(w, len(result), result)
    
    #! using with batches <- PREFERRED
    result = embedder(sentences)

    #! result.shape => [3, 5, 30]
    print(sentences, len(result), result)
