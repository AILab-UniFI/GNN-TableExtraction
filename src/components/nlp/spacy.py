import spacy
import numpy
from tqdm import tqdm
from bs4 import UnicodeDammit
from src.components.nlp.embedder import Embedder
from src.utils.paths import FEATURES


class Spacy(Embedder):

    def __init__(self,
                specifics=None,
                input=None,
                name='SPACY'):
        super(Spacy, self).__init__(specifics, input, name)

        print(f"allert-> implementation with input: {self.input}")

    def inizialize(self):
        self.model = spacy.load(self.specifics.model_path)
        if self.specifics.is_cuda:
            self.model = self.model.cuda()
        
    def _online_batch_(self, bboxs, texts, titles):
        return self.__call__(bboxs, texts, titles)

    def __call__(self, bboxs, texts, titles, split=None):

        # assert split != None, 'split must exists'
        
        print('SPACY', end='\n')

        # todo -> not sure if it works also with list
        sentences = texts if type(texts) == list else [texts]
        sentences = [[''.join(bbox_texts.split(' ')) for bbox_texts in page_texts] for page_texts in sentences] #tqdm(sentences, desc='- single texts join [paper->bbox]')]

        # todo -> check what happen with list
        all_results = []
        for page_sentences in sentences: # tqdm(sentences, desc='- extracting vectors'):
            
            page_sentences_concatenated = ' '.join([el.encode('utf-8', 'surrogatepass').decode("utf-8", "ignore") for el in page_sentences])
            
            # todo -> what kind of list is it?
            tokens = self.model(page_sentences_concatenated)

            # todo -> align texts splitted to original
            # "this" "is,%" "m@i" "catt!"
            # "this" "is" "," "%" "m" "@" "i" "catt" "!"
            id_token = 0
            texts_index = [[] for _ in page_sentences]
            
            #! invert cycle over page_sentences instead tokens
            for id_words, words in enumerate(page_sentences):

                # print(token.text, token.has_vector, token.vector_norm, token.is_oov)
                                
                while id_token<tokens.tensor.shape[0] and tokens[id_token].text in words:
                    words = words[len(tokens[id_token].text):]
                    texts_index[id_words].append(id_token)
                    id_token += 1
                    continue
            
            # reconstructed_texts = [ None if len(index)==0 else tokens[index[0]].text if len(index)==1 else ''.join(map(lambda x: x.text,map(tokens.__getitem__, index))) for index in texts_index ]
            
            page_results = [ None if len(index)==0 else tokens[index[0]].vector if len(index)==1 else numpy.mean(list(map(lambda x: x.vector, map(tokens.__getitem__, index))), axis=0) for index in texts_index ]
            
            # if not (reconstructed_texts == page_sentences):
            #     set_intersection = list(set(reconstructed_texts) - set(page_sentences))
            #     print(set_intersection)
            all_results.append(page_results)
        
        return all_results

        # todo -> add extention pickle
        feature_path = FEATURES / split / 'BBOX'
        
        save_info(feature_path, features)
        
        return True

if __name__ == '__main__':
    print('Repr test start')
    # create the embedder
    embedder = Spacy()
    sentences = ['banana33', 'p-value', '33', '1.1', '(1.1,']
    
    #? using singularly
    for w in sentences:
        result = embedder(w)
        print(w, len(result), result)
    
    #? using with batches
    result = embedder(sentences)
    print(sentences, len(result), result)