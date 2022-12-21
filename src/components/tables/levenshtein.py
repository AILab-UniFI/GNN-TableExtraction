import numpy as np
from weighted_levenshtein import lev

class LevenshteinSimilitudes():
    ins_c: np.array
    del_c: np.array
    sub_c: np.array

    def __init__(self):
        self.init_weights_uniform()

        
    def init_weights_uniform(self):    
        self.ins_c = np.ones(10001, dtype=np.float64) # np.ones(128, dtype=np.float64)  # make an array of all 1's of size 128, the number of ASCII characters
        self.del_c = np.ones(10001, dtype=np.float64) # np.ones(128, dtype=np.float64)
        self.sub_c = np.ones((10001, 10001), dtype=np.float64) # np.ones((128, 128), dtype=np.float64)  # make a 2D array of 1's
        return self
        
        
    def init_weights(self):
        """
        Input:
        - weights_file: path

        Output:
        - ins_c, del_c, sub_c (Modified with weights_file)
        """
        # todo we could store weights into a file
        # todo and load/apply them here
        
        # self.init_weights_uniform()

        self.del_c[ord('%')] = 1.5  
        self.del_c[ord('.')] = 2  
        self.del_c[ord(',')] = 2  

        self.ins_c[ord('%')] = 1.5  
        self.ins_c[ord('.')] = 2  
        self.ins_c[ord(',')] = 2

        self.sub_c[ord('.'), ord(',')] = 1.9 
        self.sub_c[ord(','), ord('.')] = 1.9 

        self.sub_c[ord('w'), ord('x')] = 1.9 
        self.sub_c[ord('x'), ord('w')] = 1.9 
        self.sub_c[ord('x'), ord('%')] = 5 

        return self

    def calculate_similarity(self, words, weights=None):
        if weights!=None: self.init_weights()
        print(len(words))
        matrix = np.zeros((len(words), len(words)))
        for i, w2 in enumerate(words):
            for j, w1 in enumerate(words):
                # print(i,j)
                w1 = w1.replace('\ufeff', '')
                #! must support unicode -> see README.md section -> weighted_levenshtein Installation
                matrix[i,j] = -1*lev(str(w1), str(w2), insert_costs=self.ins_c, delete_costs=self.del_c, substitute_costs=self.sub_c)


                
        return matrix

