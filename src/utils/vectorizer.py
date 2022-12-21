# -*- coding: utf-8 -*-
"""A class to turn a list of strings into concatenated one hot vectors of
each char."""
import re
import numpy as np

class CharVectorizer(object):


    def __init__(self, accepted_chars, map_unknown_chars_to='',
                 fill_left_char='', fill_right_char='',
                 auto_lowercase=True, auto_uppercase=False):
      
        self.auto_lowercase = auto_lowercase
        self.auto_uppercase = auto_uppercase
        self.map_unknown_chars_to = map_unknown_chars_to
        self.fill_left_char = fill_left_char
        self.fill_right_char = fill_right_char

        self.accepted_chars = accepted_chars
        # self.accepted_chars = self._unique_keep_order(
        #     list(accepted_chars) ]
        # )
        self.accepted_chars_indirect = list(self.accepted_chars)

       
        self.char_to_index = dict()
        for i, charr in enumerate(self.accepted_chars):
            self.char_to_index[charr] = i

            if auto_lowercase:
                charr_upp = charr.upper()
                if charr_upp != charr and charr_upp not in self.accepted_chars:
                    self.accepted_chars_indirect.append(charr_upp)
                    self.char_to_index[charr_upp] = i

            if auto_uppercase:
                charr_low = charr.lower()
                if charr_low != charr and charr_low not in self.accepted_chars:
                    self.accepted_chars_indirect.append(charr_low)
                    self.char_to_index[charr_low] = i

        
        self.unknown_chars_regex = re.compile(
            r"[^%s]" % re.escape("".join(self.accepted_chars_indirect))
        )

    def fit(self, _):
        return self

    def fit_transform(self, texts, per_string_length, fill_right=True,
                      dtype=np.int):
        return self.transform(texts, per_string_length,
                              fill_right=fill_right, dtype=dtype)

    def transform(self, texts, per_string_length, fill_right=True,
                  dtype=np.int):
        
        rows = len(texts)
        cols = per_string_length
        matrix = np.zeros((rows, cols), dtype=np.int)

        for i, text in enumerate(texts):
            matrix[i, :] = self._text_to_char_indexes(
                text,
                per_string_length=per_string_length,
                fill_right=fill_right
            )

        return self._one_hot_matrix(matrix, per_string_length,
                                    len(self.accepted_chars) - 1,
                                    dtype=dtype)

    def sanitizer(self, text, one_hot_matrix):
        words_list = [matrix[:len(text[id])] for id, matrix in enumerate(one_hot_matrix)]
        return words_list

    def transform_string(self, text, per_string_length, fill_right=True,
                         dtype=np.int):
        
        return self.transform([text],
                              per_string_length,
                              fill_right=fill_right,
                              dtype=dtype)

    def transform_char(self, char, dtype=np.int):
        
        return self.transform_string(char,
                                     1,
                                     dtype=dtype)

    def _text_to_char_indexes(self, text, per_string_length,
                              fill_right=True):
        
        text = self.unknown_chars_regex.sub(self.map_unknown_chars_to, text)

        result = np.zeros((per_string_length,), dtype=np.int)

        lenn = len(text)
        if lenn > per_string_length:
            text = text[0:per_string_length]
        elif lenn < per_string_length:
            diff = per_string_length - lenn
            if fill_right:
                filler = self.fill_right_char * diff
                text = text + filler
            else:
                filler = self.fill_left_char * diff
                text = filler + text

        for i, charr in enumerate(list(text)):
            index = self.char_to_index[charr]
            result[i] = index

        return result

    def reverse_transform(self, matrix):
        assert type(matrix).__module__ == np.__name__

        result = []
        for row in matrix:
            result.append(self.reverse_transform_string(row))
        return result

    def reverse_transform_maxval(self, matrix):
        assert type(matrix).__module__ == np.__name__

        result = []
        for row in matrix:
            result.append(self.reverse_transform_string_maxval(row))
        return result

    def reverse_transform_string(self, vectorized):
        assert type(vectorized).__module__ == np.__name__

        length_per_char = self.get_one_char_vector_length()
        vecs = self._list_to_chunks(vectorized, length_per_char)

        text = [self.reverse_transform_char(vec) for vec in vecs]

        return "".join(text)

    def reverse_transform_string_maxval(self, vectorized):
        assert type(vectorized).__module__ == np.__name__

        length_per_char = self.get_one_char_vector_length()

        vecs = self._list_to_chunks(vectorized, length_per_char)
        
        text = []
        for vec in vecs:
            text.append(self.reverse_transform_char_maxval(vec))

        return "".join(text)

    def reverse_transform_char(self, char_one_hot_vector):
        return self.reverse_transform_char_maxval(char_one_hot_vector)

    def reverse_transform_char_maxval(self, fuzzy_one_hot_vector):
        assert type(fuzzy_one_hot_vector).__module__ == np.__name__
        assert fuzzy_one_hot_vector.shape == (len(self.accepted_chars),)

        max_index = np.argmax(fuzzy_one_hot_vector)
        return self.accepted_chars[max_index]

    def get_one_char_vector_length(self):
        return len(self.accepted_chars)

    def get_vector_length(self, per_string_length):
        return per_string_length * self.get_one_char_vector_length()

    def _list_to_chunks(self, lst, chunk_length):
        for i in xrange(0, len(lst), chunk_length):
            yield lst[i:i + chunk_length]

    def _unique_keep_order(self, seq):
        seen = set()
        seen_add = seen.add
        return [x for x in seq if not (x in seen or seen_add(x))]

    def _one_hot_matrix(self, char_indexes, per_string_length, max_char_index,
                        dtype=np.int):

        n_rows = char_indexes.shape[0]
        n_features = per_string_length
        n_values_per_feature = max_char_index + 1

        arr = np.zeros((n_rows, n_features * n_values_per_feature),
                       dtype=dtype)


        offset = np.repeat([n_values_per_feature], n_rows * per_string_length)
        offset[0] = 0
        offset = np.cumsum(offset)

        
        arr.flat[offset + char_indexes.ravel()] = 1

        return np.reshape(arr, (n_rows, n_features, len(self.accepted_chars)))