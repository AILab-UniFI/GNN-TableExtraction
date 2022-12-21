from itertools import groupby
import numpy as np
import pandas as pd
from typing import List, Union

from src.components.tables.utils import UNICODE_DICT
from src.components.tables.manager import Manager
from src.components.tables.types import Method, Task, TableInstance, Cell

class ExtendedDf(pd.DataFrame):
    id: str
    number: Union[int, str]
    column_list: List[List[Union[int, str, float]]]

    def __init__(self, structure_id, *args):
        super(ExtendedDf, self).__init__(args[0])
        if structure_id is not None:
            self.id, _, self.number = structure_id.split("_")

    def calculate_column_list(self):
        def decode(value_list):
            def single_decode(element):
                try: return element.decode("utf-8")
                except: return element.decode("latin1")
            return [single_decode(element) for element in value_list]

        self.column_list = [[self.id, self.number] + decode(value_list) for value_list in np.asarray(self.transpose())]
        return self

    @property
    def get_column_list(self):
        return self.column_list

    def get_csv(self):
        return self.to_csv(header=False)


class Table(TableInstance):
    paper_id: str
    table_number: int

    # rows_length: int
    # cols_length: int

    mask_matrix: np.ndarray
    value_matrix: np.ndarray
    sub_tables: List[ExtendedDf]

    _debug: bool
    _keys_dict: dict
    _v_counter: int

    def __init__(self, dictionary):
        super(Table, self).__init__(dictionary)
        # id, _, number = structure_id.split("_")
        self.paper_id, _, self.table_number = self.structure_id.split("_")

        if hasattr(self, 'rows') and hasattr(self, 'columns'):
            self.mask_matrix = np.empty((len(self.rows), len(self.columns)))
            self.value_matrix = np.empty((len(self.rows), len(self.columns)), dtype="S64")
        
        self.sub_tables = []

        self._debug = False

    @classmethod
    def _from_tables(mask_matrix_df, value_matrix_df):
        table = Table({'structure_id':'nocode__0'})
        table.mask_matrix = mask_matrix_df.to_numpy()
        table.value_matrix = value_matrix_df.to_numpy()
        return table


    @property
    def rows_length(self):
        return self.mask_matrix.shape[0]

    @property
    def cols_length(self):
        return self.mask_matrix.shape[1]

    @property
    def debug(self):
        self._debug = True
        return self

    @property
    def no_debug(self):
        self._debug = False
        return self
    
    @property
    def mask_matrix_df(self):
        return pd.DataFrame(self.mask_matrix)

    @property
    def value_matrix_df(self, i=None, j=None):
        # if i==None and j==None:
        #     return pd.DataFrame(self.value_matrix)
        if i==None: i=range(self.rows_length)
        if j==None: j=range(self.cols_length)
        return pd.DataFrame(self.value_matrix).iloc[i, j]

    def unroll(self):
        return np.concatenate(self.value_matrix)
    
    def table_extraction(self, config) -> any:
        self._keys_dict = dict()
        self._v_counter = 1

        is_vertical_span = False

        for cell_ns in self.cells:
            cell = Cell(cell_ns)

            header_value = 0

            # and not cell.is_projected_row_header:
            if len(cell.column_nums_set) > 1 and not cell.is_column_header:
                header_value = -1

            # and not cell.is_projected_row_header:
            if len(cell.row_nums_set) > 1 and not cell.is_column_header:
                # v_counter indicates whether you have 
                # at least one cell that spans in multiple rows
                self._v_counter += 1
                is_vertical_span = True

            for r in cell.row_nums_set:

                for c in cell.column_nums_set:

                    if self._debug: print(r, c, cell.is_column_header, cell.xml_text_content)
                    if self._debug: print(cell.text_contents)

                    self.mask_matrix[r, c] = header_value if header_value < 0 else self._v_counter if is_vertical_span else cell.is_column_header 

                    # manager = Manager(cell.xml_text_content, UNICODE_DICT).set_new_content()
                    manager = Manager(cell.xml_text_content).set_new_content().set_replace_content()

                    if "\n" in cell.xml_text_content:
                        # print(f"ATTENTION {cell.xml_text_content}!")
                        cell.xml_text_content = cell.xml_text_content.replace("\n", "")

                    if config.GENERAL.task == Task.substitution.value:
                        
                        if self._debug: manager.debug
                        # substitute all numbers with x and chars with w 
                        # -x -> x
                        cell.xml_text_content = manager \
                                                .replace_chars_and_digits() \
                                                .remove_number_sign() \
                                                .new_content                       
                    
                    # substitute unicode value from UNICODE_DICT
                    # " +- " -> "+-"
                    cell.xml_text_content = manager \
                                            .remove_unicode() \
                                            .new_content # .remove_spaces_between_symbols() \
                    
                    if config.GENERAL.method == Method.split.value:
                        splitted_content = cell.xml_text_content.split(' ')
                        for content in splitted_content:
                            self._keys_dict[content] = self._keys_dict.get(content, 0) + 1
                    else:
                        self._keys_dict[cell.xml_text_content] = self._keys_dict.get(cell.xml_text_content, 0) + 1

                    # content.encode("utf-8") if content == raw_content else make_indent(content, raw_content, debug).encode("utf-8")
                    self.value_matrix[r, c] = cell.xml_text_content.encode("utf-8") 

            is_vertical_span = False  

        return self
    
    def table_splits(self) -> any:
        if self._debug: print()

        # TODO v_counter start from 1.0, but it"s the header!
        group = self.mask_matrix.groupby(0).indices

        def no_vertical_span(table, group) -> List[pd.DataFrame]:
            sub_tables = []
            list_group = group.get(0.0, [])

            if table._debug:
                print(list_group)

            for _, y in groupby(enumerate(list_group), lambda x: x[1]-x[0]):
                indexes = [i for _y, i in list(y)]
                sub_body = table.value_matrix_df.iloc[indexes, :]
                sub_tables.append(ExtendedDf(table.structure_id, sub_body))
            return sub_tables

        def vertical_span(table, group) -> List[pd.DataFrame]:
            sub_tables = []

            for count in range(2, table._v_counter + 1):

                list_group = group.get(float(count), [])

                if table._debug: print(list_group)

                for _, y in groupby(enumerate(list_group), lambda x: x[1]-x[0]):
                    indexes = [i for _y, i in list(y)]
                    sub_body = table.value_matrix_df.iloc[indexes, :]
                    sub_tables.append(ExtendedDf(table.structure_id, sub_body))

            return sub_tables
        

        if self._v_counter > 1:
            self.sub_tables = vertical_span(self, group)

        self.sub_tables = no_vertical_span(self, group)

        return self


    def get_matrices(self):
        return (self.mask_matrix_df, self.value_matrix_df)
        
        


