from enum import Enum
from typing import List, Tuple
from types import SimpleNamespace

class Task(Enum):
    substitution = 'substitution'
    no_substitution = 'no_substitution'

class Method(Enum):
    keepit = 'keepit'
    split = 'split'

class Divider(Enum):
    # ['u-32'] and [1997/07] 
    strictly_words = 'strictly_words'
    # ['u','-','32'] and ['1997','/','7']
    words_and_numbers = 'word_and_numbers'
    
class Cell:
    row_nums: List[int]
    column_nums: List[int]
    align: str
    style: str
    indented: bool
    is_column_header: bool
    pdf_bbox: Tuple[float, float, float, float]
    pdf_text_tight_bbox: Tuple[float, float, float, float]
    is_projected_row_header: bool
    xml_text_content: str
    xml_raw_text_content: str
    pdf_text_content: str

    def __init__(self, dictionary):
        if isinstance(dictionary, SimpleNamespace): dictionary = dictionary.__dict__
        for k, v in dictionary.items():
            setattr(self, k, v)

    @property
    def row_nums_set(self):
        return set(self.row_nums)

    @property
    def column_nums_set(self):
        return set(self.column_nums)

    @property
    def text_contents(self):
        return " ".join([self.xml_text_content, '|', self.xml_raw_text_content, '|', self.pdf_text_content])

class Row:
    pdf_row_bbox: Tuple[float, float, float, float]
    is_column_header: bool

class Column: 
    pdf_column_bbox: Tuple[float, float, float, float]

class TableInstance:
    structure_id: str

    cells: List[Cell]
    rows: List[Row]
    columns: List[Column]

    pmc_id: str
    pdf_file_name: str
    xml_file_name: str
    split: str
    exclude_for_structure: bool
    exclude_for_detection: bool
    xml_table_index: int
    pdf_page_index: int

    pdf_full_page_bbox: Tuple[float, float, float, float]
    pdf_table_bbox: Tuple[float, float, float, float]
    pdf_table_wrap_bbox: Tuple[float, float, float, float]

    xml_table_wrap_start_character_index: int
    xml_table_wrap_end_character_index: int

    def __init__(self, dictionary):
        # for k in self.keys(): setattr(self, k, None)
        if isinstance(dictionary, SimpleNamespace): dictionary = dictionary.__dict__
        for k, v in dictionary.items():
            # if hasattr(self, k): 
            setattr(self, k, v)
            # else: print(f"Nothing to do with {k}")


class Paper:
    tables: List[TableInstance]

    def __init__(self, tables_list: List[TableInstance]):
        self.tables = tables_list

class CountDict(dict):

    def append(self, d2):
        for k in d2.keys():
            self[k] = d2[k] + self.get(k, 0)

    def add_count(self, list):
        for element in list:
            self[element] = self.get(element, 0) + 1
