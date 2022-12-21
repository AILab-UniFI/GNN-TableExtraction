import re
from typing import List
from src.utils.nums import inter_digit
from src.components.tables.utils import UNICODE_DICT
from bs4 import UnicodeDammit

class Manager:
    content: str
    new_content: str
    replace_content: str

    unicode_dict: dict

    _debug: bool
    
    def __init__(self, *args): # content, unicode_dict=None
        if isinstance(args[0], Manager):
            # constructor by copy
            for k, v in args[0].__dict__.items():
                # if hasattr(self, k):
                try:
                    setattr(self, k, v)
                except:
                # else: 
                    print(f"Nothing to do with {k}")
        else:
            assert isinstance(args[0], str), f'args {args[0]} of arg {args} is not str'
            self.content = args[0]
            self.content = self.content.replace('\ufeff', '')
            try:
                assert isinstance(args[1], dict)
                self.unicode_dict = args[1]
            except:
                # print(f"No unicode_dict has been detected")
                self.unicode_dict = None
            finally:
                self.new_content = None
                self.replace_content = None
                self._debug = False
    
    @property
    def debug(self):
        self._debug = True
        return self

    @property
    def no_debug(self):
        self._debug = False
        return self

    def set_new_content(self, new_content=None):
        self.new_content = self.content if new_content is None else new_content
        return self

    def set_replace_content(self, replace_content=None):
        self.replace_content = self.content if replace_content is None else replace_content
        return self

    def set_unicode_dict(self, unicode_dict=None):
        self.unicode_dict = UNICODE_DICT if unicode_dict is None else unicode_dict
        return self
    
    def remove_unicode(self):
        """
        Given a UNICODE_DICT with all 
        utf-8 symbol and corresponding non-uft-8 elements
        it substitutes the elements with respective notation.
        """
        assert self.new_content != None, "NEW CONTENT not set"
        assert self.replace_content != None, "REPLACE CONTENT not set"

        if not self.unicode_dict:
            # print(f'self.unicode_dict value: {self.unicode_dict}')
            self.new_content = UnicodeDammit(self.new_content).unicode_markup
            self.replace_content = UnicodeDammit(self.replace_content).unicode_markup
        else:
            self.new_content = ''.join([self.unicode_dict.get(c, c) for c in self.new_content])
            self.replace_content = ''.join([self.unicode_dict.get(c, c) for c in self.replace_content])
        return self

    def remove_nextline(self):
        """
        Given a UNICODE_DICT with all 
        utf-8 symbol and corresponding non-uft-8 elements
        it substitutes the elements with respective notation.
        """
        assert self.new_content != None, "NEW CONTENT not set"
        if "\n" in self.new_content:
            self.new_content = self.new_content.replace("\n", "")
        
        assert self.replace_content != None, "REPLACE CONTENT not set"
        if "\n" in self.replace_content:
            self.replace_content = self.replace_content.replace("\n", "")
        return self

    def remove_spaces_between_symbols(self):
        assert self.new_content != None, "NEW CONTENT not set"
        assert self.replace_content != None, "REPLACE CONTENT not set"
        
        if not self.unicode_dict:
            self.new_content = re.sub(r" ?± ?", r"±", self.new_content)
            self.replace_content = re.sub(r" ?± ?", r"±", self.replace_content)
        else:
            self.new_content = re.sub(r" ?\+- ?", r"+-", self.new_content)
            self.replace_content = re.sub(r" ?\+- ?", r"+-", self.replace_content)
        return self

    def comma_to_point(self):
        assert self.new_content != None, "NEW CONTENT not set"
        assert self.replace_content != None, "REPLACE CONTENT not set"
        self.new_content = ''.join([inter_digit(self.new_content[i-1:i+1]) if c == ',' else c for i, c in enumerate(self.new_content)])
        self.replace_content = ''.join([inter_digit(self.replace_content[i-1:i+1]) if c == ',' else c for i, c in enumerate(self.replace_content)])
        return self

    def replace_chars_and_digits(self):
        assert self.replace_content != None, "REPLACE CONTENT not set"
        word_list = [''.join(['x' if character.isdigit() else 'w' if character.isalpha() else character for character in word]) for word in self.replace_content.split()]
        self.replace_content = ' '.join([re.sub(r"(.)\1+", r"\1", subs) for subs in word_list])
        return self

    def remove_number_sign(self):
        assert self.replace_content != None, "REPLACE CONTENT not set"
        founds = list(re.finditer(r"-x", self.replace_content))
        if len(founds)>0:
            to_remove = [m[0] for m in founds[0].regs if (m[0] == 0 or self.replace_content[m[0]-1] not in ['+', 'w', 'x'])]
            self.replace_content = ''.join([el for i, el in enumerate(self.replace_content) if i not in to_remove])
        return self
