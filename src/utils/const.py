from attrdict import AttrDict
from enum import Enum

class Categories_names(Enum):
    """ Original categories names """
    OTHER       = 0
    TEXT        = 1
    TITLE       = 2
    LIST        = 3
    TABLE       = 4 #! NO NODES
    FIGURE      = 5
    CAPTION     = 6
    TABLE_COLH  = 7
    TABLE_SP    = 8
    TABLE_GCELL = 9 #! NO NODES
    TABLE_TCELL = 10
    TABLE_COL   = 11 #! NO NODES
    TABLE_ROW   = 12 #! NO NODES

#* categories_colors
categories_colors = AttrDict({
    0:  [ 127, 127, 127 ],   # 'OTHER'
    1:  [ 255,  0, 0 ],   # 'TEXT'
    2:  [ 0, 255, 0 ],   # 'TITLE'
    3:  [  0, 0, 255 ],   # 'LIST'
    4:  [ 255, 255, 0 ],   # 'TABLE'
    5:  [ 255,  0, 255 ],   # 'FIGURE'
    6:  [  0,  255, 255 ],   # 'CAPTION'
    7:  [ 255, 150,  0 ],   # 'TABLE-COLH'
    8:  [ 0,   255,   150 ],   # 'TABLE-SPANNING-CELL'
    9:  [ 150, 0, 255 ],   # 'TABLE-GRID-CELL'
    10: [ 255,  155, 155 ],   # 'TABLE-TEXT-CELL'
    11: [ 155, 255, 155 ],   # 'TABLE-COLUMN'
    12: [  155,  155,  255 ],   # 'TABLE-ROW'
})

categories_names = AttrDict({
    0:  'OTHER',
    1:  'TEXT',
    2:  'TITLE',
    3:  'LIST',
    4:  'TABLE',
    5:  'FIGURE',
    6:  'CAPTION',
    7:  'TABLE-COLH',
    8:  'TABLE-SP',
    9:  'TABLE-GCELL',
    10: 'TABLE-TCELL',
    11: 'TABLE-COL',
    12: 'TABLE-ROW'
})

categories_names_underscore = AttrDict({
    'OTHER':  'OTHER',
    'TEXT': 'TEXT',
    'TITLE':  'TITLE',
    'LIST':  'LIST',
    'TABLE':  'TABLE',
    'FIGURE':  'FIGURE',
    'CAPTION':  'CAPTION',
    'TABLE_COLH':  'TABLE-COLH',
    'TABLE_SP':  'TABLE-SP',
    'TABLE_GCELL':  'TABLE-GCELL',
    'TABLE_TCELL': 'TABLE-TCELL',
    'TABLE_COL': 'TABLE-COL',
    'TABLE_ROW': 'TABLE-ROW'
})

SCALE_FACTOR = 0.36

RANDOM_SEED = 42