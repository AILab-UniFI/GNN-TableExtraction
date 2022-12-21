
from attrdict import AttrDict


class Embedder():

    def __init__(self, 
                specifics=None,
                input=None,
                name=None):
        
        assert name != None, 'Value Error: name is None'
        self.specifics = None if specifics == None else AttrDict(specifics)
        self.input = None if input == None else AttrDict(input)
        self.name = name

        self.inizialize()

    def inizialize(self):
        raise NotImplementedError

    def __call__(self, bboxs, string):
        raise NotImplementedError
        