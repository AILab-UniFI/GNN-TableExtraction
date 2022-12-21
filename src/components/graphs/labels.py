from itertools import count
import numpy as np
from src.utils.const import Categories_names

MAX_CLASS_COUNT = len(Categories_names)

class LableModification:
            
    def __init__(self, config):
        self.config = config
        self.to_remove = config['LABELS']['to_remove']
        self.to_remove = np.array(self.to_remove)
        self.origin_to_conv = dict(
            {key:value for key, value in zip(
                range(MAX_CLASS_COUNT), 
                [id - len(self.to_remove[self.to_remove < id]) if id not in self.to_remove else None for id in range(MAX_CLASS_COUNT)]
            )}
        )
        self.conv_to_origin = dict(
            {key:value for value,key in self.origin_to_conv.items()}
        )

    def convert(self, orig_labels):
        return [self.origin_to_conv.get(ol) for ol in orig_labels]

    def revert(self, converted_labels):
        return [self.conv_to_origin.get(cl) for cl in converted_labels]

