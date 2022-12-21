import random   # python
import torch    # pytorch
import numpy    # numpy


#################################
#######  set the #! SEEDS
#################################

def set_seeds(seed):
    random.seed(seed)
    torch.manual_seed(seed)
    numpy.random.seed(seed)

