import torch 
import numpy as np 

def as_tt(frame): 
    return torch.tensor(frame.values.astype(np.float32))
