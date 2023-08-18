from functools import partial

import torch
import numpy as np

def multi_apply(func, *args, **kwargs):
    pfunc = partial(func, **kwargs) if kwargs else func
    map_results = map(pfunc, *args)
    return tuple(map(list, zip(*map_results)))


def torch_to_numpy(x):
    assert isinstance(x, torch.Tensor)
    return x.detach().cpu().numpy()

# def decompress_partseg(mask_compressed):
#     return np.stack([x.toarray() for x in mask_compressed], axis=0)