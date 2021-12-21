import numpy as np


def get_persistence(dataloader,target_idx,lag=30):
    N_inds = len(dataloader)
    sg_indices = dataloader.sg_indices 
    pers_ind = sg_indices[:,-1]-lag 
    return dataloader.supermag_data[pers_ind,:,target_idx]

