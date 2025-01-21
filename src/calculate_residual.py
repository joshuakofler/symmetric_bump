# WIP
# TODO: Done
from global_var import *
import numpy as np
import calculate_flux as cF

def update_residual():
    # update flux vector using the currently saved cell properties
    cF.update_flux()
    
    R[:,:,0] = np.sum(F[:,:,:,0], axis=2)
    R[:,:,1] = np.sum(F[:,:,:,1], axis=2)
    R[:,:,2] = np.sum(F[:,:,:,2], axis=2)
    R[:,:,3] = np.sum(F[:,:,:,3], axis=2)

    return None