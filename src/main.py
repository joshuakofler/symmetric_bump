#%%
# import modules

# import all global variables from globals 
from global_var import *
from global_var import c_infty

import numpy as np
import importlib

# own libs
import calculate_artificial_dissipation as aD
import calculate_flux as cF
import calculate_residual as cR
import RK
import plot

import mesh
import cell

# import global_var

# importlib.reload(global_var)
importlib.reload(mesh)
importlib.reload(aD)
importlib.reload(cF)
importlib.reload(cR)
importlib.reload(RK)
importlib.reload(plot)
importlib.reload(cell)



mesh.initialize()

cell.calculate_inlet_properties()

cell.initialize()

plot.Mach_number()

#plot.Mach_number()

# for iter in range(10):
#     iteration += 1
    
#     RK.run_iteration(state_vector)

#     plot.Mach_number()
