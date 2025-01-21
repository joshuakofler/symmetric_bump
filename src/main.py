#%%
# import modules

# import all global variables from globals 
from constants import *
import global_vars as gv
import numpy as np

import importlib

# own libs
import mesh
import cell
import calculate_artificial_dissipation as aD
import calculate_flux as cF
import calculate_residual as cR
import RK
import plot

import data_io as io

importlib.reload(gv)
importlib.reload(mesh)
importlib.reload(cell)
importlib.reload(aD)
importlib.reload(cF)
importlib.reload(cR)
importlib.reload(RK)
importlib.reload(plot)
importlib.reload(io)

# Specify at which iterations the output should be saved
gv.output_iterations = {10, 20, 50, 100}

io.initialize_folder_structure()

# first initialize the mesh
mesh.initialize()

# initialize the cell paramerters over the whole domain
cell.initialize()

# initialize the residuals 
cR.update_residual()

# start simulation
gv.iteration = 0

for iter in range(MAX_ITERATIONS):
    RK.run_iteration()

    if gv.iteration in gv.output_iterations:
        io.save_iteration(gv.iteration)

    gv.iteration += 1

plot.Massflow()
    
#plot.Mach_number()