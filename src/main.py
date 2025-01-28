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
from matplotlib import pyplot as plt

importlib.reload(gv)
importlib.reload(mesh)
importlib.reload(cell)
importlib.reload(aD)
importlib.reload(cF)
importlib.reload(cR)
importlib.reload(RK)
importlib.reload(plot)
importlib.reload(io)

runSimulation = True
# Specify at which iterations the output should be saved
gv.output_iterations = {5, 1000, 1400, 1425, 1450, 1475, 1480, 1490, 1500, 2000, 3000, 4000, 5000}

loadSimulation = False
iteration = 5000
sim_file_path = OUTPUT_DIR / str(iteration) / f"{iteration}.vtp"

OUTPUT_DIR_STORED = PROJECT_DIR / "output_M_0_1"
sim_file_path = OUTPUT_DIR_STORED / str(iteration) / f"{iteration}.vtp"

# clear the output folder
clear_output_folder = False
if clear_output_folder:
    io.clear_folder_structure()

# first initialize the mesh
mesh.initialize()

if runSimulation:
    # initialize the folder structure to save the iterations
    io.initialize_folder_structure()

    # initialize the cell paramerters over the whole domain
    cell.initialize()

    # initialize the residuals 
    cR.update_residual(gv.state_vector)

    # start simulation

    for iter in range(MAX_ITERATIONS):        
        gv.iteration += 1 
        
        RK.run_iteration()

        if gv.iteration in gv.output_iterations:
            io.save_iteration(gv.iteration)

    # plot the results
    fig1, ax1 = plt.subplots(figsize=(8,6))
    plot.plot_convergence_history(fig1)

    # plot the Mach number
    fig2, ax2 = plt.subplots(figsize=(24,8))
    plot.plot_cell_data(fig2, gv.M, "Mach number", "M")

if loadSimulation:
    gv.iteration = iteration

    [gv.state_vector, gv.m_in, gv.m_out] = io.read_iteration(sim_file_path)

    cell.update_cell_properties(gv.state_vector)

    fig0, ax0 = plt.subplots(figsize=(8,6))
    plot.plot_convergence_history(fig0)

    fig1, ax1 = plt.subplots(figsize=(24,8))
    plot.plot_cell_data(fig1, gv.M, "Mach number", "M")

    fig2, ax2 = plt.subplots(figsize=(24,8))
    plot.plot_cell_data(fig2, gv.u[:,:,1], "Tangential Velocity", "Uy")
# %%
