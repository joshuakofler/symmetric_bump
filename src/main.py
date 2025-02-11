#%%
# Internal flow in a channel with a thin symmetric bump

# Import necessary modules

# Import all global variables from globals
from constants import *  # Import constants used throughout the code
import global_vars as gv  # Import global variables
import numpy as np  # Import numpy for numerical operations
import os

import importlib  # Import importlib for module reloading

# Import own libraries
import mesh  # Mesh handling
import cell  # Cell properties
import calculate_artificial_dissipation as aD  # Artificial dissipation calculation
import calculate_flux as cF  # Flux calculation
import calculate_residual as cR  # Residual computation
import RK  # Runge-Kutta solver
import plot  # Plotting
import data_io as io  # Input/output operations

from matplotlib import pyplot as plt

# Reload modules to ensure the latest changes are applied
importlib.reload(gv)
importlib.reload(mesh)
importlib.reload(cell)
importlib.reload(aD)
importlib.reload(cF)
importlib.reload(cR)
importlib.reload(RK)
importlib.reload(plot)
importlib.reload(io)

# Simulation mode selection
# -----------------------------------------------------------
# mode = 0 -> Run the simulation
# mode = 1 -> Run a previous simulation 
# mode = 2 -> Load a previous simulation file to analyze
# mode = 3 -> Clear output folder
mode = 0

# Simulation parameters
# -----------------------------------------------------------
# Specify the iterations at which output should be saved
gv.output_iterations = np.arange(100, MAX_ITERATIONS + 1, 100)

# Load simulation parameters
# -----------------------------------------------------------
# Define the iteration number to load when analyzing an old simulation
iteration = 5000

# Optionally override the OUTPUT parent directory (uncomment if needed)
# OUTPUT_DIR_STORED = PROJECT_DIR / "results" / "output_M_0_85"
# OUTPUT_DIR = OUTPUT_DIR_STORED

# Construct the file path for the stored simulation data
# sim_dir = os.path.join(OUTPUT_DIR, str(iteration))
# sim_file_path = os.path.join(sim_dir, f"{iteration}.vts")

sim_dir = os.path.join(RESULT_DIR, "bump_0_08", "M_0_2")
sim_iteration_dir = os.path.join(sim_dir, str(iteration))
sim_file_path = os.path.join(sim_iteration_dir, f"{iteration}.vts")

# .pvd file name
# for running a previous simulation
pvd_filename = os.path.join(OUTPUT_DIR, "2025-02-10_16-36-40_solution_225_113.pvd")

# Initialize the mesh before starting simulation
mesh.initialize()

# Run the simulation if mode is set to 0
if mode == 0:
    # Initialize folder structure for saving iteration outputs
    io.initialize_folder_structure()

    # Initialize cell parameters across the computational domain
    cell.initialize()

    # Compute initial residuals
    cR.update_residual(gv.state_vector)
    
    # Start the main simulation loop
    for iter in range(MAX_ITERATIONS):        
        gv.iteration += 1  # Increment iteration counter
        
        # Run a single iteration of the solver
        RK.run_iteration()
        
        # Save output at specified iterations
        if gv.iteration in gv.output_iterations:
            io.save_iteration(gv.iteration)
    
    # Save output in ParaView format
    io.save_pvd()
    
    # Plot convergence history
    fig1, ax1 = plt.subplots(figsize=(8,6))
    plot.plot_convergence_history(fig1)
    
    # Plot Mach number distribution
    fig2, ax2 = plt.subplots(figsize=(24,8))
    plot.plot_cell_data(fig2, gv.M, "Mach number", "M")

if mode == 1:
    gv.iteration = iteration  # Set iteration number

    gv.output_iterations = np.arange(gv.iteration + 100, MAX_ITERATIONS + 1, 100)

    io.initialize_folder_structure()

    # Load pvd file
    io.read_existing_pvd(pvd_filename)

    # Load state vector and mass flow data from the stored file
    io.read_iteration(sim_file_path)

    gv.state_vector[:, :, 0] = gv.rho[:,:]
    gv.state_vector[:, :, 1] = gv.rho[:,:] * gv.u[:,:,0]
    gv.state_vector[:, :, 2] = gv.rho[:,:] * gv.u[:,:,1]
    gv.state_vector[:, :, 3] = gv.rho[:,:] * gv.E[:,:]

    # Define stagnation properties
    cell.calculate_inlet_properties()
    # Initialize cell parameters across the computational domain
    cell.update_cell_properties(gv.state_vector)

    # Compute initial residuals
    cR.update_residual(gv.state_vector)

    # Start the main simulation loop
    for iter in range(iteration, MAX_ITERATIONS):        
        gv.iteration += 1  # Increment iteration counter
        
        # Run a single iteration of the solver
        RK.run_iteration()
        
        # Save output at specified iterations
        if gv.iteration in gv.output_iterations:
            io.save_iteration(gv.iteration)

    # Save output in ParaView format
    io.save_pvd()
    
    # Plot convergence history
    fig1, ax1 = plt.subplots(figsize=(8,6))
    plot.plot_convergence_history(fig1)
    
    # Plot Mach number distribution
    fig2, ax2 = plt.subplots(figsize=(12,4))
    plot.plot_cell_data(fig2, gv.M, "Mach number", "M")


# Load and analyze a previous simulation if mode is set to 1
if mode == 2:
    gv.iteration = iteration  # Set iteration number
    
    # Load state vector and mass flow data from the stored file
    io.read_iteration(sim_file_path)

    gv.state_vector[:, :, 0] = gv.rho[:,:]
    gv.state_vector[:, :, 1] = gv.rho[:,:] * gv.u[:,:,0]
    gv.state_vector[:, :, 2] = gv.rho[:,:] * gv.u[:,:,1]
    gv.state_vector[:, :, 3] = gv.rho[:,:] * gv.E[:,:]

    # Define stagnation properties
    cell.calculate_inlet_properties()
    # Initialize cell parameters across the computational domain
    cell.update_cell_properties(gv.state_vector)
    
    # # Plot convergence history of the loaded simulation
    # fig0, ax0 = plt.subplots(figsize=(8,6))
    # plot.plot_convergence_history(fig0, sim_dir)
    
    # Plot Mach number distribution
    fig1, ax1 = plt.subplots(figsize=(24,8))
    plot.plot_cell_data(fig1, gv.M, "Mach number", "M", sim_dir)
    
# Option to clear the output folder before running a new simulation
if mode == 3:
    io.clear_folder_structure()