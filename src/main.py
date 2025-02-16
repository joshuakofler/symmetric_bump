#%%
# Internal Flow Simulation in a Channel with a Thin Symmetric Bump
# -----------------------------------------------------------
# This script performs a numerical simulation of an internal flow in a two-dimensional channel
# with a thin symmetric bump on the bottom wall. The bump introduces a perturbation to the flow, 
# allowing for the study of flow features such as pressure distribution, velocity profiles. 
#
# The simulation employs a finite volume approach to solve the compressible Euler equations, 
# using a structured, quasi-uniform grid and a Runge-Kutta scheme for time integration. Artificial 
# dissipation terms are included to stabilize the solution, and subsonic or supersonic conditions 
# can be specified based on the upstream Mach number.
#
# Features of the Simulation:
# - Structured grid with customizable dimensions (NUM_CELLS_X, NUM_CELLS_Y).
# - Artificial dissipation handling for both subsonic and supersonic flows.
# - Output control for intermediate results at specified iterations.
# - Modular design, with separate components for grid generation, flux calculation, residual 
#   computation, and post-processing.
#
# Usage Modes:
# -----------------------------------------------------------
# This script supports four distinct modes of operation:
# - **mode = 0**: Run a new simulation from the initial conditions.
# - **mode = 1**: Resume and continue a previously saved simulation.
# - **mode = 2**: Load and analyze results from a specific simulation iteration.
# - **mode = 3**: Clear all output files in the simulation directory.
#
# Output:
# -----------------------------------------------------------
# Simulation results, including pressure, velocity, and Mach number distributions, 
# are saved to the OUTPUT_DIR directory as .vts and .pvd files. These files can be 
# visualized using external tools such as ParaView.
#
# To adjust key parameters such as grid resolution, bump height, or Mach number, 
# modify the constants in `constants.py`.
#
# The modular design also allows easy customization of methods for artificial dissipation, 
# flux computation, and numerical solvers. Use the imports section below to include and 
# reload any modified modules during development.


# Import Necessary Modules
# -----------------------------------------------------------
# Global constants and variables
from constants import *  # Import constants used throughout the code
import global_vars as gv  # Import global variables

# Numerical and file handling libraries
import numpy as np  # For numerical operations
import os  # For directory and file path handling
import importlib  # For reloading modules during development

# Custom modules
import mesh  # Mesh generation and handling
import cell  # Cell property calculations
import calculate_artificial_dissipation as aD  # Artificial dissipation calculation
import calculate_flux as cF  # Flux computation
import calculate_residual as cR  # Residual calculation
import RK  # Runge-Kutta time integration solver
import plot  # Visualization and plotting
import data_io as io  # Input/output operations

# Plotting library
from matplotlib import pyplot as plt

# Reload modules to ensure the latest changes are applied
# (Useful during iterative development)
importlib.reload(io)

# Simulation Mode Selection
# -----------------------------------------------------------
# Select one of the following simulation modes:
# mode = 0 -> Run a new simulation
# mode = 1 -> Continue a previous simulation
# mode = 2 -> Load and analyze results from a previous simulation
# mode = 3 -> Clear the output folder
mode = 2

# Simulation Parameters
# -----------------------------------------------------------
# Specify the iteration numbers at which output data should be saved
# This can be defined as a list of specific integers:
gv.output_iterations = [100, 200, 300, 400, 500]

# Alternatively, use np.arange to define output intervals (uncomment if needed)
# gv.output_iterations = np.arange(100, MAX_ITERATIONS + 1, 100)

# Simulation Directory and File Management
# -----------------------------------------------------------
# Define the iteration number to load for analysis or continuing a simulation
iteration = 100

# Define the parent directory for simulation output files
# DEFAULT: OUTPUT_DIR from constants (uncomment if needed)
gv.sim_dir = os.path.join(RESULT_DIR, "bump_0_08", "M_0_2")

# Construct the directory and file paths for loading simulation data
sim_iteration_dir = os.path.join(gv.sim_dir, str(iteration))  # Directory for a specific iteration
sim_file_path = os.path.join(sim_iteration_dir, f"{iteration}.vts")  # .vts file path

# -----------------------------------------------------------
# Programm start
# -----------------------------------------------------------

# Initialize the mesh before starting simulation
mesh.initialize()

# Run the simulation if mode is set to 0
if mode == 0:
    # Display initial simulation parameters in the console
    io.print_simulation_info()

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
    plot.plot_convergence_history(fig1, sim_iteration_dir, False)
    
    # Plot Mach number distribution
    fig2, ax2 = plt.subplots(figsize=(24,8))
    plot.plot_cell_data(fig2, gv.M, "Mach number", "M", sim_iteration_dir, False)

# Continue previous simulation if mode is set to 1
if mode == 1:
    gv.iteration = iteration  # Set iteration number

    # Display initial simulation parameters in the console
    io.print_simulation_info()

    # Initialize folder structure for saving iteration outputs
    io.initialize_folder_structure()

    # Load pvd file
    io.read_existing_pvd()

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
    plot.plot_convergence_history(fig1, sim_iteration_dir, False)
    
    # Plot Mach number distribution
    fig2, ax2 = plt.subplots(figsize=(24,8))
    plot.plot_cell_data(fig2, gv.M, "Mach number", "M", sim_iteration_dir, False)

# Load and analyze a previous simulation if mode is set to 2
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
    fig0, ax0 = plt.subplots(figsize=(8,6))
    plot.plot_convergence_history(fig0, sim_iteration_dir, False)

    # Plot pressure coefficient at the bottom wall
    fig1, ax1 = plt.subplots(figsize=(8,6))
    plot.plot_Cp(fig1, sim_iteration_dir, False)
    
    # Plot Mach number distribution
    fig2, ax2 = plt.subplots(figsize=(24,8))
    plot.plot_cell_data(fig2, gv.M, "Mach number", "M", sim_iteration_dir, False)

# Clear the output folder if mode is set to 3
if mode == 3:
    io.clear_folder_structure()