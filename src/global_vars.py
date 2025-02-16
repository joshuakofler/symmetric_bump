"""
Global Variables for Simulation:
--------------------------------

This module defines and initializes global variables used throughout the simulation.
The variables are grouped into categories such as grid information, primitive variables,
boundary conditions, and other numerical properties.
"""
import numpy as np
from constants import *

# ------------------------------ Grid Variables ------------------------------
# Grid spacing and coordinates
cell_dx = np.array([0.0], dtype='d') # (float): Spacing between cell centers in the x-direction.
cell_x_coords = None                 # (ndarray): x-coordinates of cell centers.
cell_y0 = None                       # (ndarray): Bump height at each cell center (x-direction).
cell_dy = None                       # (ndarray): Spacing between cell centers in the y-direction.
cell_y_coords = None                 # (ndarray): y-coordinates of cell centers.

# Face variables (for flux calculations)
face_x_coords = None                 # (ndarray): x-coordinates of cell faces.
face_y0 = None                       # (ndarray): Bump height at each cell face (x-direction).
face_dy = None                       # (ndarray): Spacing between cell faces in the y-direction.
face_y_coords = None                 # (ndarray): y-coordinates of cell faces.

# Cell area
cell_area = np.zeros([NUM_CELLS_X, NUM_CELLS_Y], dtype='d')  # (ndarray): Area of each computational cell.

# ndS Vector (normal vector for faces)
# Shape: [NUM_CELLS_X, NUM_CELLS_Y, 4, 2]
# Face index convention:
#   0 - south, 1 - east, 2 - north, 3 - west
ndS = np.zeros([NUM_CELLS_X, NUM_CELLS_Y, 4, 2], dtype='d')  # Normal vectors for cell faces.

# ------------------------------ Flow Properties ------------------------------
# Primitive variables
rho = np.zeros([NUM_CELLS_X, NUM_CELLS_Y], dtype='d')  # Density
u = np.zeros([NUM_CELLS_X, NUM_CELLS_Y, 2], dtype='d')  # Velocity vector (u_x, u_y)
E = np.zeros([NUM_CELLS_X, NUM_CELLS_Y], dtype='d')     # Total energy
e = np.zeros([NUM_CELLS_X, NUM_CELLS_Y], dtype='d')     # Internal energy
T = np.zeros([NUM_CELLS_X, NUM_CELLS_Y], dtype='d')     # Temperature
c = np.zeros([NUM_CELLS_X, NUM_CELLS_Y], dtype='d')     # Speed of sound
p = np.zeros([NUM_CELLS_X, NUM_CELLS_Y], dtype='d')     # Pressure
H = np.zeros([NUM_CELLS_X, NUM_CELLS_Y], dtype='d')     # Total enthalpy
M = np.zeros([NUM_CELLS_X, NUM_CELLS_Y], dtype='d')     # Mach number

# ------------------------------ Boundary Conditions ------------------------------
# Freestream conditions
p_infty = np.zeros([NUM_CELLS_Y])    # Freestream pressure
T_infty = np.zeros([NUM_CELLS_Y])    # Freestream temperature
rho_infty = np.zeros([NUM_CELLS_Y])  # Freestream density
c_infty = np.zeros([NUM_CELLS_Y])    # Freestream speed of sound
u_infty = np.zeros([NUM_CELLS_Y])    # Freestream velocity

# Boundary flow properties
# Inlet
p_in = np.zeros([NUM_CELLS_Y])
T_in = np.zeros([NUM_CELLS_Y])
rho_in = np.zeros([NUM_CELLS_Y])
u_in = np.zeros([NUM_CELLS_Y])
M_in = np.zeros([NUM_CELLS_Y])
H_in = np.zeros([NUM_CELLS_Y])

# Outlet
p_out = np.zeros([NUM_CELLS_Y])
T_out = np.zeros([NUM_CELLS_Y])
rho_out = np.zeros([NUM_CELLS_Y])
u_out = np.zeros([NUM_CELLS_Y])
M_out = np.zeros([NUM_CELLS_Y])
H_out = np.zeros([NUM_CELLS_Y])

# ------------------------------ Numerical Properties ------------------------------
# Vector of conserved variables U - here denoted as state vector
# Face index convention (third index):
#   0 - south, 1 - east, 2 - north, 3 - west
state_vector = np.zeros([NUM_CELLS_X, NUM_CELLS_Y, 4], dtype='d')

# Flux vector
# Face index convention (third index):
#   0 - south, 1 - east, 2 - north, 3 - west
F = np.zeros([NUM_CELLS_X, NUM_CELLS_Y, 4, 4], dtype='d')
F_star = np.zeros([NUM_CELLS_X, NUM_CELLS_Y, 4, 4], dtype='d')

f = np.zeros([NUM_CELLS_X, NUM_CELLS_Y, 4], dtype='d')  # Flux in x-direction
g = np.zeros([NUM_CELLS_X, NUM_CELLS_Y, 4], dtype='d')  # Flux in y-direction

# Residual
R = np.zeros([NUM_CELLS_X, NUM_CELLS_Y, 4], dtype='d')  # Residuals
# (Local) Time Step
dt = np.zeros([NUM_CELLS_X, NUM_CELLS_Y], dtype='d')    # Time step (local/global)


# ------------------------------ Artificial Dissipation ------------------------------
# Artificial dissipation
# Face index convention (third index):
#   0 - south, 1 - east, 2 - north, 3 - west
artificial_dissipation = np.zeros([NUM_CELLS_X, NUM_CELLS_Y, 4, 4])

# Artificial dissipation coefficients
a_d_coefficient_gamma = np.zeros([NUM_CELLS_X, NUM_CELLS_Y, 4])
a_d_coefficient_eta = np.zeros([NUM_CELLS_X, NUM_CELLS_Y, 4])

# Pressure sensor 
nu = np.zeros([NUM_CELLS_X, NUM_CELLS_Y], 'd')
nu_max = np.zeros([NUM_CELLS_X, NUM_CELLS_Y, 4], 'd')

# ------------------------------ Mass Flow ------------------------------
# Mass flow at inlet and outlet
m_in = np.zeros([MAX_ITERATIONS + 1], dtype='d')
m_out = np.zeros([MAX_ITERATIONS + 1], dtype='d')


# ------------------------------ Simulation Control ------------------------------
iteration = 0  # Current iteration
output_iterations = {}  # Output control (e.g., snapshots)
pvd_entries = []  # PVD file entries for vtk output

sim_dir = OUTPUT_DIR