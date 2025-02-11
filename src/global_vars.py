import numpy as np
from constants import *

# Declare all global variables used throughout the function
cell_dx = np.array([0.0], 'd')      # (float): Grid spacing between cell centers in the x-direction.

cell_x_coords = None    # (ndarray): x-coordinates of cell centers.
cell_y0 = None          # (ndarray): Bump height at each cell center in the x-direction.
cell_dy = None          # (ndarray): Grid spacing between cell centers in the y-direction.
cell_y_coords = None    # (ndarray): y-coordinates of cell centers.

face_x_coords = None    # (ndarray): x-coordinates of cell faces.
face_y0 = None          # (ndarray): Bump height at each cell face in the x-direction.
face_dy = None          # (ndarray): Grid spacing between cell faces in the y-direction.
face_y_coords = None    # (ndarray): y-coordinates of cell faces.

# Cell area
# (ndarray): Area of each computational cell.
cell_area = np.zeros([NUM_CELLS_X, NUM_CELLS_Y])

# ndS vector
#
# The ndS vector is a field defined on the faces of a grid cell.
# The third index specifies the face, and the fourth index can be used to access specific components:
#
# dS[i,j,0,:]  - Accesses the ndS vector at the south face of cell (i,j).
# dS[i,j,1,:]  - Accesses the ndS vector at the east face of cell (i,j).
# dS[i,j,2,:]  - Accesses the ndS vector at the north face of cell (i,j).
# dS[i,j,3,:]  - Accesses the ndS vector at the west face of cell (i,j).
#
# Face index convention:
#   0 - south
#   1 - east
#   2 - north
#   3 - west
ndS = np.zeros([NUM_CELLS_X, NUM_CELLS_Y, 4, 2], 'd')

# State vector
state_vector = np.zeros([NUM_CELLS_X, NUM_CELLS_Y, 4], 'd')

# Primitive variables
rho = np.zeros([NUM_CELLS_X, NUM_CELLS_Y], 'd')  # Density
u = np.zeros([NUM_CELLS_X, NUM_CELLS_Y, 2], 'd')  # Velocity vector
E = np.zeros([NUM_CELLS_X, NUM_CELLS_Y], 'd')    # Total energy
e = np.zeros([NUM_CELLS_X, NUM_CELLS_Y], 'd')    # Internal energy
T = np.zeros([NUM_CELLS_X, NUM_CELLS_Y], 'd')    # Temperature
c = np.zeros([NUM_CELLS_X, NUM_CELLS_Y], 'd')    # Speed of sound
p = np.zeros([NUM_CELLS_X, NUM_CELLS_Y], 'd')    # Pressure
H = np.zeros([NUM_CELLS_X, NUM_CELLS_Y], 'd')    # Total enthalpy
M = np.zeros([NUM_CELLS_X, NUM_CELLS_Y], 'd')    # Mach number


# massflow
# inlet
m_in = np.zeros([MAX_ITERATIONS+1], 'd')
# outlet
m_out = np.zeros([MAX_ITERATIONS+1], 'd')

# Upstream fluid values
p_infty = np.zeros([NUM_CELLS_Y])
T_infty = np.zeros([NUM_CELLS_Y])
rho_infty = np.zeros([NUM_CELLS_Y])
c_infty = np.zeros([NUM_CELLS_Y])
u_infty = np.zeros([NUM_CELLS_Y])


p_in = np.zeros([NUM_CELLS_Y])
T_in = np.zeros([NUM_CELLS_Y])
c_in = np.zeros([NUM_CELLS_Y])
u_in = np.zeros([NUM_CELLS_Y])
M_in = np.zeros([NUM_CELLS_Y])
rho_in = np.zeros([NUM_CELLS_Y])
H_in = np.zeros([NUM_CELLS_Y])


p_out = np.zeros([NUM_CELLS_Y])
T_out = np.zeros([NUM_CELLS_Y])
c_out = np.zeros([NUM_CELLS_Y])
u_out = np.zeros([NUM_CELLS_Y])
M_out = np.zeros([NUM_CELLS_Y])
rho_out = np.zeros([NUM_CELLS_Y])
H_out = np.zeros([NUM_CELLS_Y])


f = np.zeros([NUM_CELLS_X, NUM_CELLS_Y, 4], 'd')
g = np.zeros([NUM_CELLS_X, NUM_CELLS_Y, 4], 'd')

# Flux vector
F = np.zeros([NUM_CELLS_X, NUM_CELLS_Y, 4, 4], 'd')
F_star = np.zeros([NUM_CELLS_X, NUM_CELLS_Y, 4, 4], 'd')


# Dissipation
artificial_dissipation = np.zeros([NUM_CELLS_X, NUM_CELLS_Y, 4, 4])

a_d_coefficient_gamma = np.zeros([NUM_CELLS_X, NUM_CELLS_Y, 4])
a_d_coefficient_eta = np.zeros([NUM_CELLS_X, NUM_CELLS_Y, 4])

nu = np.zeros([NUM_CELLS_X, NUM_CELLS_Y], 'd')
nu_max = np.zeros([NUM_CELLS_X, NUM_CELLS_Y, 4], 'd')
nu_max_2 = np.zeros([NUM_CELLS_X, NUM_CELLS_Y, 4], 'd')

# Residuals
R = np.zeros([NUM_CELLS_X, NUM_CELLS_Y, 4], 'd')

dt = 0

iteration = 0

output_iterations = {}

pvd_entries = []

useCircularArc = False