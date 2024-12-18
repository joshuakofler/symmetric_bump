import numpy as np

import constants as const



# Declare all global variables used throughout the function
cell_dx = None          # (float):      Grid spacing between cell centers in the x-direction.
cell_x_coords = None    # (ndarray):    x-coordinates of cell centers.
cell_y0 = None          # (ndarray):    Bump height at each cell center in the x-direction.
cell_dy = None          # (ndarray):    Grid spacing between cell centers in the y-direction.
cell_y_coords = None    # (ndarray):    y-coordinates of cell centers.

face_x_coords = None    # (ndarray):    x-coordinates of cell faces.
face_y0 = None          # (ndarray):    Bump height at each cell face in the x-direction.
face_dy = None          # (ndarray):    Grid spacing between cell faces in the y-direction.
face_y_coords = None    # (ndarray):    y-coordinates of cell faces.

# cell area
# (ndarray):    Area of each computational cell.
cell_area = np.zeros([const.NUM_CELLS_X, const.NUM_CELLS_Y])

# ndS vector
#
# The ndS vector is a field defined on the faces of a grid cell.
# The third index specifies the face, and the fourth index can be used to access specific components:
#
# dS[i,j,0,:]  - Accesses the ndS vector at the south face of cell (i,j).
# dS[i,j,1,:]  - Accesses the ndS vector at the west face of cell (i,j).
# dS[i,j,2,:]  - Accesses the ndS vector at the north face of cell (i,j).
# dS[i,j,3,:]  - Accesses the ndS vector at the east face of cell (i,j).
#
# To access specific components (e.g., x-component):
# dS[i,j,1,0]  - Accesses the x-component at the west face of cell (i,j).
#
# Face index convention:
#   0 - south
#   1 - west
#   2 - north
#   3 - east
ndS = np.zeros([const.NUM_CELLS_X, const.NUM_CELLS_Y, 4, 2], 'd')



# flux vector
F = np.zeros([const.NUM_CELLS_X, const.NUM_CELLS_Y, 4], 'd')

# residuals
R = np.zeros([const.NUM_CELLS_X, const.NUM_CELLS_Y], 'd')

