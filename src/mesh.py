#%%
# this field generates the mesh

# import modules
from fileinput import filename
import numpy as np
import importlib
from matplotlib import pyplot as plt
fig, ax = plt.subplots()


import constants as const

# reload constant lib
importlib.reload(const)

def get_y0(x):
    try:
        # Compute y0 using np.where
        y0 = np.where(
            (x < 0) | (x > const.L),  # Use bitwise OR for the condition
            0,                        # Value when condition is True
            const.epsilon * x * (1 - x / const.L)  # Value when condition is False
        )
        
        # Check if any values in y0 are negative
        if np.any(y0 < 0):
            raise ValueError("[Mesh] y0 contains negative values.")
        
        return y0
    except ValueError as e:
        print(f"Error: {e}")
        return None  # Optionally return a default value or re-raise the error

#######################################################################
# Plot parameters
#######################################################################
show_horizontal_grid = True
show_vertical_grid = True
show_face_points = True
show_center_points = True
show_diagonal_lines = True

x_axis_limits = [-1, 2]
y_axis_limits = [0, 1]

# ax.set_xlim([-const.L-dx/2, 2*const.L+dx/2]);
# ax.set_ylim([0-dy[0]/2, const.channel_height+dy[0]/2]);

save_grid_plot = True
if save_grid_plot:
    grid_plot_filename = "mesh.pdf"

#######################################################################
# Initialize the grid center coordinates
#######################################################################

# Calculate grid spacing in the x-direction
# dx is the spacing between the centers (or faces) of the cells in the x-direction
dx = const.channel_length / const.Nx_c

# Generate the x-coordinates for the centers of the cells
cell_x_coords = np.linspace(dx / 2, const.channel_length - dx / 2, const.Nx_c)

# Shift the x-coordinates by L to align with the desired coordinate system
cell_x_coords -= const.L

# Compute the bump height at the center of each x-cell
# This uses the function get_y0 to calculate y0 values based on cell_x_coords
cell_y0 = get_y0(cell_x_coords)

# Calculate the grid spacing in the y-direction
# The spacing is adjusted to account for the varying bump height y0_c
dy = (const.channel_height - cell_y0) / const.Ny_c

# Generate the y-coordinates for the centers of the cells in the y-direction
cell_y_coords = np.transpose(np.linspace(cell_y0 + dy / 2, const.channel_height - dy / 2, const.Ny_c))

#######################################################################
# Initialize the grid face coordinates
#######################################################################
# Generate the x-coordinates for the faces of the cells
face_x_coords = np.linspace(0, const.channel_length, const.Nx_f)

# Shift the x-coordinates by L to align with the desired coordinate system
face_x_coords -= const.L

# Compute the bump height at the faces of each x-cell
face_y0 = get_y0(face_x_coords)

# Calculate the grid spacing in the y-direction
# The spacing is adjusted to account for the varying bump height y0_f
face_y_coords = np.linspace(face_y0, const.channel_height, const.Ny_f)
# Transpose the face_y_coords array so that the indexing is consistent.
# After transposing, face_y_coords(x, index_y) will correctly match the structure where
# the first index corresponds to the x-position and the second to the y-position.
face_y_coords = np.transpose(face_y_coords)

#######################################################################
# Plot the Grid
#######################################################################

if show_vertical_grid:
    # Sweep over the x-direction to plot vertical grid lines connecting points at each x-face.
    # This loop iterates over cell centers along the x-direction.
    for i in range(const.Nx_c):
        # For each x-cell, iterate over the y-faces and plot vertical lines between consecutive x-faces.
        for j in range(const.Ny_f):
            ax.plot(
                [face_x_coords[i], face_x_coords[i+1]],  # x-coordinates for the vertical line
                [face_y_coords[i, j], face_y_coords[i+1, j]],  # y-coordinates for the line at this vertical slice
                '--', color="gray", linewidth=0.5  # Dashed gray line with thin width
            )

if show_horizontal_grid:
    # Plot horizontal grid lines at each x-face.
    # This loop iterates over all x-faces and connects consecutive y-points at fixed x.
    for i in range(const.Nx_f):
        for j in range(const.Ny_c):  # Iterate over y-cell centers
            ax.plot(
                [face_x_coords[i], face_x_coords[i]],  # x-coordinate remains constant (vertical face)
                [face_y_coords[i, j], face_y_coords[i, j+1]],  # Connect y-coordinates for this x-face
                '--', color="gray", linewidth=0.5  # Dashed gray line
            )

if show_diagonal_lines:
    # Plot diagonal lines for each cell to form a criss-cross pattern.
    # This ensures the diagonals of each cell are displayed for visual clarity.
    for i in range(const.Nx_c):
        for j in range(const.Ny_c):
            # Diagonal from bottom-left to top-right
            ax.plot(
                [face_x_coords[i], face_x_coords[i+1]],  # x-coordinates of the diagonal
                [face_y_coords[i, j], face_y_coords[i+1, j+1]],  # y-coordinates of the diagonal
                '--', color="gray", linewidth=0.5  # Dashed gray line
            )
            # Diagonal from top-left to bottom-right
            ax.plot(
                [face_x_coords[i], face_x_coords[i+1]],  # x-coordinates of the diagonal
                [face_y_coords[i, j+1], face_y_coords[i+1, j]],  # y-coordinates of the diagonal
                '--', color="gray", linewidth=0.5  # Dashed gray line
            )

if show_center_points:
    # Plot markers for the cell centers.
    # These correspond to cell_x_coords (center of each cell in the x-direction) and cell_y_coords (center for each x).
    for index, x in enumerate(cell_x_coords):
        ax.scatter(
            np.full_like(cell_y_coords[index, :], x),  # Fill x-values to match cell_y_coords array for this x-index
            cell_y_coords[index, :],  # y-coordinates at this x
            marker='o', s=13, color='k'  # Black circles as markers
        )

if show_face_points:
    # Plot markers for the cell faces.
    # These correspond to face_x_coords (faces in the x-direction) and face_y_coords (faces in the y-direction).
    for index, x in enumerate(face_x_coords):
        ax.scatter(
            np.full_like(face_y_coords[index, :], x),  # Fill x-values to match face_y_coords array for this x-index
            face_y_coords[index, :],  # y-coordinates at this x-face
            marker='s', s=15, color='g'  # Green squares as markers
        )

ax.set_xlim(x_axis_limits)
ax.set_ylim(y_axis_limits)

if save_grid_plot:
    fig.savefig(grid_plot_filename)