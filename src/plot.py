# import modules
from global_var import *
import numpy as np
from matplotlib import pyplot as plt

def Mach_number():
    fig, ax = plt.subplots()

    #######################################################################
    # Plot parameters
    #######################################################################
    show_horizontal_grid = True
    show_vertical_grid = True
    show_face_points = True
    show_center_points = True
    show_diagonal_lines = False
    show_cell_area = False
    
    show_normal_vector = False
    show_normal_vector_south = True
    show_normal_vector_north = True
    show_normal_vector_west = True
    show_normal_vector_east = True

    x_axis_limits = [-DOMAIN_LENGTH-cell_dx/2, 2*DOMAIN_LENGTH+cell_dx/2]
    y_axis_limits = [0-cell_dy[0]/2, CHANNEL_HEIGHT+cell_dy[0]/2]

    save_grid_plot = False
    if save_grid_plot:
        filename = "../img/Mach_number.pdf"
    #######################################################################
    # Plot the Grid
    #######################################################################
    
    if show_vertical_grid:
        # Plot vertical grid lines at each y-face.
        for i, j in np.ndindex(NUM_CELLS_X, NUM_FACES_Y):
            ax.plot(
                [face_x_coords[i], face_x_coords[i+1]],  # x-coordinates for the vertical line
                [face_y_coords[i, j], face_y_coords[i+1, j]],  # y-coordinates for the line at this vertical slice
                '--', color="gray", linewidth=0.5  # Dashed gray line with thin width
            )

    if show_horizontal_grid:
        # Plot horizontal grid lines at each x-face.
        for i, j in np.ndindex(NUM_FACES_X, NUM_CELLS_Y):
            ax.plot(
                [face_x_coords[i], face_x_coords[i]],  # x-coordinate remains constant (vertical face)
                [face_y_coords[i, j], face_y_coords[i, j+1]],  # Connect y-coordinates for this x-face
                '--', color="gray", linewidth=0.5  # Dashed gray line
            )

    ax.set_xlim(x_axis_limits)
    ax.set_ylim(y_axis_limits)

    ax.set_title("Mach number")
    ax.set_xlabel(r"$x$")
    ax.set_ylabel(r"$y$")

    if save_grid_plot:
        fig.savefig(filename)

    return None