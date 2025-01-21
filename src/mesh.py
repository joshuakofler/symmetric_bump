#%%
"""
Initialize the mesh for the computational domain.

This module sets up the mesh structure by calculating the coordinates of cell centers and faces
in both the x- and y-directions. It also computes the varying bump height in the domain, adjusts the
spacing to account for this geometry, and calculates the area of each computational cell.


Notes:
    - The grid alignment and spacing are adjusted based on the bump height (cell_y0 and face_y0).
    - The function modifies global variables directly and does not return any value.
"""

# import modules
from constants import *
import global_vars as gv
import numpy as np
from matplotlib import pyplot as plt

# define the functions
def get_point_coordinates(cell_x_index, cell_y_index):
    """
    Return the cell coordinates based on the cell indices.

    Args:
        cell_x_index (int): Index of the cell along the x-axis.
        cell_y_index (int): Index of the cell along the y-axis.

    Returns:
        np.ndarray: [x-coordinate, y-coordinate] of the point.
    """
    return np.array([gv.cell_x_coords[cell_x_index], gv.cell_y_coords[cell_x_index, cell_y_index]])

def get_vertex_coordinates(cell_x_index, cell_y_index, vertex):
    """
    Return the coordinates of a cell edge point based on the specified direction.

    Args:
        cell_x_index (int): Index of the cell along the x-axis.
        cell_y_index (int): Index of the cell along the y-axis.
        vertex (str): Direction of the point ("SW", "NW", "NE", "SE") or ("A", "B", "C", "D").

    Returns:
        np.ndarray: [x-coordinate, y-coordinate] of the cell edge point.

    Raises:
        ValueError: If the direction argument is invalid.
    """
    try:
        match vertex:
            case ("SW" | "A"):  # Southwest or direction A
                return np.array([gv.face_x_coords[cell_x_index + 1], gv.face_y_coords[cell_x_index + 1, cell_y_index]])
            case ("NW" | "B"):  # Northwest or direction B
                return np.array([gv.face_x_coords[cell_x_index + 1], gv.face_y_coords[cell_x_index + 1, cell_y_index + 1]])
            case ("NE" | "C"):  # Northeast or direction C
                return np.array([gv.face_x_coords[cell_x_index], gv.face_y_coords[cell_x_index, cell_y_index + 1]])
            case ("SE" | "D"):  # Southeast or direction D
                return np.array([gv.face_x_coords[cell_x_index], gv.face_y_coords[cell_x_index, cell_y_index]])
            case _:  # Invalid direction
                raise ValueError("Invalid direction argument: direction must be 'SW', 'NW', 'NE', 'SE'.")
    except ValueError as error:
        print(f"Error: {error}")
        return None

def get_cell_dy(cell_x_index):
    return gv.cell_dy[cell_x_index]

def get_cell_area(cell_x_index, cell_y_index):
    """
    Return the area of the cell at the specified indices.

    Args:
        cell_x_index (int): Index of the cell along the x-axis.
        cell_y_index (int): Index of the cell along the y-axis.

    Returns:
        float: Area of the cell at the specified indices.
    """
    return gv.cell_area[cell_x_index, cell_y_index]

def get_normal_vector(cell_x_index, cell_y_index, face):
    """
        Calculate the normalized unit normal vector for a specified face of a specified cell.

        This function computes the outward-facing unit normal vector for a specified face 
        ("S", "W", "N", or "E") of a cell in a structured 2D grid, given the cell's center 
        indices `(cell_x, cell_y)`.

        Args:
            cell_x_index (int): The x-coordinate (index) of the center of the cell.
            cell_y_index (int): The y-coordinate (index) of the center of the cell.
            face (str): The face of the cell for which the normal vector is to be calculated.
                        Must be one of the following:
                        - "S" (South): The bottom face of the cell.
                        - "W" (West): The left face of the cell.
                        - "N" (North): The top face of the cell.
                        - "E" (East): The right face of the cell.

        Returns:
            np.ndarray: [x-component, y-component] of the normalized unit normal vector at the specified face of the given cell.
    """

    try:
        # Calculate dx and dy based on the specified direction
        match face:
            case "S":
                # Get the coordinates of points A and D
                [x_point_a, y_point_a] = get_vertex_coordinates(cell_x_index, cell_y_index, 'A')
                [x_point_d, y_point_d] = get_vertex_coordinates(cell_x_index, cell_y_index, 'D')
                
                # Calculate the vector components (yA - yD, -(xA - xD))
                vector = np.array([
                    (y_point_a - y_point_d),  # x-component
                    -(x_point_a - x_point_d)  # y-component
                ])
            case "E":
                # Get the coordinates of points A and B
                [x_point_a, y_point_a] = get_vertex_coordinates(cell_x_index, cell_y_index, 'A')
                [x_point_b, y_point_b] = get_vertex_coordinates(cell_x_index, cell_y_index, 'B')

                # Calculate the vector components (yB - yA, -(xB - xA))
                vector = np.array([
                    (y_point_b - y_point_a),  # x-component
                    -(x_point_b - x_point_a)  # y-component
                ])
            case "N":
                # Get the coordinates of points B and C
                [x_point_b, y_point_b] = get_vertex_coordinates(cell_x_index, cell_y_index, 'B')
                [x_point_c, y_point_c] = get_vertex_coordinates(cell_x_index, cell_y_index, 'C')
                
                # Calculate the vector components (yC - yB, -(xC - xB))
                vector = np.array([
                    (y_point_c - y_point_b),  # x-component
                    -(x_point_c - x_point_b)  # y-component
                ])
            case "W":            
                # Get the coordinates of points C and D
                [x_point_c, y_point_c] = get_vertex_coordinates(cell_x_index, cell_y_index, 'C')
                [x_point_d, y_point_d] = get_vertex_coordinates(cell_x_index, cell_y_index, 'D')
                
                # Calculate the vector components (yD - yC, -(xD - xC))
                vector = np.array([
                    (y_point_d - y_point_c),  # x-component
                    -(x_point_d - x_point_c)  # y-component
                ])

            # If the direction is not valid, handle it below
            case _:
                raise ValueError(f"Invalid direction: {face}. Must be one of 'S', 'W', 'N', or 'E'.")
                            
        # Compute the magnitude (Euclidean norm) of the vector
        magnitude = np.sqrt(vector[0]**2 + vector[1]**2)
        
        # Normalize the vector (divide each component by the magnitude)
        normalized_vector = vector / magnitude
        
        # Return the normalized vector
        return normalized_vector
    
    except IndexError:
        # Handle the case where index is out of bounds
        print(f"Error: Index out of bounds. cell_x_index={cell_x_index}, cell_y_index={cell_y_index}")
        return [0, 0]
    
    except Exception as e:
        # Catch any other unexpected errors
        print(f"Error: {e}")
        return [0, 0]

def _calculate_bump_height(x):
    """
    Calculate the bump height (y0) as a function of the x-coordinate.

    Args:
        x (float or ndarray): The x-coordinate(s) at which to compute the bump height.

    Returns:
        ndarray: An array of computed bump heights (y0) corresponding to the input x-coordinate(s).
    """

    # Compute y0 using np.where
    y0 = np.where(
        (x < 0) | (x > DOMAIN_LENGTH),  # Condition: x < 0 or x > L
        0,                        # Value when condition is True: y0 will be set to 0
        BUMP_COEFFICIENT * x * (1 - x / DOMAIN_LENGTH)  # Value when condition is False: computed as a function of x
    )
    
    # Return the computed y0 values
    return y0

def _calculate_cell_area():
    """
    Compute the area of each computational cell in the grid.

    This function calculates the area of every cell in the grid using the coordinates of its corners
    (A, B, C, and D). The areas are stored in the global variable `cell_area`.

    Updates:
        - Global variable `cell_area` (ndarray): A 2D array where each element represents the area 
        of a cell at the corresponding indices.

    Raises:
        ValueError: If a negative cell area is detected during calculation.

    Returns:
        None: The function directly updates the global `cell_area` variable.

    Note:
        - The cell area is calculated using the determinant formula for quadrilaterals.
        - This function iterates over all cells in the grid defined by `NUM_CELLS_X` and `NUM_CELLS_Y`.
    """
    try:
        for cell_x_index, cell_y_index in np.ndindex(NUM_CELLS_X, NUM_CELLS_Y):
            # Retrieve the edge points of the cell
            [x_point_b, y_point_b] = get_vertex_coordinates(cell_x_index, cell_y_index, "B")  # Northwest
            [x_point_c, y_point_c] = get_vertex_coordinates(cell_x_index, cell_y_index, "C")  # Northeast
            [x_point_d, y_point_d] = get_vertex_coordinates(cell_x_index, cell_y_index, "D")  # Southeast
            [x_point_a, y_point_a] = get_vertex_coordinates(cell_x_index, cell_y_index, "A")  # Southwest

            # Calculate the area using the determinant formula
            gv.cell_area[cell_x_index, cell_y_index] = 0.5 * ((x_point_c - x_point_a) * (y_point_d - y_point_b) - (x_point_d - x_point_b) * (y_point_c - y_point_a))

            # Check if the area is negative
            if gv.cell_area[cell_x_index, cell_y_index] < 0:
                raise ValueError(f"Negative cell area calculated: {gv.cell_area[cell_x_index, cell_y_index]} for cell ({cell_x_index}, {cell_y_index})")

    except ValueError as e:
        print(f"Error: {e}")
        return None  # Optionally return None to handle the error gracefully    
    return None

def _calculate_ndS():
    """
    Calculate the face-normal vectors (n * dS) for all cell faces in the grid.

    This function computes the face-normal vector multiplied by the face length (ndS) for each face
    of every computational cell in the grid. The results are stored in the global variable `ndS`.

    Updates:
        - Global variable `ndS` (ndarray): A 4D array where the indices represent:
        [cell_x_index, cell_y_index, face_index, vector_component].

    Face Index Mapping:
        - Face 0: South face (between points D and A).
        - Face 1: West face (between points A and B).
        - Face 2: North face (between points B and C).
        - Face 3: East face (between points C and D).

    Returns:
        None: The function directly updates the global `ndS` variable.

    Note:
        - The face-normal vector is calculated for each face using its endpoints.
        - The function loops over all cells in the grid and calculates ndS for each face.
    """
    # Loop through all grid cells using their x and y indices
    for cell_x_index, cell_y_index in np.ndindex(NUM_CELLS_X, NUM_CELLS_Y):
        
        # Get the coordinates of the four face corners (A, B, C, D) for the current cell
        [x_point_a, y_point_a] = get_vertex_coordinates(cell_x_index, cell_y_index, 'A')  # Corner A (South-West)
        [x_point_b, y_point_b] = get_vertex_coordinates(cell_x_index, cell_y_index, 'B')  # Corner B (North-West)
        [x_point_c, y_point_c] = get_vertex_coordinates(cell_x_index, cell_y_index, 'C')  # Corner C (North-East)
        [x_point_d, y_point_d] = get_vertex_coordinates(cell_x_index, cell_y_index, 'D')  # Corner D (South-East)
        
        # Calculate the normal vector ndS for each face of the current cell:
        # Face 0 (South): Between points D and A
        gv.ndS[cell_x_index, cell_y_index, 0, :] = np.array([
            (y_point_a - y_point_d),  # x-component
            -(x_point_a - x_point_d)  # y-component
        ])
        
        # Face 1 (East): Between points A and B
        gv.ndS[cell_x_index, cell_y_index, 1, :] = np.array([
            (y_point_b - y_point_a),  # x-component
            -(x_point_b - x_point_a)  # y-component
        ])
        
        # Face 2 (North): Between points B and C
        gv.ndS[cell_x_index, cell_y_index, 2, :] = np.array([
            (y_point_c - y_point_b),  # x-component
            -(x_point_c - x_point_b)  # y-component
        ])
        
        # Face 3 (West): Between points C and D
        gv.ndS[cell_x_index, cell_y_index, 3, :] = np.array([
            (y_point_d - y_point_c),  # x-component
            -(x_point_d - x_point_c)  # y-component
        ])
    
    # The function doesn't return anything; it directly updates the global variable `ndS`.
    return None

def plot_mesh():
    """
    Plot the computational mesh in 2D, showing the grid structure and bump profile.

    Args:
        None

    Returns:
        None: The function directly generates and displays the plot.
    """
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

    x_axis_limits = [-DOMAIN_LENGTH-gv.cell_dx/2, 2*DOMAIN_LENGTH+gv.cell_dx/2]
    y_axis_limits = [0-gv.cell_dy[0]/2, CHANNEL_HEIGHT+gv.cell_dy[0]/2]

    save_grid_plot = False
    if save_grid_plot:
        grid_plot_filename = "../img/mesh_with_normal_vectors.pdf"
    #######################################################################
    # Plot the Grid
    #######################################################################
    
    # Add text outside the plot in the upper-right corner
    fig_width, fig_height = fig.get_size_inches()  # Get figure dimensions
    dpi = fig.dpi  # Dots per inch
    offset_x = 10 / dpi  # Offset in inches (e.g., 10 pixels)
    offset_y = 10 / dpi  # Offset in inches

    # Coordinates for the text placement
    text_x = ax.get_position().x1 + offset_x / fig_width  # Right side of the plot + offset
    text_y = ax.get_position().y1 + offset_y / fig_height  # Top side of the plot + offset

    # Add the parameter info
    plt.text(text_x, text_y, 'Parameters', transform=fig.transFigure, ha='left', va='bottom')
    plt.text(text_x, text_y-0.05, f'  Nx = {NUM_CELLS_X} ', transform=fig.transFigure, ha='left', va='bottom')
    plt.text(text_x, text_y-0.1, f'  Ny = {NUM_CELLS_Y}', transform=fig.transFigure, ha='left', va='bottom')

    if show_vertical_grid:
        # Plot vertical grid lines at each y-face.
        for i, j in np.ndindex(NUM_CELLS_X, NUM_FACES_Y):
            ax.plot(
                [gv.face_x_coords[i], gv.face_x_coords[i+1]],  # x-coordinates for the vertical line
                [gv.face_y_coords[i, j], gv.face_y_coords[i+1, j]],  # y-coordinates for the line at this vertical slice
                '--', color="gray", linewidth=0.5  # Dashed gray line with thin width
            )

    if show_horizontal_grid:
        # Plot horizontal grid lines at each x-face.
        for i, j in np.ndindex(NUM_FACES_X, NUM_CELLS_Y):
            ax.plot(
                [gv.face_x_coords[i], gv.face_x_coords[i]],  # x-coordinate remains constant (vertical face)
                [gv.face_y_coords[i, j], gv.face_y_coords[i, j+1]],  # Connect y-coordinates for this x-face
                '--', color="gray", linewidth=0.5  # Dashed gray line
            )

    if show_diagonal_lines:
        # Plot diagonal lines for each cell to form a criss-cross pattern.
        for i, j in np.ndindex(NUM_CELLS_X, NUM_CELLS_Y):
            # Diagonal from bottom-left to top-right
            ax.plot(
                [gv.face_x_coords[i], gv.face_x_coords[i+1]],  # x-coordinates of the diagonal
                [gv.face_y_coords[i, j], gv.face_y_coords[i+1, j+1]],  # y-coordinates of the diagonal
                '--', color="gray", linewidth=0.5  # Dashed gray line
            )
            # Diagonal from top-left to bottom-right
            ax.plot(
                [gv.face_x_coords[i], gv.face_x_coords[i+1]],  # x-coordinates of the diagonal
                [gv.face_y_coords[i, j+1], gv.face_y_coords[i+1, j]],  # y-coordinates of the diagonal
                '--', color="gray", linewidth=0.5  # Dashed gray line
            )

    if show_center_points:
        # Plot markers for the cell centers.
        # These correspond to cell_x_coords (center of each cell in the x-direction) and cell_y_coords (center of each cell in the y-direction).
        for index, x in enumerate(gv.cell_x_coords):
            ax.scatter(
                np.full_like(gv.cell_y_coords[index, :], x),  # Fill x-values to match cell_y_coords array for this x-index
                gv.cell_y_coords[index, :],  # y-coordinates at this x
                marker='o', s=13, color='k'  # Black circles as markers
            )

    if show_face_points:
        # Plot markers for the cell faces.
        for index, x in enumerate(gv.face_x_coords):
            ax.scatter(
                np.full_like(gv.face_y_coords[index, :], x),  # Fill x-values to match face_y_coords array for this x-index
                gv.face_y_coords[index, :],  # y-coordinates at this x-face
                marker='s', s=15, color='g'  # Green squares as markers
            )

    if show_cell_area:
        for i, j in np.ndindex(NUM_CELLS_X, NUM_CELLS_Y):
            # Check if the current cell's x and y coordinates are within the axis limits
            if (
                x_axis_limits[0] <= gv.cell_x_coords[i] <= x_axis_limits[1] and  # Check x-axis bounds
                y_axis_limits[0] <= gv.cell_y_coords[i, j] <= y_axis_limits[1]  # Check y-axis bounds
            ):
                # Plot the cell area value as text inside the axes
                ax.text(
                    gv.cell_x_coords[i],
                    gv.cell_y_coords[i, j],
                    s=np.round(gv.cell_area[i, j], decimals=6),  # Rounded cell area value as text
                    fontsize=6,
                    horizontalalignment='center',
                    verticalalignment='center',
                )

    if show_normal_vector:
        for i, j in np.ndindex(NUM_CELLS_X, NUM_CELLS_Y):
            arrow_scale = 0.05
            if show_normal_vector_south:
                # south arrow
                ax.arrow(get_point_coordinates(i,j)[0] + gv.cell_dx/8, get_point_coordinates(i,j)[1] - gv.cell_dy[j]/2, arrow_scale*get_normal_vector(i,j,'S')[0], arrow_scale*get_normal_vector(i,j,'S')[1], head_width = 0.5 * arrow_scale, head_length = 0.5 * arrow_scale, color='red')
            if show_normal_vector_north:    
                # north arrow
                ax.arrow(get_point_coordinates(i,j)[0] - gv.cell_dx/8, get_point_coordinates(i,j)[1] + gv.cell_dy[j]/2, arrow_scale*get_normal_vector(i,j,'N')[0], arrow_scale*get_normal_vector(i,j,'N')[1], head_width = 0.5 * arrow_scale, head_length = 0.5 * arrow_scale, color='blue')
            if show_normal_vector_west:
                # west arrow
                ax.arrow(get_point_coordinates(i,j)[0] - gv.cell_dx/2, get_point_coordinates(i,j)[1] + gv.cell_dy[j]/8, arrow_scale*get_normal_vector(i,j,'W')[0], arrow_scale*get_normal_vector(i,j,'W')[1], head_width = 0.5 * arrow_scale, head_length = 0.5 * arrow_scale, color='red')
            if show_normal_vector_east:
                # east arrow
                ax.arrow(get_point_coordinates(i,j)[0] + gv.cell_dx/2, get_point_coordinates(i,j)[1] - gv.cell_dy[j]/8, arrow_scale*get_normal_vector(i,j,'E')[0], arrow_scale*get_normal_vector(i,j,'E')[1], head_width = 0.5 * arrow_scale, head_length = 0.5 * arrow_scale, color='blue')
        ax.plot([],[], color='red', label=r"$n_{south}$ / $n_{west}$")
        ax.plot([],[], color='blue', label=r"$n_{north}$ / $n_{east}$")

        ax.legend(loc=[0,1.1])

        ax.set_aspect('equal', adjustable='box')


    ax.set_xlim(x_axis_limits)
    ax.set_ylim(y_axis_limits)

    ax.set_title("Mesh")
    ax.set_xlabel(r"$x$")
    ax.set_ylabel(r"$y$")


    if save_grid_plot:
        fig.savefig(grid_plot_filename)

def initialize():
    #######################################################################
    # Initialize the grid cell coordinates
    #######################################################################

    # Calculate grid spacing in the x-direction
    # cell_dx is the spacing between the centers (or faces) of the cells in the x-direction
    # It is constant throughout the entire domain
    gv.cell_dx = CHANNEL_LENGTH / NUM_CELLS_X

    # Generate the x-coordinates for the centers of the cells
    # This creates an array of evenly spaced coordinates for the cell centers
    gv.cell_x_coords = np.linspace(gv.cell_dx / 2, CHANNEL_LENGTH - gv.cell_dx / 2, NUM_CELLS_X)

    # Shift the x-coordinates by L to align with the desired coordinate system
    gv.cell_x_coords -= DOMAIN_LENGTH

    # Compute the bump height at the center of each cell
    # The bump height, cell_y0, is determined using the get_y0 function
    gv.cell_y0 = _calculate_bump_height(gv.cell_x_coords)

    # Calculate the grid spacing in the y-direction for each x-coordinate
    # cell_dy is determined by dividing the height above the bump by the number of cells
    # This ensures the spacing varies appropriately with the domain geometry
    gv.cell_dy = (CHANNEL_HEIGHT - gv.cell_y0) / NUM_CELLS_Y

    # Generate the y-coordinates for the centers of the cells in the y-direction
    # The coordinates are offset to ensure the cell centers align correctly in the domain
    gv.cell_y_coords = np.linspace(gv.cell_y0 + gv.cell_dy / 2, CHANNEL_HEIGHT - gv.cell_dy / 2, NUM_CELLS_Y)

    # Transpose the face_y_coords array so that indexing is consistent
    # After transposing, face_y_coords(x, index_y) will match the expected structure:
    # the first index corresponds to the x-position, and the second to the y-position
    gv.cell_y_coords = np.transpose(gv.cell_y_coords)

    #######################################################################
    # Initialize the grid face coordinates
    #######################################################################

    # Generate the x-coordinates for the faces of the cells
    # This creates evenly spaced coordinates for the vertical faces of the cells
    gv.face_x_coords = np.linspace(0, CHANNEL_LENGTH, NUM_FACES_X)

    # Shift the x-coordinates by L to align with the desired coordinate system
    # This ensures the face coordinates match the shifted cell coordinates
    gv.face_x_coords -= DOMAIN_LENGTH

    # Compute the bump height at the faces of each x-cell
    # Similar to cell_y0, face_y0 adjusts for the domain geometry at the cell faces
    gv.face_y0 = _calculate_bump_height(gv.face_x_coords)

    # Calculate the grid spacing in the y-direction (at x-coordinates of the faces)
    # The spacing varies depending on the bump height, face_y0
    gv.face_dy = (CHANNEL_HEIGHT - gv.face_y0) / NUM_CELLS_Y

    # Generate the y-coordinates for the faces in the y-direction
    # This creates a 2D array of face y-coordinates that varies with x and y
    gv.face_y_coords = np.linspace(gv.face_y0, CHANNEL_HEIGHT, NUM_FACES_Y)

    # Transpose the face_y_coords array so that indexing is consistent
    # After transposing, face_y_coords(x, index_y) will match the expected structure:
    # the first index corresponds to the x-position, and the second to the y-position
    gv.face_y_coords = np.transpose(gv.face_y_coords)

    #######################################################################
    # Calculate Grid Parameters
    #######################################################################

    # Calculate the area of each cell in the grid
    _calculate_cell_area()

    # Calculate the normal-face vector (n * dS) for all cells
    _calculate_ndS()
    
    return None