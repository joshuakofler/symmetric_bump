# This module handles the data input and output for the project.
# Specifically, it manages the output of data at designated iteration points
# to track the simulation progress and store results at key milestones.
import numpy as np
from constants import *
import global_vars as gv

from datetime import datetime

import os
import vtk

def clear_folder_structure():
    # Clear the output folder structure

    # Check if the base directory exists
    if os.path.exists(OUTPUT_DIR):
        # Loop through each folder inside the base directory and remove it
        for folder in os.listdir(OUTPUT_DIR):
            folder_path = os.path.join(OUTPUT_DIR, folder)
            if os.path.isdir(folder_path):
                # Delete the contents of the folder manually
                for root, dirs, files in os.walk(folder_path, topdown=False):
                    for file in files:
                        file_path = os.path.join(root, file)
                        os.remove(file_path)
                    for subdir in dirs:
                        subdir_path = os.path.join(root, subdir)
                        os.rmdir(subdir_path)
                os.rmdir(folder_path)  # Remove the now-empty folder
                print(f"Deleted folder: {folder_path}")
    else:
        print(f"The directory {OUTPUT_DIR} does not exist.")

    print("\n")
    return None

def initialize_folder_structure():
    # Initialize the output folder structure
    
    # Create folders for each iteration
    for iteration in gv.output_iterations:
        folder_name = os.path.join(OUTPUT_DIR, f"{iteration}")
        os.makedirs(folder_name, exist_ok=True)  # Creates the folder if it doesn't exist
        print(f"Created folder: {folder_name}")

    print("\n")
    return None

def save_iteration(iteration):
    # Flatten the coordinates into a list of (x, y, z=0) tuples
    points = []
    for i, x in enumerate(gv.cell_x_coords):  # Iterate over x-coordinates
        for j, y in enumerate(gv.cell_y_coords[i]):  # Use y-coordinates corresponding to the current x
            points.append((x, y, 0))  # Add z=0 for 2D points
    points = np.array(points)

    # Create a VTK PolyData object
    poly_data = vtk.vtkPolyData()

    # Create a vtkPoints object to hold the points
    vtk_points = vtk.vtkPoints()
    for point in points:
        vtk_points.InsertNextPoint(point)

    # Set points in the PolyData object
    poly_data.SetPoints(vtk_points)

    # Shape (nx, ny, 4)
    state_vectors = gv.state_vector

    # Create a VTK FloatArray to store the 4-component state vector
    vtk_state_vectors = vtk.vtkFloatArray()
    vtk_state_vectors.SetName("StateVector")  # Name for the state vector field
    vtk_state_vectors.SetNumberOfComponents(4)  # 4 components per point

    # Flatten the 3D array (nx, ny, 4) to iterate over all grid points
    for i,j in np.ndindex(NUM_CELLS_X, NUM_CELLS_Y):
        # Extract the 4-component vector at (i, j)
        vector = state_vectors[i, j, :]
        vtk_state_vectors.InsertNextTuple(vector.tolist())  # Convert to list

    # Attach the state vectors to the PolyData
    poly_data.GetPointData().AddArray(vtk_state_vectors)
    
    # Add m_in to the PolyData
    vtk_m_in = vtk.vtkFloatArray()
    vtk_m_in.SetName("m_in")  # Name for the m_in field
    vtk_m_in.SetNumberOfComponents(1)  # 1 component per point

    # Ensure that gv.m_in is a list or numpy array with the same length as the number of points
    for index, value in enumerate(gv.m_in):
        if(index <= gv.iteration):
            vtk_m_in.InsertNextValue(value)  # Insert each value of m_in into the vtk array

    poly_data.GetFieldData().AddArray(vtk_m_in)

    # Add m_out to the PolyData
    vtk_m_out = vtk.vtkFloatArray()
    vtk_m_out.SetName("m_out")  # Name for the m_out field
    vtk_m_out.SetNumberOfComponents(1)  # 1 component per point

    # Ensure that gv.m_out is a list or numpy array with the same length as the number of points
    for index, value in enumerate(gv.m_out):
        if(index <= gv.iteration):
            vtk_m_out.InsertNextValue(value)  # Insert each value of m_out into the vtk array

    poly_data.GetFieldData().AddArray(vtk_m_out)

    # Create the output directory if it doesn't exist
    iteration_dir = os.path.join(OUTPUT_DIR, str(iteration))
    # Create a writer to save the data to a file
    file_path = os.path.join(iteration_dir, f"{iteration}.vtp")
    writer = vtk.vtkXMLPolyDataWriter()
    writer.SetInputData(poly_data)  # Ensure the PolyData is provided to the writer
    writer.SetFileName(file_path)
    writer.Write()
    
    # Get the size of the file
    file_size = os.path.getsize(file_path)
    
    # Get the current time
    current_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    print_data_save_info(iteration, file_path, file_size, current_time)

    return None

def read_iteration(file_path):
    # Create a reader for the VTK file
    reader = vtk.vtkXMLPolyDataReader()
    reader.SetFileName(file_path)
    reader.Update()

    # Get the PolyData object from the reader
    poly_data = reader.GetOutput()

    # Check if the PolyData has points
    vtk_points = poly_data.GetPoints()
    if vtk_points is None:
        print(f"Error: No points found in the VTK file {file_path}.")
        return None, None, None, None  # Return None to indicate an error

    # Retrieve state vector (assuming it's stored with 4 components)
    vtk_state_vectors = poly_data.GetPointData().GetArray("StateVector")
    if vtk_state_vectors is None:
        print(f"Error: StateVector array not found in the VTK file {file_path}.")
        state_vectors = None
    else:
        state_vectors = np.array([vtk_state_vectors.GetTuple(i) for i in range(vtk_state_vectors.GetNumberOfTuples())])

    state_vector_grid = state_vectors.reshape((NUM_CELLS_X, NUM_CELLS_Y, 4))
    
    # Retrieve m_in field
    vtk_m_in = poly_data.GetFieldData().GetArray("m_in")
    if vtk_m_in is None:
        print(f"Warning: m_in array not found in the VTK file {file_path}.")
        m_in = None  # Handle case if m_in is not present in the file
    else:
        m_in = np.array([vtk_m_in.GetValue(i) for i in range(vtk_m_in.GetNumberOfValues())])

    # Retrieve m_out field
    vtk_m_out = poly_data.GetFieldData().GetArray("m_out")
    if vtk_m_out is None:
        print(f"Warning: m_out array not found in the VTK file {file_path}.")
        m_out = None  # Handle case if m_out is not present in the file
    else:
        m_out = np.array([vtk_m_out.GetValue(i) for i in range(vtk_m_out.GetNumberOfValues())])

    # Optionally, you can return the retrieved data for further processing
    return state_vector_grid, m_in, m_out

def simplify_file_path(file_path, base_path):
    """Simplify the file path by stripping the base path."""
    return file_path.replace(base_path, "").lstrip(os.sep)

def print_data_save_info(iteration, file_path, file_size, current_time):
    # Simplify the file path by removing the base path part
    simplified_file_path = simplify_file_path(file_path, str(PROJECT_DIR))
    
    # Calculate the maximum width for the divider lines based on the simplified file path
    max_width = 50
    
    print("\n")
    # Print the divider line
    print("~" * max_width)
    print("  -- Data Saved Successfully --  ")
    print("~" * max_width)

    # Print iteration information
    print("Iteration Information")
    print("-" * max_width)
    print(f" Iteration:         {iteration}")
    
    # Print the simplified file path
    print(f" File Path:        {simplified_file_path}")
    
    print(f" File Size:        {file_size / 1024:.2f} KB")  # Size in KB
    print(f" Timestamp:        {current_time}")
    print("~" * max_width)

def print_iteration_residual(iteration, R_start, R_final):
    """
    Print residuals at the start and end of a specific iteration.
    
    Parameters:
    iteration (int): The iteration number.
    R_start (dict): Residuals at the start of the iteration (keys: 'rho', 'qux', 'quy', 'qE').
    R_final (dict): Residuals at the end of the iteration (keys: 'rho', 'qux', 'quy', 'qE').
    """
    # Maximum width for divider lines
    max_width = 50

    print("\n")
    # Print the divider line
    print("~" * max_width)
    print(f"  -- Iteration {iteration} Residuals --  ")
    print("~" * max_width)

    # Print iteration information
    print("Residual Information")
    print("-" * max_width)

    # Print residuals at the start
    print(" Start Residuals:")
    print(f"   rho:  {R_start[0]:.5e}")
    print(f"   qux:  {R_start[1]:.5e}")
    print(f"   quy:  {R_start[2]:.5e}")
    print(f"   qE:   {R_start[3]:.5e}")
    
    # Print a separator
    print("-" * max_width)

    # Print residuals at the end
    print(" Final Residuals:")
    print(f"   rho:  {R_final[0]:.5e}")
    print(f"   qux:  {R_final[1]:.5e}")
    print(f"   quy:  {R_final[2]:.5e}")
    print(f"   qE:   {R_final[3]:.5e}")

    # Print a closing divider
    print("~" * max_width)
    
    return None