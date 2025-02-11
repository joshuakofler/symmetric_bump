# This module handles the data input and output for the project.
# Specifically, it manages the output of data at designated iteration points
# to track the simulation progress and store results at key milestones.
import numpy as np
from constants import *
import global_vars as gv

from datetime import datetime

import os
import re
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
        folder_name = os.path.join(OUTPUT_DIR, f"{int(iteration)}")
        os.makedirs(folder_name, exist_ok=True)  # Creates the folder if it doesn't exist
        print(f"Created folder: {folder_name}")

    print("\n")
    return None

def save_iteration(iteration):
    # Create VTK Points for the grid with face coordinates
    vtk_points = vtk.vtkPoints()
    for j in range(NUM_FACES_Y):
        for i in range(NUM_FACES_X):
            # Extract face coordinates from gv (vertex coordinates)
            x = gv.face_x_coords[i]
            y = gv.face_y_coords[i, j]
            vtk_points.InsertNextPoint(x, y, 0)  # Assuming 2D grid, z=0

    # Create the structured grid (vtkStructuredGrid)
    structured_grid = vtk.vtkStructuredGrid()
    structured_grid.SetDimensions(NUM_FACES_X, NUM_FACES_Y, 1)  # Set the grid dimensions

    # Set the points for the grid
    structured_grid.SetPoints(vtk_points)

    # Create density field (rho)
    vtk_rho = vtk.vtkFloatArray()
    vtk_rho.SetName("Density")
    vtk_rho.SetNumberOfComponents(1)

    for j in range(NUM_CELLS_Y):
        for i in range(NUM_CELLS_X):
            vtk_rho.InsertNextValue(gv.rho[i, j])

    structured_grid.GetCellData().SetScalars(vtk_rho)

    # Create velocity vector field (U) with 2 components (u, v)
    vtk_velocity = vtk.vtkFloatArray()
    vtk_velocity.SetName("Velocity")
    vtk_velocity.SetNumberOfComponents(2)  # 2 components for the vector (u, v)

    for j in range(NUM_CELLS_Y):
        for i in range(NUM_CELLS_X):
            vtk_velocity.InsertNextTuple2(gv.u[i, j, 0], gv.u[i, j, 1])  # Assuming gv.u contains 2 components for velocity in x and y

    structured_grid.GetCellData().AddArray(vtk_velocity)

    # Create total energy field (E)
    vtk_energy = vtk.vtkFloatArray()
    vtk_energy.SetName("Total Energy")
    vtk_energy.SetNumberOfComponents(1)

    for j in range(NUM_CELLS_Y):
        for i in range(NUM_CELLS_X):
            vtk_energy.InsertNextValue(gv.E[i, j])

    structured_grid.GetCellData().AddArray(vtk_energy)

    # Create internal energy field (e)
    vtk_internal_energy = vtk.vtkFloatArray()
    vtk_internal_energy.SetName("Internal Energy")
    vtk_internal_energy.SetNumberOfComponents(1)

    for j in range(NUM_CELLS_Y):
        for i in range(NUM_CELLS_X):
            vtk_internal_energy.InsertNextValue(gv.e[i, j])

    structured_grid.GetCellData().AddArray(vtk_internal_energy)

    # Create temperature field (T)
    vtk_temperature = vtk.vtkFloatArray()
    vtk_temperature.SetName("Temperature")
    vtk_temperature.SetNumberOfComponents(1)

    for j in range(NUM_CELLS_Y):
        for i in range(NUM_CELLS_X):
            vtk_temperature.InsertNextValue(gv.T[i, j])

    structured_grid.GetCellData().AddArray(vtk_temperature)

    # Create speed of sound field (c)
    vtk_speed_of_sound = vtk.vtkFloatArray()
    vtk_speed_of_sound.SetName("Speed of Sound")
    vtk_speed_of_sound.SetNumberOfComponents(1)

    for j in range(NUM_CELLS_Y):
        for i in range(NUM_CELLS_X):
            vtk_speed_of_sound.InsertNextValue(gv.c[i, j])

    structured_grid.GetCellData().AddArray(vtk_speed_of_sound)

    # Create pressure field (p)
    vtk_pressure = vtk.vtkFloatArray()
    vtk_pressure.SetName("Pressure")
    vtk_pressure.SetNumberOfComponents(1)

    for j in range(NUM_CELLS_Y):
        for i in range(NUM_CELLS_X):
            vtk_pressure.InsertNextValue(gv.p[i, j])

    structured_grid.GetCellData().AddArray(vtk_pressure)

    # Create total enthalpy field (H)
    vtk_enthalpy = vtk.vtkFloatArray()
    vtk_enthalpy.SetName("Total Enthalpy")
    vtk_enthalpy.SetNumberOfComponents(1)

    for j in range(NUM_CELLS_Y):
        for i in range(NUM_CELLS_X):
            vtk_enthalpy.InsertNextValue(gv.H[i, j])

    structured_grid.GetCellData().AddArray(vtk_enthalpy)

    # Create Mach number field (M)
    vtk_mach = vtk.vtkFloatArray()
    vtk_mach.SetName("Mach Number")
    vtk_mach.SetNumberOfComponents(1)

    for j in range(NUM_CELLS_Y):
        for i in range(NUM_CELLS_X):
            vtk_mach.InsertNextValue(gv.M[i, j])

    structured_grid.GetCellData().AddArray(vtk_mach)
    
    # Add m_in to the PolyData
    vtk_m_in = vtk.vtkFloatArray()
    vtk_m_in.SetName("Inlet Massflow")  # Name for the m_in field
    vtk_m_in.SetNumberOfComponents(1)  # 1 component per point

    # Ensure that gv.m_in is a list or numpy array with the same length as the number of points
    for index, value in enumerate(gv.m_in):
        if(index <= gv.iteration):
            vtk_m_in.InsertNextValue(value)  # Insert each value of m_in into the vtk array

    structured_grid.GetFieldData().AddArray(vtk_m_in)

    # Add m_out to the PolyData
    vtk_m_out = vtk.vtkFloatArray()
    vtk_m_out.SetName("Outlet Massflow")  # Name for the m_out field
    vtk_m_out.SetNumberOfComponents(1)  # 1 component per point

    # Ensure that gv.m_out is a list or numpy array with the same length as the number of points
    for index, value in enumerate(gv.m_out):
        if(index <= gv.iteration):
            vtk_m_out.InsertNextValue(value)  # Insert each value of m_out into the vtk array

    structured_grid.GetFieldData().AddArray(vtk_m_out)

    # Create the output directory if it doesn't exist
    file_path = os.path.join(OUTPUT_DIR, f"{iteration}\{iteration}.vts")

    # Save the structured grid to file using vtkXMLStructuredGridWriter
    writer = vtk.vtkXMLStructuredGridWriter()
    writer.SetFileName(file_path)
    writer.SetInputData(structured_grid)
    writer.Write()

    # Add entry for PVD file (used for time series visualization)
    gv.pvd_entries.append(f'    <DataSet timestep="{iteration}" file="{iteration}/{iteration}.vts"/>')
    
    # Get the size of the file
    file_size = os.path.getsize(file_path)
    
    # Get the current time
    current_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    print_data_save_info(iteration, file_path, file_size, current_time)

    return None

def read_existing_pvd(pvd_filename):
    if not os.path.exists(pvd_filename):
        return []

    with open(pvd_filename, "r") as f:
        content = f.read()

    # Extrahiere vorhandene <DataSet>-Einträge
    existing_entries = re.findall(r"<DataSet[\s\S]*?/>\s*", content)

    # Falls keine Einträge existieren, gib eine leere Liste zurück
    if not existing_entries:
        return []

    # Erster Eintrag bleibt ohne Tab, alle anderen bekommen eine Einrückung
    gv.pvd_entries.extend([existing_entries[0].strip()] + [f"\t{entry.strip()}" for entry in existing_entries[1:]])

    return None

def save_pvd():
    """Creates a .pvd file referencing all saved .vtp files."""
    pvd_content = """<?xml version="1.0"?>
    <VTKFile type="Collection" version="0.1" byte_order="LittleEndian">
        <Collection>
            {}    
        </Collection>
    </VTKFile>
    """.format("\n".join(gv.pvd_entries))  # Ensure entries are correctly indented


    # Get the current time
    current_time = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")

    pvd_filename = os.path.join(OUTPUT_DIR, f"{current_time}_solution_{NUM_CELLS_X}_{NUM_CELLS_Y}.pvd")
    with open(pvd_filename, "w") as f:
        f.write(pvd_content)

    print(f"Saved PVD file: {pvd_filename}")

    return None

def read_iteration(file_path):
    # Initialize the VTK reader for .vts files
    reader = vtk.vtkXMLStructuredGridReader()
    reader.SetFileName(file_path)
    reader.Update()  # Read the file

    # Get the structured grid data
    structured_grid = reader.GetOutput()

    # Extract grid dimensions
    dimensions = structured_grid.GetDimensions()
    NUM_FACES_X, NUM_FACES_Y = dimensions[0], dimensions[1]

    # Extract point coordinates (grid face positions)
    points = structured_grid.GetPoints()
    gv.face_x_coords = np.zeros(NUM_FACES_X)
    gv.face_y_coords = np.zeros((NUM_FACES_X, NUM_FACES_Y))

    for j in range(NUM_FACES_Y):
        for i in range(NUM_FACES_X):
            x, y, _ = points.GetPoint(i + j * NUM_FACES_X)  # Extract x, y (ignoring z)
            gv.face_x_coords[i] = x
            gv.face_y_coords[i, j] = y

    # Extract cell-based scalar fields
    cell_data = structured_grid.GetCellData()

    def extract_field(field_name):
        """Helper function to extract a field from cell data."""
        vtk_array = cell_data.GetArray(field_name)
        if not vtk_array:
            return None  # Field might not be present
        data = np.zeros((NUM_FACES_X - 1, NUM_FACES_Y - 1))  # Adjust for cell-centered data
        for j in range(NUM_FACES_Y - 1):
            for i in range(NUM_FACES_X - 1):
                data[i, j] = vtk_array.GetValue(i + j * (NUM_FACES_X - 1))
        return data

    # Load each field into gv
    gv.rho = extract_field("Density")
    
    # Extract velocity as a 2-component vector field
    gv.u = np.zeros((NUM_FACES_X - 1, NUM_FACES_Y - 1, 2))
    velocity_array = cell_data.GetArray("Velocity")
    if velocity_array:
        for j in range(NUM_FACES_Y - 1):
            for i in range(NUM_FACES_X - 1):
                gv.u[i, j, 0], gv.u[i, j, 1] = velocity_array.GetTuple2(i + j * (NUM_FACES_X - 1))

    gv.E = extract_field("Total Energy")
    gv.e = extract_field("Internal Energy")
    gv.T = extract_field("Temperature")
    gv.c = extract_field("Speed of Sound")
    gv.p = extract_field("Pressure")
    gv.H = extract_field("Total Enthalpy")
    gv.M = extract_field("Mach Number")

    # Extract global field data (e.g., mass flow rates)
    field_data = structured_grid.GetFieldData()

    def extract_field_data(field_name):
        """Helper function to extract global field data."""
        vtk_array = field_data.GetArray(field_name)
        if not vtk_array:
            return None
        return np.array([vtk_array.GetValue(i) for i in range(vtk_array.GetNumberOfTuples())])

    gv.m_in[:-(MAX_ITERATIONS-gv.iteration)] = extract_field_data("Inlet Massflow")
    gv.m_out[:-(MAX_ITERATIONS-gv.iteration)] = extract_field_data("Outlet Massflow")

    # Fill the state vector
    gv.state_vector[:, :, 0] = gv.rho[:,:]
    gv.state_vector[:, :, 1] = gv.rho[:,:] * gv.u[:,:,0]
    gv.state_vector[:, :, 2] = gv.rho[:,:] * gv.u[:,:,1]
    gv.state_vector[:, :, 3] = gv.rho[:,:] * gv.E[:,:]

    print(f"Successfully loaded {file_path} into global variables.")

    return None

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