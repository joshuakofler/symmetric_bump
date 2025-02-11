#%%

import pyvista as pv
import numpy as np

# Load the VTK file#
reader = pv.VTKDataSetReader("..//output//contour_final.vtk")

mesh = reader.read()

#mesh = pv.read("..//output//contour_final.vtk")

print(mesh)

print(mesh['Mach Number'])
print(mesh.points)


# Create a PyVista mesh from the coordinates and scalar field

points = np.column_stack((X, Y, Z))  # Combine X, Y, Z into a (N_points, 3) array
mesh_with_scalars = pv.PolyData(points)
mesh_with_scalars['Mach Number'] = mesh['Mach Number']  # Assign Mach Number as a scalar field

# Plot the mesh with the scalar field
mesh_with_scalars.plot(scalars='Mach Number', cmap='viridis', show_edges=True)



