#%%
# import modules
import numpy as np
import importlib
from matplotlib import pyplot as plt

# own libs
import constants as const
import mesh

# reload constant lib
importlib.reload(const)
importlib.reload(mesh)


from variables import F, R, ndS
# initialize the mesh
mesh.init()












def get_inlet_pressure(atmospheric_pressure, heat_capacity_ratio, mach_number):
    """
    Calculates the inlet pressure based on the given parameters.

    Args:
        atmospheric_pressure (float): atmospheric pressure (Pa).
        heat_capacity_ratio (float): Heat capacity ratio (γ).
        mach_number (float): Mach number.

    Returns:
        float: Inlet pressure.
    """
    return atmospheric_pressure * np.power(1 + (heat_capacity_ratio - 1) / 2 * mach_number**2, 
                                        heat_capacity_ratio / (heat_capacity_ratio - 1))


def get_inlet_temperature(atmospheric_temperature, heat_capacity_ratio, mach_number):
    """
    Calculates the inlet temperature based on the given parameters.

    Args:
        atmospheric_temperature (float): atmospheric temperature (K).
        heat_capacity_ratio (float): Heat capacity ratio (γ).
        mach_number (float): Mach number.

    Returns:
        float: Inlet temperature.
    """
    return atmospheric_temperature * (1 + (heat_capacity_ratio - 1) / 2 * mach_number**2)

mesh.plot_mesh()

# plot the mesh
#mesh.plot_mesh()
