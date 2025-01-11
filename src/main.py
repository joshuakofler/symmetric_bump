#%%
# import modules
import numpy as np

# own libs
import constants
import mesh
import cell
import calculate_artificial_dissipation as aD
import importlib

import calculate_residual as cR
import calculate_flux as cF


# import all global variables from globals 
from globals import *
from constants import *

importlib.reload(mesh)
importlib.reload(aD)
importlib.reload(cell)
importlib.reload(constants)
importlib.reload(cF)




#mesh.plot_mesh()

cell.initialize_domain()

cF.update_flux()

aD.calculate_coefficient()


aD.update_artificial_dissipation()


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