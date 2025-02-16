"""
Global Constants for CFD Simulation
------------------------------------
This file contains the constants used throughout the CFD simulation,
including physical constants, domain dimensions, numerical settings,
and artificial dissipation parameters.

Constants are grouped into sections for better readability and maintainability.
"""

# -------------------------------- Booleans / Flags --------------------------------
# Enable or disable local time-stepping.
# When enabled (True), each computational cell uses an individual time step (dt) based on local conditions.
# When disabled (False), a single global time step (uniform dt) is used for the entire domain.
USE_LOCAL_TIME_STEP = True

# Maximum number of iterations for the solver
MAX_ITERATIONS = 30

# Upstream flow conditions
UPSTREAM_MACH_NUMBER = 0.2

# Option to manually specify the calculation method for artificial dissipation.
# - If True, only third-order dissipation is used (assuming subsonic conditions).
# - If False, first-order dissipation is added to third-order dissipation (for supersonic conditions).
USE_SUBSONIC_AD = False

# -------------------------------- Directories --------------------------------
from pathlib import Path

# Project directory paths
CURRENT_DIR = Path(__file__).parent  # Current directory (inside 'src' folder)
PROJECT_DIR = CURRENT_DIR.parent     # Project root directory
OUTPUT_DIR = PROJECT_DIR / "output"  # Directory for simulation output files
RESULT_DIR = PROJECT_DIR / "results" # Directory for result files

# -------------------------------- Computational Domain --------------------------------
# Base length scale
DOMAIN_LENGTH = 1.0  # Base length scale (m)

# Channel dimensions
CHANNEL_HEIGHT = DOMAIN_LENGTH         # Channel height (m)
CHANNEL_LENGTH = 3 * DOMAIN_LENGTH     # Channel length (m)

# Bump configuration (geometry of the channel)
USE_CIRCULAR_ARC = False               # Flag to use a circular arc for the bump
if USE_CIRCULAR_ARC:
    BUMP_COEFFICIENT = 0.1  # Bump height coefficient for circular arc
else:
    BUMP_COEFFICIENT = 0.08  # Default bump height coefficient for non-circular arc

# -------------------------------- Grid Configuration --------------------------------
# Number of computational cells
NUM_CELLS_X = 65  # Number of cells in the x-direction
NUM_CELLS_Y = 33  # Number of cells in the y-direction

# Number of faces (grid boundaries)
NUM_FACES_X = NUM_CELLS_X + 1  # Number of faces in the x-direction
NUM_FACES_Y = NUM_CELLS_Y + 1  # Number of faces in the y-direction

# -------------------------------- Physical Constants --------------------------------
# Gas properties for dry air
GAS_CONSTANT = 287.05          # Gas constant (J / kgK)
HEAT_CAPACITY_RATIO = 1.4      # Heat capacity ratio (γ)

# Specific heat capacities (for air)
SPECIFIC_HEAT_CV = GAS_CONSTANT / (HEAT_CAPACITY_RATIO - 1)  # Cv (J / kgK)
SPECIFIC_HEAT_CP = SPECIFIC_HEAT_CV + GAS_CONSTANT           # Cp (J / kgK)

# Atmospheric conditions
ATMOSPHERIC_PRESSURE = 101300   # Atmospheric pressure (Pascals)
ATMOSPHERIC_TEMPERATURE = 288   # Atmospheric temperature (Kelvin)

# -------------------------------- Numerical Settings --------------------------------
# Runge-Kutta scheme coefficients
RK_ALPHA = [1/4, 1/3, 1/2, 1]  # Runge-Kutta alpha values for multi-stage time integration

# Courant-Friedrichs-Lewy (CFL) condition
CFL = 2.0  # CFL number for time-step calculation

# -------------------------------- Artificial Dissipation --------------------------------
# Artificial dissipation coefficients
ARTIFICIAL_DISSIPATION_KAPPA_2 = 1
ARTIFICIAL_DISSIPATION_KAPPA_4 = 1 / 256