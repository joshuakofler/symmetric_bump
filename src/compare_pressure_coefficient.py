#%%
import csv
import matplotlib.pyplot as plt

from constants import RESULT_DIR

# Function to read data from a CSV file
def read_csv(filename):
    """
    Reads data from a CSV file with the assumption that the first column contains 'x' values 
    and the second column contains 'Cp' values. The header row is skipped.

    Parameters:
        filename (str): Path to the CSV file to read.

    Returns:
        tuple: Two lists, one for 'x' values and one for 'Cp' values.
    """
    x_values = []  # List to store x values
    Cp_values = []  # List to store Cp values

    try:
        # Open the CSV file in read mode
        with open(filename, mode='r') as file:
            reader = csv.reader(file)
            next(reader)  # Skip the header row
            for row in reader:
                x_values.append(float(row[0]))  # First column: x values
                Cp_values.append(float(row[1]))  # Second column: Cp values

    except Exception as e:
        # Handle any errors (e.g., file not found or read error)
        print(f"Error reading {filename}: {e}")
        return [], []  # Return empty lists in case of error

    return x_values, Cp_values

# Function to plot Cp comparison from two CSV files on the same plot
def plot_Cp_comparison(file1, file2, parent_directory="OUTPUT_DIR"):
    """
    Plots the Cp values from two CSV files and compares them on the same plot.

    Parameters:
        file1 (str): Path to the first CSV file.
        file2 (str): Path to the second CSV file.
        parent_directory (str): The parent directory for the output files (default is "OUTPUT_DIR").
    """
    global M  # Assume M is globally defined (e.g., M = "M_0_85")

    # Read data from both CSV files
    x1, Cp1 = read_csv(file1)
    x2, Cp2 = read_csv(file2)

    # Check if the data from both files were successfully read
    if not x1 or not Cp1 or not x2 or not Cp2:
        print("Error: One or both datasets are empty or could not be read.")
        return

    # Create a plot with specified size
    fig, ax = plt.subplots(figsize=(6, 5))

    # Plot Cp values from the first dataset using scatter plot
    ax.scatter(x1, Cp1, label=r"$65\times33$", s=250, color='gray', marker='+', linewidths=0.5)

    # Plot Cp values from the second dataset using a line plot
    ax.plot(x2, Cp2, label=r"$225\times113$", linestyle='-', color='black')

    # Set plot limits
    ax.set_xlim([-1, 2])
    ax.set_ylim([-0.6, 0.3])

    # Customize y-axis ticks
    ax.set_yticks([-0.6, -0.3, 0, 0.3])

    # Set axis labels and title
    ax.set_xlabel(r"$x$", fontsize=18)
    ax.set_ylabel(r"$C_p$", fontsize=18)
    ax.tick_params(axis='both', which='major', labelsize=16)

    # Enable grid lines for better visibility
    ax.grid(True, which='both', linestyle='--', linewidth=0.5)

    # Save the plot as a PDF file
    fig.savefig(f"Comparison_{M}_Cp.pdf", bbox_inches="tight")

    return None

M = "M_0_85" # Define Mach number (M)

iteration1 = 3000  # Iteration number for the first file
iteration2 = 12000  # Iteration number for the second file

# Example usage: File names for comparison (adjust paths as needed)
file1 = f"..//results//bump_0_08//{M}//{iteration1}_Cp_65_33.csv"
file2 = f"..//results//bump_0_08//{M}_fine//{iteration2}_Cp_225_113.csv"

# Plot the comparison of Cp values from both files
plot_Cp_comparison(file1, file2, parent_directory=RESULT_DIR)