import numpy as np
from scipy.optimize import root_scalar
from matplotlib import pyplot as plt
from scipy.io import loadmat
def area_mach_relation(M, gamma, area_ratio):
    """Computes the area-Mach number relation."""
    term1 = 1 / M
    term2 = (2 / (gamma + 1)) * (1 + (gamma - 1) / 2 * M**2)
    term2 = term2**((gamma + 1) / (2 * (gamma - 1)))
    return term1 * term2 - area_ratio

def get_theoritical_values(area_ratios, gamma=1.4):
    Mach_nums = []
    pressure_ratios = []
    for area_ratio in area_ratios:
        Mach_nums.append(find_subsonic_mach(area_ratio, gamma))
        pressure_ratio = (1+(gamma-1)/2*Mach_nums[-1]**2)**(gamma/(gamma-1))
        pressure_ratios.append(pressure_ratio)
    return Mach_nums, pressure_ratios


def area_mach_relation(M, gamma, area_ratio):
    """Computes the area-Mach number relation."""
    term1 = 1 / M
    term2 = (2 / (gamma + 1)) * (1 + (gamma - 1) / 2 * M**2)
    term2 = term2**((gamma + 1) / (2 * (gamma - 1)))
    return term1 * term2 - area_ratio

def find_subsonic_mach(area_ratio, gamma):
    """
    Solves for the subsonic Mach number given an area ratio.
    
    Parameters:
        area_ratio (float): A/A*, area ratio
        gamma (float): Specific heat ratio, default is 1.4 for air
    
    Returns:
        float: Subsonic Mach number
    """
    # Define a reasonable range for subsonic Mach numbers (0 < M < 1)
    mach_min = 0.01
    mach_max = 1
    
    # Use a numerical solver to find the Mach number
    solution = root_scalar(area_mach_relation, args=(gamma, area_ratio), bracket=[mach_min, mach_max], method='bisect')
    
    if solution.converged:
        return solution.root
    else:
        raise ValueError("Numerical solver did not converge!")
    
A_Astar = [1.06, 1, 1.05, 1.15, 1.23, 1.27, 1.28, 1.3, 1.3, 1.3, 1.3]

Mach, pressure = get_theoritical_values(A_Astar)
pressure = np.array(pressure)



subsonic_static_offset_mean = loadmat('241114_Group01_Subsonic_Static_offset')
subsonic_static_offset_mean = subsonic_static_offset_mean['v_offset_mean'][0]
subsonic_static_total_1_mean = loadmat('241114_Group01_Subsonic_Total_1')
subsonic_static_total_1_mean = subsonic_static_total_1_mean['dataP_mean']
subsonic_static_total_2_mean = loadmat('241114_Group01_Subsonic_Total_2')
subsonic_static_total_2_mean = subsonic_static_total_2_mean['dataP_mean']
subsonic_static_total_3_mean = loadmat('241114_Group01_Subsonic_Total_3')
subsonic_static_total_3_mean = subsonic_static_total_3_mean['dataP_mean']
subsonic_static_total_4_mean = loadmat('241114_Group01_Subsonic_Total_4')
subsonic_static_total_4_mean = subsonic_static_total_4_mean['dataP_mean']
subsonic_static_total_5_mean = loadmat('241114_Group01_Subsonic_Total_5')
subsonic_static_total_5_mean = subsonic_static_total_5_mean['dataP_mean']
subsonic_static_total_6_mean = loadmat('241114_Group01_Subsonic_Total_6')
subsonic_static_total_6_mean = subsonic_static_total_6_mean['dataP_mean']
subsonic_static_total_7_mean = loadmat('241114_Group01_Subsonic_Total_7')
subsonic_static_total_7_mean = subsonic_static_total_7_mean['dataP_mean']


subsonic_pressures = [subsonic_static_total_1_mean, subsonic_static_total_2_mean, subsonic_static_total_3_mean, subsonic_static_total_4_mean, subsonic_static_total_5_mean, subsonic_static_total_6_mean, subsonic_static_total_7_mean]

for i in range(len(subsonic_pressures)):
    subsonic_pressures[i] = (subsonic_pressures[i] +101325)[0]
    total_pressure = subsonic_pressures[i][-1]
    subsonic_pressures[i] = subsonic_pressures[i][:-1]
    last_index = subsonic_pressures[i][-1]
    subsonic_pressures[i][-1] = subsonic_pressures[i][-2]
    subsonic_pressures[i][-2] = last_index
    pressure=total_pressure/pressure
    plt.plot(pressure, label='Expected')
    plt.plot(subsonic_pressures[i], '-o', label='Measured')
    plt.title('Expected Pressure vs. Measured')
    plt.legend()
    plt.xlabel('Port Number')
    plt.ylabel('Presure Number (Pa)')
    plt.show()
    subsonic_mach = (2/0.4*((total_pressure/subsonic_pressures[i])**(0.4/1.4)-1))**0.5

    plt.plot(Mach, label='Expected')
    plt.plot(subsonic_mach, '-o', label='Measured')
    plt.title('Expected Pressure vs. Measured')
    plt.legend()
    plt.xlabel('Port Number')
    plt.ylabel('Ma')
    plt.show()
    pressure=total_pressure/pressure
