import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
import pandas as pd

# Parameters for the WE model:
D_WE = 63e-3  # Cylinder tube diameter (m)
d_WE = 40e-3  # Piston rod diameter (m)
A_WE_1 = (D_WE ** 2) * (np.pi / 4)  # Area of chamber A (m^2)
A_WE_2 = (D_WE ** 2) * (np.pi / 4) - (d_WE ** 2) * (np.pi / 4)  # Area of chamber B (m^2)
L_WE = 0.49  # Length of the piston chamber "The stroke" (m)
V_WE_1_0 = 0.001  # Initial volume in chamber A (m^3)
V_WE_2_0 = 0.001  # Initial volume in chamber B (m^3)
p_T_val = 1e5  # Tank pressure at equilibrium (Pa)
alpha = 0.02 # is the air ratio in the fluid
n_polyIndx = 1.4  # Polytropic index
beta_0 = 16e8  # Bulk modulus of fluid (Pa)
Q_n = 40 / 60000  # Nominal flow for the piston
p_n = 35e5  # Nominal pressure for the piston
C_le = 5e-13  # Leakage coefficient
J = 181.02  # Moment of inertia (kg*m^2) 
c = 15000#12810.039656431883 # Damping coefficient (Nms)
k = 1000#582.6066345764121 # Spring constant (N/m)
M = 681.97 # Mass of the "part" (kg)
g = 9.82  # Gravitational acceleration (m/s^2)
theta_0 = np.deg2rad(0) # Initial angle (rad)
x_com = 0.21  # Center of mass in the x-direction (m)
y_com = 0.06  # Center of mass in the y-direction (m)
L_com = np.sqrt(x_com**2 + y_com**2)  # Distance from the rotation point to the center of mass (m)
a_WE = 1726.59e-3 # Distance from the rotation point to the point where the top part of the we piston is connected (m)
c_WE = 1369e-3 # Distance from the rotation point to the point where the force from the WE acts (m)
L_WE_0 = 718e-3 # The length of the WE base (m)
x_WE_0 = 266e-3 # the lengthe of the WE rod, in the initial position (m)
a_PTO = 976.8e-3 # Distance from the rotation point to the point where the PTO piston is connected (m)
c_PTO = 540e-3 # Distance from the rotation point to the point where the force from the PTO acts (m)
L_PTO_0 = 624e-3 # The length of the PTO base (m)
Dead_PTO = 0.0999488938200001 # The length of the PTO rod, in the final position (m)
x_PTO_0 = 190e-3 # the lengthe of the PTO rod, in the initial position (m)
x_PTO = 0.18011024645 # The length of the PTO rod, from start to end (m)
A_PTO_1_1 = A_WE_2  #The area of chamber 1 of PTO cylinder 1 (m^2)
A_PTO_2_1 = A_WE_1  #The area of chamber 2 of PTO cylinder 1 (m^2)
A_PTO_1_2 = A_WE_2  #The area of chamber 1 of PTO cylinder 2 (m^2)
A_PTO_2_2 = A_WE_1  #The area of chamber 2 of PTO cylinder 2 (m^2)
theta_WE_level = np.acos((a_WE**2+c_WE**2-(L_WE_0+x_WE_0)**2)/(2*a_WE*c_WE))
theta_WE_top = np.acos((a_WE**2+c_WE**2-L_WE_0**2)/(2*a_WE*c_WE))
theta_PTO_level = np.acos((a_PTO**2+c_PTO**2-(L_PTO_0+x_PTO_0)**2)/(2*a_PTO*c_PTO))
theta_PTO_top = np.acos((a_PTO**2+c_PTO**2-(L_PTO_0+Dead_PTO)**2)/(2*a_PTO*c_PTO))

# Wave data and simulation parameters: 
Ts = 5e-5  # Time step (s)
SimulationTime = 4 # Total simulation time (s)
numSteps = int(SimulationTime / Ts+1)  # Number of simulation steps
wave = np.zeros(numSteps)  # Wave data
wave[0:20000] = 0.001  # Wave height (m)
wave[20000:] = 0.1  # Wave height (m)

# Read data from Excel sheet
tout_data = pd.read_excel("C:/Users/Rober/OneDrive - Aalborg Universitet/MP.dok/Studie/Master/1. Semester/Project/testsinusstep.xlsx", sheet_name='tout', header=0)
yout_data = pd.read_excel("C:/Users/Rober/OneDrive - Aalborg Universitet/MP.dok/Studie/Master/1. Semester/Project/testsinusstep.xlsx", sheet_name='yout', header=0)

# Extract relevant columns from the Excel data
time_data = tout_data.iloc[:, 0].values
num_datapionts = len(time_data)
yoke_p_data = yout_data.iloc[:,7].values
p_WE_1_data = yout_data.iloc[:,10].values*1e5
p_WE_2_data = yout_data.iloc[:,11].values*1e5
p_C1R1_data = yout_data.iloc[:,12].values*1e5
p_C1B2_data = yout_data.iloc[:,13].values*1e5
p_C2R1_data = yout_data.iloc[:,14].values*1e5
p_C2B2_data = yout_data.iloc[:,15].values*1e5
x_cyl2_data = yoke_p_data*(x_PTO/100)+Dead_PTO
x_cyl1_data = Dead_PTO + x_PTO - (x_cyl2_data-Dead_PTO)
F_WE_data = A_WE_1 * p_WE_1_data - A_WE_2 * p_WE_2_data
F_cyl1_data = A_PTO_1_1 * p_C1R1_data - A_PTO_2_1 * p_C1B2_data
F_cyl2_data = A_PTO_1_2 * p_C2R1_data - A_PTO_2_2 * p_C2B2_data
tau_cyl1_data = F_cyl1_data * c_PTO * np.sin(np.acos(((L_PTO_0+x_cyl1_data)**2+c_PTO**2 - a_PTO**2)/(2*(L_PTO_0+x_cyl1_data)*c_PTO)))
tau_cyl2_data = F_cyl2_data * c_PTO * np.sin(np.acos(((L_PTO_0+x_cyl2_data)**2+c_PTO**2 - a_PTO**2)/(2*(L_PTO_0+x_cyl2_data)*c_PTO)))

# Select data points from 
start_row = int(200*7.5)
#end_row = start_row + 2500
end_row = start_row + int(SimulationTime*200)
# Update the time and Excel data to the selected range
time_data = time_data[start_row:end_row]-(start_row-1)/200-0.01
p_WE_1_data = p_WE_1_data[start_row:end_row]
p_WE_2_data = p_WE_2_data[start_row:end_row]
p_C1R1_data = p_C1R1_data[start_row:end_row]
p_C1B2_data = p_C1B2_data[start_row:end_row]
p_C2R1_data = p_C2R1_data[start_row:end_row]
p_C2B2_data = p_C2B2_data[start_row:end_row]
x_cyl2_data = x_cyl2_data[start_row:end_row]
x_cyl1_data = x_cyl1_data[start_row:end_row]
F_cyl1_data = F_cyl1_data[start_row:end_row]
F_cyl2_data = F_cyl2_data[start_row:end_row]
tau_cyl1_data = tau_cyl1_data[start_row:end_row]
tau_cyl2_data = tau_cyl2_data[start_row:end_row]

# Resizing the excel data to match the simulation data
time_data = np.interp(np.linspace(0, SimulationTime, numSteps), np.linspace(0, SimulationTime, len(time_data)), time_data)
p_WE_1_data = np.interp(np.linspace(0, SimulationTime, numSteps), np.linspace(0, SimulationTime, len(p_WE_1_data)), p_WE_1_data)
p_WE_2_data = np.interp(np.linspace(0, SimulationTime, numSteps), np.linspace(0, SimulationTime, len(p_WE_2_data)), p_WE_2_data)
p_C1R1_data = np.interp(np.linspace(0, SimulationTime, numSteps), np.linspace(0, SimulationTime, len(p_C1R1_data)), p_C1R1_data)
p_C1B2_data = np.interp(np.linspace(0, SimulationTime, numSteps), np.linspace(0, SimulationTime, len(p_C1B2_data)), p_C1B2_data)
p_C2R1_data = np.interp(np.linspace(0, SimulationTime, numSteps), np.linspace(0, SimulationTime, len(p_C2R1_data)), p_C2R1_data)
p_C2B2_data = np.interp(np.linspace(0, SimulationTime, numSteps), np.linspace(0, SimulationTime, len(p_C2B2_data)), p_C2B2_data)
x_cyl2_data = np.interp(np.linspace(0, SimulationTime, numSteps), np.linspace(0, SimulationTime, len(x_cyl2_data)), x_cyl2_data)
x_cyl1_data = np.interp(np.linspace(0, SimulationTime, numSteps), np.linspace(0, SimulationTime, len(x_cyl1_data)), x_cyl1_data)
F_cyl1_data = np.interp(np.linspace(0, SimulationTime, numSteps), np.linspace(0, SimulationTime, len(F_cyl1_data)), F_cyl1_data)
F_cyl2_data = np.interp(np.linspace(0, SimulationTime, numSteps), np.linspace(0, SimulationTime, len(F_cyl2_data)), F_cyl2_data)
tau_cyl1_data = np.interp(np.linspace(0, SimulationTime, numSteps), np.linspace(0, SimulationTime, len(tau_cyl1_data)), tau_cyl1_data)
tau_cyl2_data = np.interp(np.linspace(0, SimulationTime, numSteps), np.linspace(0, SimulationTime, len(tau_cyl2_data)), tau_cyl2_data)

# Initialize arrays to store simulation data
# Piston motion
yoke_p_arr = np.zeros(numSteps)
yoke_p_Ref_arr = np.zeros(numSteps)
x_v_arr = np.zeros(numSteps)
Q_1_arr = np.zeros(numSteps)
Q_2_arr = np.zeros(numSteps)
x_WE_arr = np.zeros(numSteps)
beta_1_arr = np.zeros(numSteps)
beta_2_arr = np.zeros(numSteps)
dot_x_WE_arr = np.zeros(numSteps)
ddot_x_WE_arr = np.zeros(numSteps)
p_S_arr = np.zeros(numSteps)
p_T_arr = np.zeros(numSteps)
p_WE_1_arr = np.zeros(numSteps)
p_WE_2_arr = np.zeros(numSteps)
dot_p_WE_1_arr = np.zeros(numSteps)
dot_p_WE_2_arr = np.zeros(numSteps)
# Dynamic model
theta_arr = np.zeros(numSteps)
dot_theta_arr = np.zeros(numSteps)
ddot_theta_arr = np.zeros(numSteps)
F_WE_arr = np.zeros(numSteps)
F_cyl1_arr = np.zeros(numSteps)
F_cyl2_arr = np.zeros(numSteps)
x_cyl1_arr = np.zeros(numSteps)
x_cyl2_arr = np.zeros(numSteps)
tau_g_arr = np.zeros(numSteps)
tau_cyl1_arr = np.zeros(numSteps)
tau_cyl2_arr = np.zeros(numSteps)
tau_WE_arr = np.zeros(numSteps)
tau_F_D_arr = np.zeros(numSteps)

# Initialize variables
yoke_p_arr[0] = 50
yoke_p_Ref_arr[0] = 50
x_v_arr[0] = 0
p_S_arr[0] = 180e5
p_T_arr[0] = 1e5
p_WE_1_arr[0] = p_WE_1_data[0]
p_WE_2_arr[0] = p_WE_2_data[0]
dot_p_WE_1_arr[0] = 0
dot_p_WE_2_arr[0] = 0
theta_arr[0] = 0
dot_theta_arr[0] = 0
ddot_theta_arr[0] = 0
x_cyl1_arr[0] = x_PTO 
x_cyl2_arr[0] = x_PTO + 0.01
x_WE_arr[0] = np.sqrt(a_WE**2 + c_WE**2 - 2*a_WE*c_WE*np.cos(theta_WE_level-theta_arr[0])) - np.sqrt(a_WE**2 + c_WE**2 - 2*a_WE*c_WE*np.cos(theta_WE_top))
dot_x_WE_arr[0] = 0
Q_1_arr[0] = Q_n / np.sqrt(p_n) * x_v_arr[0] * np.sqrt(np.abs(p_S_arr[0] - p_WE_1_arr[0])) * ((p_S_arr[0] - p_WE_1_arr[0]) / np.abs(p_S_arr[0] - p_WE_1_arr[0]))
Q_2_arr[0] = Q_n / np.sqrt(p_n) * x_v_arr[0] * np.sqrt(np.abs(p_WE_2_arr[0] - p_T_arr[0])) * ((p_WE_2_arr[0] - p_T_arr[0]) / np.abs(p_WE_2_arr[0] - p_T_arr[0]))
beta_1_arr[0] = ((1-alpha)*np.exp((p_T_val-p_WE_1_arr[0])/beta_0)+alpha*(p_T_val/p_WE_1_arr[0])**(1/n_polyIndx))/(((1-alpha)/beta_0)*np.exp((p_T_val-p_WE_1_arr[0])/beta_0)+(alpha/(n_polyIndx*p_T_val))*(p_T_val/p_WE_1_arr[0])**((n_polyIndx+1)/n_polyIndx))
beta_2_arr[0] = ((1-alpha)*np.exp((p_T_val-p_WE_2_arr[0])/beta_0)+alpha*(p_T_val/p_WE_2_arr[0])**(1/n_polyIndx))/(((1-alpha)/beta_0)*np.exp((p_T_val-p_WE_2_arr[0])/beta_0)+(alpha/(n_polyIndx*p_T_val))*(p_T_val/p_WE_2_arr[0])**((n_polyIndx+1)/n_polyIndx))
# Limit the pressure to a maximum of 400 bar and a minimum threshold
max_pressure = 250e5  #
min_pressure = 1e5  #
# limit angle acceleration
max_ddot_theta = np.deg2rad(100)
min_ddot_theta = np.deg2rad(-100)
# Dynamic model
F_WE_arr[0] = A_WE_1 * p_WE_1_arr[0] - A_WE_2 * p_WE_2_arr[0]
F_cyl1_arr[0] = F_cyl1_data[0]
F_cyl2_arr[0] = F_cyl2_data[0]
tau_g_arr[0] = M * g * np.sqrt(x_com**2 + y_com**2) * np.cos(np.atan(y_com/x_com)+theta_arr[0])
tau_cyl1_arr[0] = tau_cyl1_data[0]
tau_cyl2_arr[0] = tau_cyl2_data[0]
tau_WE_arr[0] = F_WE_arr[0] * np.sin(np.acos(((L_WE_0 + x_WE_arr[0])**2 + c_WE**2 - a_WE**2)/(2*(L_WE_0 + x_WE_arr[0])*c_WE))) * c_WE
tau_F_D_arr[0] = c * dot_theta_arr[0] + k * dot_theta_arr[0]
print("")

# Simulate the system with progress bar
for n in tqdm(range(1, numSteps), desc="Simulation Progress"):    
    
    # Piston motion
    yoke_p_Ref_arr[n] = 50 - wave[n] * 50
    yoke_p_arr[n] = (x_cyl2_arr[n-1]-Dead_PTO) / x_PTO * 100
    # Apply upper and lower limits to x_v
    x_v_arr[n] = ((yoke_p_Ref_arr[n] - yoke_p_arr[n]) * 0.12 + (((yoke_p_Ref_arr[n] - yoke_p_arr[n])-(yoke_p_Ref_arr[n-1] - yoke_p_arr[n-1]))/((Ts*n)-(Ts*n-1))*0.4))
    p_S_arr[n] = p_S_arr[0]
    p_T_arr[n] = p_T_arr[0]
    
    # Calculate flow rates based on pressure difference
    if x_v_arr[n] >= 0:
        Q_1_arr[n] = Q_n / np.sqrt(p_n) * x_v_arr[n] * np.sqrt(np.abs(p_S_arr[n] - p_WE_1_arr[n-1])) * ((p_S_arr[n] - p_WE_1_arr[n-1]) / np.abs(p_S_arr[n] - p_WE_1_arr[n-1]))
    else:
        Q_1_arr[n] = Q_n / np.sqrt(p_n) * x_v_arr[n] * np.sqrt(np.abs(p_WE_1_arr[n-1] - p_T_arr[n])) * ((p_WE_1_arr[n-1] - p_T_arr[n]) / np.abs(p_WE_1_arr[n-1] - p_T_arr[n]))
    
    if x_v_arr[n] >= 0:
        Q_2_arr[n] = Q_n / np.sqrt(p_n) * x_v_arr[n] * np.sqrt(np.abs(p_WE_2_arr[n-1] - p_T_arr[n])) * ((p_WE_2_arr[n-1] - p_T_arr[n]) / np.abs(p_WE_2_arr[n-1] - p_T_arr[n]))
    else:
        Q_2_arr[n] = Q_n / np.sqrt(p_n) * x_v_arr[n] * np.sqrt(np.abs(p_S_arr[n] - p_WE_2_arr[n-1])) * ((p_S_arr[n] - p_WE_2_arr[n-1]) / np.abs(p_S_arr[n] - p_WE_2_arr[n-1]))

    beta_1_arr[n] = ((1-alpha)*np.exp((p_T_val-p_WE_1_arr[n-1])/beta_0)+alpha*(p_T_val/p_WE_1_arr[n-1])**(1/n_polyIndx))/(((1-alpha)/beta_0)*np.exp((p_T_val-p_WE_1_arr[n-1])/beta_0)+(alpha/(n_polyIndx*p_T_val))*(p_T_val/p_WE_1_arr[n-1])**((n_polyIndx+1)/n_polyIndx))
    beta_2_arr[n] = ((1-alpha)*np.exp((p_T_val-p_WE_2_arr[n-1])/beta_0)+alpha*(p_T_val/p_WE_2_arr[n-1])**(1/n_polyIndx))/(((1-alpha)/beta_0)*np.exp((p_T_val-p_WE_2_arr[n-1])/beta_0)+(alpha/(n_polyIndx*p_T_val))*(p_T_val/p_WE_2_arr[n-1])**((n_polyIndx+1)/n_polyIndx))
    # Calculate pressure changes in chambers A and B
    dot_p_WE_1_arr[n] = (beta_1_arr[n]) / (V_WE_1_0 + A_WE_1 * (x_WE_arr[n-1])) * (Q_1_arr[n] - C_le * (p_WE_1_arr[n-1] - p_WE_2_arr[n-1]) - A_WE_1 * dot_x_WE_arr[n-1])
    dot_p_WE_2_arr[n] = (beta_2_arr[n]) / (V_WE_2_0 - A_WE_2 * (L_WE-x_WE_arr[n-1])) * (C_le * (p_WE_1_arr[n-1] - p_WE_2_arr[n-1]) - Q_2_arr[n] + A_WE_2 * dot_x_WE_arr[n-1])
    
    # Calculate the pressure in chambers A and B
    p_WE_1_arr[n] = np.clip(dot_p_WE_1_arr[n] * Ts + p_WE_1_arr[n-1], min_pressure, max_pressure)
    p_WE_2_arr[n] = np.clip(dot_p_WE_2_arr[n] * Ts + p_WE_2_arr[n-1], min_pressure, max_pressure)

    # Calculate the force acting from the WE, cylinder 1 and 2:
    F_WE_arr[n] = A_WE_1 * p_WE_1_arr[n] - A_WE_2 * p_WE_2_arr[n]
    F_cyl1_arr[n] = F_cyl1_data[n]*0
    F_cyl2_arr[n] = F_cyl2_data[n]*0

    # Dynamic model
    # Calculate the torque of the gravity:
    tau_g_arr[n] = M * g * np.sqrt(x_com**2 + y_com**2) * np.cos(np.atan(y_com/x_com)+theta_arr[n-1])
    
    # Calculate the torque of the cylinder 1 and 2:
    tau_cyl1_arr[n] = tau_cyl1_data[n]*0
    tau_cyl2_arr[n] = tau_cyl2_data[n]*0

    # Calculate the torque of the wave emulater:
    tau_WE_arr[n] = F_WE_arr[n] * np.sin(np.acos(((L_WE_0 + x_WE_arr[n-1])**2 + c_WE**2 - a_WE**2)/(2*(L_WE_0 + x_WE_arr[n-1])*c_WE))) * c_WE

    # Calculate the torque of the friction and the damping:
    tau_F_D_arr[n] = c * dot_theta_arr[n-1] + k * np.sign(dot_theta_arr[n-1])

    # Calculate ddot_theta:
    if x_v_arr[n] >= 0:
        ddot_theta_arr[n] = (tau_cyl1_arr[n] + tau_cyl2_arr[n] - tau_WE_arr[n] - tau_g_arr[n] - tau_F_D_arr[n])/J
    else:
        ddot_theta_arr[n] = (-tau_cyl1_arr[n] - tau_cyl2_arr[n] - tau_WE_arr[n] - tau_g_arr[n] - tau_F_D_arr[n])/J
    
    # Ensure ddot_theta is within the range of -15 to 15 degrees/s^2
    if ddot_theta_arr[n] > max_ddot_theta:
        ddot_theta_arr[n] = max_ddot_theta
    elif ddot_theta_arr[n] < min_ddot_theta:
        ddot_theta_arr[n] = min_ddot_theta
    
    # Calculate dot_theta and theta:
    dot_theta_arr[n] = dot_theta_arr[n-1] + ddot_theta_arr[n] * Ts
    theta_arr[n] = theta_arr[n-1] + dot_theta_arr[n] * Ts 
    
    # Ensure theta is within the range of -theta_0 to theta_0 degrees
    max_theta = np.deg2rad(11)
    if theta_arr[n] > max_theta:
        theta_arr[n] = max_theta
        dot_theta_arr[n] = 0  # Stop angular velocity if limit is reached
    elif theta_arr[n] < -max_theta:
        theta_arr[n] = -max_theta
        dot_theta_arr[n] = 0  # Stop angular velocity if limit is reached
    
    # Update the position of the cylinder WE, 1 and 2:
    x_WE_arr[n] = np.sqrt(a_WE**2 + c_WE**2 - 2*a_WE*c_WE*np.cos(theta_WE_level-theta_arr[n])) - np.sqrt(a_WE**2 + c_WE**2 - 2*a_WE*c_WE*np.cos(theta_WE_top))
    x_cyl2_arr[n] = Dead_PTO + np.sqrt(a_PTO**2 + c_PTO**2 - 2*a_PTO*c_PTO*np.cos(theta_PTO_level - theta_arr[n])) - np.sqrt(a_PTO**2 + c_PTO**2 - 2*a_PTO*c_PTO*np.cos(theta_PTO_top))
    x_cyl1_arr[n] = Dead_PTO + x_PTO - (x_cyl2_arr[n]-Dead_PTO)
    dot_x_WE_arr[n] = (x_WE_arr[n] - x_WE_arr[n-1])/Ts

    # Store the results
    yoke_p_arr[n] = round(yoke_p_arr[n], 10)
    x_v_arr[n] = round(x_v_arr[n], 10)
    Q_1_arr[n] = round(Q_1_arr[n], 10)
    Q_2_arr[n] = round(Q_2_arr[n], 10)
    beta_1_arr[n] = round(beta_1_arr[n], 10)
    beta_2_arr[n] = round(beta_2_arr[n], 10)
    dot_p_WE_1_arr[n] = round(dot_p_WE_1_arr[n], 10)
    dot_p_WE_2_arr[n] = round(dot_p_WE_2_arr[n], 10)
    p_WE_1_arr[n] = round(p_WE_1_arr[n], 10)
    p_WE_2_arr[n] = round(p_WE_2_arr[n], 10)
    F_WE_arr[n] = round(F_WE_arr[n], 10)
    F_cyl1_arr[n] = round(F_cyl1_arr[n], 10)
    F_cyl2_arr[n] = round(F_cyl2_arr[n], 10)
    x_cyl1_arr[n] = round(x_cyl1_arr[n], 10)
    x_cyl2_arr[n] = round(x_cyl2_arr[n], 10)
    tau_g_arr[n] = round(tau_g_arr[n], 10)
    tau_cyl1_arr[n] = round(tau_cyl1_arr[n], 10)
    tau_cyl2_arr[n] = round(tau_cyl2_arr[n], 10)
    tau_WE_arr[n] = round(tau_WE_arr[n], 10)
    tau_F_D_arr[n] = round(tau_F_D_arr[n], 10)
    ddot_theta_arr[n] = round(ddot_theta_arr[n], 10)
    dot_theta_arr[n] = round(dot_theta_arr[n], 10)
    theta_arr[n] = round(theta_arr[n], 10)
    x_WE_arr[n] = round(x_WE_arr[n], 10)
    dot_x_WE_arr[n] = round(dot_x_WE_arr[n], 10)

    if n < 2 or n % 5000 == 0 or n == 19999 or n == 20000:
        # Print the results
        print("Step:", n)
        print("Yoke_p:", yoke_p_arr[n])
        print("Yoke_p_Ref:", yoke_p_Ref_arr[n])
        print("x_v sevor input:", x_v_arr[n])
        print("The diff part:",((yoke_p_Ref_arr[n] - yoke_p_arr[n])-(yoke_p_Ref_arr[n-1] - yoke_p_arr[n-1]))/((Ts*n)-(Ts*n-1)))
        print("Flow Rate Q1:", Q_1_arr[n])
        print("Flow Rate Q2:", Q_2_arr[n])
        print("Beta 1:", beta_1_arr[n]/1e5)
        print("Beta 2:", beta_2_arr[n]/1e5)
        print("Pressure change in Chamber A:", dot_p_WE_1_arr[n]/1e5)
        print("Pressure change in Chamber B:", dot_p_WE_2_arr[n]/1e5)
        print("Pressure in Chamber A:", p_WE_1_arr[n]/1e5)
        print("Pressure in Chamber B:", p_WE_2_arr[n]/1e5)
        print("Force from WE:", F_WE_arr[n])
        print("Force from Cylinder 1:", F_cyl1_arr[n])
        print("Force from Cylinder 2:", F_cyl2_arr[n])
        print("Torque from Gravity:", tau_g_arr[n])
        print("Torque from Cylinder 1:", tau_cyl1_arr[n])
        print("Torque from Cylinder 2:", tau_cyl2_arr[n])
        print("Torque from WE:", tau_WE_arr[n])
        print("Torque from Friction and Damping:", tau_F_D_arr[n])
        print("x_cyl1:", x_cyl1_arr[n])
        print("x_cyl2:", x_cyl2_arr[n])
        print("Theta:", np.rad2deg(theta_arr[n]))
        print("Angular Velocity:", dot_theta_arr[n])
        print("Angular Acceleration:", ddot_theta_arr[n])
        print("Position of WE:", x_WE_arr[n])
        print("Velocity of WE:", dot_x_WE_arr[n])
        print("")

# Plot results
time = np.linspace(0, SimulationTime, int(SimulationTime / Ts + 1))

plt.figure(figsize=(12, 8))
plt.subplot(3, 2, 1)
plt.plot(time, yoke_p_arr, label="Yoke Position")
plt.xlabel("Time [s]")
plt.ylabel("Yoke Position [%]")
plt.legend()
plt.grid()

plt.subplot(3, 2, 2)
plt.plot(time, x_v_arr, label="Servo Input")
plt.xlabel("Time [s]")
plt.ylabel("Servo Input [m]")
plt.legend()
plt.grid()

plt.subplot(3, 2, 3)
plt.plot(time, Q_1_arr, label="Flow Rate Q1")
plt.plot(time, Q_2_arr, label="Flow Rate Q2")
plt.xlabel("Time [s]")
plt.ylabel("Flow Rate [m^3/s]")
plt.legend()
plt.grid()

plt.subplot(3, 2, 4)
plt.plot(time, p_WE_1_arr/1e5, label="Pressure in Chamber A")
plt.plot(time, p_WE_2_arr/1e5, label="Pressure in Chamber B")
plt.xlabel("Time [s]")
plt.ylabel("Pressure [Bar]")
plt.legend()
plt.grid()

plt.subplot(3, 2, 5)
plt.plot(time, tau_WE_arr, label="Torque from WE")
plt.xlabel("Time [s]")
plt.ylabel("Torque [Nm]")
plt.legend()
plt.grid()

plt.subplot(3, 2, 6)
plt.plot(time, np.rad2deg(theta_arr), label="Angle")
plt.xlabel("Time [s]")
plt.ylabel("Angle [deg]")
plt.legend()
plt.grid()

plt.tight_layout()
plt.show()

