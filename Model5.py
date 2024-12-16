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
theta_0 = np.deg2rad(9.6) # Initial angle (rad)
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
Ts = 1e-4  # Time step (s)
SimulationTime = 8.9 # Total simulation time (s)
numSteps = int(SimulationTime / Ts+1)  # Number of simulation steps

# Define the step function parameters
Period = 3  # 3 second period
step_0 = 0
step_1 = 2.5
step_2 = 5
step_3 = -5
step_4 = 0

# Initialize the step function array
Step = np.zeros(numSteps)

# Generate the step function over the simulation time
for i in range(numSteps):
    counter = i * Ts
    if  counter < Period * 1:
        Step[i] = step_1
    elif counter >= Period * 1 and counter < Period * 2:
        Step[i] = step_2
    elif counter >= Period * 2 and counter < Period * 3:
        Step[i] = step_3
    elif counter >= Period * 3 and counter < Period * 4:
        Step[i] = step_4
    elif counter >= Period * 4:
        Step[i] = step_0
# Read data from Excel sheet
tout_data = pd.read_excel("C:/Users/Rober/OneDrive - Aalborg Universitet/MP.dok/Studie/Master/1. Semester/Project/testStep1.xlsx", sheet_name='tout', header=0)
yout_data = pd.read_excel("C:/Users/Rober/OneDrive - Aalborg Universitet/MP.dok/Studie/Master/1. Semester/Project/testStep1.xlsx", sheet_name='yout', header=0)

# Extract relevant columns from the Excel data
time_data = tout_data.iloc[:, 0].values
num_datapionts = len(time_data)
yoke_p_data = yout_data.iloc[:,7].values
p_S_data = yout_data.iloc[:,8].values*1e5
p_T_data = yout_data.iloc[:,9].values*1e5
p_WE_1_data = yout_data.iloc[:,10].values*1e5
dot_p_WE_1_data = np.gradient(p_WE_1_data, time_data)
p_WE_2_data = yout_data.iloc[:,11].values*1e5
dot_p_WE_2_data = np.gradient(p_WE_2_data, time_data)
p_C1R1_data = yout_data.iloc[:,12].values*1e5
p_C1B2_data = yout_data.iloc[:,13].values*1e5
p_C2R1_data = yout_data.iloc[:,14].values*1e5
p_C2B2_data = yout_data.iloc[:,15].values*1e5
x_v_data = yout_data.iloc[:,20].values
x_cyl2_data = yoke_p_data*(x_PTO/100)+Dead_PTO
x_cyl1_data = Dead_PTO + x_PTO - (x_cyl2_data-Dead_PTO)
beta_1_data = ((1-alpha)*np.exp((1e5-p_WE_1_data)/beta_0)+alpha*(1e5/p_WE_1_data)**(1/n_polyIndx))/(((1-alpha)/beta_0)*np.exp((1e5-p_WE_1_data)/beta_0)+(alpha/(n_polyIndx*1e5))*(1e5/p_WE_1_data)**((n_polyIndx+1)/n_polyIndx))
beta_2_data = ((1-alpha)*np.exp((1e5-p_WE_2_data)/beta_0)+alpha*(1e5/p_WE_2_data)**(1/n_polyIndx))/(((1-alpha)/beta_0)*np.exp((1e5-p_WE_2_data)/beta_0)+(alpha/(n_polyIndx*1e5))*(1e5/p_WE_2_data)**((n_polyIndx+1)/n_polyIndx))
theta_data = np.deg2rad(90) - np.acos((c_PTO**2 + a_PTO**2 - (L_PTO_0 + x_cyl2_data)**2)/(2*c_PTO*a_PTO)) - np.deg2rad(33.56)
dot_theta_data = np.gradient(theta_data, time_data)
x_WE_data = np.sqrt(a_WE**2 + c_WE**2 - 2*a_WE*c_WE*np.cos(theta_WE_level-theta_data)) - np.sqrt(a_WE**2 + c_WE**2 - 2*a_WE*c_WE*np.cos(theta_WE_top))
dot_x_WE_data = np.gradient(x_WE_data, time_data)
Q_1_data = A_WE_1 * dot_x_WE_data
Q_2_data = A_WE_2 * dot_x_WE_data
F_WE_data = A_WE_1 * p_WE_1_data - A_WE_2 * p_WE_2_data
F_cyl1_data = A_PTO_1_1 * p_C1R1_data - A_PTO_2_1 * p_C1B2_data
F_cyl2_data = A_PTO_1_2 * p_C2R1_data - A_PTO_2_2 * p_C2B2_data
tau_cyl1_data = F_cyl1_data * c_PTO * np.sin(np.acos(((L_PTO_0+x_cyl1_data)**2+c_PTO**2 - a_PTO**2)/(2*(L_PTO_0+x_cyl1_data)*c_PTO)))
tau_cyl2_data = F_cyl2_data * c_PTO * np.sin(np.acos(((L_PTO_0+x_cyl2_data)**2+c_PTO**2 - a_PTO**2)/(2*(L_PTO_0+x_cyl2_data)*c_PTO)))
tau_WE_data = F_WE_data * np.sin(np.acos(((L_WE_0 + x_WE_data)**2 + c_WE**2 - a_WE**2)/(2*(L_WE_0 + x_WE_data)*c_WE))) * c_WE
tau_F_D_data = c * dot_theta_data + k * np.sign(dot_theta_data)

# Select data points from 
start_row = int(200*8.2)
#end_row = start_row + 2500
end_row = start_row + int(SimulationTime*200)
# Update the time and Excel data to the selected range
time_data = time_data[start_row:end_row]-(start_row-1)/200-0.01
yoke_p_data = yoke_p_data[start_row:end_row]
p_S_data = p_S_data[start_row:end_row]
p_T_data = p_T_data[start_row:end_row]
p_WE_1_data = p_WE_1_data[start_row:end_row]
dot_p_WE_1_data = dot_p_WE_1_data[start_row:end_row]
p_WE_2_data = p_WE_2_data[start_row:end_row]
dot_p_WE_2_data = dot_p_WE_2_data[start_row:end_row]
p_C1R1_data = p_C1R1_data[start_row:end_row]
p_C1B2_data = p_C1B2_data[start_row:end_row]
p_C2R1_data = p_C2R1_data[start_row:end_row]
p_C2B2_data = p_C2B2_data[start_row:end_row]
x_v_data = x_v_data[start_row:end_row]
x_cyl2_data = x_cyl2_data[start_row:end_row]
x_cyl1_data = x_cyl1_data[start_row:end_row]
dot_theta_data = dot_theta_data[start_row:end_row]
Q_1_data = Q_1_data[start_row:end_row]
Q_2_data = Q_2_data[start_row:end_row]
beta_1_data = beta_1_data[start_row:end_row]
beta_2_data = beta_2_data[start_row:end_row]
theta_data = theta_data[start_row:end_row]
x_WE_data = x_WE_data[start_row:end_row]
dot_x_WE_data = dot_x_WE_data[start_row:end_row]
F_WE_data = F_WE_data[start_row:end_row]
F_cyl1_data = F_cyl1_data[start_row:end_row]
F_cyl2_data = F_cyl2_data[start_row:end_row]
tau_cyl1_data = tau_cyl1_data[start_row:end_row]
tau_cyl2_data = tau_cyl2_data[start_row:end_row]
tau_WE_data = tau_WE_data[start_row:end_row]
tau_F_D_data = tau_F_D_data[start_row:end_row]

# Resizing the excel data to match the simulation data
time_data = np.interp(np.linspace(0, SimulationTime, numSteps), np.linspace(0, SimulationTime, len(time_data)), time_data)
yoke_p_data = np.interp(np.linspace(0, SimulationTime, numSteps), np.linspace(0, SimulationTime, len(yoke_p_data)), yoke_p_data)
p_S_data = np.interp(np.linspace(0, SimulationTime, numSteps), np.linspace(0, SimulationTime, len(p_S_data)), p_S_data)
p_T_data = np.interp(np.linspace(0, SimulationTime, numSteps), np.linspace(0, SimulationTime, len(p_T_data)), p_T_data)
p_WE_1_data = np.interp(np.linspace(0, SimulationTime, numSteps), np.linspace(0, SimulationTime, len(p_WE_1_data)), p_WE_1_data)
dot_p_WE_1_data = np.interp(np.linspace(0, SimulationTime, numSteps), np.linspace(0, SimulationTime, len(dot_p_WE_1_data)), dot_p_WE_1_data)
p_WE_2_data = np.interp(np.linspace(0, SimulationTime, numSteps), np.linspace(0, SimulationTime, len(p_WE_2_data)), p_WE_2_data)
dot_p_WE_2_data = np.interp(np.linspace(0, SimulationTime, numSteps), np.linspace(0, SimulationTime, len(dot_p_WE_2_data)), dot_p_WE_2_data)
p_C1R1_data = np.interp(np.linspace(0, SimulationTime, numSteps), np.linspace(0, SimulationTime, len(p_C1R1_data)), p_C1R1_data)
p_C1B2_data = np.interp(np.linspace(0, SimulationTime, numSteps), np.linspace(0, SimulationTime, len(p_C1B2_data)), p_C1B2_data)
p_C2R1_data = np.interp(np.linspace(0, SimulationTime, numSteps), np.linspace(0, SimulationTime, len(p_C2R1_data)), p_C2R1_data)
p_C2B2_data = np.interp(np.linspace(0, SimulationTime, numSteps), np.linspace(0, SimulationTime, len(p_C2B2_data)), p_C2B2_data)
x_v_data = np.interp(np.linspace(0, SimulationTime, numSteps), np.linspace(0, SimulationTime, len(x_v_data)), x_v_data)
x_cyl2_data = np.interp(np.linspace(0, SimulationTime, numSteps), np.linspace(0, SimulationTime, len(x_cyl2_data)), x_cyl2_data)
x_cyl1_data = np.interp(np.linspace(0, SimulationTime, numSteps), np.linspace(0, SimulationTime, len(x_cyl1_data)), x_cyl1_data)
dot_theta_data = np.interp(np.linspace(0, SimulationTime, numSteps), np.linspace(0, SimulationTime, len(dot_theta_data)), dot_theta_data)
Q_1_data = np.interp(np.linspace(0, SimulationTime, numSteps), np.linspace(0, SimulationTime, len(Q_1_data)), Q_1_data)
Q_2_data = np.interp(np.linspace(0, SimulationTime, numSteps), np.linspace(0, SimulationTime, len(Q_2_data)), Q_2_data)
beta_1_data = np.interp(np.linspace(0, SimulationTime, numSteps), np.linspace(0, SimulationTime, len(beta_1_data)), beta_1_data)
beta_2_data = np.interp(np.linspace(0, SimulationTime, numSteps), np.linspace(0, SimulationTime, len(beta_2_data)), beta_2_data)
theta_data = np.interp(np.linspace(0, SimulationTime, numSteps), np.linspace(0, SimulationTime, len(theta_data)), theta_data)
x_WE_data = np.interp(np.linspace(0, SimulationTime, numSteps), np.linspace(0, SimulationTime, len(x_WE_data)), x_WE_data)
dot_x_WE_data = np.interp(np.linspace(0, SimulationTime, numSteps), np.linspace(0, SimulationTime, len(dot_x_WE_data)), dot_x_WE_data)
F_WE_data = np.interp(np.linspace(0, SimulationTime, numSteps), np.linspace(0, SimulationTime, len(F_WE_data)), F_WE_data)
F_cyl1_data = np.interp(np.linspace(0, SimulationTime, numSteps), np.linspace(0, SimulationTime, len(F_cyl1_data)), F_cyl1_data)
F_cyl2_data = np.interp(np.linspace(0, SimulationTime, numSteps), np.linspace(0, SimulationTime, len(F_cyl2_data)), F_cyl2_data)
tau_cyl1_data = np.interp(np.linspace(0, SimulationTime, numSteps), np.linspace(0, SimulationTime, len(tau_cyl1_data)), tau_cyl1_data)
tau_cyl2_data = np.interp(np.linspace(0, SimulationTime, numSteps), np.linspace(0, SimulationTime, len(tau_cyl2_data)), tau_cyl2_data)
tau_WE_data = np.interp(np.linspace(0, SimulationTime, numSteps), np.linspace(0, SimulationTime, len(tau_WE_data)), tau_WE_data)
tau_F_D_data = np.interp(np.linspace(0, SimulationTime, numSteps), np.linspace(0, SimulationTime, len(tau_F_D_data)), tau_F_D_data)

# Initialize arrays to store simulation data
# Piston motion
yoke_p_arr = np.zeros(numSteps)
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

p_S_avg = np.mean(p_S_data)
p_T_avg = np.mean(p_T_data)
p_C1R1_avg = np.mean(p_C1R1_data)
p_C1B2_avg = np.mean(p_C1B2_data)
p_C2R1_avg = np.mean(p_C2R1_data)
p_C2B2_avg = np.mean(p_C2B2_data)
F_cyl1_avg = np.mean(F_cyl1_data)
F_cyl2_avg = np.mean(F_cyl2_data)
tau_cyl1_avg = np.mean(tau_cyl1_data)
tau_cyl2_avg = np.mean(tau_cyl2_data) 

# Initialize variables
yoke_p_arr[0] = yoke_p_data[0]
x_v_arr[0] = -0.025
p_S_arr[0] = p_S_data[0]
p_T_arr[0] = p_T_data[0]
p_WE_1_arr[0] = p_WE_1_data[0]
p_WE_2_arr[0] = p_WE_2_data[0]
dot_p_WE_1_arr[0] = dot_p_WE_1_data[0]
dot_p_WE_2_arr[0] = dot_p_WE_2_data[0]
theta_arr[0] = theta_data[0]
dot_theta_arr[0] = dot_theta_data[0]
ddot_theta_arr[0] = 0
x_cyl1_arr[0] = x_cyl1_data[0]
x_cyl2_arr[0] = x_cyl2_data[0]
x_WE_arr[0] = np.sqrt(a_WE**2 + c_WE**2 - 2*a_WE*c_WE*np.cos(theta_WE_level-theta_arr[0])) - np.sqrt(a_WE**2 + c_WE**2 - 2*a_WE*c_WE*np.cos(theta_WE_top))
dot_x_WE_arr[0] = dot_x_WE_data[0]
Q_1_arr[0] = Q_1_data[0]
Q_2_arr[0] = Q_2_data[0]
beta_1_arr[0] = ((1-alpha)*np.exp((p_T_val-p_WE_1_arr[0])/beta_0)+alpha*(p_T_val/p_WE_1_arr[0])**(1/n_polyIndx))/(((1-alpha)/beta_0)*np.exp((p_T_val-p_WE_1_arr[0])/beta_0)+(alpha/(n_polyIndx*p_T_val))*(p_T_val/p_WE_1_arr[0])**((n_polyIndx+1)/n_polyIndx))
beta_2_arr[0] = ((1-alpha)*np.exp((p_T_val-p_WE_2_arr[0])/beta_0)+alpha*(p_T_val/p_WE_2_arr[0])**(1/n_polyIndx))/(((1-alpha)/beta_0)*np.exp((p_T_val-p_WE_2_arr[0])/beta_0)+(alpha/(n_polyIndx*p_T_val))*(p_T_val/p_WE_2_arr[0])**((n_polyIndx+1)/n_polyIndx))
# Limit the pressure to a maximum of 400 bar and a minimum threshold
max_pressure = 250e5  #
min_pressure = 1e5  #
# limit angle acceleration
max_ddot_theta = np.deg2rad(50)
min_ddot_theta = np.deg2rad(-50)
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
    yoke_p_arr[n] = (x_cyl2_arr[n-1]-Dead_PTO) / x_PTO * 100
    # Apply upper and lower limits to x_v
    x_v_arr[n] = -Step[n]/70
    p_S_arr[n] = p_S_data[n]
    p_T_arr[n] = p_T_data[n]
    
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
    F_cyl1_arr[n] = F_cyl1_data[n]
    F_cyl2_arr[n] = F_cyl2_data[n]

    # Dynamic model
    # Calculate the torque of the gravity:
    tau_g_arr[n] = M * g * np.sqrt(x_com**2 + y_com**2) * np.cos(np.atan(y_com/x_com)+theta_arr[n-1])
    
    # Calculate the torque of the cylinder 1 and 2:
    tau_cyl1_arr[n] = tau_cyl1_data[n]
    tau_cyl2_arr[n] = tau_cyl2_data[n]

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
    #if ddot_theta_arr[n] > max_ddot_theta:
    #    ddot_theta_arr[n] = max_ddot_theta
    #elif ddot_theta_arr[n] < min_ddot_theta:
    #    ddot_theta_arr[n] = min_ddot_theta
    
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

    if n < 2 or n % 5000 == 0:
        # Print the results
        print("Step:", n)
        print("Yoke_p:", yoke_p_arr[n])
        print("x_v sevor input:", x_v_arr[n])
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

# Exclude the first x rows of the simulation arrays
start_index = 0
x_v_arr = x_v_arr[start_index:]
Q_1_arr = Q_1_arr[start_index:]
Q_2_arr = Q_2_arr[start_index:]
x_WE_arr = x_WE_arr[start_index:]
beta_1_arr = beta_1_arr[start_index:]
beta_2_arr = beta_2_arr[start_index:]
dot_x_WE_arr = dot_x_WE_arr[start_index:]
ddot_x_WE_arr = ddot_x_WE_arr[start_index:]
p_WE_1_arr = p_WE_1_arr[start_index:]
p_WE_2_arr = p_WE_2_arr[start_index:]
dot_p_WE_1_arr = dot_p_WE_1_arr[start_index:]
dot_p_WE_2_arr = dot_p_WE_2_arr[start_index:]
theta_arr = theta_arr[start_index:]
dot_theta_arr = dot_theta_arr[start_index:]
ddot_theta_arr = ddot_theta_arr[start_index:]
F_WE_arr = F_WE_arr[start_index:]
F_cyl1_arr = F_cyl1_arr[start_index:]
F_cyl2_arr = F_cyl2_arr[start_index:]
x_cyl1_arr = x_cyl1_arr[start_index:]
x_cyl2_arr = x_cyl2_arr[start_index:]
tau_g_arr = tau_g_arr[start_index:]
tau_cyl1_arr = tau_cyl1_arr[start_index:]
tau_cyl2_arr = tau_cyl2_arr[start_index:]
tau_WE_arr = tau_WE_arr[start_index:]
tau_F_D_arr = tau_F_D_arr[start_index:]
time = time[start_index:]

# Exclude the first x rows of the data arrays
time_data = time_data[start_index:]
yoke_p_data = yoke_p_data[start_index:]
p_S_data = p_S_data[start_index:]
p_T_data = p_T_data[start_index:]
p_WE_1_data = p_WE_1_data[start_index:]
dot_p_WE_1_data = dot_p_WE_1_data[start_index:]
p_WE_2_data = p_WE_2_data[start_index:]
dot_p_WE_2_data = dot_p_WE_2_data[start_index:]
p_C1R1_data = p_C1R1_data[start_index:]
p_C1B2_data = p_C1B2_data[start_index:]
p_C2R1_data = p_C2R1_data[start_index:]
p_C2B2_data = p_C2B2_data[start_index:]
x_v_data = x_v_data[start_index:]
x_cyl2_data = x_cyl2_data[start_index:]
x_cyl1_data = x_cyl1_data[start_index:]
dot_theta_data = dot_theta_data[start_index:]
Q_1_data = Q_1_data[start_index:]
Q_2_data = Q_2_data[start_index:]
beta_1_data = beta_1_data[start_index:]
beta_2_data = beta_2_data[start_index:]
theta_data = theta_data[start_index:]
x_WE_data = x_WE_data[start_index:]
dot_x_WE_data = dot_x_WE_data[start_index:]
F_WE_data = F_WE_data[start_index:]
F_cyl1_data = F_cyl1_data[start_index:]
F_cyl2_data = F_cyl2_data[start_index:]
tau_cyl1_data = tau_cyl1_data[start_index:]
tau_cyl2_data = tau_cyl2_data[start_index:]
tau_WE_data = tau_WE_data[start_index:]
tau_F_D_data = tau_F_D_data[start_index:]

# Set font globally to Times New Roman
plt.rcParams['font.family'] = 'Times New Roman'

# Plot valve input:
plt.figure(figsize=(12, 6))
plt.plot(time, -x_v_arr*10, label='Valve Input (simulation)', color='blue', linewidth=2)
plt.plot(time_data, x_v_data, label='Valve Input (data)', linestyle='dashed', color='red', linewidth=1)
plt.xlabel('Time [s]', fontname="Times New Roman", fontsize=12)
plt.ylabel('Valve Input', fontname="Times New Roman", fontsize=12)
plt.title('Valve Input vs Time', fontname="Times New Roman", fontsize=14)
plt.legend(fontsize=10, loc='upper left')
plt.grid(color='gray', linestyle='--', linewidth=0.5)
plt.tick_params(axis='both', which='major', labelsize=10)

# Plot Flow Rate Q1 and Q2
plt.figure(figsize=(12, 6))
plt.plot(time, Q_1_arr*6e4, label='Flow Rate Q1 (simulation)', color='blue', linewidth=2)
plt.plot(time, Q_2_arr*6e4, label='Flow Rate Q2 (simulation)', color='green', linewidth=2)
plt.plot(time_data, Q_1_data*6e4, label='Flow Rate Q1 (data)', linestyle='dashed', color='red', linewidth=1)
plt.plot(time_data, Q_2_data*6e4, label='Flow Rate Q2 (data)', linestyle='dashed', color='orange', linewidth=1)
plt.xlabel('Time [s]', fontname="Times New Roman", fontsize=12)
plt.ylabel('Flow Rate [L/min]', fontname="Times New Roman", fontsize=12)
plt.title('Flow Rate Q1 and Q2 vs Time', fontname="Times New Roman", fontsize=14)
plt.legend(fontsize=10, loc='upper left')
plt.grid(color='gray', linestyle='--', linewidth=0.5)
plt.tick_params(axis='both', which='major', labelsize=10)

# Plot bulk modulus beta_1 and beta_2
plt.figure(figsize=(12, 6))
plt.plot(time, beta_1_arr/1e5, label='Bulk Modulus in Chamber A (simulation)', color='blue', linewidth=2)
plt.plot(time, beta_2_arr/1e5, label='Bulk Modulus in Chamber B (simulation)', color='green', linewidth=2)
plt.plot(time_data, beta_1_data/1e5, label='Bulk Modulus in Chamber A (data)', linestyle='dashed', color='red', linewidth=1)
plt.plot(time_data, beta_2_data/1e5, label='Bulk Modulus in Chamber B (data)', linestyle='dashed', color='orange', linewidth=1)
plt.xlabel('Time [s]', fontname="Times New Roman", fontsize=12)
plt.ylabel('Bulk Modulus [Bar]', fontname="Times New Roman", fontsize=12)
plt.title('Bulk Modulus in Chambers A and B vs Time', fontname="Times New Roman", fontsize=14)
plt.legend(fontsize=10, loc='upper left')
plt.grid(color='gray', linestyle='--', linewidth=0.5)
plt.tick_params(axis='both', which='major', labelsize=10)

# Plot Pressure change in Chamber A and B
plt.figure(figsize=(12, 6))
plt.plot(time, dot_p_WE_1_arr/1e5, label='Pressure change in Chamber A (simulation)', color='blue', linewidth=2)
plt.plot(time, dot_p_WE_2_arr/1e5, label='Pressure change in Chamber B (simulation)', color='green', linewidth=2)
plt.plot(time_data, dot_p_WE_1_data/1e5, label='Pressure change in Chamber A (data)', linestyle='dashed', color='red', linewidth=1)
plt.plot(time_data, dot_p_WE_2_data/1e5, label='Pressure change in Chamber B (data)', linestyle='dashed', color='orange', linewidth=1)
plt.xlabel('Time [s]', fontname="Times New Roman", fontsize=12)
plt.ylabel('Pressure Change [Bar/s]', fontname="Times New Roman", fontsize=12)
plt.title('Pressure Change in Chambers A and B vs Time', fontname="Times New Roman", fontsize=14)
plt.legend(fontsize=10, loc='upper left')
plt.grid(color='gray', linestyle='--', linewidth=0.5)
plt.tick_params(axis='both', which='major', labelsize=10)

# Plot Pressure in Chamber A and B
plt.figure(figsize=(12, 6))
plt.plot(time, p_WE_1_arr/1e5, label='Pressure in Chamber A (simulation)', color='blue', linewidth=2)
plt.plot(time, p_WE_2_arr/1e5, label='Pressure in Chamber B (simulation)', color='green', linewidth=2)
plt.plot(time_data, p_WE_1_data/1e5, label='Pressure in Chamber A (data)', linestyle='dashed', color='red', linewidth=1)
plt.plot(time_data, p_WE_2_data/1e5, label='Pressure in Chamber B (data)', linestyle='dashed', color='orange', linewidth=1)
plt.xlabel('Time [s]', fontname="Times New Roman", fontsize=12)
plt.ylabel('Pressure [Bar]', fontname="Times New Roman", fontsize=12)
plt.title('Pressure in Chambers A and B vs Time', fontname="Times New Roman", fontsize=14)
plt.legend(fontsize=10, loc='upper left')
plt.grid(color='gray', linestyle='--', linewidth=0.5)
plt.tick_params(axis='both', which='major', labelsize=10)

# Plot force from WE, Cylinder 1, and Cylinder 2
plt.figure(figsize=(12, 6))
plt.plot(time, F_WE_arr, label='Force from WE', color='blue', linewidth=2)
plt.plot(time_data, F_WE_data, label='Force from WE (data)', linestyle='dashed', color='red', linewidth=1)
plt.xlabel('Time [s]', fontname="Times New Roman", fontsize=12)
plt.ylabel('Force [N]', fontname="Times New Roman", fontsize=12)
plt.title('Force from WE vs Time', fontname="Times New Roman", fontsize=14)
plt.legend(fontsize=10, loc='upper left')
plt.grid(color='gray', linestyle='--', linewidth=0.5)
plt.tick_params(axis='both', which='major', labelsize=10)

# Plot Torque from Gravity, Cylinder 1, Cylinder 2, WE, and Friction and Damping
plt.figure(figsize=(12, 6))
plt.plot(time, tau_WE_arr, label='Torque from WE', color='purple', linewidth=2)
plt.plot(time, tau_F_D_arr, label='Torque from Friction and Damping', color='brown', linewidth=2)
plt.plot(time_data, tau_WE_data, label='Torque from WE (data)', linestyle='dashed', color='cyan', linewidth=1)
plt.plot(time_data, tau_F_D_data, label='Torque from Friction and Damping (data)', linestyle='dashed', color='black', linewidth=1)
plt.xlabel('Time [s]', fontname="Times New Roman", fontsize=12)
plt.ylabel('Torque [Nm]', fontname="Times New Roman", fontsize=12)
plt.title('Torque from Various Sources vs Time', fontname="Times New Roman", fontsize=14)
plt.legend(fontsize=10, loc='upper left')
plt.grid(color='gray', linestyle='--', linewidth=0.5)
plt.tick_params(axis='both', which='major', labelsize=10)

# Plot Position of Cylinder 1 and 2, Theta, and Position of WE
plt.figure(figsize=(12, 6))
plt.plot(time, x_WE_arr, label='Position of WE (simulation)', color='blue', linewidth=2)
plt.plot(time, x_cyl1_arr, label='Position of Cylinder 1 (simulation)', color='green', linewidth=2)
plt.plot(time, x_cyl2_arr, label='Position of Cylinder 2 (simulation)', color='orange', linewidth=2)
plt.plot(time_data, x_WE_data, label='Position of WE (data)', linestyle='dashed', color='red', linewidth=1)
plt.plot(time_data, x_cyl1_data, label='Position of Cylinder 1 (data)', linestyle='dashed', color='purple', linewidth=1)
plt.plot(time_data, x_cyl2_data, label='Position of Cylinder 2 (data)', linestyle='dashed', color='brown', linewidth=1)
plt.xlabel('Time [s]', fontname="Times New Roman", fontsize=12)
plt.ylabel('Position [m]', fontname="Times New Roman", fontsize=12)
plt.title('Position of WE and Cylinders 1 and 2 vs Time', fontname="Times New Roman", fontsize=14)
plt.legend(fontsize=10, loc='upper left')
plt.grid(color='gray', linestyle='--', linewidth=0.5)
plt.tick_params(axis='both', which='major', labelsize=10)

# Plot angle Theta
plt.figure(figsize=(12, 6))
plt.plot(time, np.rad2deg(theta_arr), label='Angle Theta (simulation)', color='blue', linewidth=2)
plt.plot(time_data, np.rad2deg(theta_data), label='Angle Theta (data)', linestyle='dashed', color='red', linewidth=1)
plt.xlabel('Time [s]', fontname="Times New Roman", fontsize=12)
plt.ylabel('Angle [deg]', fontname="Times New Roman", fontsize=12)
plt.title('Angle Theta vs Time', fontname="Times New Roman", fontsize=14)
plt.legend(fontsize=10, loc='upper left')
plt.grid(color='gray', linestyle='--', linewidth=0.5)
plt.tick_params(axis='both', which='major', labelsize=10)

# Plot Angular Velocity
plt.figure(figsize=(12, 6))
plt.plot(time, np.rad2deg(dot_theta_arr), label='Angular Velocity (simulation)', color='blue', linewidth=2)
plt.plot(time_data, np.rad2deg(dot_theta_data), label='Angular Velocity (data)', linestyle='dashed', color='red', linewidth=1)
plt.xlabel('Time [s]', fontname="Times New Roman", fontsize=12)
plt.ylabel('Angular Velocity [deg/s]', fontname="Times New Roman", fontsize=12)
plt.title('Angular Velocity vs Time', fontname="Times New Roman", fontsize=14)
plt.legend(fontsize=10, loc='upper left')
plt.grid(color='gray', linestyle='--', linewidth=0.5)
plt.tick_params(axis='both', which='major', labelsize=10)

# Plot Angular Acceleration
plt.figure(figsize=(12, 6))
plt.plot(time, np.rad2deg(ddot_theta_arr), label='Angular Acceleration (simulation)', color='blue', linewidth=2)
plt.xlabel('Time [s]', fontname="Times New Roman", fontsize=12)
plt.ylabel('Angular Acceleration [deg/s^2]', fontname="Times New Roman", fontsize=12)
plt.title('Angular Acceleration vs Time', fontname="Times New Roman", fontsize=14)
plt.legend(fontsize=10, loc='upper left')
plt.grid(color='gray', linestyle='--', linewidth=0.5)
plt.tick_params(axis='both', which='major', labelsize=10)

# Plot error
plt.figure(figsize=(12, 6))
error = yoke_p_data - yoke_p_arr
plt.plot(time_data, error, label='Error', color='blue', linewidth=2)
plt.xlabel('Time [s]', fontname="Times New Roman", fontsize=12)
plt.ylabel('Error [%]', fontname="Times New Roman", fontsize=12)
plt.title('Error vs Time', fontname="Times New Roman", fontsize=14)
plt.legend(fontsize=10, loc='upper left')
plt.grid(color='gray', linestyle='--', linewidth=0.5)
plt.tick_params(axis='both', which='major', labelsize=10)
plt.show()