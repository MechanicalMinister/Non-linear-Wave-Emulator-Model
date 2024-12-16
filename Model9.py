import numpy as np
import matplotlib.pyplot as plt
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
Ts = 1e-4  # Time step (s)
SimulationTime = 25 # Total simulation time (s)
numSteps = int(SimulationTime / Ts+1)  # Number of simulation steps
wave = (15*np.sin(0.2*np.pi*np.linspace(0, SimulationTime, numSteps))+50)

# Extract relevant columns from the Excel data
def DataMakerStep(tout_data, yout_data):
    start = 0
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
    # Find the first time x_v_data has a larger difference than 0.2
    for i in range(1, len(x_v_data)):
        avg_first_500 = np.mean(x_v_data[:500])
        if abs(x_v_data[i] - avg_first_500) > 0.3 or abs(x_v_data[i] - avg_first_500) < -0.3: 
            start = time_data[i]
            break
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
    start_row = int(200*start)
    print(start_row)
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

    return time_data, yoke_p_data, p_S_data, p_T_data, p_WE_1_data, dot_p_WE_1_data, p_WE_2_data, dot_p_WE_2_data, p_C1R1_data, p_C1B2_data, p_C2R1_data, p_C2B2_data, x_v_data, x_cyl2_data, x_cyl1_data, dot_theta_data, Q_1_data, Q_2_data, beta_1_data, beta_2_data, theta_data, x_WE_data, dot_x_WE_data, F_WE_data, F_cyl1_data, F_cyl2_data, tau_cyl1_data, tau_cyl2_data, tau_WE_data, tau_F_D_data

# Extract relevant columns from the Excel data
def DataMakerSinus(tout_data, yout_data):
    start = 0
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
    # Find the first time x_PTO cosses 190 mm from less to larger than 190 mm
    for i in range(1, len(time_data)):
        avg_first_100 = np.mean(x_v_data[:100])
        if abs(x_v_data[i] - avg_first_100) > 0.2 or abs(x_v_data[i] - avg_first_100) < -0.2: 
            if (yoke_p_data[i]/100)*x_PTO+Dead_PTO > 0.190 and (yoke_p_data[i-1]/100)*x_PTO+Dead_PTO < 0.190:
                start = time_data[i]
                break
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
    start_row = int(200*start)
    print(start_row)
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

    return time_data, yoke_p_data, p_S_data, p_T_data, p_WE_1_data, dot_p_WE_1_data, p_WE_2_data, dot_p_WE_2_data, p_C1R1_data, p_C1B2_data, p_C2R1_data, p_C2B2_data, x_v_data, x_cyl2_data, x_cyl1_data, dot_theta_data, Q_1_data, Q_2_data, beta_1_data, beta_2_data, theta_data, x_WE_data, dot_x_WE_data, F_WE_data, F_cyl1_data, F_cyl2_data, tau_cyl1_data, tau_cyl2_data, tau_WE_data, tau_F_D_data

# Function to calculate phase difference between two waves
def calculate_phase_difference(wave1, wave2, time):
    # Find the indices of the peaks of the waves
    peaks_wave1 = np.where((wave1[1:-1] > wave1[:-2]) & (wave1[1:-1] > wave1[2:]))[0] + 1
    peaks_wave2 = np.where((wave2[1:-1] > wave2[:-2]) & (wave2[1:-1] > wave2[2:]))[0] + 1
    
    # Calculate the time difference between the first peaks
    time_diff = time[peaks_wave2[0]] - time[peaks_wave1[0]]
    
    # Calculate the period of the waves
    period = time[peaks_wave1[1]] - time[peaks_wave1[0]]
    
    # Calculate the phase difference in degrees
    phase_difference = (time_diff / period) * 360
    
    return phase_difference

# Function to calculate amplitude difference between two waves
def calculate_amplitude_difference(wave1, wave2):
    # Calculate the amplitude of each wave
    amplitude_wave1 = (np.max(wave1) - np.min(wave1)) / 2
    amplitude_wave2 = (np.max(wave2) - np.min(wave2)) / 2
    
    # Calculate the amplitude difference
    amplitude_difference = amplitude_wave2 - amplitude_wave1
    
    return amplitude_difference

# Set font globally to Times New Roman
plt.rcParams['font.family'] = 'Times New Roman'

time = np.linspace(0, SimulationTime, int(SimulationTime / Ts + 1))

plt.figure(figsize=(12, 6))
# Plot Sinus input:
plt.subplot(2, 1, 1)

# Read data from Excel sheet
tout_data = pd.read_excel("C:/Users/Rober/OneDrive - Aalborg Universitet/MP.dok/Studie/Master/1. Semester/Project/ControllerTest/P_Control_Sinus.xlsx", sheet_name='tout', header=0)
yout_data = pd.read_excel("C:/Users/Rober/OneDrive - Aalborg Universitet/MP.dok/Studie/Master/1. Semester/Project/ControllerTest/P_Control_Sinus.xlsx", sheet_name='yout', header=0)
P_Contol_Sinus = DataMakerSinus(tout_data, yout_data)
plt.plot(P_Contol_Sinus[0], Dead_PTO + np.sqrt(a_PTO**2 + c_PTO**2 - 2*a_PTO*c_PTO*np.cos(theta_PTO_level - P_Contol_Sinus[20])) - np.sqrt(a_PTO**2 + c_PTO**2 - 2*a_PTO*c_PTO*np.cos(theta_PTO_top)), label='P controller (k_p = 0.12)')

# Read data from Excel sheet
tout_data = pd.read_excel("C:/Users/Rober/OneDrive - Aalborg Universitet/MP.dok/Studie/Master/1. Semester/Project/ControllerTest/P_Control_Sinus2.xlsx", sheet_name='tout', header=0)
yout_data = pd.read_excel("C:/Users/Rober/OneDrive - Aalborg Universitet/MP.dok/Studie/Master/1. Semester/Project/ControllerTest/P_Control_Sinus2.xlsx", sheet_name='yout', header=0)
P_Contol_Sinus2 = DataMakerSinus(tout_data, yout_data)
plt.plot(P_Contol_Sinus2[0], Dead_PTO + np.sqrt(a_PTO**2 + c_PTO**2 - 2*a_PTO*c_PTO*np.cos(theta_PTO_level - P_Contol_Sinus2[20])) - np.sqrt(a_PTO**2 + c_PTO**2 - 2*a_PTO*c_PTO*np.cos(theta_PTO_top)), label='P controller (k_p = 0.5)')

# Read data from Excel sheet
tout_data = pd.read_excel("C:/Users/Rober/OneDrive - Aalborg Universitet/MP.dok/Studie/Master/1. Semester/Project/ControllerTest/PI_Control_Sinus.xlsx", sheet_name='tout', header=0)
yout_data = pd.read_excel("C:/Users/Rober/OneDrive - Aalborg Universitet/MP.dok/Studie/Master/1. Semester/Project/ControllerTest/PI_Control_Sinus.xlsx", sheet_name='yout', header=0)
PI_Contol_Sinus = DataMakerSinus(tout_data, yout_data)
plt.plot(PI_Contol_Sinus[0], Dead_PTO + np.sqrt(a_PTO**2 + c_PTO**2 - 2*a_PTO*c_PTO*np.cos(theta_PTO_level - PI_Contol_Sinus[20])) - np.sqrt(a_PTO**2 + c_PTO**2 - 2*a_PTO*c_PTO*np.cos(theta_PTO_top)), label='PI controller (k_p = 0.1  - k_i = 0.002)')

# Read data from Excel sheet
tout_data = pd.read_excel("C:/Users/Rober/OneDrive - Aalborg Universitet/MP.dok/Studie/Master/1. Semester/Project/ControllerTest/PI_Control_Sinus2.xlsx", sheet_name='tout', header=0)
yout_data = pd.read_excel("C:/Users/Rober/OneDrive - Aalborg Universitet/MP.dok/Studie/Master/1. Semester/Project/ControllerTest/PI_Control_Sinus2.xlsx", sheet_name='yout', header=0)
PI_Contol_Sinus2 = DataMakerSinus(tout_data, yout_data)
plt.plot(PI_Contol_Sinus2[0], Dead_PTO + np.sqrt(a_PTO**2 + c_PTO**2 - 2*a_PTO*c_PTO*np.cos(theta_PTO_level - PI_Contol_Sinus2[20])) - np.sqrt(a_PTO**2 + c_PTO**2 - 2*a_PTO*c_PTO*np.cos(theta_PTO_top)), label='PI controller (k_p = 0.1 - k_i = 0.05)')

# Read data from Excel sheet
tout_data = pd.read_excel("C:/Users/Rober/OneDrive - Aalborg Universitet/MP.dok/Studie/Master/1. Semester/Project/ControllerTest/PID_Control_Sinus.xlsx", sheet_name='tout', header=0)
yout_data = pd.read_excel("C:/Users/Rober/OneDrive - Aalborg Universitet/MP.dok/Studie/Master/1. Semester/Project/ControllerTest/PID_Control_Sinus.xlsx", sheet_name='yout', header=0)
PID_Contol_Sinus = DataMakerSinus(tout_data, yout_data)
plt.plot(PID_Contol_Sinus[0], Dead_PTO + np.sqrt(a_PTO**2 + c_PTO**2 - 2*a_PTO*c_PTO*np.cos(theta_PTO_level - PID_Contol_Sinus[20])) - np.sqrt(a_PTO**2 + c_PTO**2 - 2*a_PTO*c_PTO*np.cos(theta_PTO_top)), label='PID controller (k_p = 0.02 - k_i = 0.0001 - k_d = 0.001)')

# Read data from Excel sheet
tout_data = pd.read_excel("C:/Users/Rober/OneDrive - Aalborg Universitet/MP.dok/Studie/Master/1. Semester/Project/ControllerTest/PID_Control_Sinus2.xlsx", sheet_name='tout', header=0)
yout_data = pd.read_excel("C:/Users/Rober/OneDrive - Aalborg Universitet/MP.dok/Studie/Master/1. Semester/Project/ControllerTest/PID_Control_Sinus2.xlsx", sheet_name='yout', header=0)
PID_Contol_Sinus2 = DataMakerSinus(tout_data, yout_data)
plt.plot(PID_Contol_Sinus2[0], Dead_PTO + np.sqrt(a_PTO**2 + c_PTO**2 - 2*a_PTO*c_PTO*np.cos(theta_PTO_level - PID_Contol_Sinus2[20])) - np.sqrt(a_PTO**2 + c_PTO**2 - 2*a_PTO*c_PTO*np.cos(theta_PTO_top)), label='PID controller (k_p = 0.1 - k_i = 0.05 - k_d = 0.001)')

# Read data from Excel sheet
tout_data = pd.read_excel("C:/Users/Rober/OneDrive - Aalborg Universitet/MP.dok/Studie/Master/1. Semester/Project/ControllerTest/PID_Control_Sinus3.xlsx", sheet_name='tout', header=0)
yout_data = pd.read_excel("C:/Users/Rober/OneDrive - Aalborg Universitet/MP.dok/Studie/Master/1. Semester/Project/ControllerTest/PID_Control_Sinus3.xlsx", sheet_name='yout', header=0)
PID_Contol_Sinus3 = DataMakerSinus(tout_data, yout_data)
plt.plot(PID_Contol_Sinus3[0], Dead_PTO + np.sqrt(a_PTO**2 + c_PTO**2 - 2*a_PTO*c_PTO*np.cos(theta_PTO_level - PID_Contol_Sinus3[20])) - np.sqrt(a_PTO**2 + c_PTO**2 - 2*a_PTO*c_PTO*np.cos(theta_PTO_top)), label='PID controller (k_p = 0.1 - k_i = 0.05 - k_d = 0.01)')

# Plot wave input:
plt.plot(time, (wave/100)*x_PTO+Dead_PTO, label='X_PTO', linestyle='--', color='black', linewidth=2)
plt.xlabel('Time [s]')
plt.ylabel('Float position [m]')
plt.title('Sinusoidal Wave')
plt.legend()
plt.grid()

# Plot Step input:
plt.subplot(2, 1, 2)

# Read data from Excel sheet
tout_data = pd.read_excel("C:/Users/Rober/OneDrive - Aalborg Universitet/MP.dok/Studie/Master/1. Semester/Project/ControllerTest/P_Control_Step.xlsx", sheet_name='tout', header=0)
yout_data = pd.read_excel("C:/Users/Rober/OneDrive - Aalborg Universitet/MP.dok/Studie/Master/1. Semester/Project/ControllerTest/P_Control_Step.xlsx", sheet_name='yout', header=0)
P_Contol_Step = DataMakerStep(tout_data, yout_data)
plt.plot(P_Contol_Step[0], -(Dead_PTO + np.sqrt(a_PTO**2 + c_PTO**2 - 2*a_PTO*c_PTO*np.cos(theta_PTO_level - P_Contol_Step[20])) - np.sqrt(a_PTO**2 + c_PTO**2 - 2*a_PTO*c_PTO*np.cos(theta_PTO_top))), label='P controller (k_p = 0.12)')

# Read data from Excel sheet
tout_data = pd.read_excel("C:/Users/Rober/OneDrive - Aalborg Universitet/MP.dok/Studie/Master/1. Semester/Project/ControllerTest/P_Control_Step2.xlsx", sheet_name='tout', header=0)
yout_data = pd.read_excel("C:/Users/Rober/OneDrive - Aalborg Universitet/MP.dok/Studie/Master/1. Semester/Project/ControllerTest/P_Control_Step2.xlsx", sheet_name='yout', header=0)
P_Contol_Step2 = DataMakerStep(tout_data, yout_data)
plt.plot(P_Contol_Step2[0], -(Dead_PTO + np.sqrt(a_PTO**2 + c_PTO**2 - 2*a_PTO*c_PTO*np.cos(theta_PTO_level - P_Contol_Step2[20])) - np.sqrt(a_PTO**2 + c_PTO**2 - 2*a_PTO*c_PTO*np.cos(theta_PTO_top))), label='P controller (k_p = 0.5)')

# Read data from Excel sheet
tout_data = pd.read_excel("C:/Users/Rober/OneDrive - Aalborg Universitet/MP.dok/Studie/Master/1. Semester/Project/ControllerTest/PI_Control_Step.xlsx", sheet_name='tout', header=0)
yout_data = pd.read_excel("C:/Users/Rober/OneDrive - Aalborg Universitet/MP.dok/Studie/Master/1. Semester/Project/ControllerTest/PI_Control_Step.xlsx", sheet_name='yout', header=0)
PI_Contol_Step = DataMakerStep(tout_data, yout_data)
plt.plot(PI_Contol_Step[0], -(Dead_PTO + np.sqrt(a_PTO**2 + c_PTO**2 - 2*a_PTO*c_PTO*np.cos(theta_PTO_level - PI_Contol_Step[20])) - np.sqrt(a_PTO**2 + c_PTO**2 - 2*a_PTO*c_PTO*np.cos(theta_PTO_top))), label='PI controller (k_p = 0.1  - k_i = 0.002)')

# Read data from Excel sheet
tout_data = pd.read_excel("C:/Users/Rober/OneDrive - Aalborg Universitet/MP.dok/Studie/Master/1. Semester/Project/ControllerTest/PI_Control_Step2.xlsx", sheet_name='tout', header=0)
yout_data = pd.read_excel("C:/Users/Rober/OneDrive - Aalborg Universitet/MP.dok/Studie/Master/1. Semester/Project/ControllerTest/PI_Control_Step2.xlsx", sheet_name='yout', header=0)
PI_Contol_Step2 = DataMakerStep(tout_data, yout_data)
plt.plot(PI_Contol_Step2[0], -(Dead_PTO + np.sqrt(a_PTO**2 + c_PTO**2 - 2*a_PTO*c_PTO*np.cos(theta_PTO_level - PI_Contol_Step2[20])) - np.sqrt(a_PTO**2 + c_PTO**2 - 2*a_PTO*c_PTO*np.cos(theta_PTO_top))), label='PI controller (k_p = 0.1 - k_i = 0.05)')

# Read data from Excel sheet
tout_data = pd.read_excel("C:/Users/Rober/OneDrive - Aalborg Universitet/MP.dok/Studie/Master/1. Semester/Project/ControllerTest/PID_Control_Step.xlsx", sheet_name='tout', header=0)
yout_data = pd.read_excel("C:/Users/Rober/OneDrive - Aalborg Universitet/MP.dok/Studie/Master/1. Semester/Project/ControllerTest/PID_Control_Step.xlsx", sheet_name='yout', header=0)
PID_Contol_Step = DataMakerStep(tout_data, yout_data)
plt.plot(PID_Contol_Step[0], -(Dead_PTO + np.sqrt(a_PTO**2 + c_PTO**2 - 2*a_PTO*c_PTO*np.cos(theta_PTO_level - PID_Contol_Step[20])) - np.sqrt(a_PTO**2 + c_PTO**2 - 2*a_PTO*c_PTO*np.cos(theta_PTO_top))), label='PID controller (k_p = 0.02 - k_i = 0.0001 - k_d = 0.001)')

# Read data from Excel sheet
tout_data = pd.read_excel("C:/Users/Rober/OneDrive - Aalborg Universitet/MP.dok/Studie/Master/1. Semester/Project/ControllerTest/PID_Control_Step2.xlsx", sheet_name='tout', header=0)
yout_data = pd.read_excel("C:/Users/Rober/OneDrive - Aalborg Universitet/MP.dok/Studie/Master/1. Semester/Project/ControllerTest/PID_Control_Step2.xlsx", sheet_name='yout', header=0)
PID_Contol_Step2 = DataMakerStep(tout_data, yout_data)
plt.plot(PID_Contol_Step2[0], -(Dead_PTO + np.sqrt(a_PTO**2 + c_PTO**2 - 2*a_PTO*c_PTO*np.cos(theta_PTO_level - PID_Contol_Step2[20])) - np.sqrt(a_PTO**2 + c_PTO**2 - 2*a_PTO*c_PTO*np.cos(theta_PTO_top))), label='PID controller (k_p = 0.1 - k_i = 0.05 - k_d = 0.001)')

# Read data from Excel sheet
tout_data = pd.read_excel("C:/Users/Rober/OneDrive - Aalborg Universitet/MP.dok/Studie/Master/1. Semester/Project/ControllerTest/PID_Control_Step3.xlsx", sheet_name='tout', header=0)
yout_data = pd.read_excel("C:/Users/Rober/OneDrive - Aalborg Universitet/MP.dok/Studie/Master/1. Semester/Project/ControllerTest/PID_Control_Step3.xlsx", sheet_name='yout', header=0)
PID_Contol_Step3 = DataMakerStep(tout_data, yout_data)
plt.plot(PID_Contol_Step3[0], -(Dead_PTO + np.sqrt(a_PTO**2 + c_PTO**2 - 2*a_PTO*c_PTO*np.cos(theta_PTO_level - PID_Contol_Step3[20])) - np.sqrt(a_PTO**2 + c_PTO**2 - 2*a_PTO*c_PTO*np.cos(theta_PTO_top))), label='PID controller (k_p = 0.1 - k_i = 0.05 - k_d = 0.01)')

plt.xlabel('Time [s]')
plt.ylabel('Float position [m]')
plt.title('Step input from bottom to level position')
plt.legend()
plt.grid()
plt.show()

# Calculate phase differences for Sinus input
phase_diff_P_Sinus = calculate_phase_difference((wave/100)*x_PTO+Dead_PTO, Dead_PTO + np.sqrt(a_PTO**2 + c_PTO**2 - 2*a_PTO*c_PTO*np.cos(theta_PTO_level - P_Contol_Sinus[20])) - np.sqrt(a_PTO**2 + c_PTO**2 - 2*a_PTO*c_PTO*np.cos(theta_PTO_top)), time)
phase_diff_P_Sinus2 = calculate_phase_difference((wave/100)*x_PTO+Dead_PTO, Dead_PTO + np.sqrt(a_PTO**2 + c_PTO**2 - 2*a_PTO*c_PTO*np.cos(theta_PTO_level - P_Contol_Sinus2[20])) - np.sqrt(a_PTO**2 + c_PTO**2 - 2*a_PTO*c_PTO*np.cos(theta_PTO_top)), time)
phase_diff_PI_Sinus = calculate_phase_difference((wave/100)*x_PTO+Dead_PTO, Dead_PTO + np.sqrt(a_PTO**2 + c_PTO**2 - 2*a_PTO*c_PTO*np.cos(theta_PTO_level - PI_Contol_Sinus[20])) - np.sqrt(a_PTO**2 + c_PTO**2 - 2*a_PTO*c_PTO*np.cos(theta_PTO_top)), time)
phase_diff_PI_Sinus2 = calculate_phase_difference((wave/100)*x_PTO+Dead_PTO, Dead_PTO + np.sqrt(a_PTO**2 + c_PTO**2 - 2*a_PTO*c_PTO*np.cos(theta_PTO_level - PI_Contol_Sinus2[20])) - np.sqrt(a_PTO**2 + c_PTO**2 - 2*a_PTO*c_PTO*np.cos(theta_PTO_top)), time)
phase_diff_PID_Sinus = calculate_phase_difference((wave/100)*x_PTO+Dead_PTO, Dead_PTO + np.sqrt(a_PTO**2 + c_PTO**2 - 2*a_PTO*c_PTO*np.cos(theta_PTO_level - PID_Contol_Sinus[20])) - np.sqrt(a_PTO**2 + c_PTO**2 - 2*a_PTO*c_PTO*np.cos(theta_PTO_top)), time)
phase_diff_PID_Sinus2 = calculate_phase_difference((wave/100)*x_PTO+Dead_PTO, Dead_PTO + np.sqrt(a_PTO**2 + c_PTO**2 - 2*a_PTO*c_PTO*np.cos(theta_PTO_level - PID_Contol_Sinus2[20])) - np.sqrt(a_PTO**2 + c_PTO**2 - 2*a_PTO*c_PTO*np.cos(theta_PTO_top)), time)
phase_diff_PID_Sinus3 = calculate_phase_difference((wave/100)*x_PTO+Dead_PTO, Dead_PTO + np.sqrt(a_PTO**2 + c_PTO**2 - 2*a_PTO*c_PTO*np.cos(theta_PTO_level - PID_Contol_Sinus3[20])) - np.sqrt(a_PTO**2 + c_PTO**2 - 2*a_PTO*c_PTO*np.cos(theta_PTO_top)), time)

print('Phase difference for P controller (k_p = 0.5):', (phase_diff_P_Sinus2 / 180) * 100, '%')

# Calculate amplitude differences for Sinus input
amplitude_diff_P_Sinus = calculate_amplitude_difference((wave/100)*x_PTO+Dead_PTO, Dead_PTO + np.sqrt(a_PTO**2 + c_PTO**2 - 2*a_PTO*c_PTO*np.cos(theta_PTO_level - P_Contol_Sinus[20])) - np.sqrt(a_PTO**2 + c_PTO**2 - 2*a_PTO*c_PTO*np.cos(theta_PTO_top)))
amplitude_diff_P_Sinus2 = calculate_amplitude_difference((wave/100)*x_PTO+Dead_PTO, Dead_PTO + np.sqrt(a_PTO**2 + c_PTO**2 - 2*a_PTO*c_PTO*np.cos(theta_PTO_level - P_Contol_Sinus2[20])) - np.sqrt(a_PTO**2 + c_PTO**2 - 2*a_PTO*c_PTO*np.cos(theta_PTO_top)))
amplitude_diff_PI_Sinus = calculate_amplitude_difference((wave/100)*x_PTO+Dead_PTO, Dead_PTO + np.sqrt(a_PTO**2 + c_PTO**2 - 2*a_PTO*c_PTO*np.cos(theta_PTO_level - PI_Contol_Sinus[20])) - np.sqrt(a_PTO**2 + c_PTO**2 - 2*a_PTO*c_PTO*np.cos(theta_PTO_top)))
amplitude_diff_PI_Sinus2 = calculate_amplitude_difference((wave/100)*x_PTO+Dead_PTO, Dead_PTO + np.sqrt(a_PTO**2 + c_PTO**2 - 2*a_PTO*c_PTO*np.cos(theta_PTO_level - PI_Contol_Sinus2[20])) - np.sqrt(a_PTO**2 + c_PTO**2 - 2*a_PTO*c_PTO*np.cos(theta_PTO_top)))
amplitude_diff_PID_Sinus = calculate_amplitude_difference((wave/100)*x_PTO+Dead_PTO, Dead_PTO + np.sqrt(a_PTO**2 + c_PTO**2 - 2*a_PTO*c_PTO*np.cos(theta_PTO_level - PID_Contol_Sinus[20])) - np.sqrt(a_PTO**2 + c_PTO**2 - 2*a_PTO*c_PTO*np.cos(theta_PTO_top)))
amplitude_diff_PID_Sinus2 = calculate_amplitude_difference((wave/100)*x_PTO+Dead_PTO, Dead_PTO + np.sqrt(a_PTO**2 + c_PTO**2 - 2*a_PTO*c_PTO*np.cos(theta_PTO_level - PID_Contol_Sinus2[20])) - np.sqrt(a_PTO**2 + c_PTO**2 - 2*a_PTO*c_PTO*np.cos(theta_PTO_top)))
amplitude_diff_PID_Sinus3 = calculate_amplitude_difference((wave/100)*x_PTO+Dead_PTO, Dead_PTO + np.sqrt(a_PTO**2 + c_PTO**2 - 2*a_PTO*c_PTO*np.cos(theta_PTO_level - PID_Contol_Sinus3[20])) - np.sqrt(a_PTO**2 + c_PTO**2 - 2*a_PTO*c_PTO*np.cos(theta_PTO_top)))

print('Amplitude difference for P controller (k_p = 0.5):', amplitude_diff_P_Sinus2/0.28*100, '%')

# Plot amplitude and phase differences
controllers = ['P_1', 'P_2', 'PI_1', 'PI_2', 'PID_1', 'PID_2', 'PID_3']
amplitude_diffs = [amplitude_diff_P_Sinus, amplitude_diff_P_Sinus2, amplitude_diff_PI_Sinus, amplitude_diff_PI_Sinus2, amplitude_diff_PID_Sinus, amplitude_diff_PID_Sinus2, amplitude_diff_PID_Sinus3]
phase_diffs = [phase_diff_P_Sinus, phase_diff_P_Sinus2, phase_diff_PI_Sinus, phase_diff_PI_Sinus2, phase_diff_PID_Sinus, phase_diff_PID_Sinus2, phase_diff_PID_Sinus3]

fig, ax1 = plt.subplots(figsize=(12, 6))

color = 'tab:blue'
ax1.set_xlabel('Controller')
ax1.set_ylabel('Amplitude Difference [m]', color=color)
ax1.bar(controllers, amplitude_diffs, color=color, alpha=0.6, label='Amplitude Difference')
ax1.tick_params(axis='y', labelcolor=color)
ax1.legend(loc='upper left')

ax2 = ax1.twinx()
color = 'tab:red'
ax2.set_ylabel('Phase Difference [degrees]', color=color)
ax2.plot(controllers, phase_diffs, color=color, marker='o', linestyle='-', linewidth=2, label='Phase Difference')
ax2.tick_params(axis='y', labelcolor=color)
ax2.legend(loc='upper right')
plt.title('Amplitude and Phase Differences for Different Controllers')
plt.grid()
plt.show()