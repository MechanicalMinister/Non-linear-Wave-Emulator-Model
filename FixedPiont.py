import numpy as np
from scipy.optimize import fsolve

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
Ts = 1e-5  # Time step (s)
SimulationTime = 1 # Total simulation time (s)
numSteps = int(SimulationTime / Ts+1)  # Number of simulation steps
wave = 0

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

# Initialize variables
yoke_p_arr[0] = 50
theta_arr[0] = np.deg2rad(0)
dot_theta_arr[0] = np.deg2rad(0)
x_v_arr[0] = 0
p_S_arr[0] = 180e5
p_T_arr[0] = 1.5e5
p_WE_1_arr[0] = 75e5
p_WE_2_arr[0] = 133.222133130104e5
Q_1_arr[0] = Q_n / np.sqrt(p_n) * x_v_arr[0] * np.sqrt(np.abs(p_S_arr[0] - p_WE_1_arr[0])) * ((p_S_arr[0] - p_WE_1_arr[0]) / np.abs(p_S_arr[0] - p_WE_1_arr[0]))
Q_2_arr[0] = Q_n / np.sqrt(p_n) * x_v_arr[0] * np.sqrt(np.abs(p_WE_2_arr[0] - p_T_arr[0])) * ((p_WE_2_arr[0] - p_T_arr[0]) / np.abs(p_WE_2_arr[0] - p_T_arr[0]))
beta_1_arr[0] = ((1-alpha)*np.exp((p_T_val-p_WE_1_arr[0])/beta_0)+alpha*(p_T_val/p_WE_1_arr[0])**(1/n_polyIndx))/(((1-alpha)/beta_0)*np.exp((p_T_val-p_WE_1_arr[0])/beta_0)+(alpha/(n_polyIndx*p_T_val))*(p_T_val/p_WE_1_arr[0])**((n_polyIndx+1)/n_polyIndx))
beta_2_arr[0] = ((1-alpha)*np.exp((p_T_val-p_WE_2_arr[0])/beta_0)+alpha*(p_T_val/p_WE_2_arr[0])**(1/n_polyIndx))/(((1-alpha)/beta_0)*np.exp((p_T_val-p_WE_2_arr[0])/beta_0)+(alpha/(n_polyIndx*p_T_val))*(p_T_val/p_WE_2_arr[0])**((n_polyIndx+1)/n_polyIndx))
dot_p_WE_1_arr[0] = (beta_1_arr[0]) / (V_WE_1_0 + A_WE_1 * (x_WE_arr[0])) * (Q_1_arr[0] - C_le * (p_WE_1_arr[0] - p_WE_2_arr[0]) - A_WE_1 * dot_x_WE_arr[0])
dot_p_WE_2_arr[0] = (beta_2_arr[0]) / (V_WE_2_0 - A_WE_2 * (L_WE-x_WE_arr[0])) * (C_le * (p_WE_1_arr[0] - p_WE_2_arr[0]) - Q_2_arr[0] + A_WE_2 * dot_x_WE_arr[0])
# Limit the pressure to a maximum of 400 bar and a minimum threshold
max_pressure = 250e5  #
min_pressure = 1e5  #
F_WE_arr[0] = A_WE_1 * p_WE_1_arr[0] - A_WE_2 * p_WE_2_arr[0]
F_cyl1_arr[0] = -400
F_cyl2_arr[0] = -400
tau_g_arr[0] = M * g * np.sqrt(x_com**2 + y_com**2) * np.cos(np.atan(y_com/x_com)+theta_arr[0])
tau_cyl1_arr[0] = F_cyl1_arr[0] * c_PTO
tau_cyl2_arr[0] = F_cyl2_arr[0] * c_PTO
tau_WE_arr[0] = F_WE_arr[0] * np.sin(np.acos(((L_WE_0 + x_WE_arr[0])**2 + c_WE**2 - a_WE**2)/(2*(L_WE_0 + x_WE_arr[0])*c_WE))) * c_WE
tau_F_D_arr[0] = c * dot_theta_arr[0] + k * dot_theta_arr[0]
ddot_theta_arr[0] = (tau_cyl1_arr[0] + tau_cyl2_arr[0] - tau_WE_arr[0] - tau_g_arr[0] - tau_F_D_arr[0])/J
x_cyl1_arr[0] = x_PTO_0
x_cyl2_arr[0] = x_PTO_0
x_WE_arr[0] = np.sqrt(a_WE**2 + c_WE**2 - 2*a_WE*c_WE*np.cos(theta_WE_level-theta_arr[0])) - np.sqrt(a_WE**2 + c_WE**2 - 2*a_WE*c_WE*np.cos(theta_WE_top))
dot_x_WE_arr[0] = 0

def equations(p_WE_2):
    F_WE = A_WE_1 * 75e5 - A_WE_2 * p_WE_2
    tau_WE = F_WE * np.sin(np.acos(((L_WE_0 + x_WE_arr[0])**2 + c_WE**2 - a_WE**2)/(2*(L_WE_0 + x_WE_arr[0])*c_WE))) * c_WE
    return tau_cyl1_arr[0] + tau_cyl2_arr[0] - tau_WE - tau_g_arr[0] - tau_F_D_arr[0]

p_WE_2_solution = fsolve(equations, p_WE_2_arr[0])
print("Solved p_WE_2:", p_WE_2_solution[0])

print("")
n=0
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
