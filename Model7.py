import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
from MMM import p_WE_1_sol, p_WE_2_sol, x_v_sol
from scipy import signal

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
B_p = c  # Damping coefficient (Ns/m)
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
SimulationTime = 1 # Total simulation time (s)
numSteps = int(SimulationTime / Ts+1)  # Number of simulation steps
wave = (0.75*np.sin(2*np.pi*0.1*np.linspace(0, SimulationTime, numSteps)))*0.1
# Generate step input for x_v
#wave = np.zeros(numSteps)
#wave[:int(1 / Ts)] = 0.1

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

# Simulate the system with progress bar
for n in tqdm(range(1, numSteps), desc="Non-Linear Simulation Progress"):    
    x_v_arr[n] = wave[n]/7

x_WE0 = x_WE_0
p_S0 = 180e5
p_T0 = 1e5
p_a0 = p_WE_1_sol
p_b0 = p_WE_2_sol
x_v0 = x_v_sol
dot_x_WE0 = 0 
F_L = 4000

#The linearized function of the partial derivative of dot_p_1 with respect to x_WE at x_WE0 is:
dot_p_WE_1_ParDiff_x_WE__P = -A_WE_1*beta_0*(-A_WE_1*dot_x_WE0 - C_le*(p_a0 - p_b0) + Q_n*x_v0*((-p_T0 + p_a0)**2)**(1/4)/np.sqrt(p_n))/(A_WE_1*x_WE0 + V_WE_1_0)**2

#The linearized function of the partial derivative of dot_p_1 with respect to x_v at x_WE0 is:
dot_p_WE_1_ParDiff_x_v__P = Q_n*beta_0*((-p_T0 + p_a0)**2)**(1/4)/(np.sqrt(p_n)*(A_WE_1*x_WE0 + V_WE_1_0))

#The linearized function of the partial derivative of dot_p_1 with respect to p_S at x_WE0 is:
dot_p_WE_1_ParDiff_p_S__P = Q_n*beta_0*x_v0*(p_T0/2 - p_a0/2)*((-p_T0 + p_a0)**2)**(1/4)/(np.sqrt(p_n)*(-p_T0 + p_a0)**2*(A_WE_1*x_WE0 + V_WE_1_0))

#The linearized function of the partial derivative of dot_p_1 with respect to p_1 at x_WE0 is:
dot_p_WE_1_ParDiff_p_1__P = beta_0*(-C_le + Q_n*x_v0*(-p_T0/2 + p_a0/2)*((-p_T0 + p_a0)**2)**(1/4)/(np.sqrt(p_n)*(-p_T0 + p_a0)**2))/(A_WE_1*x_WE0 + V_WE_1_0)

#The linearized function of the partial derivative of dot_p_1 with respect to p_2 at x_WE0 is:
dot_p_WE_1_ParDiff_p_2__P = C_le*beta_0/(A_WE_1*x_WE0 + V_WE_1_0)

#The linearized function of the partial derivative of dot_p_1 with respect to dot_x_WE at x_WE0 is:
dot_p_WE_1_ParDiff_dot_x_WE__P = -A_WE_1*beta_0/(A_WE_1*x_WE0 + V_WE_1_0)

#The linearized function of the partial derivative of dot_p_2 with respect to x_WE at x_WE0 is:
dot_p_WE_2_ParDiff_x_WE__P = -A_WE_2*beta_0*(A_WE_2*dot_x_WE0 + C_le*(p_a0 - p_b0) - Q_n*x_v0*((p_S0 - p_b0)**2)**(1/4)/np.sqrt(p_n))/(-A_WE_2*(L_WE - x_WE0) + V_WE_2_0)**2

#The linearized function of the partial derivative of dot_p_2 with respect to x_v at x_WE0 is:
dot_p_WE_2_ParDiff_x_v__P = -Q_n*beta_0*((p_S0 - p_b0)**2)**(1/4)/(np.sqrt(p_n)*(-A_WE_2*(L_WE - x_WE0) + V_WE_2_0))

#The linearized function of the partial derivative of dot_p_2 with respect to p_1 at x_WE0 is:
dot_p_WE_2_ParDiff_p_1__P = C_le*beta_0/(-A_WE_2*(L_WE - x_WE0) + V_WE_2_0)

#The linearized function of the partial derivative of dot_p_2 with respect to p_2 at x_WE0 is:
dot_p_WE_2_ParDiff_p_2__P = beta_0*(-C_le - Q_n*x_v0*(-p_S0/2 + p_b0/2)*((p_S0 - p_b0)**2)**(1/4)/(np.sqrt(p_n)*(p_S0 - p_b0)**2))/(-A_WE_2*(L_WE - x_WE0) + V_WE_2_0)

#The linearized function of the partial derivative of dot_p_2 with respect to p_T at x_WE0 is:
dot_p_WE_2_ParDiff_p_T__P = -Q_n*beta_0*x_v0*(p_S0/2 - p_b0/2)*((p_S0 - p_b0)**2)**(1/4)/(np.sqrt(p_n)*(p_S0 - p_b0)**2*(-A_WE_2*(L_WE - x_WE0) + V_WE_2_0))

#The linearized function of the partial derivative of dot_p_2 with respect to dot_x_WE at x_WE0 is:
dot_p_WE_2_ParDiff_dot_x_WE__P = A_WE_2*beta_0/(-A_WE_2*(L_WE - x_WE0) + V_WE_2_0)

#The state space representation matrix A is:
A_P = np.array([
    [0, 1, 0, 0],
    [0, -B_p / M, A_WE_1 / M, -A_WE_2 / M],
    [dot_p_WE_1_ParDiff_x_WE__P, dot_p_WE_1_ParDiff_dot_x_WE__P, dot_p_WE_1_ParDiff_p_1__P, dot_p_WE_1_ParDiff_p_2__P],
    [dot_p_WE_2_ParDiff_x_WE__P, dot_p_WE_2_ParDiff_dot_x_WE__P, dot_p_WE_2_ParDiff_p_1__P, dot_p_WE_2_ParDiff_p_2__P]
])

#The state space representation matrix B is:
B_P = np.array([
    [0, 0, 0, 0],
    [0, 0, 0, 1 / M],
    [dot_p_WE_1_ParDiff_p_S__P, 0, dot_p_WE_1_ParDiff_x_v__P, 0],
    [0, dot_p_WE_2_ParDiff_p_T__P, dot_p_WE_2_ParDiff_x_v__P, 0]
])

#The linearized function of The partial derivative of dot_p_1 with respect to x_WE at x_WE0 is:
dot_p_WE_1_ParDiff_x_WE__N = -A_WE_1*beta_0*(-A_WE_1*dot_x_WE0 - C_le*(p_a0 - p_b0) + Q_n*x_v0*((p_S0 - p_a0)**2)**(1/4)/np.sqrt(p_n))/(A_WE_1*x_WE0 + V_WE_1_0)**2

#The linearized function of The partial derivative of dot_p_1 with respect to x_v at x_WE0 is:
dot_p_WE_1_ParDiff_x_v__N = Q_n*beta_0*((p_S0 - p_a0)**2)**(1/4)/(np.sqrt(p_n)*(A_WE_1*x_WE0 + V_WE_1_0))

#The linearized function of The partial derivative of dot_p_1 with respect to p_T at x_WE0 is:
dot_p_WE_1_ParDiff_p_T__N = Q_n*beta_0*x_v0*(p_S0/2 - p_a0/2)*((p_S0 - p_a0)**2)**(1/4)/(np.sqrt(p_n)*(p_S0 - p_a0)**2*(A_WE_1*x_WE0 + V_WE_1_0))

#The linearized function of The partial derivative of dot_p_1 with respect to p_1 at x_WE0 is:
dot_p_WE_1_ParDiff_p_1__N = beta_0*(-C_le + Q_n*x_v0*(-p_S0/2 + p_a0/2)*((p_S0 - p_a0)**2)**(1/4)/(np.sqrt(p_n)*(p_S0 - p_a0)**2))/(A_WE_1*x_WE0 + V_WE_1_0)

#The linearized function of The partial derivative of dot_p_1 with respect to p_2 at x_WE0 is:
dot_p_WE_1_ParDiff_p_2__N = C_le*beta_0/(A_WE_1*x_WE0 + V_WE_1_0)

#The linearized function of The partial derivative of dot_p_1 with respect to dot_x_WE at x_WE0 is:
dot_p_WE_1_ParDiff_dot_x_WE__N = -A_WE_1*beta_0/(A_WE_1*x_WE0 + V_WE_1_0)

#The linearized function of The partial derivative of dot_p_2 with respect to x_WE at x_WE0 is:
dot_p_WE_2_ParDiff_x_WE__N = -A_WE_2*beta_0*(A_WE_2*dot_x_WE0 + C_le*(p_a0 - p_b0) - Q_n*x_v0*((-p_T0 + p_b0)**2)**(1/4)/np.sqrt(p_n))/(-A_WE_2*(L_WE - x_WE0) + V_WE_2_0)**2

#The linearized function of The partial derivative of dot_p_2 with respect to x_v at x_WE0 is:
dot_p_WE_2_ParDiff_x_v__N = -Q_n*beta_0*((-p_T0 + p_b0)**2)**(1/4)/(np.sqrt(p_n)*(-A_WE_2*(L_WE - x_WE0) + V_WE_2_0))

#The linearized function of The partial derivative of dot_p_2 with respect to p_1 at x_WE0 is:
dot_p_WE_2_ParDiff_p_1__N = C_le*beta_0/(-A_WE_2*(L_WE - x_WE0) + V_WE_2_0)

#The linearized function of the partial derivative of dot_p_2 with respect to p_2 at x_WE0 is:
dot_p_WE_2_ParDiff_p_2__N = beta_0*(-C_le - Q_n*x_v0*(-p_T0/2 + p_b0/2)*((-p_T0 + p_b0)**2)**(1/4)/(np.sqrt(p_n)*(-p_T0 + p_b0)**2))/(-A_WE_2*(L_WE - x_WE0) + V_WE_2_0)

#The linearized function of the partial derivative of dot_p_2 with respect to p_S at x_WE0 is:
dot_p_WE_2_ParDiff_p_S__N = -Q_n*beta_0*x_v0*(p_T0/2 - p_b0/2)*((-p_T0 + p_b0)**2)**(1/4)/(np.sqrt(p_n)*(-p_T0 + p_b0)**2*(-A_WE_2*(L_WE - x_WE0) + V_WE_2_0))

#The linearized function of the partial derivative of dot_p_2 with respect to dot_x_WE at x_WE0 is:
dot_p_WE_2_ParDiff_dot_x_WE__N = A_WE_2*beta_0/(-A_WE_2*(L_WE - x_WE0) + V_WE_2_0)

#The state space representation matrix A is:
A_N = np.array([
    [0, 1, 0, 0],
    [0, -B_p / M, A_WE_1 / M, -A_WE_2 / M],
    [dot_p_WE_1_ParDiff_x_WE__N, dot_p_WE_1_ParDiff_dot_x_WE__N, dot_p_WE_1_ParDiff_p_1__N, dot_p_WE_1_ParDiff_p_2__N],
    [dot_p_WE_2_ParDiff_x_WE__N, dot_p_WE_2_ParDiff_dot_x_WE__N, dot_p_WE_2_ParDiff_p_1__N, dot_p_WE_2_ParDiff_p_2__N]
])

#The state space representation matrix B is:
B_N = np.array([
    [0, 0, 0, 0],
    [0, 0, 0, 1 / M],
    [0, dot_p_WE_1_ParDiff_p_T__N, dot_p_WE_1_ParDiff_x_v__N, 0],
    [dot_p_WE_2_ParDiff_p_S__N, 0, dot_p_WE_2_ParDiff_x_v__N, 0]
])

#The state space representation matrix C is:
C = np.array([
    [1, 0, 0, 0],
    [0, 0, 0, 0],
    [0, 0, 0, 0],
    [0, 0, 0, 0]
])

#The state space representation matrix D is:
D = np.array([
    [0, 0, 0, 0],
    [0, 0, 0, 0],
    [0, 0, 0, 0],
    [0, 0, 0, 0]
])

#________________________________________________________________________________#
# Define the state-space system
SS = signal.StateSpace(A_P, B_P, C, D)

# Compute the transfer function from the state-space representation
TF = signal.ss2tf(A_P, B_P, C, D)

# Print the transfer function
print(f"Transfer Function:")
print(f"Numerator: {TF[0][0]}")
print(f"Denominator: {TF[1]}")
print("")

# Plot the Pole zero plot for the transfer function
poles, zeros, gain = signal.tf2zpk(TF[0][0], TF[1])
plt.figure()
plt.scatter(poles.real, poles.imag, marker='x', color='red', label='Poles')
plt.scatter(zeros.real, zeros.imag, marker='o', color='blue', label='Zeros')
plt.title(f'Pole-Zero Plot for Transfer Function (Gain: {gain:.2f})')
plt.xlabel('Real')
plt.ylabel('Imaginary')
plt.legend()
plt.grid()

# Plot the Bode plot for the transfer function
w, mag, phase = signal.bode((TF[0][0], TF[1]))
plt.figure()
plt.subplot(2, 1, 1)
plt.semilogx(w, mag)  # Bode magnitude plot
plt.title(f'Bode Plot for Transfer Function')
plt.xlabel('Frequency [rad/s]')
plt.ylabel('Magnitude [dB]')
plt.grid()
plt.subplot(2, 1, 2)
plt.semilogx(w, phase)  # Bode phase plot
plt.xlabel('Frequency [rad/s]')
plt.ylabel('Phase [degrees]')
plt.grid()

plt.show()