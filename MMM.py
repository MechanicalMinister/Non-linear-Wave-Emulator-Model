import sympy as sp
import numpy as np
from scipy.optimize import fsolve

# Define the variables
x_WE, dot_x_WE, beta, A_WE_1, A_WE_2, V_WE_1_0, V_WE_2_0, L_WE, x_WE0, p_1, p_2, x_v, Q_n, p_n, B_p, M, p_S, p_T, F_L, alpha, beta_0, n_polyIndx, C_le, c_WE = sp.symbols(
    'x_WE dot_x_WE beta A_WE_1 A_WE_2 V_WE_1_0 V_WE_2_0 L_WE x_WE0 p_1 p_2 x_v Q_n p_n B_p M p_S p_T F_L alpha beta_0 n_polyIndx C_le c_WE')
 
# Define the linearization point
x_WE0 = sp.Symbol('x_WE0')
x_v0 = sp.Symbol('x_v0')
p_S0 = sp.Symbol('p_S0')
p_T0 = sp.Symbol('p_T0')
p_a0 = sp.Symbol('p_a0')
p_b0 = sp.Symbol('p_b0')  
dot_x_WE0 = sp.Symbol('dot_x_WE0')

# Define the function

Q_A_Positive = Q_n / sp.sqrt(p_n) * x_v * sp.sqrt(sp.sqrt((p_S - p_1)**2))
Q_A_Negative = Q_n / sp.sqrt(p_n) * x_v * sp.sqrt(sp.sqrt((p_1 - p_T)**2)) 
Q_B_Positive = Q_n / sp.sqrt(p_n) * x_v * sp.sqrt(sp.sqrt((p_2 - p_T)**2)) 
Q_B_Negative = Q_n / sp.sqrt(p_n) * x_v * sp.sqrt(sp.sqrt((p_S - p_2)**2))

I = 0
while I < 2:
    if I == 0:
        Q_A = Q_A_Positive
        Q_B = Q_B_Positive
        p1 = p_1; p2 = p_2; p3 = p_S; p4 = p_T

    else:
        Q_A = Q_A_Negative
        Q_B = Q_B_Negative
        p1 = p_T; p2 = p_S; p3 = p_1; p4 = p_2

    dot_p_1 = beta_0 / (V_WE_1_0 + A_WE_1 * x_WE) * ( Q_A - C_le * (p_1 - p_2) - A_WE_1 * dot_x_WE)
    dot_p_2 = beta_0 / (V_WE_2_0 - A_WE_2 * (L_WE - x_WE)) * (- Q_B + C_le * (p_1 - p_2) + A_WE_2 * dot_x_WE)

    # Compute the partial derivative with respect to:
    d_dot_p_1_d_x_WE = sp.diff(dot_p_1, x_WE)
    d_dot_p_1_d_x_v = sp.diff(dot_p_1, x_v)
    d_dot_p_1_d_p_1 = sp.diff(dot_p_1, p_1)
    d_dot_p_1_d_p_2 = sp.diff(dot_p_1, p_2)
    if I == 0:
        d_dot_p_1_d_p_S = sp.diff(dot_p_1, p_S)
    else:
        d_dot_p_1_d_p_T = sp.diff(dot_p_1, p_T)
    d_dot_p_1_d_dot_x_WE = sp.diff(dot_p_1, dot_x_WE) 

    d_dot_p_2_d_x_WE = sp.diff(dot_p_2, x_WE) 
    d_dot_p_2_d_x_v = sp.diff(dot_p_2, x_v)
    d_dot_p_2_d_p_1 = sp.diff(dot_p_2, p_1)
    d_dot_p_2_d_p_2 = sp.diff(dot_p_2, p_2)
    if I == 0:
        d_dot_p_2_d_p_T = sp.diff(dot_p_2, p_T)
    else:
        d_dot_p_2_d_p_S = sp.diff(dot_p_2, p_S)
    d_dot_p_2_d_dot_x_WE = sp.diff(dot_p_2, dot_x_WE) 

    # Evaluate the partial derivative at the linearization point
    dot_p_1_linearized_part1 = d_dot_p_1_d_x_WE.subs({x_WE: x_WE0, x_v: x_v0, p_S: p_S0, p_T: p_T0, p_1: p_a0, p_2: p_b0, dot_x_WE: dot_x_WE0})
    dot_p_1_linearized_part2 = d_dot_p_1_d_x_v.subs({x_WE: x_WE0, x_v: x_v0, p_S: p_S0, p_T: p_T0, p_1: p_a0, p_2: p_b0, dot_x_WE: dot_x_WE0})
    if I == 0:
        dot_p_1_linearized_part3 = d_dot_p_1_d_p_S.subs({x_WE: x_WE0, x_v: x_v0, p_S: p_S0, p_T: p_T0, p_1: p_a0, p_2: p_b0, dot_x_WE: dot_x_WE0})
    else:
        dot_p_1_linearized_part3 = d_dot_p_1_d_p_T.subs({x_WE: x_WE0, x_v: x_v0, p_S: p_S0, p_T: p_T0, p_1: p_a0, p_2: p_b0, dot_x_WE: dot_x_WE0})
    dot_p_1_linearized_part4 = d_dot_p_1_d_p_1.subs({x_WE: x_WE0, x_v: x_v0, p_S: p_S0, p_T: p_T0, p_1: p_a0, p_2: p_b0, dot_x_WE: dot_x_WE0})
    dot_p_1_linearized_part5 = d_dot_p_1_d_p_2.subs({x_WE: x_WE0, x_v: x_v0, p_S: p_S0, p_T: p_T0, p_1: p_a0, p_2: p_b0, dot_x_WE: dot_x_WE0})
    dot_p_1_linearized_part6 = d_dot_p_1_d_dot_x_WE.subs({x_WE: x_WE0, x_v: x_v0, p_S: p_S0, p_T: p_T0, p_1: p_a0, p_2: p_b0, dot_x_WE: dot_x_WE0})

    dot_p_2_linearized_part1 = d_dot_p_2_d_x_WE.subs({x_WE: x_WE0, x_v: x_v0, p_S: p_S0, p_T: p_T0, p_1: p_a0, p_2: p_b0, dot_x_WE: dot_x_WE0})
    dot_p_2_linearized_part2 = d_dot_p_2_d_x_v.subs({x_WE: x_WE0, x_v: x_v0, p_S: p_S0, p_T: p_T0, p_1: p_a0, p_2: p_b0, dot_x_WE: dot_x_WE0})
    dot_p_2_linearized_part3 = d_dot_p_2_d_p_1.subs({x_WE: x_WE0, x_v: x_v0, p_S: p_S0, p_T: p_T0, p_1: p_a0, p_2: p_b0, dot_x_WE: dot_x_WE0})
    dot_p_2_linearized_part4 = d_dot_p_2_d_p_2.subs({x_WE: x_WE0, x_v: x_v0, p_S: p_S0, p_T: p_T0, p_1: p_a0, p_2: p_b0, dot_x_WE: dot_x_WE0})
    if I == 0:
        dot_p_2_linearized_part5 = d_dot_p_2_d_p_T.subs({x_WE: x_WE0, x_v: x_v0, p_S: p_S0, p_T: p_T0, p_1: p_a0, p_2: p_b0, dot_x_WE: dot_x_WE0})
    else:
        dot_p_2_linearized_part5 = d_dot_p_2_d_p_S.subs({x_WE: x_WE0, x_v: x_v0, p_S: p_S0, p_T: p_T0, p_1: p_a0, p_2: p_b0, dot_x_WE: dot_x_WE0})
    dot_p_2_linearized_part6 = d_dot_p_2_d_dot_x_WE.subs({x_WE: x_WE0, x_v: x_v0, p_S: p_S0, p_T: p_T0, p_1: p_a0, p_2: p_b0, dot_x_WE: dot_x_WE0})

    # Write the outputs to a note block file
    if I == 1:
        with open('output_note_block(P).txt', 'w') as f:
            f.write("The linearized function of the partial derivative of dot_p_1 with respect to x_WE at x_WE0 is:\n")
            f.write(str(dot_p_1_linearized_part1) + "\n\n")
            f.write("The linearized function of the partial derivative of dot_p_1 with respect to x_v at x_WE0 is:\n")
            f.write(str(dot_p_1_linearized_part2) + "\n\n")
            f.write("The linearized function of the partial derivative of dot_p_1 with respect to p_S at x_WE0 is:\n")
            f.write(str(dot_p_1_linearized_part3) + "\n\n")
            f.write("The linearized function of the partial derivative of dot_p_1 with respect to p_1 at x_WE0 is:\n")
            f.write(str(dot_p_1_linearized_part4) + "\n\n")
            f.write("The linearized function of the partial derivative of dot_p_1 with respect to p_2 at x_WE0 is:\n")
            f.write(str(dot_p_1_linearized_part5) + "\n\n")
            f.write("The linearized function of the partial derivative of dot_p_1 with respect to dot_x_WE at x_WE0 is:\n")
            f.write(str(dot_p_1_linearized_part6) + "\n\n")
            f.write("The linearized function of the partial derivative of dot_p_2 with respect to x_WE at x_WE0 is:\n")
            f.write(str(dot_p_2_linearized_part1) + "\n\n")
            f.write("The linearized function of the partial derivative of dot_p_2 with respect to x_v at x_WE0 is:\n")
            f.write(str(dot_p_2_linearized_part2) + "\n\n")
            f.write("The linearized function of the partial derivative of dot_p_2 with respect to p_1 at x_WE0 is:\n")
            f.write(str(dot_p_2_linearized_part3) + "\n\n")
            f.write("The linearized function of the partial derivative of dot_p_2 with respect to p_2 at x_WE0 is:\n")
            f.write(str(dot_p_2_linearized_part4) + "\n\n")
            f.write("The linearized function of the partial derivative of dot_p_2 with respect to p_T at x_WE0 is:\n")
            f.write(str(dot_p_2_linearized_part5) + "\n\n")
            f.write("The linearized function of the partial derivative of dot_p_2 with respect to dot_x_WE at x_WE0 is:\n")
            f.write(str(dot_p_2_linearized_part6) + "\n\n")

    else:
        with open('output_note_block(N).txt', 'w') as f:
            f.write("The linearized function of the partial derivative of dot_p_1 with respect to x_WE at x_WE0 is:\n")
            f.write(str(dot_p_1_linearized_part1) + "\n\n")
            f.write("The linearized function of the partial derivative of dot_p_1 with respect to x_v at x_WE0 is:\n")
            f.write(str(dot_p_1_linearized_part2) + "\n\n")
            f.write("The linearized function of the partial derivative of dot_p_1 with respect to p_T at x_WE0 is:\n")
            f.write(str(dot_p_1_linearized_part3) + "\n\n")
            f.write("The linearized function of the partial derivative of dot_p_1 with respect to p_1 at x_WE0 is:\n")
            f.write(str(dot_p_1_linearized_part4) + "\n\n")
            f.write("The linearized function of the partial derivative of dot_p_1 with respect to p_2 at x_WE0 is:\n")
            f.write(str(dot_p_1_linearized_part5) + "\n\n")
            f.write("The linearized function of the partial derivative of dot_p_1 with respect to dot_x_WE at x_WE0 is:\n")
            f.write(str(dot_p_1_linearized_part6) + "\n\n")
            f.write("The linearized function of the partial derivative of dot_p_2 with respect to x_WE at x_WE0 is:\n")
            f.write(str(dot_p_2_linearized_part1) + "\n\n")
            f.write("The linearized function of the partial derivative of dot_p_2 with respect to x_v at x_WE0 is:\n")
            f.write(str(dot_p_2_linearized_part2) + "\n\n")
            f.write("The linearized function of the partial derivative of dot_p_2 with respect to p_1 at x_WE0 is:\n")
            f.write(str(dot_p_2_linearized_part3) + "\n\n")
            f.write("The linearized function of the partial derivative of dot_p_2 with respect to p_2 at x_WE0 is:\n")
            f.write(str(dot_p_2_linearized_part4) + "\n\n")
            f.write("The linearized function of the partial derivative of dot_p_2 with respect to p_S at x_WE0 is:\n")
            f.write(str(dot_p_2_linearized_part5) + "\n\n")
            f.write("The linearized function of the partial derivative of dot_p_2 with respect to dot_x_WE at x_WE0 is:\n")
            f.write(str(dot_p_2_linearized_part6) + "\n\n")
    I += 1
# The equilibrium point is found by solving the following equations:

M = 681.97
Q_n = 40 / 60000
p_n = 35e5
x_WE = 0.245
dot_x_WE = 0
p_S = 180e5
p_T = 1e5
F_L = 4000
c_WE = 1369e-3
g = 9.82
L_WE = 0.49
x_com = 0.21
y_com = 0.06
L_com = np.sqrt(x_com**2 + y_com**2)
J = 181.02
m_eq = J / (-0.738)**2
beta_0 = 16e8
V_WE_1_0 = 0.001
V_WE_2_0 = 0.001
C_le = 5e-13
D_WE = 63e-3
d_WE = 40e-3
A_WE_1 = (D_WE ** 2) * (np.pi / 4)
A_WE_2 = (D_WE ** 2) * (np.pi / 4) - (d_WE ** 2) * (np.pi / 4) 
B_p = 15000
F_c = 1000
F_F_D = B_p * dot_x_WE + F_c*np.sign(dot_x_WE)
F_g = M * g * np.cos(np.atan(y_com/x_com)) 


def equations(vars):
    p_WE_1, p_WE_2, x_v = vars
    eq1 = beta_0 / (V_WE_1_0 + A_WE_1 * x_WE) * ((Q_n / np.sqrt(p_n) * x_v * np.sqrt(np.abs(p_S - p_WE_1)) * np.sign(p_S - p_WE_1)) - C_le * (p_WE_1 - p_WE_2) - A_WE_1 * dot_x_WE)
    eq2 = beta_0 / (V_WE_2_0 - A_WE_2 * (L_WE - x_WE)) * (- (Q_n / np.sqrt(p_n) * x_v * np.sqrt(np.abs(p_WE_2 - p_T)) * np.sign(p_WE_2 - p_T)) + C_le * (p_WE_1 - p_WE_2) + A_WE_2 * dot_x_WE)
    eq3 = (A_WE_1 * p_WE_1 - A_WE_2 * p_WE_2 - F_F_D - F_L - F_g) * 1/m_eq
    return [eq1, eq2, eq3]

initial_guess = [75e5, 110e5, 0.1]
solution = fsolve(equations, initial_guess)
p_WE_1_sol, p_WE_2_sol, x_v_sol = solution
print(f"p_WE_1: {p_WE_1_sol/1e5}, p_WE_2: {p_WE_2_sol/1e5}, x_v: {x_v_sol}")

