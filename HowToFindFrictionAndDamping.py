import pandas as pd
import numpy as np
from scipy.optimize import curve_fit
import matplotlib.pyplot as plt

# Load the Excel file
file_path = r'C:\Users\Rober\OneDrive - Aalborg Universitet\MP.dok\Studie\Libary\torque and angular momentum relation.xlsx'  # Replace with your file path
df = pd.read_excel(file_path, header=None)

N003 = df.iloc[0, 1] 
N003_row_count = df.iloc[:, 1].count()
N003_x_values = np.full(N003_row_count-1, N003)
N003_y_values = df.iloc[1:N003_row_count, 1]

N00233 = df.iloc[0, 2] 
N00233_row_count = df.iloc[:, 3].count()
N00233_x_values = np.full(N00233_row_count-1, N00233)
N00233_y_values = df.iloc[1:N00233_row_count, 1]

N002 = df.iloc[0, 3]
N002_row_count = df.iloc[:,3].count()
N002_x_values = np.full(N002_row_count-1, N002)
N002_y_values = df.iloc[1:N002_row_count, 3]

N00166 = df.iloc[0, 4]
N00166_row_count = df.iloc[:, 4].count()
N00166_x_values = np.full(N00166_row_count-1, N00166)
N00166_y_values = df.iloc[1:N00166_row_count, 4]

N00133 = df.iloc[0, 5]
N00133_row_count = df.iloc[:, 5].count()
N00133_x_values = np.full(N00133_row_count-1, N00133)
N00133_y_values = df.iloc[1:N00133_row_count, 5]

N001 = df.iloc[0, 6]
N001_row_count = df.iloc[:, 6].count()
N001_x_values = np.full(N001_row_count-1, N001)
N001_y_values = df.iloc[1:N001_row_count, 6]

N000667 = df.iloc[0, 7]
N000667_row_count = df.iloc[:, 7].count()
N000667_x_values = np.full(N000667_row_count-1, N000667)
N000667_y_values = df.iloc[1:N000667_row_count, 7]

P000667 = df.iloc[0, 8]
P000667_row_count = df.iloc[:, 8].count()
P000667_x_values = np.full(P000667_row_count-1, P000667)
P000667_y_values = df.iloc[1:P000667_row_count, 8]

P001 = df.iloc[0, 9]
P001_row_count = df.iloc[:, 9].count()
P001_x_values = np.full(P001_row_count-1, P001)
P001_y_values = df.iloc[1:P001_row_count, 9]

P00133 = df.iloc[0, 10]
P00133_row_count = df.iloc[:, 10].count()
P00133_x_values = np.full(P00133_row_count-1, P00133)
P00133_y_values = df.iloc[1:P00133_row_count, 10]

P00166 = df.iloc[0, 11]
P00166_row_count = df.iloc[:, 11].count()
P00166_x_values = np.full(P00166_row_count-1, P00166)
P00166_y_values = df.iloc[1:P00166_row_count, 11]

P002 = df.iloc[0, 12]
P002_row_count = df.iloc[:, 12].count()
P002_x_values = np.full(P002_row_count-1, P002)
P002_y_values = df.iloc[1:P002_row_count, 12]

P00233 = df.iloc[0, 13]
P00233_row_count = df.iloc[:, 13].count()
P00233_x_values = np.full(P00233_row_count-1, P00233)
P00233_y_values = df.iloc[1:P00233_row_count, 13]

P003 = df.iloc[0, 14]
P003_row_count = df.iloc[:, 14].count()
P003_x_values = np.full(P003_row_count-1, P003)
P003_y_values = df.iloc[1:P003_row_count, 14]

N0001667394 = df.iloc[0, 15]
N0001667394_row_count = df.iloc[:, 15].count()
N0001667394_x_values = np.full(N0001667394_row_count-1, N0001667394)
N0001667394_y_values = df.iloc[1:N0001667394_row_count, 15]

P0001667394 = df.iloc[0, 16]
P0001667394_row_count = df.iloc[:, 16].count()
P0001667394_x_values = np.full(P0001667394_row_count-1, P0001667394)
P0001667394_y_values = df.iloc[1:P0001667394_row_count, 16]

P000300192 = df.iloc[0, 17]
P000300192_row_count = df.iloc[:, 17].count()
P000300192_x_values = np.full(P000300192_row_count-1, P000300192)
P000300192_y_values = df.iloc[1:P000300192_row_count, 17]

N000300192 = df.iloc[0, 18]
N000300192_row_count = df.iloc[:, 18].count()
N000300192_x_values = np.full(N000300192_row_count-1, N000300192)
N000300192_y_values = df.iloc[1:N000300192_row_count, 18]

P0066500449 = df.iloc[0, 19]
P0066500449_row_count = df.iloc[:, 19].count()
P0066500449_x_values = np.full(P0066500449_row_count-1, P0066500449)
P0066500449_y_values = df.iloc[1:P0066500449_row_count, 19]

N0066500449 = df.iloc[0, 20]
N0066500449_row_count = df.iloc[:, 20].count()
N0066500449_x_values = np.full(N0066500449_row_count-1, N0066500449)
N0066500449_y_values = df.iloc[1:N0066500449_row_count, 20]

P0198521049 = df.iloc[0, 21]
P0198521049_row_count = df.iloc[:, 21].count()
P0198521049_x_values = np.full(P0198521049_row_count-1, P0198521049)
P0198521049_y_values = df.iloc[1:P0198521049_row_count, 21]

N0168799505 = df.iloc[0, 22]
N0168799505_row_count = df.iloc[:, 22].count()
N0168799505_x_values = np.full(N0168799505_row_count-1, N0168799505)
N0168799505_y_values = df.iloc[1:N0168799505_row_count, 22]

P0101940416 = df.iloc[0, 23]
P0101940416_row_count = df.iloc[:, 23].count()
P0101940416_x_values = np.full(P0101940416_row_count-1, P0101940416)
P0101940416_y_values = df.iloc[1:P0101940416_row_count, 23]

N0096814392 = df.iloc[0, 24]
N0096814392_row_count = df.iloc[:, 24].count()
N0096814392_x_values = np.full(N0096814392_row_count-1, N0096814392)
N0096814392_y_values = df.iloc[1:N0096814392_row_count, 24]

# Plot the data
plt.figure(figsize=(12, 6))
plt.plot(N003_x_values, N003_y_values, 'o')
plt.plot(N00233_x_values, N00233_y_values, 'o')
plt.plot(N002_x_values, N002_y_values, 'o')
plt.plot(N00166_x_values, N00166_y_values, 'o')
plt.plot(N00133_x_values, N00133_y_values, 'o')
plt.plot(N001_x_values, N001_y_values, 'o')
plt.plot(N000667_x_values, N000667_y_values, 'o')
plt.plot(P000667_x_values, P000667_y_values, 'o')
plt.plot(P001_x_values, P001_y_values, 'o')
plt.plot(P00133_x_values, P00133_y_values, 'o')
plt.plot(P00166_x_values, P00166_y_values, 'o')
plt.plot(P002_x_values, P002_y_values, 'o')
plt.plot(P00233_x_values, P00233_y_values, 'o')
plt.plot(P003_x_values, P003_y_values, 'o')
plt.plot(N0001667394_x_values, N0001667394_y_values, 'o')
plt.plot(P0001667394_x_values, P0001667394_y_values, 'o')
plt.plot(P000300192_x_values, P000300192_y_values, 'o')
plt.plot(N000300192_x_values, N000300192_y_values, 'o')
plt.plot(P0066500449_x_values, P0066500449_y_values, 'o')
plt.plot(N0066500449_x_values, N0066500449_y_values, 'o')
plt.plot(P0198521049_x_values, P0198521049_y_values, 'o')
plt.plot(N0168799505_x_values, N0168799505_y_values, 'o')
plt.plot(P0101940416_x_values, P0101940416_y_values, 'o')
plt.plot(N0096814392_x_values, N0096814392_y_values, 'o')

# Combine all x and y values into single arrays
x_values = np.concatenate([
    N003_x_values, N00233_x_values, N002_x_values, N00166_x_values, N00133_x_values, N001_x_values, N000667_x_values,
    P000667_x_values, P001_x_values, P00133_x_values, P00166_x_values, P002_x_values, P00233_x_values, P003_x_values,
    N0001667394_x_values, P0001667394_x_values, P000300192_x_values, N000300192_x_values, P0066500449_x_values,
    N0066500449_x_values, P0198521049_x_values, N0168799505_x_values, P0101940416_x_values, N0096814392_x_values
])
y_values = np.concatenate([
    N003_y_values, N00233_y_values, N002_y_values, N00166_y_values, N00133_y_values, N001_y_values, N000667_y_values,
    P000667_y_values, P001_y_values, P00133_y_values, P00166_y_values, P002_y_values, P00233_y_values, P003_y_values,
    N0001667394_y_values, P0001667394_y_values, P000300192_y_values, N000300192_y_values, P0066500449_y_values,
    N0066500449_y_values, P0198521049_y_values, N0168799505_y_values, P0101940416_y_values, N0096814392_y_values
])

# Remove NaN and infinite values
mask = ~np.isnan(x_values) & ~np.isnan(y_values) & ~np.isinf(x_values) & ~np.isinf(y_values)
x_values_clean = x_values[mask]
y_values_clean = y_values[mask]

# Define the model function
def friction_model(dot_theta, k, B):
    return k * np.sign(dot_theta) + B * dot_theta

# Fit the model to the data
popt, pcov = curve_fit(friction_model, x_values_clean, y_values_clean)

# Extract the coefficients
k, B = popt
residuals = y_values_clean- friction_model(x_values_clean, *popt)
ss_res = np.sum(residuals**2)
ss_tot = np.sum((y_values_clean-np.mean(y_values_clean))**2)
r_squared = 1 - (ss_res / ss_tot)
print(f"Fitted coefficients: k = {k}, B = {B}")
print(f"The error is: {r_squared}")

# Plot the fitted model
dot_theta_fit = np.linspace(min(x_values), max(x_values), 1000)
tau_F_D_fit_1 = friction_model(dot_theta_fit, k, B)
tau_F_D_fit_2 = friction_model(dot_theta_fit, 1000, 15000)

# Set font globally to Times New Roman
plt.rcParams['font.family'] = 'Times New Roman'

# Generate your plot
#plt.plot(dot_theta_fit, tau_F_D_fit_1, label='Fitted Model', color='red', linewidth=2)
plt.plot(dot_theta_fit, tau_F_D_fit_2, label='Fitted Model', color='blue', linewidth=2)

# Add axis labels and title
plt.xlabel('Angular Velocity (rad/s)', fontname="Times New Roman", fontsize=12)
plt.ylabel('Frictional Torque (Nm)', fontname="Times New Roman", fontsize=12)
plt.title('Frictional Torque vs Angular Velocity', fontname="Times New Roman", fontsize=14, fontweight='bold')

# Customize grid
plt.grid(color='gray', linestyle='--', linewidth=0.5)

# Customize ticks
plt.xticks(fontsize=10)
plt.yticks(fontsize=10)

# Add legend
plt.legend(fontsize=10, loc='upper left')

# Display the plot
plt.tight_layout()  # Ensure no elements are clipped
plt.show()