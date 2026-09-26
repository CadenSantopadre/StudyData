import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
import matplotlib.pyplot as plt

df = pd.read_csv("study.csv")

variables = ["Hours", "Velocity", "Happiness", "Stress"]

X = df[variables].iloc[:-1]
Y = df[variables].shift(-1).iloc[:-1]

model = LinearRegression()
model.fit(X, Y)

A = model.coef_
b = model.intercept_

DAYS = 10
SIMCOUNT = 1000000
EPS_SCALE = 1.0

initial_state = df[variables].iloc[-1].to_numpy() 

best_cum_utility = -float('inf')
best_path = None
best_decisions = None

for sim in range(SIMCOUNT):
    current_state = initial_state.copy()
    sim_path = []
    sim_decisions = []
    cumulative_utility = 0.0

    for t in range(DAYS):
        next_state = np.dot(current_state, A.T) + b
        
        eps_h = np.random.uniform(-EPS_SCALE, EPS_SCALE)
        eps_v = np.random.uniform(-EPS_SCALE, EPS_SCALE)

        next_state[0] += eps_h
        next_state[1] += eps_v
        
        next_state = np.clip(next_state, a_min=0, a_max=None)

        step_utility = next_state[2] - next_state[3]
        cumulative_utility += step_utility

        sim_path.append(next_state.copy())
        sim_decisions.append((eps_h, eps_v))

        current_state = next_state

    if cumulative_utility > best_cum_utility:
        best_cum_utility = cumulative_utility
        best_path = np.array(sim_path)
        best_decisions = np.array(sim_decisions)

print(f"--- Optimization Results over a {DAYS}-Day Window ---")
print(f"Maximized Cumulative Utility Found: {best_cum_utility:.4f}")
print("\nOptimal Path Profiles:")
for day in range(DAYS):
    print(f"Day {day+1} -> Choice Adjustments (dH: {best_decisions[day][0]:.2f}, dV: {best_decisions[day][1]:.2f})")
    print(f"         State Matrix -> Hours: {best_path[day][0]:.2f} | Velocity: {best_path[day][1]:.2f} | Happiness: {best_path[day][2]:.2f} | Stress: {best_path[day][3]:.2f}")

# --- 3D TRAJECTORY PLOT ---
# Calculate running cumulative utility tracking along the selected optimal path
running_utility = []
current_sum = 0.0
for day in range(DAYS):
    current_sum += (best_path[day][2] - best_path[day][3])
    running_utility.append(current_sum)

x_hours = best_path[:, 0]
y_velocity = best_path[:, 1]
z_utility = np.array(running_utility)

fig = plt.figure(figsize=(10, 7))
ax = fig.add_subplot(111, projection='3d')

# Plot the 3D trajectory line and scatter points for each day
ax.plot(x_hours, y_velocity, z_utility, color='blue', linewidth=2.5, label='Optimal Path Trajectory')
scatter = ax.scatter(x_hours, y_velocity, z_utility, c=np.arange(1, DAYS + 1), cmap='viridis', s=60, edgecolors='black')

# Label individual nodes with their respective day number
for day in range(DAYS):
    ax.text(x_hours[day] + 0.05, y_velocity[day] + 0.05, z_utility[day], f'D{day+1}', fontsize=9, fontweight='bold')

# Configure labels and axes
ax.set_title('3D State Space Optimization Path', fontsize=14, fontweight='bold', pad=20)
ax.set_xlabel('Hours Spent ($X$)', fontsize=11, labelpad=10)
ax.set_ylabel('Velocity ($Y$)', fontsize=11, labelpad=10)
ax.set_zlabel('Cumulative Utility ($Z$)', fontsize=11, labelpad=10)

# Add a colorbar to anchor the chronological progression across the 10 days
cbar = fig.colorbar(scatter, ax=ax, pad=0.1, shrink=0.6)
cbar.set_label('Timeline Progression (Days)', fontsize=10)
cbar.set_ticks(np.arange(1, DAYS + 1))

plt.tight_layout()
plt.show()