import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# Load data
df = pd.read_csv("study.csv")
v_max = 10
lag = 1
window_size = 3  # Define the window size for the rolling average

# 1. Pre-calculate standard Poincaré data
df['hours_today'] = df['Hours']
df['hours_tomorrow'] = df['Hours'].shift(-lag)

# 2. Pre-calculate rolling average Poincaré data
df['hours_roll'] = df['Hours'].rolling(window=window_size, center=True).mean()
df['hours_roll_today'] = df['hours_roll']
df['hours_roll_tomorrow'] = df['hours_roll'].shift(-lag)

# Create side-by-side subplots
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5), sharex=True, sharey=True)

# Left Subplot: Standard Poincaré Plot
ax1.scatter(df['hours_today'], df['hours_tomorrow'], color='blue', alpha=0.6, edgecolors='k')
ax1.plot([0, v_max], [0, v_max], 'r--')
ax1.set_title("Standard Poincaré Plot")
ax1.set_xlabel("Hours ($t$)")
ax1.set_ylabel("Hours ($t + 1$)")
ax1.grid(True, linestyle=':', alpha=0.6)

# Right Subplot: Rolling Average Poincaré Plot
ax2.scatter(df['hours_roll_today'], df['hours_roll_tomorrow'], color='green', alpha=0.6, edgecolors='k')
ax2.plot([0, v_max], [0, v_max], 'r--')
ax2.set_title(f"Rolling Average Poincaré Plot (Window={window_size})")
ax2.set_xlabel(f"Smoothed Hours ($t$)")
ax2.set_ylabel(f"Smoothed Hours ($t + 1$)")
ax2.grid(True, linestyle=':', alpha=0.6)

plt.tight_layout()
plt.show()
