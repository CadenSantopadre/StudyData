import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy import stats
from statsmodels.tsa.stattools import acf
from statsmodels.graphics.tsaplots import plot_acf

df = pd.read_csv("study.csv")
lag = 1
window_size = 3

variables = [
    "Hours",
    "Velocity",
    "Happiness",
    "Stress"
]

#First we're going to make poincare plots and compare them relative to the identity line

import numpy as np
import matplotlib.pyplot as plt

fig, axes = plt.subplots(1, len(variables), figsize=(len(variables) * 5, 5))
axes = np.array(axes).flatten()

for i, variable in enumerate(variables):
    ax = axes[i]
    x = df[variable].iloc[:-lag].values
    y = df[variable].iloc[lag:].values

    # Scatter plot on the specific axis
    ax.scatter(x, y)

    # Diagonal identity line
    low = min(np.min(x), np.min(y))
    high = max(np.max(x), np.max(y))
    ax.plot([low, high], [low, high], linestyle="--", color="gray", label="Identity")

    # --- LINE OF BEST FIT ---
    # Calculate slope (m) and intercept (b) for linear regression (degree 1)
    m, b = np.polyfit(x, y, 1)
    
    # Generate points for the line using the min and max of x
    x_line = np.array([np.min(x), np.max(x)])
    y_line = m * x_line + b
    
    # Plot the line of best fit
    ax.plot(x_line, y_line, color="red", linestyle="-", label=f"Fit (m={m:.2f})")
    # ------------------------

    # Labels and titles
    ax.set_xlabel(f"t")
    ax.set_ylabel(f"t + {lag}")
    ax.set_title(f"Poincare: {variable}")
    ax.grid(True)
    ax.legend()  # Optional: Adds a legend to distinguish the lines

plt.show()




print("------------------STATS---------------------")
print(df[variables].describe()) #doign df[variables] means it'll go through all of them
#also, .describe() is a nifty tool that displays basic stats
#It gives mean, stddev, Q0, Q1, Q2, Q3, Q4, and count

print("------------------CORR----------------------")
print(df[variables].corr(method="pearson")) #We use pearson because idk what the others do

#It might look something like this:
#              Hours  Velocity  Happiness    Stress
#Hours      1.000000  0.390560   0.045721  0.531636
#Velocity   0.390560  1.000000   0.061721  0.908541
#Happiness  0.045721  0.061721   1.000000 -0.091287
#Stress     0.531636  0.908541  -0.091287  1.000000

#Notice an identity through 1.00, that doesn't matter
#But a 0.53 with hours and stress means more hours = more stressed at an r = 0.53


#Next, we'll visualize some of the big correlations that we found from CORR
x = df["Hours"]
y = df["Stress"]

slope, intercept, r, p, se = stats.linregress( #linregress gives m,y0,r,p,stderror(std/sqrt(n))
    x,y
)

plt.figure(figsize=(7, 5))

plt.scatter(x, y)

plt.plot(
    x,
    intercept + slope * x,
    linestyle="--"
)

plt.xlabel("Study hours")
plt.ylabel("Stress")
plt.title(f"Study Time vs Stress (r = {r:.2f})")
plt.grid(True)
plt.show()

x = df["Velocity"]
y = df["Stress"]

slope, intercept, r, p, se = stats.linregress(
    x,y
)

plt.figure(figsize=(7, 5))

plt.scatter(x, y)

plt.plot(
    x,
    intercept + slope * x,
    linestyle="--"
)

plt.xlabel("Velocity")
plt.ylabel("Stress")
plt.title(f"Velocity vs Stress (r = {r:.2f})")
plt.grid(True)
plt.show()

#Next lets maek an acf function for everything
max_lags = len(df) - 1
fig, axes = plt.subplots(1, len(variables), figsize=(18, 5))
axes_flat = axes.flatten()
for i, var in enumerate(variables):
    plot_acf(df[var], lags=max_lags, ax=axes[i])

    axes_flat[i].set_title(f"ACF - {var}")
    axes_flat[i].set_xlabel("Lags")
    axes_flat[i].set_ylabel("Autocorrelation")
    axes_flat[i].grid(True)

plt.tight_layout()
plt.show()
#if something is inside the blue zones, it's statistically INsiginficant
#But this shows you the memory pattern, does x predict y based on what has happened?

