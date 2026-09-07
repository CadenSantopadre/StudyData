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
fig, axes = plt.subplots(1, len(variables), figsize=(len(variables) * 5, 5))
axes = np.array(axes).flatten()

#As we loop through each plot, loop through each variable too
for i, variable in enumerate(variables):
    ax = axes[i]
    x = df[variable]
    y = df[variable].shift(-1)

    #Make a scatter plot first
    ax.scatter(x, y)

    #Then make the identity line
    low = min(np.min(x), np.min(y))
    high = max(np.max(x), np.max(y))
    ax.plot([low, high], [low, high], linestyle="--", color="gray")

    #np.polyfit will return slope and y0
    m, b = np.polyfit(x.drop(df.tail(1).index), y.dropna(), 1) #NEED .dropna() because the last one is shifted so its NaN there
    
    #Then make an array of x values from min to max
    x_line = np.array([np.min(x), np.max(x)])
    #Correspondingly, make an array multiplying m and adding b to it
    y_line = m * x_line + b
    
    # Plot the line of best fit
    ax.plot(x_line, y_line, color="red", linestyle="-")

    ax.set_xlabel("t")
    ax.set_ylabel(f"t + {lag}")
    ax.set_title(f"Poincare: {variable}")
    ax.grid(True)
plt.show()

#Btw, some interpretation for poincare plots goes like this:
#Close to identity line = minimal change
#Negative line = long study -> short study
#Flat line = almost no correlation with anything
#Positive line = long study -> longer study

#Tight clusters = discipline to routine
#scattered points = flexibility

#Bottom right = post-exam crash
#Top left = Panic study spike



print("------------------STATS---------------------")
print(df[variables].describe()) #doign df[variables] means it'll go through all of them
#also, .describe() is a nifty tool that displays basic stats
#It gives mean, stddev, min, Q1, med, Q3, max, and count

print("------------------CORR----------------------")
corr_matrix = df[variables].corr(method="pearson")

# 1. Define ANSI codes and color thresholds
GREEN = "\033[92m"  # Strong positive
RED = "\033[91m"    # Strong negative
RESET = "\033[0m"   # Reset back to default terminal text
threshold = 0.4     # Adjust this to change what counts as "strong"

# 2. Print the column headers with spacing
header_row = f"{'':<12}" + "".join([f"{var:<12}" for var in variables])
print(header_row)

# 3. Loop through rows and color individual values
for row in variables:
    row_str = f"{row:<12}" # Row label aligned left
    for col in variables:
        val = corr_matrix.loc[row, col]
        
        # Determine color based on strength and direction
        if row == col:
            color = RESET  # Keep self-correlation (1.00) neutral
        elif val >= threshold:
            color = GREEN
        elif val <= -threshold:
            color = RED
        else:
            color = RESET
            
        # Format the number to 4 decimal places and wrap in ANSI colors
        row_str += f"{color}{val:>11.4f}{RESET} "
        
    print(row_str)
print("--------------------------------------------")
 #We use pearson because idk what the others do

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

