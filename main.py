import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy import stats
from statsmodels.graphics.tsaplots import plot_acf
import statsmodels.api as sm

df = pd.read_csv("study.csv")
lag = 1
window_size = 3

variables = [ 
    "Hours",
    "Velocity",
    "Happiness",
    "Stress"
]

print("------------------STATS---------------------")
print(df[variables].describe())


print("------------------CORR----------------------")
corr_matrix = df[variables].corr(method="pearson")


GREEN = "\033[92m"
RED = "\033[91m"
RESET = "\033[0m"
threshold = 0.4

header_row = f"{'':<12}" + "".join([f"{var:<12}" for var in variables])
print(header_row)


for row in variables:
    row_str = f"{row:<12}"
    for col in variables:
        val = corr_matrix.loc[row, col]
        
        if row == col:
            color = RESET
        elif val >= threshold:
            color = GREEN
        elif val <= -threshold:
            color = RED
        else:
            color = RESET
            
        
        row_str += f"{color}{val:>11.4f}{RESET} "
    print(row_str) 
print("--------------------------------------------")


#------------------------------------------------------
#                        PLOTS
#------------------------------------------------------
fig, axes = plt.subplots(1, len(variables), figsize=(len(variables) * 5, 5)) 
axes = np.array(axes).flatten()

for i, variable in enumerate(variables): 
    ax = axes[i]
    x = df[variable]
    y = df[variable].shift(-lag)

    ax.scatter(x, y)

    low = min(np.min(x), np.min(y))
    high = max(np.max(x), np.max(y))
    ax.plot([low, high], [low, high], linestyle="--", color="gray")

    m, b = np.polyfit(x.drop(df.tail(1).index), y.dropna(), 1)
    
    x_line = np.array([np.min(x), np.max(x)])
    y_line = m * x_line + b
    
    ax.plot(x_line, y_line, color="red", linestyle="-")

    ax.set_xlabel("t")
    ax.set_ylabel(f"t + {lag}")
    ax.set_title(f"Poincare: {variable}")
    ax.grid(True)
plt.show()

def poincare_metrics(series, lag=1):
    x = series.to_numpy()

    x1 = x[:-lag]
    x2 = x[lag:]

    differences = x2 - x1

    sd1 = np.sqrt(0.5 * np.var(differences, ddof=1))

    sd2 = np.sqrt(2 * np.var(x, ddof=1)- 0.5 * np.var(differences, ddof=1))

    ratio = sd2 / sd1
    
    r = np.corrcoef(x1, x2)[0, 1]

    return sd1, sd2, ratio, r


print("\n------------- POINCARE METRICS -------------")

for variable in variables:
    sd1, sd2, ratio, r = poincare_metrics(df[variable], lag)

    print(
        f"{variable:<12}   "
        f"SD1={sd1:.3f}   "
        f"SD2={sd2:.3f}   "
        f"SD1/SD2={ratio:.3f}   "
        f"r={r:.3f}   "
    )

lagged_pairs = [
    ("Hours", "Hours"),
    ("Velocity", "Velocity"),
    ("Happiness", "Happiness"),
    ("Stress", "Stress"),
    

    ("Hours", "Velocity"),
    ("Hours", "Happiness"),
    ("Hours", "Stress"),
    
    ("Velocity", "Hours"),
    ("Velocity", "Happiness"),
    ("Velocity", "Stress"),
    
    ("Happiness", "Hours"),
    ("Happiness", "Velocity"),
    ("Happiness", "Stress"),
    
    ("Stress", "Hours"),
    ("Stress", "Velocity"),
    ("Stress", "Happiness")
]


print("\n------------- LAGGED CORRELATIONS -------------")
threshold = 0.45
for x_var, y_var in lagged_pairs:
    x = df[x_var]
    y = df[y_var].shift(-1)
    data = pd.concat([x, y], axis=1).dropna()
    r, p = stats.pearsonr(data.iloc[:, 0], data.iloc[:, 1])

    if r >= threshold:
        color = GREEN
    elif r <= -threshold:
        color = RED
    else:
        color = RESET

    print(
        f"{x_var:<8} -> {y_var:<10} tomorrow: "
        f"{color}r={r:>6.3f}{RESET}, "
        f"p={p:.4f}"
    )

x = df["Hours"]
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

x = df['Stress']
y = df['Hours'].shift(-1)

slope, intercept, r, p, se = stats.linregress(x.drop(df.tail(1).index), y.dropna())


plt.figure(figsize=(7, 5))

plt.scatter(x, y)

plt.plot(
    x,
    intercept + slope * x,
    linestyle="--"
)

plt.xlabel("Stress")
plt.ylabel("Hours(t+1)")
plt.title(f"Stress v. Hours(t+1) (r = {r:.2f})")
plt.grid(True)
plt.show()



df['d-stress'] = df["Stress"].shift(-1) - df['Stress']
df['d-hours'] = df["Hours"].shift(-1) - df['Hours']

plt.figure(figsize=(8,6))

plt.quiver(df['Stress'].iloc[:-1], df['Hours'].iloc[:-1], 
           df['d-stress'].iloc[:-1], df['d-hours'].iloc[:-1], 
           angles='xy', scale_units='xy', scale=1, color='blue', alpha=0.6, label='System Vector Field')
plt.scatter(df['Stress'], df['Hours'], color='black')
plt.title("Phase Space Vector Field: (Stress_t, Hours_t) Multi-Day Dynamics")
plt.xlabel("Stress (t)")
plt.ylabel("Hours (t)")
plt.grid(True)
plt.legend()
plt.show()

X = df[variables]
Y = df["Hours"].shift(-1)


X = X.iloc[:-1]
Y = Y.iloc[:-1]


X = sm.add_constant(X)


model = sm.OLS(Y, X).fit()
print("\n------------------ OLS REGRESSION RESULTS ------------------")
print(model.summary())

latest_day = df.iloc[-1]
intercept = model.params['const']

predicted_tomorrow = (
    intercept +
    (model.params['Hours'] * latest_day['Hours']) +
    (model.params['Velocity'] * latest_day['Velocity']) +
    (model.params['Happiness'] * latest_day['Happiness']) +
    (model.params['Stress'] * latest_day['Stress'])
)

print("\n------------------------TOMORROW PREDICTION---------------------------")
print(f"Based on today's state, you are mathematically on track to study:")
print(f"{predicted_tomorrow:.2f} hours tomorrow.")
print("-----------------------------------------------------------------------")
