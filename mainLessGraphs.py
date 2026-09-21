import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy import stats
import statsmodels.api as sm

df = pd.read_csv("study.csv")
base_lag = 1
window_size = 3

variables = [ 
    "Hours",
    "Velocity",
    "Happiness",
    "Stress"
]

print("------------------STATS---------------------")
print(df[variables].describe())


max_lag = 3
GREEN = "\033[92m"
RED = "\033[91m"
RESET = "\033[0m"
threshold = 0.4
for lag in range(max_lag + 1):
    print(f"\n------------------ LAGGED CORR (Lag = {lag}) ------------------")
    
    lagged_corr_dict = {}
    
    for row in variables:
        lagged_corr_dict[row] = {}
        for col in variables:

            val = df[row].corr(df[col].shift(lag), method="pearson")
            lagged_corr_dict[row][col] = val
            
    corr_matrix = pd.DataFrame(lagged_corr_dict).T
    
    header_row = f"{'':<12}" + "".join([f"{var:<12}" for var in variables])
    print(header_row)
    
    for row in variables:
        row_str = f"{row:<12}"
        for col in variables:
            val = corr_matrix.loc[row, col]
            
            if pd.isna(val):
                row_str += f"{RESET}{'NaN':>11} "
                continue
            

            if lag == 0 and row == col:
                color = RESET
            elif val >= threshold:
                color = GREEN
            elif val <= -threshold:
                color = RED
            else:
                color = RESET
                
            row_str += f"{color}{val:>11.4f}{RESET} "
        print(row_str) 
    print("------------------------------------------------------------")

fig, axes = plt.subplots(1, len(variables), figsize=(len(variables) * 5, 5)) 
axes = np.array(axes).flatten()

for i, variable in enumerate(variables): 
    ax = axes[i]
    x = df[variable]
    y = df[variable].shift(-base_lag)

    ax.scatter(x, y)

    low = min(np.min(x), np.min(y))
    high = max(np.max(x), np.max(y))
    ax.plot([low, high], [low, high], linestyle="--", color="gray")

    m, b = np.polyfit(x.drop(df.tail(1).index), y.dropna(), 1)
    
    x_line = np.array([np.min(x), np.max(x)])
    y_line = m * x_line + b
    
    ax.plot(x_line, y_line, color="red", linestyle="-")

    ax.set_xlabel("t")
    ax.set_ylabel(f"t + {base_lag}")
    ax.set_title(f"Poincare: {variable}")
    ax.grid(True)
plt.show()

def poincare_metrics(series, base_lag=1):
    x = series.to_numpy()

    x1 = x[:-base_lag]
    x2 = x[base_lag:]

    differences = x2 - x1

    sd1 = np.sqrt(0.5 * np.var(differences, ddof=1))

    sd2 = np.sqrt(2 * np.var(x, ddof=1)- 0.5 * np.var(differences, ddof=1))

    ratio = sd2 / sd1
    
    r = np.corrcoef(x1, x2)[0, 1]

    return sd1, sd2, ratio, r


print("\n------------- POINCARE METRICS -------------")

for variable in variables:
    sd1, sd2, ratio, r = poincare_metrics(df[variable], base_lag)

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


df['d-stress'] = df["Stress"].shift(-1) - df['Stress']
df['d-hours'] = df["Hours"].shift(-1) - df['Hours']

X = df['Stress'].iloc[:-1].values
Y = df['Hours'].iloc[:-1].values
U = df['d-stress'].iloc[:-1].values
V = df['d-hours'].iloc[:-1].values

magnitude = np.sqrt(U**2 + V**2)
magnitude_safe = np.where(magnitude == 0, 1, magnitude) 
U_norm = U / magnitude_safe
V_norm = V / magnitude_safe

stride = 1
X_s, Y_s, U_s, V_s, M_s = X[::stride], Y[::stride], U_norm[::stride], V_norm[::stride], magnitude[::stride]

plt.figure(figsize=(10, 8))

Q = plt.quiver(X_s, Y_s, U_s, V_s, M_s,
               cmap='viridis',
               angles='xy', 
               scale=15,
               pivot='tail', 
               alpha=0.8)

cbar = plt.colorbar(Q)
cbar.set_label('Transition Magnitude (Speed)', rotation=270, labelpad=15)

plt.plot(df['Stress'], df['Hours'], color='black', alpha=0.3, linestyle='--', label='Trajectory Path')
plt.scatter(df['Stress'], df['Hours'], color='black', s=15, zorder=3)

plt.title("Phase Space Vector Field: Normalized Multi-Day Dynamics")
plt.xlabel("Stress (t)")
plt.ylabel("Hours (t)")
plt.grid(True, linestyle=':', alpha=0.6)
plt.legend()
plt.tight_layout()
plt.show()

print("\n------------------------TOMORROW PREDICTIONS---------------------------")
print("Based on today's state, you are mathematically on track for:")

latest_day = df.iloc[-1]

for target_var in variables:
    X = df[variables].iloc[:-1]
    Y = df[target_var].shift(-1).iloc[:-1]
    
    X = sm.add_constant(X)
    
    model = sm.OLS(Y, X).fit()
    
    if target_var == "Hours":
        print("\n------------------ OLS REGRESSION RESULTS (HOURS) ------------------")
        print(model.summary())
        print("--------------------------------------------------------------------\n")

    intercept = model.params['const']
    predicted_tomorrow = intercept + sum(model.params[var] * latest_day[var] for var in variables)
    
    print(f"{predicted_tomorrow:.2f} {target_var.lower()} tomorrow")

print("-----------------------------------------------------------------------")
