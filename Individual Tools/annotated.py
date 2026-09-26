import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy import stats
from statsmodels.graphics.tsaplots import plot_acf
import statsmodels.api as sm

df = pd.read_csv("study.csv") #make our dataframe from the csv
lag = 1 #We can adjust this to look at differnt lags, particualy 1 and 3 for poincare plots
window_size = 3 #rolling average window

variables = [ #If we make this in an array, doing df[variables] will do ALL of these
    "Hours",
    "Velocity",
    "Happiness",
    "Stress"
]

#------------------------------------------------------
#                        STATS
#------------------------------------------------------
print("------------------STATS---------------------")
print(df[variables].describe()) #.describe() gives basic stats


print("------------------CORR----------------------")
corr_matrix = df[variables].corr(method="pearson") #This gives an r value for each correlation


GREEN = "\033[92m"
RED = "\033[91m"
RESET = "\033[0m"
threshold = 0.4 #Threshold for significance in the correlation matrix

header_row = f"{'':<12}" + "".join([f"{var:<12}" for var in variables])
print(header_row) #Gotta use weird formatting since we're doing color coding for some stuff


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
                    #ANSI code + val + Rest ANSI code
    print(row_str) 
print("--------------------------------------------")


#------------------------------------------------------
#                        PLOTS
#------------------------------------------------------
fig, axes = plt.subplots(1, len(variables), figsize=(len(variables) * 5, 5)) #We're making multiple poincares for each variable
axes = np.array(axes).flatten() #This changes the 1, 4 array into a 1x1 so we can do a for loop

for i, variable in enumerate(variables): #for i, for each variable in a list of 4 variables...
    ax = axes[i] #Let ax be the index of the axis we are looking at
    x = df[variable] #let x
    y = df[variable].shift(-lag) #let y be a poincare with lag k 

    ax.scatter(x, y) #Plot the points

    low = min(np.min(x), np.min(y))
    high = max(np.max(x), np.max(y))
    ax.plot([low, high], [low, high], linestyle="--", color="gray") #Make an identity line

    m, b = np.polyfit(x.drop(df.tail(lag).index), y.dropna(), 1) #Make a slope and base for an LBF after dropping the shifted empty part from y
                                                               #And drop the last value from x because otherwise there's a hang
    
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

    x1 = x[:-lag] #Removes the last lag elements
    x2 = x[lag:] #Removes the first lag elements

    differences = x2 - x1 #dx = x2-x1

    sd1 = np.sqrt(0.5 * np.var(differences, ddof=1)) #measures variance of step-by-step indices, 
                                                    #Also ddof=1 means we divide by n-1, since we don't have the full population
                                                    #Apparantly it's a thing called Bessel's Correction

    sd2 = np.sqrt(2 * np.var(x, ddof=1)- 0.5 * np.var(differences, ddof=1))

    #There's some cool geometry to this:
    #If you were to rotate the plot 45 degrees os the line of identity was straight, you have a new coordinate plane.
    #sd1's distance from the identity line is simply Y-X... we calculate the variance of the Y-X by doing x2-x1... y-x
    #sd2's distance along the identity line is Y+X, Total variance in a 2D space is the sum of its orthogonal parts. Therefore, Total Variance = Var(SD1) + Var(SD2).

    ratio = sd2 / sd1
    #The ratio is special too, since it measures the spread of long term variability to short term
    #High(2,3ish) means SD2 is much higher than SD1, the cloud is a long torpedo.
    #High indicates drifting over time but lacks step-by-step variability

    #Low(1ish) means SD2 is around SD1, the cloud is a circle
    #High, erratic chaos... or really really specific patterns- it's hard to tell without looking at the data
    r = np.corrcoef(x1, x2)[0, 1]
    #woah, the r value can actually be modeled by this:
    #ln(ratio) = arctanh(r)
    #And it is SUPER BEAUTIFUL like look:

    #SD1^2 = variance of differences = stddev^2(1-r)
    #SD2^2 = variance of sums = stddev^2(1+r)
    #Ratio^2 = std^2(1+r)/std^2(1-r) = 1+r/1-r
    #ln(ratio) = 0.5ln(1+r/1-r)
    #arctanh is literally what's above

    #ln(SD2/SD1) = arctanh(r)
    #Beautiful


    return sd1, sd2, ratio, r


#------------------------------------------------------
#                      POINCARE STATS
#------------------------------------------------------
print("\n------------- POINCARE METRICS -------------")

for variable in variables:
    sd1, sd2, ratio, r = poincare_metrics(df[variable], lag)

    print(
        f"{variable:<12}   " #Add extra spaces so its not clustered
        f"SD1={sd1:.3f}   "
        f"SD2={sd2:.3f}   "
        f"SD1/SD2={ratio:.3f}   "
        f"r={r:.3f}   "
    )

lagged_pairs = [
    #Self lags, how something predicts itself tomorrow
    ("Hours", "Hours"),
    ("Velocity", "Velocity"),
    ("Happiness", "Happiness"),
    ("Stress", "Stress"),
    
    #Cross lags, how something predicts another thing tomorrow
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

#------------------------------------------------------
#                      LAGGED CORRELATIONS
#------------------------------------------------------
print("\n------------- LAGGED CORRELATIONS -------------")
threshold = 0.45
for x_var, y_var in lagged_pairs: #Same deal as before with CORR matrix
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
#------------------------------------------------------
#                      IMPORTANT PLOTS
#------------------------------------------------------
#Make a plot of hours v. stress because it had a high r value (0.45)
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

#Make a plot of velocity v. stress because it had a high r value (0.89)
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
#------------------------------------------------------
#                      AUTOCORRELATION
#------------------------------------------------------
max_lags = len(df) - 1
fig, axes = plt.subplots(1, len(variables), figsize=(18, 5))
axes_flat = axes.flatten()
for i, var in enumerate(variables):
    plot_acf(df[var], lags=max_lags, ax=axes[i]) #you can literalyl just do plot_acf and it'll do it for you

    axes_flat[i].set_title(f"ACF - {var}")
    axes_flat[i].set_xlabel("Lags")
    axes_flat[i].set_ylabel("Autocorrelation")
    axes_flat[i].grid(True)

plt.tight_layout()
plt.show()

#------------------------------------------------------
#                     STRESS v. HOURS_t+1
#------------------------------------------------------ 
x = df['Stress']
y = df['Hours'].shift(-1)

slope, intercept, r, p, se = stats.linregress(x.drop(df.tail(lag).index), y.dropna())


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


#------------------------------------------------------
#                      VECTOR FIELD
#------------------------------------------------------
#Well, actually a quiver plot but it sounds cooler as a vector field
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
#Prediction formulas are really weird... and it gets into some hard math(well to me its hard)
#Say we want our next hours.
#Hours_t+1 = a1*Hours_t + b1*Stress_t + c1*Happiness_t + d1*Velocity_t
#If b1 is positive, high stress today correlates with MORE study horus tomorrow - which matches what we're dealing with

#finding hte best a1,b1,c1,d1 is through ordinary least squares regression
#If you don't know how this works, you take error: actual - predicted
#Then square it, so you try to minimize all the squared errors
#An algorithm loops through every possible combination of a1,b1... until the sum of errors is 0ish


#------------------------------------------------------
# OLS MODEL (LOOPED)
#------------------------------------------------------
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


# Create empty structures for our matrix components
intercepts = []
coef_matrix = []

# Order of variables must match exactly
ordered_vars = ["Hours", "Velocity", "Happiness", "Stress"]

for target_var in ordered_vars:
    X = df[ordered_vars].iloc[:-1]
    Y = df[target_var].shift(-1).iloc[:-1]
    X = sm.add_constant(X)
    model = sm.OLS(Y, X).fit()
    
    # Save the intercept (\beta_0)
    intercepts.append(model.params['const'])
    
    # Save the row of coefficients (\beta_1, \beta_2, \beta_3, \beta_4)
    row_coefs = [model.params[var] for var in ordered_vars]
    coef_matrix.append(row_coefs)

# Convert to NumPy arrays
B = np.array(coef_matrix)
c = np.array(intercepts)
I = np.eye(4)

try:
    # Solve (I - B)x = c for x
    steady_state = np.linalg.solve(I - B, c)
    
    print("\n---------------- MATHEMATICAL STEADY STATE ----------------")
    for var, val in zip(ordered_vars, steady_state):
        print(f"Optimal Steady-State {var:<10}: {val:.2f}")
    print("-----------------------------------------------------------")
except np.linalg.LinAlgError:
    print("\n[System Error]: The system has no unique steady state (Matrix is singular).")
