import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_squared_error, r2_score

df = pd.read_csv("study.csv")

variables = ["Hours", "Velocity", "Happiness", "Stress"]

X = df[variables].iloc[:-1]
Y = df[variables].shift(-1).iloc[:-1]

model = LinearRegression()
model.fit(X, Y)

A = model.coef_
b = model.intercept_

print("A matrix:")
print(pd.DataFrame(A, index=variables, columns=variables))
#this gives a matrix of relationships:
#Suppose: 
#              Hours  Velocity  Happiness    Stress
#Hours      0.307736  1.535595   0.191882 -0.361671
#Velocity  -0.037671  0.665752   0.246446 -0.149747
#Happiness -0.364941 -0.464152   0.082586  0.064304
#Stress     0.235414  2.405885   0.535228 -0.955344

#Row Velocity - Col Happiness means how much tomorrows V changes per unit of Happiness; ceteris paribus

print("")
print("Baseline vector:")
print(pd.Series(b, index=variables))

eigenvalues = np.linalg.eigvals(A)

print("")
print(eigenvalues)
print("Magnitudes:", np.abs(eigenvalues))
#Eigenvalues tell us how something tends toward the baseline
#Suppose:
#[0.3790312 +0.50285468j , 0.3790312 -0.50285468j , -0.60549279+0.j , -0.05183951+0.j]
#Magnitudes: [0.62970428 0.62970428 0.60549279 0.05183951]

#All magnitudes < 1, so everything reverts toward the baseline
#λ1,2 = 0.379 + 0.502j = Damped 53 degree spiral
#λ3 = -0.6 = Oscillation, drops by 40% each step
#λ4 = -0.05 = Fast drop, drops by 95% in the first step 

equilibrium = np.linalg.solve(
    np.eye(len(variables)) - A,
    b
)
print("")
print(pd.Series(equilibrium, index=variables))

#To solve the steady state, we look to linear algebra
#x_t+1 = Ax_t + b
#x = Ax +b
#(I-A)x = b

#Suppose:
#Hours        4.773666
#Velocity     2.721763
#Happiness    4.662390
#Stress       3.200463
#this means our baseline is 4.77, 2.72, 4.66, 3.2

#Now we're going to test if it is nonlinear

model = make_pipeline(
    PolynomialFeatures(degree=2, include_bias=False),
    Ridge(alpha=1.0)
)
model.fit(X, Y)
predictions = model.predict(X)


poly_step = model.named_steps['polynomialfeatures']
ridge_step = model.named_steps['ridge']

feature_names = poly_step.get_feature_names_out(input_features=variables)

print("=== Overall System Performance ===")
print(f"Total System R²: {r2_score(Y, predictions):.4f}")
print(
    f"Total System RMSE: {np.sqrt(mean_squared_error(Y, predictions)):.4f}\n"
)

for i, target in enumerate(variables):
    print(f"\n--- Target Variable: {target} ---")

    y_true = Y.iloc[:, i]
    y_pred = predictions[:, i]

    print(f"R² Score:  {r2_score(y_true, y_pred):.4f}")
    print(f"RMSE:      {np.sqrt(mean_squared_error(y_true, y_pred)):.4f}")

    print(f"Intercept: {ridge_step.intercept_[i]:.4f}")
    print("Coefficients:")
    for feature, coef in zip(feature_names, ridge_step.coef_[i]):
        print(f"  {feature}: {coef:.4f}")