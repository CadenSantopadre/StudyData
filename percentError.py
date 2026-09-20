import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import statsmodels.api as sm
from sklearn.preprocessing import PolynomialFeatures
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
from scipy.stats import pearsonr

# 1. Load data
df = pd.read_csv("study.csv")

variables = [ 
    "Hours",
    "Velocity",
    "Happiness",
    "Stress"
]
min_initial_days = 5  
degrees = {"Linear": 1, "Poly Order 2": 2, "Poly Order 3": 3}

prediction_metrics = {
        var: {name: {} for name in degrees}
        for var in variables
    }

# Initialize tracking dictionaries
all_errors = {var: {name: [] for name in degrees} for var in variables}
all_outliers = {var: {name: [] for name in degrees} for var in variables}
all_dates = {var: [] for var in variables}

# Shift targets on the global dataframe ahead of time to avoid creating NaN rows inside the loop
target_shifts = {var: df[var].shift(-1) for var in variables}

# 2. Daily Backtesting Loop
for target_var in variables:
    history_dates = []
    actuals_tracker = []
    preds_tracker = {name: [] for name in degrees}

    for t in range(min_initial_days, len(df) - 1):
        df_truncated = df.iloc[:t+1]
        
        # Raw features up to day t-1
        X_train_raw = df_truncated[variables].iloc[:-1].values
        # Safe targets matching index timeline alignment
        Y_train = target_shifts[target_var].iloc[:t].values
        
        # Today's features to predict tomorrow's target
        latest_today_raw = df_truncated[variables].iloc[-1].values.reshape(1, -1)
        actual_tomorrow = df.iloc[t + 1][target_var]
        
        actuals_tracker.append(actual_tomorrow)
        history_dates.append(t + 1)
        
        # Train and project using each model configuration
        for name, degree in degrees.items():
            if degree == 1:
                X_train = X_train_raw
                latest_today = latest_today_raw
            else:
                poly = PolynomialFeatures(degree=degree, include_bias=False)
                X_train = poly.fit_transform(X_train_raw)
                latest_today = poly.transform(latest_today_raw)
                
            # Add intercept constant to design matrices
            X_train = sm.add_constant(X_train, has_constant='add')
            latest_today_with_const = np.insert(latest_today, 0, 1.0, axis=1)

            # Fit OLS
            model = sm.OLS(Y_train, X_train).fit()
            predicted_tomorrow = model.predict(latest_today_with_const.reshape(1, -1))
            preds_tracker[name].append(predicted_tomorrow[0])
            
    actuals = np.array(actuals_tracker)
    all_dates[target_var] = np.array(history_dates)

    actuals = np.array(actuals_tracker)
    all_dates[target_var] = np.array(history_dates)

    for name in degrees:
        preds = np.array(preds_tracker[name])

        # RMSE
        rmse = np.sqrt(mean_squared_error(actuals, preds))

        #MAE
        mae = mean_absolute_error(actuals, preds)

        # Pearson correlation r
        r, p_value = pearsonr(actuals, preds)

        # R²
        r2 = r2_score(actuals, preds)

        prediction_metrics[target_var][name] = {
            "RMSE": rmse,
            "MAE": mae,
            "r": r,
            "R²": r2,
            "p-value": p_value
        }
    
    # 3. STATISTICAL OUTLIER DETECTION per model type using global IQR
    for name in degrees:
        preds = np.array(preds_tracker[name])
        percent_errors = ((preds - actuals) / (actuals + 1e-9)) * 100
        
        # Non-Rolling global IQR bounds calculated unique to each model trace matrix
        q1 = np.percentile(percent_errors, 25)
        q3 = np.percentile(percent_errors, 75)
        iqr = q3 - q1
        
        lower_fence = q1 - 1.5 * iqr
        upper_fence = q3 + 1.5 * iqr  # Fixed upper fence bounds logic
        
        outlier_mask = (percent_errors < lower_fence) | (percent_errors > upper_fence)
        
        all_errors[target_var][name] = percent_errors
        all_outliers[target_var][name] = outlier_mask

# 4. Plot Results: Sequential 2x2 grids per model configuration
colors = {"Linear": "purple", "Poly Order 2": "blue", "Poly Order 3": "teal"}

for name in degrees:
    # Open a single distinct window for this model type
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    axes = axes.flatten()
    
    for i, target_var in enumerate(variables):
        ax = axes[i]
        dates = all_dates[target_var]
        errors = all_errors[target_var][name]
        outlier_mask = all_outliers[target_var][name]
        
        # Omit outliers to create broken lines on error anomalies
        clean_errors = np.where(~outlier_mask, errors, np.nan)
        ax.plot(dates, clean_errors, marker='o', color=colors[name], 
                linestyle='-', linewidth=1.5, markersize=4, label=f'{name} Error')
        
        # Highlight individual outliers exactly on the Identity/0 Line
        outlier_dates = dates[outlier_mask]
        
        if len(outlier_dates) > 0:
            zero_line_y = np.zeros_like(outlier_dates, dtype=float)
            ax.scatter(outlier_dates, zero_line_y, facecolors='none', edgecolors='red', 
                       linestyle='--', linewidths=1.5, s=90, alpha=0.9, label='IQR Outlier')

        ax.axhline(0, color='black', linestyle='--', alpha=0.5)
        ax.set_title(f"{target_var} Error Profile", fontsize=11, fontweight='bold')
        ax.set_xlabel("Timeline Index (Day)", fontsize=9)
        ax.set_ylabel("Percent Error (%)", fontsize=9)
        ax.grid(True, linestyle=':', alpha=0.5)
        ax.legend(loc='best', fontsize=8)
        
    plt.suptitle(f"Global Layout Analysis — {name} Configuration", fontsize=14, fontweight='bold', y=0.98)
    plt.tight_layout()
    plt.show()

print("\n=== OUT-OF-SAMPLE PREDICTION PERFORMANCE ===")

for target_var in variables:
    print(f"\n{target_var}")
    for name in degrees:
        metrics = prediction_metrics[target_var][name]

        print(
            f"{name:15s} | "
            f"RMSE = {metrics['RMSE']:.4f} | "
            f"MAE = {metrics['MAE']:.4f} | "
            f"r = {metrics['r']:.4f} | "
            f"R² = {metrics['R²']:.4f} | "
            f"p = {metrics['p-value']:.4f}"
        )
