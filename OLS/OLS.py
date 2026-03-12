import pandas as pd
import statsmodels.api as sm
import os
import json
from itertools import combinations

# Define Paths
current_dir = os.path.dirname(os.path.abspath(__file__))
data_path1 = os.path.join(current_dir, "..", "data", "model_ready.csv")

# Load Dataset
df = pd.read_csv(data_path1)

dependent_var = 'feed_p_ppm'
independent_vars = ['feed_ffa_pct', 'feed_mi_pct', 'feed_iv', 'feed_dobi', 'feed_car_pv']

# Define fixed independent variables
fixed_vars = ['time_trend', 'missing_transition_phase'] + [f'month_{i}' for i in range(2, 13)]

# Prepare Data
df_reg = df[[dependent_var] + independent_vars + fixed_vars]

y = df_reg[dependent_var]

# ---------------------------------------------------------
# Step 1a: Initial Regression with 5 Independent Variables
# ---------------------------------------------------------
X_orig = df_reg[independent_vars]
X_orig = sm.add_constant(X_orig)
model_orig = sm.OLS(y, X_orig).fit()

out_orig_path = os.path.join(current_dir, "OLSresult_Original.csv")
with open(out_orig_path, "w") as f:
    f.write(f'"Original Variables: {", ".join(independent_vars)}"\n')
    f.write(f'"Adjusted R-squared: {model_orig.rsquared_adj:.4f}"\n\n')
    f.write(model_orig.summary().as_csv())
print(f"Original OLS results saved to {os.path.abspath(out_orig_path)}")

# ---------------------------------------------------------
# Step 1b: Combinations Regression with Fixed Validation
# ---------------------------------------------------------
best_adj_r2 = -float('inf')
best_model = None
best_vars = []
run_log = []

# Iterate through all possible combinations of independent variables
for r in range(0, len(independent_vars) + 1):
    for combo in combinations(independent_vars, r):
        current_vars = fixed_vars + list(combo)
        X_combo = df_reg[current_vars]
        # Add constant to features
        X_combo = sm.add_constant(X_combo)
        
        # Fit OLS Model
        model = sm.OLS(y, X_combo).fit()
        
        # Check if it has the best adjusted R-squared
        if model.rsquared_adj > best_adj_r2:
            best_adj_r2 = model.rsquared_adj
            best_model = model
            best_vars = current_vars
            
        run_log.append({
            "variables": current_vars,
            "rsquared": model.rsquared,
            "rsquared_adj": model.rsquared_adj,
            "aic": model.aic,
            "bic": model.bic
        })

# Output Results
out_ts_path = os.path.join(current_dir, "OLSresult_Time_Series.csv")
with open(out_ts_path, "w") as f:
    # Wrap in quotes so it occupies a single cell in the CSV
    f.write(f'"Best Independent Variables Selected: {", ".join(best_vars)}"\n')
    f.write(f'"Highest Adjusted R-squared: {best_adj_r2:.4f}"\n\n')
    f.write(best_model.summary().as_csv())

print(f"Time Series OLS model selected with vars: {best_vars}")
print(f"Results successfully saved to {os.path.abspath(out_ts_path)}")

log_ts_path = os.path.join(current_dir, "OLSresult_Time_Series_log.json")
with open(log_ts_path, "w") as f:
    json.dump(run_log, f, indent=4)
print(f"Running log successfully saved to {os.path.abspath(log_ts_path)}")

# ---------------------------------------------------------
# Step 2: Incorporate Lagged Variable (feed_p_ppm_lag1)
# ---------------------------------------------------------

# Ensure 'date' is a datetime object
if 'date' in df.columns:
    df['date'] = pd.to_datetime(df['date'])
    
    # Sort by date to ensure proper lag calculation
    df = df.sort_values(by='date').reset_index(drop=True)
    
    # Create the lag variable for the dependent variable
    df['feed_p_ppm_lag1'] = df[dependent_var].shift(1)
    
    # Calculate difference in days between consecutive rows
    df['date_diff'] = df['date'].diff().dt.days
    
    # If the date difference is greater than 2 days, set lag to NaN
    df.loc[df['date_diff'] > 2, 'feed_p_ppm_lag1'] = pd.NA
else:
    # Fallback if there is no 'date' column (assuming data is already sorted and consecutive)
    df['feed_p_ppm_lag1'] = df[dependent_var].shift(1)

# Remove rows containing NaN in the lag variable (includes the first row, and disconnected rows)
df_lag = df.dropna(subset=['feed_p_ppm_lag1']).copy()

# Optional: drop the date_diff column if it was created
if 'date_diff' in df_lag.columns:
    df_lag = df_lag.drop(columns=['date_diff'])

# Save the new dataset
data_lag_path = os.path.join(current_dir, "..", "data", "model_ready_lag.csv")
df_lag.to_csv(data_lag_path, index=False)
print(f"Dataset with lag variable saved to {os.path.abspath(data_lag_path)}")

# Prepare new features list including the lagged variable
lag_independent_vars = best_vars + ['feed_p_ppm_lag1']

# Prepare new regression data
y_lag = df_lag[dependent_var]
X_lag = df_lag[lag_independent_vars]
X_lag = sm.add_constant(X_lag)

# Fit OLS Model with the lagged variable
model_lag = sm.OLS(y_lag, X_lag).fit()

# Output Results for the lag model
out_lag_path = os.path.join(current_dir, "OLSresult_lag.csv")
with open(out_lag_path, "w") as f:
    # Wrap in quotes so it occupies a single cell in the CSV
    f.write(f'"Model Variables (Optimal + Lag): {", ".join(lag_independent_vars)}"\n')
    f.write(f'"Adjusted R-squared: {model_lag.rsquared_adj:.4f}"\n\n')
    f.write(model_lag.summary().as_csv())

print(f"Lag model OLS results successfully saved to {os.path.abspath(out_lag_path)}")
