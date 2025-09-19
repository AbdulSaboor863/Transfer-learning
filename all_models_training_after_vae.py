import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, r2_score
import xgboost as xgb
import matplotlib.pyplot as plt
import warnings
from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor
from sklearn.svm import SVR

# Configure warnings and matplotlib
plt.rcParams['figure.max_open_warning'] = 100  # Increased from default 50
warnings.filterwarnings('ignore', category=RuntimeWarning)  # Suppress figure warnings
warnings.filterwarnings('ignore', category=UserWarning)  # Suppress ylim warnings

# Load data
try:
    data = pd.read_csv(r"D:\Transfer_Learning_Output\combined_target_data.csv")
except FileNotFoundError:
    print(r"Error: File not found at D:\Transfer_Learning_Output\combined_target_data.csv")
    exit(1)

# Define targets
conversion_scaled = 'conversion_scaled'
selectivity_scaled = 'selectivity_scaled'

# Separate features and targets
X = data.drop(columns=[conversion_scaled, selectivity_scaled])
y1 = data[conversion_scaled]
y2 = data[selectivity_scaled]

# Verify feature count
if X.shape[1] != 41:
    print(f"Warning: Expected 41 features, found {X.shape[1]}.")

# Split for Conversion (random_state=80)
X_train_conv, X_test_conv, y1_train, y1_test = train_test_split(
    X, y1, test_size=0.2, random_state=55
)

# Split for Selectivity (random_state=104)
X_train_sel, X_test_sel, y2_train, y2_test = train_test_split(
    X, y2, test_size=0.2, random_state=36
)

# Separate scalers for each target
scaler_conv = StandardScaler()
X_train_conv_scaled = scaler_conv.fit_transform(X_train_conv)
X_test_conv_scaled = scaler_conv.transform(X_test_conv)

scaler_sel = StandardScaler()
X_train_sel_scaled = scaler_sel.fit_transform(X_train_sel)
X_test_sel_scaled = scaler_sel.transform(X_test_sel)

# XGBoost Parameter grids
param_grid1 = {
    'n_estimators': [120, 140, 160],
    'learning_rate': [0.15, 0.55, 0.89],
    'max_depth': [2, 4, 6],
    'subsample': [0.8, 1.5, 2.0],
    'colsample_bytree': [0.8, 1.5, 2.0],
    'reg_alpha': [0.8, 2.0, 2.5],
    'reg_lambda': [0.5, 1.5, 2.5]
}

param_grid2 = {
    'n_estimators': [60, 100, 140],
    'learning_rate': [0.17, 0.55, 0.64],
    'max_depth': [3, 4, 6],
    'subsample': [0.8, 1.2, 1.9],
    'colsample_bytree': [0.8, 1.8, 2.5],
    'reg_alpha': [0.2, 0.6, 0.9],
    'reg_lambda': [0.3, 0.6, 0.9]
}

# XGBoost Model for Conversion
model1 = xgb.XGBRegressor(
    objective='reg:squarederror',
    random_state=71,
    early_stopping_rounds=10,
    eval_metric='rmse'
)
grid_search1 = GridSearchCV(
    model1, param_grid1, scoring='r2', cv=5, n_jobs=-1, verbose=1
)
grid_search1.fit(
    X_train_conv_scaled, y1_train,
    eval_set=[(X_test_conv_scaled, y1_test)],
    verbose=False
)
model1 = grid_search1.best_estimator_

# XGBoost Model for Selectivity
model2 = xgb.XGBRegressor(
    objective='reg:squarederror',
    random_state=74,
    early_stopping_rounds=10,
    eval_metric='rmse'
)
grid_search2 = GridSearchCV(
    model2, param_grid2, scoring='r2', cv=5, n_jobs=-1, verbose=1
)
grid_search2.fit(
    X_train_sel_scaled, y2_train,
    eval_set=[(X_test_sel_scaled, y2_test)],
    verbose=False
)
model2 = grid_search2.best_estimator_

# Evaluate XGBoost Conversion
y1_train_pred = model1.predict(X_train_conv_scaled)
y1_test_pred = model1.predict(X_test_conv_scaled)
print("\nConversion Model Performance (XGBoost):")
print(f"Train R²: {r2_score(y1_train, y1_train_pred):.4f}")
print(f"Test R²: {r2_score(y1_test, y1_test_pred):.4f}")
print(f"Test RMSE: {np.sqrt(mean_squared_error(y1_test, y1_test_pred)):.4f}")
print(f"Train RMSE: {np.sqrt(mean_squared_error(y1_train, y1_train_pred)):.4f}")
print("Best parameters:", grid_search1.best_params_)
print("Best CV score:", grid_search1.best_score_)

# Evaluate XGBoost Selectivity
y2_train_pred = model2.predict(X_train_sel_scaled)
y2_test_pred = model2.predict(X_test_sel_scaled)
print("\nSelectivity Model Performance (XGBoost):")
print(f"Train R²: {r2_score(y2_train, y2_train_pred):.4f}")
print(f"Test R²: {r2_score(y2_test, y2_test_pred):.4f}")
print(f"Test RMSE: {np.sqrt(mean_squared_error(y2_test, y2_test_pred)):.4f}")
print(f"Train RMSE: {np.sqrt(mean_squared_error(y2_train, y2_train_pred)):.4f}")
print("Best parameters:", grid_search2.best_params_)
print("Best CV score:", grid_search2.best_score_)

# GBR Parameter grids
param_grid_gbr1 = {
    'n_estimators': [100, 120, 140],
    'learning_rate': [0.61, 0.77, 0.88],
    'max_depth': [2, 3, 4],
    'subsample': [0.8, 1, 1.5]
}

param_grid_gbr2 = {
    'n_estimators': [130, 170, 180],
    'learning_rate': [0.11, 0.23, 0.37],
    'max_depth': [2, 3, 4],
    'subsample': [0.9, 1.2, 1.5]
}

# GBR Model for Conversion
model_gbr1 = GradientBoostingRegressor(
    random_state=71
)
grid_search_gbr1 = GridSearchCV(
    model_gbr1, param_grid_gbr1, scoring='r2', cv=5, n_jobs=-1, verbose=1
)
grid_search_gbr1.fit(
    X_train_conv_scaled, y1_train
)
model_gbr1 = grid_search_gbr1.best_estimator_

# GBR Model for Selectivity
model_gbr2 = GradientBoostingRegressor(
    random_state=74
)
grid_search_gbr2 = GridSearchCV(
    model_gbr2, param_grid_gbr2, scoring='r2', cv=5, n_jobs=-1, verbose=1
)
grid_search_gbr2.fit(
    X_train_sel_scaled, y2_train
)
model_gbr2 = grid_search_gbr2.best_estimator_

# Evaluate GBR Conversion
y1_train_pred_gbr = model_gbr1.predict(X_train_conv_scaled)
y1_test_pred_gbr = model_gbr1.predict(X_test_conv_scaled)
print("\nGBR Conversion Model Performance:")
print(f"Train R²: {r2_score(y1_train, y1_train_pred_gbr):.4f}")
print(f"Test R²: {r2_score(y1_test, y1_test_pred_gbr):.4f}")
print(f"Test RMSE: {np.sqrt(mean_squared_error(y1_test, y1_test_pred_gbr)):.4f}")
print(f"Train RMSE: {np.sqrt(mean_squared_error(y1_train, y1_train_pred_gbr)):.4f}")
print("Best parameters:", grid_search_gbr1.best_params_)
print("Best CV score:", grid_search_gbr1.best_score_)

# Evaluate GBR Selectivity
y2_train_pred_gbr = model_gbr2.predict(X_train_sel_scaled)
y2_test_pred_gbr = model_gbr2.predict(X_test_sel_scaled)
print("\nGBR Selectivity Model Performance:")
print(f"Train R²: {r2_score(y2_train, y2_train_pred_gbr):.4f}")
print(f"Test R²: {r2_score(y2_test, y2_test_pred_gbr):.4f}")
print(f"Test RMSE: {np.sqrt(mean_squared_error(y2_test, y2_test_pred_gbr)):.4f}")
print(f"Train RMSE: {np.sqrt(mean_squared_error(y2_train, y2_train_pred_gbr)):.4f}")
print("Best parameters:", grid_search_gbr2.best_params_)
print("Best CV score:", grid_search_gbr2.best_score_)

# RF Parameter grids
param_grid_rf1 = {
    'n_estimators': [60, 90, 100],
    'max_depth': [2,3,4],
    'min_samples_split': [2, 4, 7],
    'min_samples_leaf': [0, 1, 2]
}

param_grid_rf2 = {
    'n_estimators': [60, 80, 100],
    'max_depth': [2, 3, 4],
    'min_samples_split': [2, 3, 4],
    'min_samples_leaf': [0, 1, 2]
}

# RF Model for Conversion
model_rf1 = RandomForestRegressor(
    random_state=71
)
grid_search_rf1 = GridSearchCV(
    model_rf1, param_grid_rf1, scoring='r2', cv=5, n_jobs=-1, verbose=1
)
grid_search_rf1.fit(
    X_train_conv_scaled, y1_train
)
model_rf1 = grid_search_rf1.best_estimator_

# RF Model for Selectivity
model_rf2 = RandomForestRegressor(
    random_state=74
)
grid_search_rf2 = GridSearchCV(
    model_rf2, param_grid_rf2, scoring='r2', cv=5, n_jobs=-1, verbose=1
)
grid_search_rf2.fit(
    X_train_sel_scaled, y2_train
)
model_rf2 = grid_search_rf2.best_estimator_

# Evaluate RF Conversion
y1_train_pred_rf = model_rf1.predict(X_train_conv_scaled)
y1_test_pred_rf = model_rf1.predict(X_test_conv_scaled)
print("\nRF Conversion Model Performance:")
print(f"Train R²: {r2_score(y1_train, y1_train_pred_rf):.4f}")
print(f"Test R²: {r2_score(y1_test, y1_test_pred_rf):.4f}")
print(f"Test RMSE: {np.sqrt(mean_squared_error(y1_test, y1_test_pred_rf)):.4f}")
print(f"Train RMSE: {np.sqrt(mean_squared_error(y1_train, y1_train_pred_rf)):.4f}")
print("Best parameters:", grid_search_rf1.best_params_)
print("Best CV score:", grid_search_rf1.best_score_)

# Evaluate RF Selectivity
y2_train_pred_rf = model_rf2.predict(X_train_sel_scaled)
y2_test_pred_rf = model_rf2.predict(X_test_sel_scaled)
print("\nRF Selectivity Model Performance:")
print(f"Train R²: {r2_score(y2_train, y2_train_pred_rf):.4f}")
print(f"Test R²: {r2_score(y2_test, y2_test_pred_rf):.4f}")
print(f"Test RMSE: {np.sqrt(mean_squared_error(y2_test, y2_test_pred_rf)):.4f}")
print(f"Train RMSE: {np.sqrt(mean_squared_error(y2_train, y2_train_pred_rf)):.4f}")
print("Best parameters:", grid_search_rf2.best_params_)
print("Best CV score:", grid_search_rf2.best_score_)

# SVR Parameter grids
param_grid_svr1 = {
    'kernel': ['rbf'],
    'C': [1.0, 2.0, 4,0],
    'epsilon': [0.1, 0.2, 0.3]
}

param_grid_svr2 = {
    'kernel': ['rbf', 'linear', 'poly'],
    'C': [0.15, 1.0, 10.5],
    'epsilon': [0.05, 0.1, 0.25]
}

# SVR Model for Conversion
model_svr1 = SVR()
grid_search_svr1 = GridSearchCV(
    model_svr1, param_grid_svr1, scoring='r2', cv=5, n_jobs=-1, verbose=1
)
grid_search_svr1.fit(
    X_train_conv_scaled, y1_train
)
model_svr1 = grid_search_svr1.best_estimator_

# SVR Model for Selectivity
model_svr2 = SVR()
grid_search_svr2 = GridSearchCV(
    model_svr2, param_grid_svr2, scoring='r2', cv=5, n_jobs=-1, verbose=1
)
grid_search_svr2.fit(
    X_train_sel_scaled, y2_train
)
model_svr2 = grid_search_svr2.best_estimator_

# Evaluate SVR Conversion
y1_train_pred_svr = model_svr1.predict(X_train_conv_scaled)
y1_test_pred_svr = model_svr1.predict(X_test_conv_scaled)
print("\nSVR Conversion Model Performance:")
print(f"Train R²: {r2_score(y1_train, y1_train_pred_svr):.4f}")
print(f"Test R²: {r2_score(y1_test, y1_test_pred_svr):.4f}")
print(f"Test RMSE: {np.sqrt(mean_squared_error(y1_test, y1_test_pred_svr)):.4f}")
print(f"Train RMSE: {np.sqrt(mean_squared_error(y1_train, y1_train_pred_svr)):.4f}")
print("Best parameters:", grid_search_svr1.best_params_)
print("Best CV score:", grid_search_svr1.best_score_)

# Evaluate SVR Selectivity
y2_train_pred_svr = model_svr2.predict(X_train_sel_scaled)
y2_test_pred_svr = model_svr2.predict(X_test_sel_scaled)
print("\nSVR Selectivity Model Performance:")
print(f"Train R²: {r2_score(y2_train, y2_train_pred_svr):.4f}")
print(f"Test R²: {r2_score(y2_test, y2_test_pred_svr):.4f}")
print(f"Test RMSE: {np.sqrt(mean_squared_error(y2_test, y2_test_pred_svr)):.4f}")
print(f"Train RMSE: {np.sqrt(mean_squared_error(y2_train, y2_train_pred_svr)):.4f}")
print("Best parameters:", grid_search_svr2.best_params_)
print("Best CV score:", grid_search_svr2.best_score_)

print("\nAll analysis completed successfully.")
