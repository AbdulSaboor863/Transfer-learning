import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, r2_score
import xgboost as xgb
import matplotlib.pyplot as plt
import shap
import warnings

# Configure warnings and matplotlib
plt.rcParams['figure.max_open_warning'] = 100  # Increased from default 50
warnings.filterwarnings('ignore', category=RuntimeWarning)  # Suppress figure warnings
warnings.filterwarnings('ignore', category=UserWarning)  # Suppress ylim warnings

# Load data
try:
    data = pd.read_csv(r"D:/Transfer_Learning_Output/combined_target_data.csv")
except FileNotFoundError:
    print(r"Error: File not found at D:/Transfer learning project/Best codes and combined target datafile/combined_target_data.csv")
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

# Split for Conversion (random_state=71)
X_train_conv, X_test_conv, y1_train, y1_test = train_test_split(
    X, y1, test_size=0.2, random_state=71
)

# Split for Selectivity (random_state=74)
X_train_sel, X_test_sel, y2_train, y2_test = train_test_split(
    X, y2, test_size=0.2, random_state=74
)

# Separate scalers for each target
scaler_conv = StandardScaler()
X_train_conv_scaled = scaler_conv.fit_transform(X_train_conv)
X_test_conv_scaled = scaler_conv.transform(X_test_conv)

scaler_sel = StandardScaler()
X_train_sel_scaled = scaler_sel.fit_transform(X_train_sel)
X_test_sel_scaled = scaler_sel.transform(X_test_sel)

# Parameter grids
param_grid1 = {
    'n_estimators': [120, 160, 180],
    'learning_rate': [0.53, 0.77, 0.83],
    'max_depth': [3, 5, 7],
    'subsample': [1.0, 2.0, 2.5],
    'colsample_bytree': [1.0, 2.5, 3.5],
    'reg_alpha': [0.48, 2, 3],
    'reg_lambda': [1, 2, 3]
}

param_grid2 = {
    'n_estimators': [60, 120, 180],
    'learning_rate': [0.78, 0.89, 0.93],
    'max_depth': [4, 6, 7],
    'subsample': [1.0, 2.5, 2.9],
    'colsample_bytree': [0.9, 1.8, 2.5],
    'reg_alpha': [0.8, 2.0, 3.5],
    'reg_lambda': [0.5, 1.5, 2.5]
}

# Model for Conversion
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

# Model for Selectivity
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

# Evaluate Conversion
y1_train_pred = model1.predict(X_train_conv_scaled)
y1_test_pred = model1.predict(X_test_conv_scaled)
print("\nConversion Model Performance:")
print(f"Train R²: {r2_score(y1_train, y1_train_pred):.4f}")
print(f"Test R²: {r2_score(y1_test, y1_test_pred):.4f}")
print(f"Test RMSE: {np.sqrt(mean_squared_error(y1_test, y1_test_pred)):.4f}")
print(f"Train RMSE: {np.sqrt(mean_squared_error(y1_train, y1_train_pred)):.4f}")
print("Best parameters:", grid_search1.best_params_)
print("Best CV score:", grid_search1.best_score_)

# Evaluate Selectivity
y2_train_pred = model2.predict(X_train_sel_scaled)
y2_test_pred = model2.predict(X_test_sel_scaled)
print("\nSelectivity Model Performance:")
print(f"Train R²: {r2_score(y2_train, y2_train_pred):.4f}")
print(f"Test R²: {r2_score(y2_test, y2_test_pred):.4f}")
print(f"Test RMSE: {np.sqrt(mean_squared_error(y2_test, y2_test_pred)):.4f}")
print(f"Train RMSE: {np.sqrt(mean_squared_error(y2_train, y2_train_pred)):.4f}")
print("Best parameters:", grid_search2.best_params_)
print("Best CV score:", grid_search2.best_score_)

# Create scatter plot DataFrames and save to CSV
conv_scatter_df = pd.DataFrame({
    'Actual': pd.concat([y1_train, y1_test]),
    'Predicted': np.concatenate([y1_train_pred, y1_test_pred]),
    'Dataset': ['train'] * len(y1_train) + ['test'] * len(y1_test)
})
conv_scatter_df.to_csv(r"D:\Transfer_Learning_Output\conversion_combined_scatter.csv", index=False)

sel_scatter_df = pd.DataFrame({
    'Actual': pd.concat([y2_train, y2_test]),
    'Predicted': np.concatenate([y2_train_pred, y2_test_pred]),
    'Dataset': ['train'] * len(y2_train) + ['test'] * len(y2_test)
})
sel_scatter_df.to_csv(r"D:\Transfer_Learning_Output\selectivity_combined_scatter.csv", index=False)

print("\nScatter plot data saved successfully.")


def create_error_csv(actual_train, predicted_train, actual_test, predicted_test, target_name):
    actual = pd.concat([actual_train, actual_test])
    predicted = np.concatenate([predicted_train, predicted_test])
    residuals = actual - predicted
    error_df = pd.DataFrame({
        f'Actual_{target_name}': actual,
        'Predicted': predicted,
        'Error': residuals,
        'Absolute_Error': np.abs(residuals),
        'Dataset': ['train'] * len(actual_train) + ['test'] * len(actual_test)
    })
    error_df = error_df.sort_values(f'Actual_{target_name}')
    error_df.to_csv(f"D:\\Transfer_Learning_Output\\{target_name}_error_distribution.csv", index=False)
    return error_df


conv_error_df = create_error_csv(y1_train, y1_train_pred, y1_test, y1_test_pred, "conversion")
sel_error_df = create_error_csv(y2_train, y2_train_pred, y2_test, y2_test_pred, "selectivity")

print("\nError distribution CSV files created successfully for:")
print(f"- conversion_error_distribution.csv")
print(f"- selectivity_error_distribution.csv")

# SHAP Analysis
print("\nComputing SHAP values for Conversion...")
explainer_conv = shap.TreeExplainer(model1)
shap_values_conv = explainer_conv.shap_values(X_train_conv_scaled)
feature_names = X.columns.tolist()
non_latent_indices = [i for i, name in enumerate(feature_names) if
                      not (name.startswith('latent_pca') or name.startswith('pca_fp'))]
non_latent_features = [feature_names[i] for i in non_latent_indices]
shap_values_conv_filtered = shap_values_conv[:, non_latent_indices]

# SHAP Summary Plot
fig = plt.figure(figsize=(10, 6))
shap.summary_plot(shap_values_conv_filtered, X_train_conv_scaled[:, non_latent_indices],
                  feature_names=non_latent_features, show=False)
plt.title("SHAP Summary Plot for Conversion")
plt.tight_layout()
plt.savefig(r"D:\Transfer_Learning_Output\shap_summary_conversion.png")
plt.close(fig)

# Only these features for SHAP dependence plots
selected_features = ['Loading', 'SSA', 'DP', 'TPV', 'D_M', 'Temp', 'Time', 'W_cat', 'P', 'SV']

# SHAP Dependence Plots for Conversion (only selected features)
for feature in selected_features:
    if feature in non_latent_features:
        idx = non_latent_features.index(feature)

        fig = plt.figure(figsize=(10, 6))
        shap.dependence_plot(
            idx,
            shap_values_conv_filtered,
            X_train_conv_scaled[:, non_latent_indices],
            feature_names=non_latent_features,
            interaction_index=None,
            show=False
        )
        plt.title(f"SHAP Dependence: {feature} (Conversion)")
        plt.tight_layout()
        plt.savefig(f"D:\\Transfer_Learning_Output\\shap_dependence_conv_{feature}.png")
        plt.close(fig)

        # Store data for CSV
        df = pd.DataFrame({
            'feature_value': X_train_conv_scaled[:, non_latent_indices][:, idx],
            'shap_value': shap_values_conv_filtered[:, idx]
        })
        df.to_csv(f"D:\\Transfer_Learning_Output\\shap_dependence_conv_{feature}.csv", index=False)

print("\nComputing SHAP values for Selectivity...")
explainer_sel = shap.TreeExplainer(model2)
shap_values_sel = explainer_sel.shap_values(X_train_sel_scaled)
shap_values_sel_filtered = shap_values_sel[:, non_latent_indices]

# SHAP Summary Plot
fig = plt.figure(figsize=(10, 6))
shap.summary_plot(shap_values_sel_filtered, X_train_sel_scaled[:, non_latent_indices],
                  feature_names=non_latent_features, show=False)
plt.title("SHAP Summary Plot for Selectivity")
plt.tight_layout()
plt.savefig(r"D:\Transfer_Learning_Output\shap_summary_selectivity.png")
plt.close(fig)

# SHAP Dependence Plots for Selectivity (only selected features)
for feature in selected_features:
    if feature in non_latent_features:
        idx = non_latent_features.index(feature)

        fig = plt.figure(figsize=(10, 6))
        shap.dependence_plot(
            idx,
            shap_values_sel_filtered,
            X_train_sel_scaled[:, non_latent_indices],
            feature_names=non_latent_features,
            interaction_index=None,
            show=False
        )
        plt.title(f"SHAP Dependence: {feature} (Selectivity)")
        plt.tight_layout()
        plt.savefig(f"D:\\Transfer_Learning_Output\\shap_dependence_sel_{feature}.png")
        plt.close(fig)

        # Store data for CSV
        df = pd.DataFrame({
            'feature_value': X_train_sel_scaled[:, non_latent_indices][:, idx],
            'shap_value': shap_values_sel_filtered[:, idx]
        })
        df.to_csv(f"D:\\Transfer_Learning_Output\\shap_dependence_sel_{feature}.csv", index=False)

print("\nSHAP analysis plots and data saved successfully.")

# Feature Importance
print("\nComputing feature importance for Conversion...")
feature_importance_conv = model1.feature_importances_
feature_importance_conv_filtered = feature_importance_conv[non_latent_indices]
importance_df_conv = pd.DataFrame({
    'Feature': non_latent_features,
    'Importance': feature_importance_conv_filtered
})
importance_df_conv = importance_df_conv.sort_values(by='Importance', ascending=False)
importance_df_conv.to_csv(r"D:\Transfer_Learning_Output\conversion_feature_importance.csv", index=False)

print("\nComputing feature importance for Selectivity...")
feature_importance_sel = model2.feature_importances_
feature_importance_sel_filtered = feature_importance_sel[non_latent_indices]
importance_df_sel = pd.DataFrame({
    'Feature': non_latent_features,
    'Importance': feature_importance_sel_filtered
})
importance_df_sel = importance_df_sel.sort_values(by='Importance', ascending=False)
importance_df_sel.to_csv(r"D:\Transfer_Learning_Output\selectivity_feature_importance.csv", index=False)

print("\nAll analysis completed successfully.")
