# ============================
# Imports
# ============================
import os
import numpy as np
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.base import BaseEstimator, TransformerMixin

from xgboost import XGBRegressor
from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor
from sklearn.svm import SVR

from rdkit import Chem
from rdkit.Chem import rdFingerprintGenerator


# ============================
# Environment Setup
# ============================
os.environ["LOKY_MAX_CPU_COUNT"] = "4"


# ============================
# Custom Transformers
# ============================
class SmilesToDescriptors(BaseEstimator, TransformerMixin):
    """Convert SMILES strings to Morgan fingerprint descriptors."""

    def __init__(self, smiles_column, radius=2, n_bits=128):
        self.smiles_column = smiles_column
        self.radius = radius
        self.n_bits = n_bits
        self.fpgen = rdFingerprintGenerator.GetMorganGenerator(
            radius=radius, fpSize=n_bits
        )

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        X = pd.DataFrame(X)
        if self.smiles_column not in X.columns:
            raise KeyError(
                f"Column '{self.smiles_column}' not found. "
                f"Available columns: {list(X.columns)}"
            )

        mols = X[self.smiles_column].apply(
            lambda x: Chem.MolFromSmiles(x) if isinstance(x, str) else None
        )

        fingerprints = []
        for mol in mols:
            if mol is not None:
                fp = self.fpgen.GetFingerprint(mol)
                fp_array = np.array(fp, dtype=int)
                fingerprints.append(fp_array)
            else:
                fingerprints.append(np.zeros(self.n_bits, dtype=int))

        fingerprint_df = pd.DataFrame(
            fingerprints, columns=[f'fp_{i}' for i in range(self.n_bits)]
        )

        X = X.drop(columns=[self.smiles_column])
        return pd.concat(
            [X.reset_index(drop=True), fingerprint_df.reset_index(drop=True)], axis=1
        )


class DataFrameSimpleImputer(BaseEstimator, TransformerMixin):
    """SimpleImputer wrapper that returns a DataFrame with column names."""

    def __init__(self, strategy="mean"):
        self.imputer = SimpleImputer(strategy=strategy)
        self.feature_names_ = None

    def fit(self, X, y=None):
        self.imputer.fit(X)
        self.feature_names_ = X.columns if isinstance(X, pd.DataFrame) else None
        return self

    def transform(self, X):
        X_imputed = self.imputer.transform(X)
        if self.feature_names_ is not None:
            return pd.DataFrame(X_imputed, columns=self.feature_names_)
        return X_imputed


class DataFrameStandardScaler(BaseEstimator, TransformerMixin):
    """StandardScaler wrapper that returns a DataFrame with column names."""

    def __init__(self):
        self.scaler = StandardScaler()
        self.feature_names_ = None

    def fit(self, X, y=None):
        self.scaler.fit(X)
        self.feature_names_ = X.columns if isinstance(X, pd.DataFrame) else None
        return self

    def transform(self, X):
        X_scaled = self.scaler.transform(X)
        if self.feature_names_ is not None:
            return pd.DataFrame(X_scaled, columns=self.feature_names_)
        return X_scaled


# ============================
# Utility Functions
# ============================
def validate_data(data):
    """Replace infinite values with NaN in numeric columns."""
    numeric_cols = data.select_dtypes(include=[np.number]).columns
    numeric_data = data[numeric_cols]
    if np.isinf(numeric_data.values).any():
        print("Warning: Infinite values found. Replacing with NaN.")
        data[numeric_cols] = numeric_data.replace([np.inf, -np.inf], np.nan)
    return data


# ============================
# Main Function
# ============================
def main():
    # ----- Load Data -----
    try:
        data = pd.read_csv(r"D:\Transfer learning project\Target domain.csv")
        print("Original data shape:", data.shape)
        print("Columns:", data.columns.tolist())
        data = validate_data(data)
    except FileNotFoundError:
        print("Error: File not found")
        return
    except Exception as e:
        print(f"Error loading data: {str(e)}")
        return

    # ----- Split Features & Targets -----
    try:
        x = data.drop(['conver', 'selec'], axis=1)
        y = data[['conver', 'selec']].fillna(data[['conver', 'selec']].mean())
    except KeyError as e:
        print(f"Error: Required columns not found. {str(e)}")
        return

    x_train, x_test, y_train, y_test = train_test_split(
        x, y, test_size=0.2, random_state=88
    )

    # ----- Feature Pipeline -----
    feature_pipeline = Pipeline([
        ('smiles_to_descriptors', SmilesToDescriptors(smiles_column='SMILES', n_bits=128)),
        ('imputer', DataFrameSimpleImputer(strategy="mean")),
        ('scaler', DataFrameStandardScaler())
    ])

    try:
        x_train_prepared = feature_pipeline.fit_transform(x_train)
        x_test_prepared = feature_pipeline.transform(x_test)
        print("\nFeature preparation completed successfully")
    except Exception as e:
        print(f"Error preparing features: {str(e)}")
        return

    # ----- Scale Targets -----
    try:
        target_scaler = StandardScaler()
        y_train_scaled = target_scaler.fit_transform(y_train)
        y_test_scaled = target_scaler.transform(y_test)
        print("Target scaling completed successfully")
    except Exception as e:
        print(f"Error scaling targets: {str(e)}")
        return

    # ----- Models -----
    models = {
        'XGBoost_conv': XGBRegressor(n_estimators=50, learning_rate=0.32, max_depth=4, random_state=78),
        'XGBoost_sel': XGBRegressor(n_estimators=50, learning_rate=0.09, max_depth=4, random_state=79),
        'GBR_conv': GradientBoostingRegressor(n_estimators=50, learning_rate=0.38, max_depth=4, random_state=74),
        'GBR_sel': GradientBoostingRegressor(n_estimators=50, learning_rate=0.25, max_depth=4, random_state=78),
        'RF_conv': RandomForestRegressor(n_estimators=50, max_depth=4, random_state=88),
        'RF_sel': RandomForestRegressor(n_estimators=50, max_depth=4, random_state=79),
        'SVR_conv': SVR(kernel='rbf', C=1, gamma='scale', epsilon=0.05),
        'SVR_sel': SVR(kernel='rbf', C=1, gamma='scale', epsilon=0.05)
    }

    # ----- Train & Evaluate -----
    results = []
    for name, model in models.items():
        target = 'conv' if 'conv' in name else 'sel'
        y_tr = y_train_scaled[:, 0] if target == 'conv' else y_train_scaled[:, 1]
        y_te = y_test_scaled[:, 0] if target == 'conv' else y_test_scaled[:, 1]

        print(f"\n=== Training {name} ===")
        try:
            model.fit(x_train_prepared, y_tr)
            train_pred = model.predict(x_train_prepared)
            test_pred = model.predict(x_test_prepared)

            metrics = {
                'Model': name,
                'Train RMSE': np.sqrt(mean_squared_error(y_tr, train_pred)),
                'Test RMSE': np.sqrt(mean_squared_error(y_te, test_pred)),
                'Train R2': r2_score(y_tr, train_pred),
                'Test R2': r2_score(y_te, test_pred)
            }
            results.append(metrics)

            print("Training completed successfully")
            print(f"  Train RMSE: {metrics['Train RMSE']:.4f}")
            print(f"  Train R2: {metrics['Train R2']:.4f}")
            print(f"  Test RMSE: {metrics['Test RMSE']:.4f}")
            print(f"  Test R2: {metrics['Test R2']:.4f}")

        except Exception as e:
            print(f"Error training {name}: {str(e)}")
            continue

    # ----- Save Results -----
    if results:
        results_df = pd.DataFrame(results)
        print("\n=== Final Results ===")
        print(results_df.to_string(index=False))
        results_df.to_csv(r"D:\Transfer learning project\model_results.csv", index=False)
        print("Results saved to model_results.csv")
    else:
        print("\nNo models were successfully trained.")


# ============================
# Entry Point
# ============================
if __name__ == "__main__":
    main()

