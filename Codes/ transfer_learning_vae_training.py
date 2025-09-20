import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader, TensorDataset
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.decomposition import PCA
from sklearn.impute import SimpleImputer
from rdkit import Chem
from rdkit.Chem import rdFingerprintGenerator
import os
import rdkit

SMILES_COL = 5

# VAE Loss with Beta
def vae_loss(recon_x, x, mu, logvar, beta=0.8):
    recon_loss = nn.MSELoss(reduction='sum')(recon_x, x)
    kl = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
    return recon_loss + beta * kl

# Data Loading
def load_and_clean_data():
    print("Loading data...")
    print(f"RDKit version: {rdkit.__version__}")
    try:
        source_df = pd.read_csv("D:/Transfer learning project/Source domain.csv", usecols=range(26))
        target_df = pd.read_csv("D:/Transfer learning project/Target domain.csv",
                                encoding='latin1', usecols=range(26))
    except Exception as e:
        print(f"Error loading CSV: {e}")
        raise

    expected_source_rows = 148
    expected_target_rows = 89
    print(f"Source data rows: {len(source_df)} (expected {expected_source_rows})")
    print(f"Target data rows: {len(target_df)} (expected {expected_target_rows})")

    if len(source_df) != expected_source_rows:
        raise ValueError(f"Source data has {len(source_df)} rows, expected {expected_source_rows}")
    if len(target_df) != expected_target_rows:
        raise ValueError(f"Target data has {len(target_df)} rows, expected {expected_target_rows}")

    target_smiles = "CC1CCCCC1"
    target_df.iloc[:, SMILES_COL] = target_smiles  # Fixed SMILES for target dataset
    source_df.iloc[:, SMILES_COL] = source_df.iloc[:, SMILES_COL].astype(str)
    source_df.iloc[:, SMILES_COL] = source_df.iloc[:, SMILES_COL].apply(
        lambda x: 'C' if x.lower() == 'nan' else x
    )

    # Check if all SMILES are identical in target dataset
    if target_df.iloc[:, SMILES_COL].nunique() == 1:
        print("Warning: All SMILES in target dataset are identical. Fingerprint features may have zero variance.")

    for col in target_df.columns[-2:]:
        target_df[col] = target_df[col].replace([np.inf, -np.inf], np.nan)
        target_df[col] = target_df[col].fillna(target_df[col].median())

    return source_df, target_df

# Generate Morgan Fingerprints
def get_morgan_fingerprints(smiles_list, radius=2, n_bits=128):
    fingerprints = []
    morgan_gen = rdFingerprintGenerator.GetMorganGenerator(radius=radius, fpSize=n_bits)
    for smiles in smiles_list:
        try:
            mol = Chem.MolFromSmiles(smiles)
            if mol is None:
                raise ValueError(f"Invalid SMILES: {smiles}")
            fp = morgan_gen.GetFingerprint(mol)
            fp_array = np.zeros((n_bits,), dtype=np.float32)
            for idx in fp.GetOnBits():
                fp_array[idx] = 1.0
            fingerprints.append(fp_array)
        except Exception as e:
            print(f"Error processing SMILES {smiles}: {e}")
            fingerprints.append(np.zeros((n_bits,), dtype=np.float32))
    return np.array(fingerprints)

# Data Processing with PCA
def process_data(data, smiles_col=SMILES_COL, n_features=24, is_target=False, pca=None, output_dir="D:/Transfer_Learning_Output"):
    original_rows = len(data)
    expected_rows = 89 if is_target else 148
    print(f"\nProcessing {'target' if is_target else 'source'} data")
    print(f"Expected rows: {expected_rows}, Input rows: {original_rows}")

    if original_rows != expected_rows:
        raise ValueError(f"Row count mismatch! Expected {expected_rows}, got {original_rows}")

    smiles = data.iloc[:, smiles_col].values
    fingerprints = get_morgan_fingerprints(smiles, radius=2, n_bits=128)
    print(f"Fingerprints shape: {fingerprints.shape}")

    # Save raw fingerprints for debugging
    fingerprint_df = pd.DataFrame(fingerprints, columns=[f'fp_{i}' for i in range(fingerprints.shape[1])])
    fingerprint_df.to_csv(os.path.join(output_dir, f"{'target' if is_target else 'source'}_fingerprints.csv"), index=False)
    print(f"Saved {'target' if is_target else 'source'}_fingerprints.csv in {output_dir}")

    # Check fingerprint variance
    fingerprint_variance = np.var(fingerprints, axis=0)
    print(f"Fingerprint variance (min, max): {fingerprint_variance.min():.4f}, {fingerprint_variance.max():.4f}")

    # Apply PCA for fingerprints
    if is_target and pca is not None:
        # Use source PCA for target dataset
        fingerprints_reduced = pca.transform(fingerprints)
        print("Applied source PCA to target dataset fingerprints")
    else:
        # Fit PCA for source dataset
        pca = PCA(n_components=8, random_state=42)
        fingerprints_reduced = pca.fit_transform(fingerprints)
        print(f"Explained variance ratio by PCA: {sum(pca.explained_variance_ratio_):.4f}")

    print(f"Fingerprints shape after processing: {fingerprints_reduced.shape}")

    # Select 23 numerical features (excluding SMILES and 2 target columns)
    other_features = data.drop(data.columns[smiles_col], axis=1).iloc[:, :n_features - 1]
    print(f"Other features shape: {other_features.shape}")

    imputed_features = []
    for col in other_features.columns:
        col_data = pd.to_numeric(other_features[col], errors='coerce').values.reshape(-1, 1)
        imputer = SimpleImputer(strategy='median')
        imputed_col = imputer.fit_transform(col_data).flatten()
        imputed_features.append(imputed_col.astype(np.float32))

    features = np.column_stack(imputed_features)
    all_features = np.hstack([features, fingerprints_reduced])
    feature_cols = other_features.columns.tolist() + [f'pca_fp_{i}' for i in range(fingerprints_reduced.shape[1])]
    X_combined = pd.DataFrame(all_features, columns=feature_cols)

    target_cols = data.columns[-2:]
    conversion = data[target_cols[0]].astype(np.float32).values.reshape(-1, 1)
    selectivity = data[target_cols[1]].astype(np.float32).values.reshape(-1, 1)
    conversion = np.nan_to_num(conversion, nan=np.nanmedian(conversion))
    selectivity = np.nan_to_num(selectivity, nan=np.nanmedian(selectivity))

    # Check for negative values in original targets
    if conversion.min() < 0 or selectivity.min() < 0:
        print(f"Warning: Negative values detected in {'target' if is_target else 'source'} dataset - "
              f"Conversion min: {conversion.min():.2f}, Selectivity min: {selectivity.min():.2f}")

    print(f"Conversion range: min={conversion.min():.2f}, max={conversion.max():.2f}")
    print(f"Selectivity range: min={selectivity.min():.2f}, max={selectivity.max():.2f}")

    X_combined['smiles'] = smiles
    X_combined['conversion'] = conversion.flatten()
    X_combined['selectivity'] = selectivity.flatten()

    print(f"Processed shape: {X_combined.shape}")
    return X_combined, conversion, selectivity, pca

# VAE
class StableVAE(nn.Module):
    def __init__(self, input_dim, latent_dim=32):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, 256),
            nn.BatchNorm1d(256),
            nn.LeakyReLU(0.2),
            nn.Dropout(0.2),
            nn.Linear(256, 128),
            nn.BatchNorm1d(128),
            nn.LeakyReLU(0.2),
            nn.Dropout(0.2),
            nn.Linear(128, latent_dim * 2)
        )
        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, 128),
            nn.BatchNorm1d(128),
            nn.LeakyReLU(0.2),
            nn.Dropout(0.2),
            nn.Linear(128, 256),
            nn.BatchNorm1d(256),
            nn.LeakyReLU(0.2),
            nn.Dropout(0.2),
            nn.Linear(256, input_dim)
        )

    def encode(self, x):
        h = self.encoder(x)
        mu, logvar = torch.chunk(h, 2, dim=-1)
        return mu, logvar

    def forward(self, x):
        mu, logvar = self.encode(x)
        z = self.reparameterize(mu, logvar)
        return self.decoder(z), mu, logvar

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

def train_vae(model, data, epochs=1500, batch_size=16, beta=0.8):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.to(device)
    optimizer = optim.Adam(model.parameters(), lr=1.09e-3, weight_decay=1e-7)
    dataset = TensorDataset(torch.tensor(data, dtype=torch.float32))
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    best_loss = float('inf')
    patience = 100
    counter = 0
    losses = {'recon': [], 'kl': [], 'total': []}

    for epoch in range(epochs):
        model.train()
        total_loss = 0
        epoch_recon_loss = 0
        epoch_kl_loss = 0
        for batch in loader:
            batch = batch[0].to(device)
            recon, mu, logvar = model(batch)
            recon_loss = nn.MSELoss(reduction='sum')(recon, batch)
            kl = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
            loss = recon_loss + beta * kl
            if torch.isnan(loss):
                print("NaN loss detected! Adjusting...")
                loss = torch.where(torch.isnan(loss), torch.zeros_like(loss), loss)
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            total_loss += loss.item()
            epoch_recon_loss += recon_loss.item()
            epoch_kl_loss += kl.item()
        avg_loss = total_loss / len(loader)
        avg_recon_loss = epoch_recon_loss / len(loader)
        avg_kl_loss = epoch_kl_loss / len(loader)
        losses['recon'].append(avg_recon_loss)
        losses['kl'].append(avg_kl_loss)
        losses['total'].append(avg_loss)
        if (epoch + 1) % 100 == 0:
            print(
                f"Epoch {epoch + 1}, Recon Loss: {avg_recon_loss:.4f}, KL: {avg_kl_loss:.4f}, Total Loss: {avg_loss:.4f}")

        if avg_loss < best_loss:
            best_loss = avg_loss
            counter = 0
            print(f"New best loss: {best_loss:.4f} at epoch {epoch + 1}")
        else:
            counter += 1
        if counter >= patience:
            print(f"Early stopping at epoch {epoch + 1}")
            break

    return losses

def plot_vae_losses(losses, output_dir):
    plt.figure(figsize=(10, 6))
    epochs = range(1, len(losses['total']) + 1)
    plt.plot(epochs, losses['recon'], label='Reconstruction Loss', color='blue')
    plt.plot(epochs, losses['kl'], label='KL Divergence', color='orange')
    plt.plot(epochs, losses['total'], label='Total Loss', color='green')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('VAE Training Losses')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'vae_loss_plot.png'))
    plt.close()
    print(f"Saved vae_loss_plot.png in {output_dir}")

# Evaluation
def evaluate_model(model, X, y, y_scaler, target_name, dataset_type, output_dir):
    y_pred_scaled = model.predict(X)
    # Inverse transform predictions and true values to original scale
    y_pred = y_scaler.inverse_transform(y_pred_scaled.reshape(-1, 1)).flatten()
    y_true = y_scaler.inverse_transform(y.reshape(-1, 1)).flatten()
    # Clip predictions to ensure non-negative values
    y_pred = np.clip(y_pred, 0, None)
    rmse = np.sqrt(mean_squared_error(y, y_pred_scaled))
    r2 = r2_score(y_true, y_pred)
    print(f"\nEvaluation for {target_name} on {dataset_type} dataset:")
    print(f"RMSE: {rmse:.4f}")
    print(f"R² Score: {r2:.4f}")
    # Save predictions in original scale
    pred_df = pd.DataFrame({
        'True': y_true,
        'Predicted': y_pred
    })
    pred_df.to_csv(os.path.join(output_dir, f"{target_name}_{dataset_type}_predictions.csv"), index=False)
    print(f"Saved {target_name}_{dataset_type}_predictions.csv in {output_dir}")
    return y_pred

# Main Execution
if __name__ == '__main__':
    torch.manual_seed(42)
    np.random.seed(42)
    output_dir = "D:/Transfer_Learning_Output"
    os.makedirs(output_dir, exist_ok=True)

    # Load and clean data
    source_df, target_df = load_and_clean_data()
    X_src_df, conversion_src, selectivity_src, pca = process_data(source_df, is_target=False, output_dir=output_dir)
    X_tgt_df, conversion_tgt, selectivity_tgt, _ = process_data(target_df, is_target=True, pca=pca, output_dir=output_dir)

    # Save processed data
    X_src_df.to_csv(os.path.join(output_dir, "source_processed.csv"), index=False)
    X_tgt_df.to_csv(os.path.join(output_dir, "target_processed.csv"), index=False)
    print(f"Saved source_processed.csv: {len(X_src_df)} rows")
    print(f"Saved target_processed.csv: {len(X_tgt_df)} rows")

    # Extract features
    X_src = X_src_df.drop(['smiles', 'conversion', 'selectivity'], axis=1)
    X_tgt = X_tgt_df.drop(['smiles', 'conversion', 'selectivity'], axis=1)

    # Print column names for debugging
    print("X_src columns:", X_src.columns.tolist())
    print("X_tgt columns:", X_tgt.columns.tolist())

    # Define features to be scaled with source scaler
    uniform_scale_features = ['Boiling Point ', 'Melting Point', 'Density', 'Molecular weight'] + \
                             [f'pca_fp_{i}' for i in range(8)]

    # Verify that all features exist
    missing_src = [f for f in uniform_scale_features if f not in X_src.columns]
    missing_tgt = [f for f in uniform_scale_features if f not in X_tgt.columns]
    if missing_src or missing_tgt:
        raise ValueError(f"Missing columns in X_src: {missing_src}, X_tgt: {missing_tgt}")

    # Debug raw feature values and variances
    print("\nRaw feature values in X_src (min, max, unique count):")
    for col in uniform_scale_features[:4]:  # Boiling Point, Melting Point, Density, Molecular weight
        values = X_src[col]
        print(f"{col}: min={values.min():.4f}, max={values.max():.4f}, unique={len(np.unique(values))}")
        if len(np.unique(values)) == 1:
            print(f"Warning: Feature {col} has only one unique value in X_src")
    print("\nRaw feature variances in X_src:")
    for col in uniform_scale_features:
        var = np.var(X_src[col])
        print(f"{col}: {var:.4f}")
        if var < 1e-3:
            print(f"Warning: Feature {col} has low variance in raw X_src")
    print("\nRaw feature values in X_tgt (min, max, unique count):")
    for col in uniform_scale_features[:4]:  # Boiling Point, Melting Point, Density, Molecular weight
        values = X_tgt[col]
        print(f"{col}: min={values.min():.4f}, max={values.max():.4f}, unique={len(np.unique(values))}")
        if len(np.unique(values)) == 1:
            print(f"Warning: Feature {col} has only one unique value in X_tgt")
    print("\nRaw feature variances in X_tgt:")
    for col in X_tgt.columns:
        var = np.var(X_tgt[col])
        print(f"{col}: {var:.4f}")
        if var < 1e-3:
            print(f"Warning: Feature {col} has low variance in raw X_tgt")

    # Fit scaler on source data for uniform scale features
    scaler_uniform = StandardScaler().fit(X_src[uniform_scale_features])
    X_src_uniform_scaled = scaler_uniform.transform(X_src[uniform_scale_features])
    X_tgt_uniform_scaled = scaler_uniform.transform(X_tgt[uniform_scale_features])

    # Define other features to be scaled independently
    other_features = [col for col in X_src.columns if col not in uniform_scale_features]
    if other_features:
        scaler_X_src_other = StandardScaler().fit(X_src[other_features])
        scaler_X_tgt_other = StandardScaler().fit(X_tgt[other_features])

        # Debug: Print original values before scaling
        print("\nOriginal target values for other features:")
        print(X_tgt[other_features].describe())

        X_src_other_scaled = scaler_X_src_other.transform(X_src[other_features])
        X_tgt_other_scaled = scaler_X_tgt_other.transform(X_tgt[other_features])

        # Debug: Print scaled values
        print("\nScaled target values for other features:")
        print(pd.DataFrame(X_tgt_other_scaled, columns=other_features).describe())

        # Check for near-zero variance features in target
        variances = np.var(X_tgt_other_scaled, axis=0)
        low_variance_mask = variances < 1e-3
        if any(low_variance_mask):
            low_var_cols = [col for col, mask in zip(other_features, low_variance_mask) if mask]
            print(f"\nWarning: Low variance in scaled target features: {low_var_cols}")
            print("Using original values for these features")

            # Replace low-variance scaled features with original values
            for i, col in enumerate(other_features):
                if low_variance_mask[i]:
                    X_tgt_other_scaled[:, i] = X_tgt[col].values

    # Fit separate scalers for other features
    if other_features:
        scaler_X_src_other = StandardScaler().fit(X_src[other_features])
        scaler_X_tgt_other = StandardScaler().fit(X_tgt[other_features])
        X_src_other_scaled = scaler_X_src_other.transform(X_src[other_features])
        X_tgt_other_scaled = scaler_X_tgt_other.transform(X_tgt[other_features])
    else:
        X_src_other_scaled = np.array([]).reshape(len(X_src), 0)
        X_tgt_other_scaled = np.array([]).reshape(len(X_tgt), 0)

    # Combine scaled features
    X_src_scaled = np.hstack([X_src_uniform_scaled, X_src_other_scaled]) if X_src_other_scaled.size else X_src_uniform_scaled
    X_tgt_scaled = np.hstack([X_tgt_uniform_scaled, X_tgt_other_scaled]) if X_tgt_other_scaled.size else X_tgt_uniform_scaled

    # Ensure column order matches
    feature_cols = uniform_scale_features + other_features
    X_src_scaled = pd.DataFrame(X_src_scaled, columns=feature_cols)
    X_tgt_scaled = pd.DataFrame(X_tgt_scaled, columns=feature_cols)

    # Debug scaled feature variances
    print("\nScaled feature variances in X_tgt_scaled:")
    for col in feature_cols:
        var = np.var(X_tgt_scaled[col])
        print(f"{col}: {var:.4f}")
        if var < 1e-3 and col in uniform_scale_features:
            print(f"Warning: Feature {col} has low variance in X_tgt_scaled, check source data variability")

    # Save scaled target features for inspection
    X_tgt_scaled.to_csv(os.path.join(output_dir, "X_tgt_scaled.csv"), index=False)
    print(f"Saved X_tgt_scaled.csv: {len(X_tgt_scaled)} rows")

    X_src_scaled = X_src_scaled.to_numpy()
    X_tgt_scaled = X_tgt_scaled.to_numpy()

    # Separate scalers for output targets
    scaler_conv_src = StandardScaler().fit(conversion_src)
    conversion_src_scaled = scaler_conv_src.transform(conversion_src)
    scaler_sel_src = StandardScaler().fit(selectivity_src)
    selectivity_src_scaled = scaler_sel_src.transform(selectivity_src)
    scaler_conv_tgt = StandardScaler().fit(conversion_tgt)
    conversion_tgt_scaled = scaler_conv_tgt.transform(conversion_tgt)
    scaler_sel_tgt = StandardScaler().fit(selectivity_tgt)
    selectivity_tgt_scaled = scaler_sel_tgt.transform(selectivity_tgt)

    print(f"vae_input shape: {X_src_scaled.shape}")
    # Train VAE on scaled source features
    vae = StableVAE(input_dim=X_src_scaled.shape[1], latent_dim=32)
    losses = train_vae(vae, X_src_scaled, beta=0.8)
    plot_vae_losses(losses, output_dir)

    # Apply PCA to the VAE latent space
    Z_tgt = vae.encode(torch.tensor(X_tgt_scaled, dtype=torch.float32))[0].detach().numpy()
    print(f"Latent space shape before PCA: {Z_tgt.shape}")
    pca_latent = PCA(n_components=10, random_state=42)
    Z_tgt_reduced = pca_latent.fit_transform(Z_tgt)
    print(f"Latent space shape after PCA: {Z_tgt_reduced.shape}")
    print(f"Explained variance ratio by latent PCA: {sum(pca_latent.explained_variance_ratio_):.4f}")

    # Combine latent and scaled target features
    X_final = np.hstack([Z_tgt_reduced, X_tgt_scaled]).astype(np.float32)
    print(f"X_final shape: {X_final.shape}")

    # Save combined data
    latent_cols = [f'latent_pca_{i}' for i in range(Z_tgt_reduced.shape[1])]
    # Use the same feature_cols as X_tgt_scaled to ensure alignment
    combined_df = pd.DataFrame(
        np.hstack([Z_tgt_reduced, X_tgt_scaled, conversion_tgt_scaled.reshape(-1, 1), selectivity_tgt_scaled.reshape(-1, 1)]),
        columns=latent_cols + feature_cols + ['conversion_scaled', 'selectivity_scaled']
    )
    combined_df.to_csv(os.path.join(output_dir, "combined_target_data.csv"), index=False)
    print(f"Saved combined_target_data.csv: {len(combined_df)} rows")

