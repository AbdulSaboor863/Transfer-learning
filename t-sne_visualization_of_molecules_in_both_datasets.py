import os
import pandas as pd
import numpy as np
from sklearn.manifold import TSNE
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
import matplotlib.pyplot as plt
import seaborn as sns
from rdkit import Chem
from rdkit.Chem import rdFingerprintGenerator
from matplotlib.patches import Ellipse

# Suppress joblib warning
os.environ["LOKY_MAX_CPU_COUNT"] = "4"

# Load dataset
try:
    df = pd.read_csv("D:/Transfer learning project/CSV files for origin plots (TL project)/t-SNE molecular types visualization/t-SNE molecules visualization.csv")
except FileNotFoundError:
    print("Error: 't-SNE molecules visualization.csv' not found.")
    exit(1)

# Expected columns
expected_columns = [
    'Loading', 'SSA', 'DP', 'TPV', 'D_M', 'Boiling Point ', 'Melting Point', 'Density',
    'Molecular weight', 'PE_M', 'FIE_M', 'MR_M', 'EA_M', 'WF_M', 'AN_M', 'BM_M',
    'MP_M', 'MP_S', 'Temp', 'Time', 'W_cat', 'P', 'SV', 'conver', 'selec',
    'SMILES', 'Molecule type'
]
if not all(col in df.columns for col in expected_columns):
    print("Error: Dataset is missing some expected columns.")
    exit(1)

# Select numerical features
numerical_features = [
    'Loading', 'SSA', 'DP', 'TPV', 'D_M', 'Boiling Point ', 'Melting Point', 'Density',
    'Molecular weight', 'PE_M', 'FIE_M', 'MR_M', 'EA_M', 'WF_M', 'AN_M', 'BM_M',
    'MP_M', 'MP_S', 'Temp', 'Time', 'W_cat', 'P', 'SV', 'conver', 'selec'
]

# Convert SMILES to Morgan fingerprints
def smiles_to_fingerprints(smiles_list, radius=2, n_bits=2048):
    fingerprints = []
    morgan_gen = rdFingerprintGenerator.GetMorganGenerator(radius=radius, fpSize=n_bits)
    for smiles in smiles_list:
        try:
            mol = Chem.MolFromSmiles(smiles)
            if mol is None:
                raise ValueError(f"Invalid SMILES: {smiles}")
            fp = morgan_gen.GetFingerprint(mol)
            fingerprints.append(np.array(fp))
        except:
            print(f"Warning: Failed to process SMILES: {smiles}. Using zero vector.")
            fingerprints.append(np.zeros(n_bits))
    return np.array(fingerprints)

# Generate fingerprints
fingerprints = smiles_to_fingerprints(df['SMILES'], radius=2, n_bits=2048)

# Combine numerical + fingerprints
X_numerical = df[numerical_features]
imputer = SimpleImputer(strategy='mean')
X_numerical_imputed = imputer.fit_transform(X_numerical)
X_combined = np.hstack((X_numerical_imputed, fingerprints))

# Standardize
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X_combined)

# Apply t-SNE
tsne = TSNE(n_components=2, perplexity=4, learning_rate=1.5, max_iter=500, random_state=25)
X_tsne = tsne.fit_transform(X_scaled)

# === Oval Projection Fix ===
X_min, X_max = X_tsne.min(axis=0), X_tsne.max(axis=0)
X_norm = 2 * (X_tsne - X_min) / (X_max - X_min) - 1

scale_x, scale_y = 1.0, 0.6
X_oval = np.zeros_like(X_norm)
for i in range(len(X_norm)):
    x, y = X_norm[i]
    r = np.sqrt(x**2 + y**2)
    if r > 0.7:  # push back inside smaller circle for closer clustering
        x, y = x / r * 0.7, y / r * 0.7
    X_oval[i, 0] = x * scale_x
    X_oval[i, 1] = y * scale_y

# Save t-SNE coordinates and molecule types to CSV
tsne_df = pd.DataFrame({
    'tSNE_Component_1': X_oval[:, 0],
    'tSNE_Component_2': X_oval[:, 1],
    'Molecule_type': df['Molecule type']
})
tsne_df.to_csv("D:/Transfer learning project/tsne_molecules_plot.csv", index=False)
print("t-SNE plot data saved to 'D:/Transfer learning project/tsne_molecules_plot.csv'")

# Style
sns.set_style("white")
plt.figure(figsize=(10, 7))

# Labels
molecule_types = df['Molecule type'].astype('category')
labels = molecule_types.cat.codes
unique_labels = molecule_types.cat.categories
palette = sns.color_palette("Set2", len(unique_labels))

# Plot with circle markers
for i, molecule_type in enumerate(unique_labels):
    idx = labels == i
    plt.scatter(
        X_oval[idx, 0], X_oval[idx, 1],
        c=[palette[i]], label=molecule_type,
        alpha=0.8, s=150, edgecolors='w', linewidth=0.5,
        marker='o'
    )

# Customize
plt.title('t-SNE Visualization of Molecules (Oval with Circle Shapes)', fontsize=14, pad=15)
plt.xlabel('t-SNE Component 1', fontsize=12)
plt.ylabel('t-SNE Component 2', fontsize=12)

# Add molecule type indicators with colored circles in the bottom-left corner
plt.scatter([-0.85], [-0.5], c=[palette[0]], s=100, edgecolors='w', linewidth=0.5, marker='o')
plt.text(-0.8, -0.5, '1', color='black', fontsize=16, ha='left', va='center', bbox=dict(facecolor='white', alpha=0.8, edgecolor='none'))
plt.scatter([-0.85], [-0.55], c=[palette[1]], s=100, edgecolors='w', linewidth=0.5, marker='o')
plt.text(-0.8, -0.55, '2', color='black', fontsize=16, ha='left', va='center', bbox=dict(facecolor='white', alpha=0.8, edgecolor='none'))
plt.scatter([-0.85], [-0.60], c=[palette[2]], s=100, edgecolors='w', linewidth=0.5, marker='o')
plt.text(-0.8, -0.60, '3', color='black', fontsize=16, ha='left', va='center', bbox=dict(facecolor='white', alpha=0.8, edgecolor='none'))
plt.scatter([-0.85], [-0.65], c=[palette[3]], s=100, edgecolors='w', linewidth=0.5, marker='o')
plt.text(-0.8, -0.65, '4', color='black', fontsize=16, ha='left', va='center', bbox=dict(facecolor='white', alpha=0.8, edgecolor='none'))

plt.tight_layout()

plt.show()
