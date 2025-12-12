# qm9_pyg_loader.py
import torch
from torch_geometric.data import DataLoader
from torch_geometric.datasets import QM9
from torch_geometric.transforms import Distance
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error
from rdkit import Chem, DataStructs
from rdkit.Chem import MolFromSmiles
from rdkit.Chem.AllChem import GetMorganFingerprintAsBitVect
from rdkit.Chem import Descriptors
import numpy as np

def mol_to_morgan(smiles, radius=2, nbits=2048):
    mol = MolFromSmiles(smiles)
    return np.array(GetMorganFingerprintAsBitVect(mol, radius, nBits=nbits))

def mol_to_descriptors(smiles):
    mol = MolFromSmiles(smiles)
    descs = [
        Descriptors.MolWt(mol),
        Descriptors.NumValenceElectrons(mol),
        Descriptors.TPSA(mol),
        Descriptors.MolLogP(mol),
        Descriptors.NumHDonors(mol),
        Descriptors.NumHAcceptors(mol),
    ]
    return np.array(descs, dtype=np.float32)

def attach_features(dataset, use_fps=False, use_desc=False):
    """
    Attaches RDKit features to each PYG Data object.
    """
    for data in dataset:
        smiles = data.smiles

        if use_fps:
            fp = mol_to_morgan(smiles)
            data.fingerprint = torch.tensor(fp, dtype=torch.float)

        if use_desc:
            d = mol_to_descriptors(smiles)
            data.descriptors = torch.tensor(d, dtype=torch.float)

    return dataset

# ----------------------------------------------------------
# MAIN LOADER
# ----------------------------------------------------------
def get_qm9_loaders(
        root="./qm9",
        batch_size=64,
        target_index=0,
        split=(0.8,0.1,0.1),
        seed=42,
        use_fingerprints=False,
        use_descriptors=False
    ):
    """
    Returns (train_loader, val_loader, test_loader, node_feat_dim, y_mean, y_std).
    """
    # Load QM9 with geometric features
    dataset = QM9(root, transform=Distance(norm=False))
    dataset = dataset.shuffle()

    # Attach RDKit-based features (NEW)
    dataset = attach_features(dataset, use_fingerprints, use_descriptors)

    # Extract target values
    ys = torch.stack([d.y for d in dataset]).squeeze()
    y = ys[:, target_index].unsqueeze(1)

    # Basic statistics (NEW)
    y_mean = y.mean().item()
    y_std = y.std().item()

    # Split indices
    n = len(dataset)
    idx = list(range(n))
    train_idx, test_idx = train_test_split(idx, test_size=1 - split[0], random_state=seed)
    val_rel = split[1] / (split[1] + split[2])
    val_idx, test_idx = train_test_split(test_idx, test_size=val_rel, random_state=seed)

    train_ds = dataset[train_idx]
    val_ds   = dataset[val_idx]
    test_ds  = dataset[test_idx]

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
    val_loader   = DataLoader(val_ds, batch_size=batch_size, shuffle=False)
    test_loader  = DataLoader(test_ds, batch_size=batch_size, shuffle=False)

    # Node feature dimensionality
    sample = dataset[0]
    node_feat_dim = sample.x.shape[1] if sample.x is not None else 0

    return train_loader, val_loader, test_loader, node_feat_dim, y_mean, y_std

def get_qm9_fingerprint_splits(
        root="./qm9",
        target_index=0,
        split=(0.8, 0.1, 0.1),
        seed=42,
    ):
    """
    Build simple fixed-length features for QM9 using only the PyG dataset:
    we use a bag-of-atoms representation (counts of atomic numbers z).

    Returns:
        X_train, y_train, X_val, y_val, X_test, y_test
    """
    # Load QM9 once (no transform needed for RF)
    dataset = QM9(root)

    n = len(dataset)

    # Targets: (N, 19) -> pick the desired column
    ys = torch.stack([d.y for d in dataset]).squeeze()
    y = ys[:, target_index].numpy()

    # Determine max atomic number present so we can size the count vector
    max_z = 0
    for data in dataset:
        if data.z.numel() > 0:
            max_z = max(max_z, int(data.z.max().item()))
    feat_dim = max_z + 1  # index 0 may stay mostly zero; that's fine

    # Build bag-of-atoms features: X[i, z] = count of atoms with atomic number z
    X = np.zeros((n, feat_dim), dtype=np.float32)
    for i, data in enumerate(dataset):
        z_vals = data.z.numpy().astype(int)
        for z in z_vals:
            X[i, z] += 1.0

    # Train / val / test split (same pattern as before)
    idx = np.arange(n)
    train_idx, test_idx = train_test_split(idx, test_size=1 - split[0], random_state=seed)
    val_rel = split[1] / (split[1] + split[2])
    val_idx, test_idx = train_test_split(test_idx, test_size=val_rel, random_state=seed)

    X_train, y_train = X[train_idx], y[train_idx]
    X_val,   y_val   = X[val_idx],   y[val_idx]
    X_test,  y_test  = X[test_idx],  y[test_idx]

    return X_train, y_train, X_val, y_val, X_test, y_test

def evaluate_regression_model(model, data_loader, device="cpu"):
    """
    Collects predictions & computes RMSE and MAE.
    Useful for parity plots.
    """
    model.eval()
    preds, trues = [], []

    with torch.no_grad():
        for batch in data_loader:
            batch = batch.to(device)

            y_hat = model(batch)
            preds.append(y_hat.cpu())
            trues.append(batch.y[:,0].cpu())

    preds = torch.cat(preds)
    trues = torch.cat(trues)

    rmse = torch.sqrt(torch.mean((preds - trues)**2)).item()
    mae  = torch.mean(torch.abs(preds - trues)).item()

    return rmse, mae, preds.numpy(), trues.numpy()

if __name__ == "__main__":

    '''
        Abstract
        ========
        This project uses machine-learning methods to predict molecular atomization energy, a quantum-mechanical measure of bond strength and molecular stability. Using the QM9 dataset from Kaggle, each molecule’s SMILES string is converted into Morgan fingerprints using RDKit. Multiple regression models, such as Random Forests, Support Vector Regression, a feed-forward neural network, and symbolic regression via PySR, are trained and compared using the same fingerprint representation to determine which algorithm best captures structure–property relationships. Model performance is evaluated through RMSE, MAE, and parity plots. After identifying the best model, we also compare fingerprint-based and descriptor-based feature sets. We expect nonlinear models to outperform linear ones, with fingerprint features providing higher accuracy than global descriptors.
    '''

    print("\n==============================")
    print("  Building Morgan Fingerprints")
    print("==============================\n")

    X_train, y_train, X_val, y_val, X_test, y_test = get_qm9_fingerprint_splits(
        root="./qm9",
        target_index=0,
        split=(0.8, 0.1, 0.1),
        seed=42,
    )

    print(f"Training set:   X={X_train.shape}, y={y_train.shape}")
    print(f"Validation set: X={X_val.shape},   y={y_val.shape}")
    print(f"Test set:       X={X_test.shape},  y={y_test.shape}")
    print("Fingerprints are binary vectors of length 2048.\n")

    print("\n==============================")
    print("   Training Random Forest")
    print("==============================\n")

    rf = RandomForestRegressor(
        n_estimators=200,
        max_depth=None,
        n_jobs=-1,
        random_state=0
    )

    print("Fitting Random Forest... (this may take a bit)")
    rf.fit(X_train, y_train)
    print("Done! 🎉\n")

    print("==============================")
    print("   Evaluating Model")
    print("==============================\n")

    def eval_split(X, y, name):
        print(f"Evaluating on {name} set...")
        y_pred = rf.predict(X)
        mse = mean_squared_error(y, y_pred)  # no 'squared' kwarg
        rmse = np.sqrt(mse)
        mae  = mean_absolute_error(y, y_pred)
        print(f"{name:5s} | RMSE = {rmse:.6f} | MAE = {mae:.6f}\n")
        return y_pred

    y_pred_train = eval_split(X_train, y_train, "Train")
    y_pred_val   = eval_split(X_val,   y_val,   "Val")
    y_pred_test  = eval_split(X_test,  y_test,  "Test")

    print("==============================")
    print("   Baseline Complete")
    print("==============================\n")
    print("You now have:")
    print("  • y_test        – true QM9 values")
    print("  • y_pred_test   – RF predictions")
    print("These are ready for a parity plot or error analysis.\n")
    print("🔥 Random Forest baseline is done! 🔥\n")
