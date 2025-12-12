# train_pysr.py
import numpy as np
from data_qm9 import get_splits
from models_pysr import build_pysr_baseline
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.metrics import mean_squared_error, mean_absolute_error
from sklearn.pipeline import Pipeline
from train_rf import train_simple_rf
from plots import plot_top_features, parity_plots_from_expression
import matplotlib.pyplot as plt
import pandas as pd
import sympy as sp
from warnings import filterwarnings
filterwarnings('ignore')

RANDOM_SEED = 42
GET_IMPORTANCES = False
BASELINE = False
MAKE_PARITY_PLOT = True

def evaluate(model, X, y, name = ""):
    y_pred = model.predict(X)
    mse = mean_squared_error(y, y_pred)  # no 'squared' kwarg
    rmse = np.sqrt(mse)
    mae = mean_absolute_error(y, y_pred)
    print(f"{name} RMSE: {rmse:.3f}, MAE: {mae:.3f}")
    return rmse, mae, y_pred

def main():
    seed = RANDOM_SEED
    rng = np.random.RandomState(seed)

    print("Loading data...")
    X_train, X_val, X_test, y_train, y_val, y_test = get_splits()

    k = 20 if BASELINE else 30

    idx_top = (np.array([294, 1923, 1152, 1292, 841, 1325, 974, 1171, 802, 1154, 699, 674, 915, 577, 926, 1917, 378, 1057, 1004, 1257, 222, 694, 1453, 695, 1380, 314, 1928, 656, 650, 807][:k]), None)[GET_IMPORTANCES]

    if GET_IMPORTANCES:
        print("Training simple RF for importance selection...")
        rf = train_simple_rf(X_train, y_train, seed=seed)
        importances = rf.feature_importances_

        # Select top-k bits by importance
        idx_top = np.argsort(importances)[-k:]
        print(f"Using top {k} fingerprint bits based on RF importance.")
        print(f"Importance indices = {idx_top.tolist()}")

        # Plot those features
        plot_top_features(idx_top, importances,
                          fname=f"pysr_top_{k}_rf_features.png")

        # Save a mapping CSV: which bits were kept + RF importance
        mapping_df = pd.DataFrame({
            "fp_index": idx_top,
            "rf_importance": importances[idx_top],
        }).sort_values("rf_importance", ascending=False)
        mapping_df.to_csv("pysr_feature_mapping.csv", index=False)
        print("Saved feature mapping to pysr_feature_mapping.csv")

    # Reduce datasets to these k bits
    X_train_k = X_train[:, idx_top]
    X_val_k = X_val[:, idx_top]
    X_test_k = X_test[:, idx_top]
    
    if MAKE_PARITY_PLOT:
        expr = -788.5488 * (sp.Symbol("fp_650") + sp.Symbol("fp_656") + sp.Symbol("fp_807")) - 10249.428

        # 2. The feature names must match the column order of X_train_k
        feature_names = ("fp_650", "fp_656", "fp_807")

        indices = [idx_top.tolist().index(650),
                   idx_top.tolist().index(656),
                   idx_top.tolist().index(807)]
        X_train_sr = X_train_k[:, indices]
        X_val_sr   = X_val_k[:, indices]
        X_test_sr  = X_test_k[:, indices]

        # 3. Make parity plots for this expression
        parity_plots_from_expression(
            expr, feature_names,
            X_train_sr, y_train,
            X_val_sr,   y_val,
            X_test_sr,  y_test,
            prefix="pysr_expr")
        return

    # Make nice feature names for PySR: fp_<original_bit_index>
    feature_names = [f"fp_{j}" for j in idx_top]

    n_train = X_train_k.shape[0]
    n_sub = max((4000, 2000)[BASELINE], int((0.1, 0.05)[BASELINE] * n_train))  # (5, 10)% or (2000, 4000)
    idx_sub = rng.choice(n_train, size=n_sub, replace=False)

    X_train_sub = X_train_k[idx_sub]
    y_train_sub = y_train[idx_sub]

    print(f"Training PySR on {n_sub} samples and {k} features...")
    print("Feature names passed to PySR:", feature_names)

    model = build_pysr_baseline(seed=seed, feature_names=feature_names, n_iterations = (1000, 100)[BASELINE])
    model.fit(X_train_sub, y_train_sub, variable_names = feature_names)

    print("\n=== PySR Results ===")
    evaluate(model, X_train_sub, y_train_sub, "Train (subset)")
    evaluate(model, X_val_k, y_val, "Validation")
    evaluate(model, X_test_k, y_test, "Test")

    print("\n=== Best PySR Equation ===")
    print(model)
    print(sp.latex(model.sympy()))

    eqs = model.equations_

    eqs.to_csv("pysr_equations_topfeat.csv", index=False)
    print("Saved equations to pysr_equations_topfeat.csv")
    print("Variables in 'equation' column should use names like fp_147, fp_330, ...")


if __name__ == "__main__":
    main()
