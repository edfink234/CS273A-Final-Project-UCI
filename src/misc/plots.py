# plots.py
import matplotlib.pyplot as plt
import numpy as np
import sympy as sp
import pandas as pd
from sklearn.metrics import mean_squared_error, mean_absolute_error

def plot_pysr_pareto_front(
    csv_path,
    idx_top,
    X_train, y_train,
    X_val, y_val,
    X_test, y_test,
    savefig="pysr_pareto_front.png",
):
    """
    Load PySR equations from CSV and generate a Pareto-front-style plot
    showing complexity vs RMSE/MAE for train/val/test, with the
    best-scoring PySR expression highlighted.
    """

    df = pd.read_csv(csv_path)

    idx_top = np.asarray(idx_top).tolist()  # so we can .index()

    # -----------------------------
    # Build models from CSV
    # -----------------------------
    exprs = []
    models = []
    for _, row in df.iterrows():
        expr = sp.sympify(row["sympy_format"])
        exprs.append(expr)
        complexity = row["complexity"]

        # Which symbols appear? e.g. {fp_650, fp_807, ...}
        syms = sorted(expr.free_symbols, key=lambda s: s.name)

        col_indices = []
        for s in syms:
            name = s.name          # e.g. "fp_650"
            bit = int(name.split("_")[1])
            col_idx = idx_top.index(bit)  # position in X columns
            col_indices.append(col_idx)

        func = sp.lambdify(syms, expr, "numpy")
        models.append((complexity, func, col_indices))

    def eval_expr(func, cols, X, y):
        args = [X[:, j] for j in cols]
        y_pred = func(*args)

        y_pred = np.asarray(y_pred)
        if y_pred.shape == ():               # scalar expression
            y_pred = np.full_like(y, y_pred, dtype=float)
        elif y_pred.shape != y.shape:
            y_pred = y_pred.reshape(-1)

        rmse = np.sqrt(mean_squared_error(y, y_pred))
        mae = mean_absolute_error(y, y_pred)
        return rmse, mae

    # -----------------------------
    # Compute metrics
    # -----------------------------
    complexities = []
    rmse_train, rmse_val, rmse_test = [], [], []
    mae_train, mae_val, mae_test   = [], [], []

    for (complexity, func, cols) in models:
        complexities.append(complexity)

        r_tr, a_tr = eval_expr(func, cols, X_train, y_train)
        r_va, a_va = eval_expr(func, cols, X_val,   y_val)
        r_te, a_te = eval_expr(func, cols, X_test,  y_test)

        rmse_train.append(r_tr); mae_train.append(a_tr)
        rmse_val.append(r_va);   mae_val.append(a_va)
        rmse_test.append(r_te);  mae_test.append(a_te)

    complexities = np.asarray(complexities)
    rmse_train = np.asarray(rmse_train)
    rmse_val   = np.asarray(rmse_val)
    rmse_test  = np.asarray(rmse_test)
    mae_train  = np.asarray(mae_train)
    mae_val    = np.asarray(mae_val)
    mae_test   = np.asarray(mae_test)

        # -----------------------------
    # Identify best expression
    # -----------------------------
    # Choose best by smallest Test RMSE
    best_idx = int(np.argmin(rmse_test))

    best_complexity = complexities[best_idx]

    best_rmse_train = rmse_train[best_idx]
    best_mae_train  = mae_train[best_idx]

    best_rmse_val   = rmse_val[best_idx]
    best_mae_val    = mae_val[best_idx]

    best_rmse_test  = rmse_test[best_idx]
    best_mae_test   = mae_test[best_idx]

    # Sympy expression → LaTeX
    best_expr = exprs[best_idx]
    best_expr_latex = sp.latex(best_expr)

    print("=== Best PySR Expression (by smallest Test RMSE) ===")
    print(f"Index:           {best_idx}")
    print(f"Complexity:      {best_complexity}")
    print(f"Train  RMSE/MAE: {best_rmse_train:.3f} / {best_mae_train:.3f}")
    print(f"Val    RMSE/MAE: {best_rmse_val:.3f} / {best_mae_val:.3f}")
    print(f"Test   RMSE/MAE: {best_rmse_test:.3f} / {best_mae_test:.3f}")
    print("LaTeX expression:")
    print(best_expr_latex)

    # -----------------------------
    # Plot Pareto-style front
    # -----------------------------
    plt.figure(figsize=(10, 6))

    # RMSE curves
    plt.plot(complexities, rmse_train, "o-", label="Train RMSE")
    plt.plot(complexities, rmse_val,   "o-", label="Val RMSE")
    plt.plot(complexities, rmse_test,  "o-", label="Test RMSE")

    # MAE curves
    plt.plot(complexities, mae_train, "o--", label="Train MAE")
    plt.plot(complexities, mae_val,   "o--", label="Val MAE")
    plt.plot(complexities, mae_test,  "o--", label="Test MAE")

    # Highlight best expression (by score) with a star on test errors
    plt.scatter(
        [best_complexity, best_complexity],
        [best_rmse_test, best_mae_test],
        marker="*",
        s=180,
        label="Best expr (Test RMSE/MAE)",
    )

    plt.xlabel("Expression Complexity")
    plt.ylabel("Error (RMSE or MAE)")
    plt.yscale("log")
    plt.title("PySR Expression Pareto Front")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(savefig, dpi=300)
    plt.close()

    print(f"Saved Pareto-front plot to: {savefig}")
    print(f"Best expression (by Test-MSE) at complexity={best_complexity}, "
          f"Test RMSE={best_rmse_test:.2f}, Test MAE={best_mae_test:.2f}")

def parity_plot(y_true, y_pred, title, fname=None):
    plt.figure()
    plt.scatter(y_true, y_pred, s=3, alpha=0.3)
    mn = min(y_true.min(), y_pred.min())
    mx = max(y_true.max(), y_pred.max())
    plt.plot([mn, mx], [mn, mx])
    plt.xlabel("True U0")
    plt.ylabel("Predicted U0")
    plt.title(title)
    plt.tight_layout()
    if fname:
        plt.savefig(fname, dpi=300)
    else:
        plt.show()

def plot_feature_importance(importances, top_k=20, fname=None):
    idx = np.argsort(importances)[-top_k:]
    vals = importances[idx]

    plt.figure()
    plt.bar(range(len(idx)), vals)
    plt.xticks(range(len(idx)), idx, rotation=90)
    plt.xlabel("Fingerprint bit index")
    plt.ylabel("Importance")
    plt.title(f"Top {top_k} fingerprint bits")
    plt.tight_layout()
    if fname:
        plt.savefig(fname, dpi=300)
    else:
        plt.show()

def plot_top_features(idx_top, importances, fname="pysr_top_rf_features.png"):
    """
    Plot the RF importances for the selected top-k fingerprint bits.

    Parameters
    ----------
    idx_top : np.ndarray
        Indices of the top-k fingerprint bits (into original 2048-dim vector).
    importances : np.ndarray
        Full importance vector of length 2048 from RF.
    fname : str
        Output filename for the PNG.
    """
    imp_top = importances[idx_top if type(idx_top) == np.ndarray else np.array(idx_top)]
    order = np.argsort(imp_top)
    idx_sorted = idx_top[order]
    imp_sorted = imp_top[order]

    labels = [r"$\mathrm{fp}_{"f"{j}"r"}$" for j in idx_sorted]

    plt.figure(figsize=(10, 4))
    plt.bar(range(len(idx_sorted)), imp_sorted)
    plt.xticks(range(len(idx_sorted)), labels, rotation=90)
    plt.ylabel("RF importance")
    plt.tight_layout()
    plt.savefig(fname, dpi=300)
    plt.close()
    print(f"Saved top-feature plot to {fname}")

def parity_plots_from_expression(expr, feature_names,
                                 X_train, y_train,
                                 X_val, y_val,
                                 X_test, y_test,
                                 prefix="sr_expr"):
    """
    Evaluate a symbolic regression expression on train/val/test sets
    and save parity plots without refitting a model.

    Parameters
    ----------
    expr : sympy expression
        The symbolic equation, e.g. -788.5*(fp_650 + fp_656 + fp_807) - 10249.
    feature_names : list[str]
        Names of variables in the SAME ORDER as columns of X_*.
    X_train, X_val, X_test : np.ndarray
        Feature matrices with shape (n_samples, k).
    y_train, y_val, y_test : np.ndarray
        Target vectors.

    Saves PNGs: prefix + "_parity_train.png", etc.
    """

    # Create callable numpy function via sympy.lambdify
    vars_sym = [sp.symbols(name) for name in feature_names]
    expr_func = sp.lambdify(vars_sym, expr, "numpy")

    def evaluate_and_plot(X, y, split_name):
        # SymPy lambdify requires unpacking columns of X
        y_pred = expr_func(*[X[:, i] for i in range(X.shape[1])])
        fname = f"{prefix}_parity_{split_name}.png"
        parity_plot(y, y_pred,
                    title=f"SR Expression Parity – {split_name.capitalize()}",
                    fname=fname)

    evaluate_and_plot(X_train, y_train, "train")
    evaluate_and_plot(X_val,   y_val,   "val")
    evaluate_and_plot(X_test,  y_test,  "test")
