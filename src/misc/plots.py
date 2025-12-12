# plots.py
import matplotlib.pyplot as plt
import numpy as np
import sympy as sp

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
