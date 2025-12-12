# train_rf.py
from data_qm9 import get_splits
from models_rf import tune_rf, evaluate
from plots import parity_plot, plot_feature_importance
from sklearn.ensemble import RandomForestRegressor

USE_TUNED_RF = True #set to True to for hyperparameter tuning, or False to run a fixed "baseline" Random Forest.
RANDOM_SEED = 42

def train_simple_rf(X_train, y_train, seed=RANDOM_SEED):
    """Baseline Random Forest without hyperparameter search."""
    rf = RandomForestRegressor(
        n_estimators=200,
        max_depth=None,
        min_samples_split=2,
        min_samples_leaf=1,
        max_features="sqrt",
        n_jobs=-1,
        random_state=seed,
    )
    rf.fit(X_train, y_train)
    return rf


def main():
    X_train, X_val, X_test, y_train, y_val, y_test = get_splits()

    if USE_TUNED_RF:
        print("=== Running TUNED Random Forest (RandomizedSearchCV) ===")
        best_rf, search = tune_rf(
            X_train,
            y_train,
            seed=RANDOM_SEED,
            subset_frac=0.1,  # fraction of TRAIN used for tuning
            n_iter=10,
            cv=3,
        )
        print("Best params:", search.best_params_)
        print("Best CV RMSE:", -search.best_score_)
    else:
        print("=== Running BASELINE Random Forest (no tuning) ===")
        best_rf = train_simple_rf(X_train, y_train, seed=RANDOM_SEED)

    # Evaluate on train / val / test and make parity plots
    for name, X, y in [
        ("Train", X_train, y_train),
        ("Val",   X_val,   y_val),
        ("Test",  X_test,  y_test),
    ]:
        rmse, mae, y_pred = evaluate(best_rf, X, y)
        print(f"{name} RMSE: {rmse:.3f}, MAE: {mae:.3f}")
        parity_plot(
            y,
            y_pred,
            title=f"RF Parity – {name}",
            fname=f"rf_parity_{name.lower()}_{'tuned' if USE_TUNED_RF else 'baseline'}.png",
        )

    # Feature importance plot (works for both baseline and tuned RF)
    importances = best_rf.feature_importances_
    plot_feature_importance(
        importances,
        top_k=20,
        fname=f"rf_feature_importance_{'tuned' if USE_TUNED_RF else 'baseline'}.png",
    )


if __name__ == "__main__":
    main()
