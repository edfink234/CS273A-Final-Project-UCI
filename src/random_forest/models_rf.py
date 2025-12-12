# models_rf.py
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.model_selection import RandomizedSearchCV

RANDOM_SEED = 42

def make_base_rf(seed=RANDOM_SEED):
    return RandomForestRegressor(
        random_state=seed,
        n_jobs=-1,
    )

def evaluate(model, X, y):
    y_pred = model.predict(X)
    mse = mean_squared_error(y, y_pred)  # no 'squared' kwarg
    rmse = np.sqrt(mse)
    mae = mean_absolute_error(y, y_pred)
    return rmse, mae, y_pred

def tune_rf(
    X_train,
    y_train,
    seed=RANDOM_SEED,
    subset_frac=0.2,      # use `subset_frac` of train for tuning
    n_iter=15,            # number of random combinations to try
    cv=3,
):
    """
    Tune a RandomForestRegressor using RandomizedSearchCV.

    Parameters
    ----------
    subset_frac : float in (0, 1]
        Fraction of training data to use for the hyperparameter search.
    n_iter : int
        Number of sampled hyperparameter combinations.
    cv : int
        Number of CV folds.
    """

    if subset_frac < 1.0:
        rng = np.random.default_rng(seed)
        n = X_train.shape[0]
        k = int(n * subset_frac)
        indices = rng.choice(n, size=k, replace=False)
        X_tune = X_train[indices]
        y_tune = y_train[indices]
        print(f"Using subset of training data for tuning: {k}/{n} samples")
    else:
        X_tune = X_train
        y_tune = y_train
        print(f"Using full training data for tuning: {X_tune.shape[0]} samples")

    rf = make_base_rf(seed)

    param_dist = {
        "n_estimators": [100, 200],        # fewer, cheaper
        "max_depth": [None, 20, 40],
        "min_samples_leaf": [1, 2, 4],
        "min_samples_split": [2, 5, 10],
        "max_features": ["sqrt", 0.25],    # just two options
    }

    search = RandomizedSearchCV(
        rf,
        param_distributions=param_dist,
        n_iter=n_iter,
        cv=cv,
        scoring="neg_root_mean_squared_error",
        n_jobs=-1,
        random_state=seed,
        verbose=1,
    )

    print("Starting RandomizedSearchCV...")
    search.fit(X_tune, y_tune)
    print("RandomizedSearchCV done.")

    return search.best_estimator_, search
