# data_qm9.py
import pandas as pd
from sklearn.model_selection import train_test_split

RANDOM_SEED = 42
def load_fingerprint_data(path="../../QM9_datasets/qm9_fp_U0.csv"):
    df = pd.read_csv(path)
    y = df["U0"].values
    X = df.loc[:, df.columns.str.startswith("fp_")].values
    return X, y

def get_splits(path="../../QM9_datasets/qm9_fp_U0.csv", seed=RANDOM_SEED, test_size=0.1, val_size=0.1):
    X, y = load_fingerprint_data(path)

    # first split off test
    X_train_val, X_test, y_train_val, y_test = train_test_split(
        X, y, test_size=test_size, random_state=seed
    )

    # split train vs val
    val_fraction_of_train_val = val_size / (1.0 - test_size)
    X_train, X_val, y_train, y_val = train_test_split(
        X_train_val, y_train_val,
        test_size=val_fraction_of_train_val,
        random_state=seed,
    )

    return X_train, X_val, X_test, y_train, y_val, y_test

