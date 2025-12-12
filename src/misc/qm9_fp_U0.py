import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error

# 0. Fix the seed to RANDOM_SEED
RANDOM_SEED = 42
# 1. Load data
df = pd.read_csv("../QM9_datasets/qm9_fp_U0.csv")

# 2. Split out target, smiles, and fingerprint matrix
y = df["U0"].values                            # shape (N,)
smiles = df["smiles"].values                   # shape (N,)
X = df.loc[:, df.columns.str.startswith("fp_")].values  # shape (N, 2048)

print("X shape:", X.shape)
print("y shape:", y.shape)

# 3. Train/val/test split (example: 80/10/10)
# First: train+val vs test
X_train_val, X_test, y_train_val, y_test = train_test_split(
    X, y, test_size=0.1, random_state=RANDOM_SEED
)

# Then: train vs val
X_train, X_val, y_train, y_val = train_test_split(
    X_train_val, y_train_val, test_size=0.1111, random_state=RANDOM_SEED
)
# 0.1111 of 0.9 ≈ 0.1, so final ≈ 80/10/10

print("Train:", X_train.shape, "Val:", X_val.shape, "Test:", X_test.shape)

# 4. Train a simple Random Forest
rf = RandomForestRegressor(
    n_estimators=200,
    max_depth=None,
    n_jobs=-1,
    random_state=RANDOM_SEED,
)
rf.fit(X_train, y_train)

# 5. Evaluate
def evaluate(model, X, y, name="set"):
    y_pred = model.predict(X)
    mse = mean_squared_error(y, y_pred)  # no 'squared' kwarg
    rmse = np.sqrt(mse)
    mae = mean_absolute_error(y, y_pred)
    print(f"{name} RMSE: {rmse:.3f}, MAE: {mae:.3f}")
    return y_pred

y_train_pred = evaluate(rf, X_train, y_train, "Train")
y_val_pred   = evaluate(rf, X_val,   y_val,   "Val")
y_test_pred  = evaluate(rf, X_test,  y_test,  "Test")

#X shape: (129012, 2048)
#y shape: (129012,)
#Train: (103210, 2048) Val: (12900, 2048) Test: (12902, 2048)
#Train RMSE: 248.927, MAE: 161.509
#Val RMSE: 661.756, MAE: 433.208
#Test RMSE: 679.305, MAE: 440.144

