import pandas as pd
import numpy as np
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.metrics import mean_absolute_percentage_error
import joblib

def load_and_preprocess(csv_path: str):
    """Load synthetic benchmark CSV and engineer features."""
    df = pd.read_csv(csv_path)

    le = LabelEncoder()
    df["gpu_id"] = le.fit_transform(df["gpu_name"])

    prec_map = {"fp32": 0, "bf16": 1, "fp16": 2}
    df["precision_id"] = df["precision"].map(prec_map)

    df["compute_load"] = (df["model_params"] * df["train_steps"]) / df["batch_size"]

    df["log_compute_load"] = np.log1p(df["compute_load"])

    feat_cols = ["gpu_id", "log_compute_load", "vram_gb", "precision_id", "batch_size"]
    scaler = StandardScaler()
    X = scaler.fit_transform(df[feat_cols])
    y = np.log1p(df["runtime_sec"].values)  

    return X, y, le, scaler, df

def train_runtime_predictor(X, y):
    """Train and evaluate an improved runtime prediction model."""
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )
    model = GradientBoostingRegressor(
        n_estimators=300, learning_rate=0.05, max_depth=4,
        subsample=0.7, min_samples_leaf=8, max_features=0.8
    )
    model.fit(X_train, y_train)

    y_pred_log = model.predict(X_test)
    y_pred = np.expm1(y_pred_log)
    y_test_orig = np.expm1(y_test)
    mape = mean_absolute_percentage_error(y_test_orig, y_pred)
    ape = np.abs(y_pred - y_test_orig) / (y_test_orig + 1e-9)
    print(f"Test MAPE:        {mape:.2%}")
    print(f"Test Median APE:  {np.median(ape):.2%}")
    print(f"Test samples: {len(y_test)}")

    cv_scores = cross_val_score(model, X, y, cv=5,
                                scoring="neg_mean_absolute_percentage_error")
    print(f"CV MAPE (5-fold): {-cv_scores.mean():.2%} ± {cv_scores.std():.2%}")

    joblib.dump(model, "runtime_model.pkl")
    joblib.dump(scaler, "scaler.pkl")
    return model, mape

if __name__ == "__main__":
    X, y, le, scaler, df = load_and_preprocess("benchmark_data.csv")
    print(f"Dataset: {len(y)} samples, {df['gpu_name'].nunique()} GPU types\n")
    model, mape = train_runtime_predictor(X, y)
