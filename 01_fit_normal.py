"""
01_fit_normal.py
Learns the "normal region" from near-optimal datapoints (unsupervised).

Input : normal_data.csv  (columns: T, pH, RPM)
Output: model_normal.pkl (robust mean + covariance) and model_info.json
"""

import json, numpy as np, pandas as pd, joblib
from pathlib import Path

from sklearn.covariance import MinCovDet
from sklearn.preprocessing import StandardScaler

DATA_PATH  = "normal_data.csv"
MODEL_PATH = "model_normal.pkl"
INFO_PATH  = "model_info.json"

# Parameters for cutoff
MAHAL_Q = 0.997   # quantile for anomaly cutoff (~3σ)
Z_BAND  = 2.5     # per-dimension band for action hysteresis

def robust_fit(X: np.ndarray):
    scaler = StandardScaler().fit(X)
    Z = scaler.transform(X)
    mcd = MinCovDet().fit(Z)
    cz = mcd.location_
    cov = mcd.covariance_
    invC = np.linalg.inv(cov)
    d2 = ((Z - cz) @ invC * (Z - cz)).sum(axis=1)
    cutoff = np.quantile(d2, MAHAL_Q)
    stds_orig = scaler.scale_
    band = Z_BAND * stds_orig
    return {
        "use_scaler": True,
        "scaler_mean": scaler.mean_.tolist(),
        "scaler_scale": scaler.scale_.tolist(),
        "center_z": cz.tolist(),
        "inv_cov_z": invC.tolist(),
        "cutoff_mahal2": float(cutoff),
        "band_per_dim": band.tolist(),
        "feature_names": ["T", "pH", "RPM"]
    }
def main():
    df = pd.read_csv(DATA_PATH)
    X = df[["T", "pH", "RPM"]].to_numpy(float)
    model = robust_fit(X)

    joblib.dump(model, MODEL_PATH)
    info = {
        "n_points": int(X.shape[0]),
        "cutoff_mahal2": model["cutoff_mahal2"],
        "band_per_dim": model["band_per_dim"],
        "features": model["feature_names"]
    }
    Path(INFO_PATH).write_text(json.dumps(info, indent=2), encoding="utf-8")
    print(f"✅ Saved {MODEL_PATH} and {INFO_PATH}")
    print(json.dumps(info, indent=2))

if __name__ == "__main__":
    main()
