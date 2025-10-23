"""
02_detect_and_decide.py
Loads the unsupervised "normal model" and, given a reading (T,pH,RPM),
returns:
 - anomaly score (Mahalanobis distance)
 - is_normal flag
 - recommended actions to bring back to normal
"""

import json, numpy as np, joblib, sys

MODEL_PATH = "model_normal.pkl"

def load_model():
    return joblib.load(MODEL_PATH)

def score_and_decide(T, pH, RPM):
    m = load_model()
    x = np.array([T, pH, RPM], dtype=float)
    names = ["T", "pH", "RPM"]

    if m.get("use_scaler", False):
        mu = np.array(m["scaler_mean"])
        sc = np.array(m["scaler_scale"])
        z = (x - mu) / sc
        cz = np.array(m["center_z"])
        invC = np.array(m["inv_cov_z"])
        d2 = float((z - cz).T @ invC @ (z - cz))
        band = np.array(m["band_per_dim"])
        center = mu + sc * cz
    else:
        center = np.array(m["center"])
        invC = np.array(m["inv_cov"])
        d2 = float((x - center).T @ invC @ (x - center))
        band = np.array(m["band_per_dim"])

    delta = x - center

    actions = {"heater": 0, "ph": 0, "stir": 0}
    # Heater
    if   delta[0] >  band[0]: actions["heater"] = -1
    elif delta[0] < -band[0]: actions["heater"] = +1
    # pH
    if   delta[1] >  band[1]: actions["ph"] = +1
    elif delta[1] < -band[1]: actions["ph"] = -1
    # Stirring
    if   delta[2] >  band[2]: actions["stir"] = -1
    elif delta[2] < -band[2]: actions["stir"] = +1

    return {
        "reading": {"T": float(T), "pH": float(pH), "RPM": float(RPM)},
        "center_estimate": {k: float(v) for k, v in zip(names, center)},
        "delta": {k: float(v) for k, v in zip(names, delta)},
        "anomaly_score_mahal2": d2,
        "cutoff_mahal2": m["cutoff_mahal2"],
        "is_normal": bool(d2 <= m["cutoff_mahal2"]),
        "actions": actions,
        "bands": {k: float(v) for k, v in zip(names, band)}
    }

if __name__ == "__main__":
    if len(sys.argv) == 4:
        T, pH, RPM = map(float, sys.argv[1:4])
        print(json.dumps(score_and_decide(T, pH, RPM), indent=2))
    else:
        print(json.dumps(score_and_decide(44.0, 6.6, 180.0), indent=2))
