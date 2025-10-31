"""
02_detect_and_decide.py
Compares current readings (T, pH, RPM) against target setpoints.
Outputs anomaly magnitude and recommended control actions.
"""

import json, sys

CONFIG_PATH = "target_config.json"

def load_config():
    with open(CONFIG_PATH, "r") as f:
        return json.load(f)

def score_and_decide(T, pH, RPM):
    cfg = load_config()
    sp = cfg["target"]
    tol = cfg["tolerance"]

    delta = {
        "T": T - sp["T"],
        "pH": pH - sp["pH"],
        "RPM": RPM - sp["RPM"]
    }

    # Compute normalized deviation (z-like score)
    z = {
        k: abs(delta[k]) / tol[k]
        for k in delta
    }

    # Threshold: anything > 1.0 is considered outside acceptable range
    is_normal = all(v <= 1.0 for v in z.values())

    # Control logic (rule-based)
    actions = {"heater": 0, "ph": 0, "stir": 0}

    # Temperature
    if delta["T"] > tol["T"]: actions["heater"] = -1  # too hot → cool
    elif delta["T"] < -tol["T"]: actions["heater"] = +1  # too cold → heat

    # pH
    if delta["pH"] > tol["pH"]: actions["ph"] = +1    # too basic → add acid (lower pH)
    elif delta["pH"] < -tol["pH"]: actions["ph"] = -1 # too acidic → add base (raise pH)

    # Stirring
    if delta["RPM"] > tol["RPM"]: actions["stir"] = -1  # too fast → slow down
    elif delta["RPM"] < -tol["RPM"]: actions["stir"] = +1  # too slow → speed up

    out = {
        "reading": {"T": T, "pH": pH, "RPM": RPM},
        "target": sp,
        "delta": delta,
        "z_score": z,
        "is_normal": is_normal,
        "actions": actions
    }
    return out

if __name__ == "__main__":
    if len(sys.argv) == 4:
        T, pH, RPM = map(float, sys.argv[1:4])
        print(json.dumps(score_and_decide(T, pH, RPM), indent=2))
    else:
        # Demo example
        print(json.dumps(score_and_decide(44.0, 6.6, 180.0), indent=2))