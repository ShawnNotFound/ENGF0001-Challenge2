"""
03_feedback_loop.py
Enhanced version:
- Includes hysteresis (start region > stop region)
- Requires 3 consecutive anomaly detections before acting
- Stops modifying once readings return to smaller stop region
- Logs full feedback process step by step
"""

import json, random, subprocess, sys, pandas as pd

# ---- System gains and noise ----
GAINS = {"T": 0.5, "pH": 0.3, "RPM": 20.0}
NOISE = {"T": 0.05, "pH": 0.01, "RPM": 2.0}

# ---- Hysteresis parameters ----
ANOMALY_TRIGGER_COUNT = 3  # must detect anomaly 3 times in a row
START_MULTIPLIER = 2.0     # start region = 2 × tolerance
STOP_MULTIPLIER = 1.0      # stop region = 1 × tolerance

def score_and_decide(x):
    """Run the detector and return parsed result."""
    out = subprocess.run(
        [sys.executable, "02_detect_and_decide.py", str(x[0]), str(x[1]), str(x[2])],
        capture_output=True, text=True
    )
    return json.loads(out.stdout)

def apply_action(x, a):
    """Simulate process dynamics."""
    T, pH, R = x
    T  += GAINS["T"] * a["heater"] + random.gauss(0, NOISE["T"])
    pH += (-GAINS["pH"] if a["ph"] == +1 else (GAINS["pH"] if a["ph"] == -1 else 0.0)) + random.gauss(0, NOISE["pH"])
    R  += GAINS["RPM"] * a["stir"] + random.gauss(0, NOISE["RPM"])
    return (T, pH, R)

def in_band(delta, tolerance, multiplier):
    """Check if readings are inside given band multiplier × tolerance."""
    return all(abs(delta[k]) <= multiplier * tolerance[k] for k in delta)

def run_loop(start, tolerance, max_steps=50):
    """Run the feedback control loop with hysteresis and anomaly counter."""
    x = start
    trace = []
    anomaly_count = 0
    modifying = False

    print("\n=============================")
    print(f"Start from: T={x[0]:.3f}, pH={x[1]:.3f}, RPM={x[2]:.3f}")
    print("-----------------------------")

    for step in range(max_steps):
        res = score_and_decide(x)
        delta = res["delta"]
        z = res["z_score"]
        a = res["actions"]

        # Check bands
        in_stop_band = in_band(delta, tolerance, STOP_MULTIPLIER)
        in_start_band = in_band(delta, tolerance, START_MULTIPLIER)

        # --- Decision logic ---
        if not in_start_band:  # strongly abnormal
            anomaly_count += 1
        else:
            anomaly_count = 0  # reset if back inside start band

        # Start modifying after 3 consecutive anomalies
        if anomaly_count >= ANOMALY_TRIGGER_COUNT:
            modifying = True

        # Stop modifying when within smaller stop band
        if in_stop_band:
            modifying = False
            anomaly_count = 0

        # Print step info
        print(f"Step {step:02d}: T={x[0]:.3f}, pH={x[1]:.3f}, RPM={x[2]:.3f} | "
              f"ΔT={delta['T']:+.3f}, ΔpH={delta['pH']:+.3f}, ΔRPM={delta['RPM']:+.3f} | "
              f"AnomalyCount={anomaly_count} | Modifying={modifying} | "
              f"Actions={a}")

        trace.append({
            "step": step, "T": x[0], "pH": x[1], "RPM": x[2],
            "delta": delta, "z": z, "anomaly_count": anomaly_count,
            "modifying": modifying, "actions": a
        })

        # Apply actions only if modification is active
        if modifying:
            x = apply_action(x, a)
        else:
            # Still random environmental drift
            x = (
                x[0] + random.gauss(0, NOISE["T"]),
                x[1] + random.gauss(0, NOISE["pH"]),
                x[2] + random.gauss(0, NOISE["RPM"])
            )

        # Stop completely if stable for a few steps
        if not modifying and in_stop_band:
            print(f"✅ Stable within stop region at step {step}")
            break

    print(f"Final State: T={x[0]:.3f}, pH={x[1]:.3f}, RPM={x[2]:.3f}")
    print("=============================\n")

    return {"start": start, "end": x, "trace": trace}

if __name__ == "__main__":
    # Load tolerance from config
    with open("target_config.json", "r") as f:
        cfg = json.load(f)
    tolerance = cfg["tolerance"]

    scenarios = [
        (44.0, 6.6, 180.0),
        (36.0, 7.5, 400.0),
        (37.2, 7.25, 312.0)
    ]

    summary = []
    for s in scenarios:
        result = run_loop(s, tolerance)
        summary.append({
            "Start_T": round(s[0], 3),
            "Start_pH": round(s[1], 3),
            "Start_RPM": round(s[2], 3),
            "End_T": round(result["end"][0], 3),
            "End_pH": round(result["end"][1], 3),
            "End_RPM": round(result["end"][2], 3)
        })
    pd.DataFrame(summary).to_csv("feedback_results_hysteresis.csv", index=False)
    print("✅ Saved feedback_results_hysteresis.csv")