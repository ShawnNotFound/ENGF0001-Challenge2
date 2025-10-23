"""
03_feedback_loop.py
Simulates a feedback loop using the unsupervised anomaly detector.
At each step:
 - score current state
 - get recommended actions
 - apply simulated process changes
Stop when reading is back within the learned normal region.
"""

import json, random, subprocess, sys, pandas as pd

GAINS = {"T": 0.5, "pH": 0.3, "RPM": 20.0}
NOISE = {"T": 0.05, "pH": 0.01, "RPM": 2.0}

def score_and_decide(x):
    out = subprocess.run(
        [sys.executable, "02_detect_and_decide.py", str(x[0]), str(x[1]), str(x[2])],
        capture_output=True, text=True
    )
    return json.loads(out.stdout)

def apply_action(x, a):
    T, pH, R = x
    T += GAINS["T"] * a["heater"] + random.gauss(0, NOISE["T"])
    pH += (-GAINS["pH"] if a["ph"] == +1 else (GAINS["pH"] if a["ph"] == -1 else 0)) + random.gauss(0, NOISE["pH"])
    R += GAINS["RPM"] * a["stir"] + random.gauss(0, NOISE["RPM"])
    return (T, pH, R)

def within_bands(delta, bands):
    return all(abs(delta[k]) <= bands[k] for k in ["T", "pH", "RPM"])

def run_loop(start, max_steps=40):
    x = start
    trace = []
    for step in range(max_steps):
        res = score_and_decide(x)
        trace.append({"step": step, **res})
        if res["is_normal"] and within_bands(res["delta"], res["bands"]):
            break
        x = apply_action(x, res["actions"])
    return {
        "start": start,
        "end": x,
        "converged": res["is_normal"] and within_bands(res["delta"], res["bands"]),
        "trace": trace
    }

if __name__ == "__main__":
    scenarios = [
        (44.0, 6.6, 180.0),
        (36.0, 7.5, 400.0),
        (37.2, 7.25, 312.0)
    ]
    rows = []
    for s in scenarios:
        out = run_loop(s, 40)
        rows.append({
            "Start_T": round(s[0], 3),
            "Start_pH": round(s[1], 3),
            "Start_RPM": round(s[2], 3),
            "End_T": round(out["end"][0], 3),
            "End_pH": round(out["end"][1], 3),
            "End_RPM": round(out["end"][2], 3),
            "Converged": out["converged"],
            "Steps": len(out["trace"]) - 1
        })
        print(f"Start {s} -> End {tuple(round(v,3) for v in out['end'])} | Converged={out['converged']} | Steps={len(out['trace'])-1}")
    pd.DataFrame(rows).to_csv("feedback_results.csv", index=False)
    print("✅ Saved feedback_results.csv")
