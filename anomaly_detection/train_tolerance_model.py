# train_tolerance_model.py
import pandas as pd, joblib, json

df = pd.read_csv("normal_data.csv")

stats = {}
for k in ["ΔT", "ΔpH", "ΔRPM"]:
    stats[k] = {
        "mean": df[k].mean(),
        "std": df[k].std()
    }

joblib.dump(stats, "tolerance_model.pkl")
print("✅ Learned tolerance bands:")
print(json.dumps(stats, indent=2))