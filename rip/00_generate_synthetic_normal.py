import random
import pandas as pd

# Define a hypothetical "normal" region (center + small natural variation)
SP = {"T": 37.0, "pH": 7.2, "RPM": 300.0}
SD = {"T": 0.3, "pH": 0.05, "RPM": 8.0}

def generate(n=1000, seed=42, out="normal_data.csv"):
    rng = random.Random(seed)
    rows = []
    for _ in range(n):
        T  = rng.gauss(SP["T"],  SD["T"])
        pH = rng.gauss(SP["pH"], SD["pH"])
        R  = rng.gauss(SP["RPM"], SD["RPM"])
        rows.append((T, pH, R))
    df = pd.DataFrame(rows, columns=["T", "pH", "RPM"])
    df.to_csv(out, index=False)
    print(f"✅ Saved {out} with shape {df.shape}")
    print(df.head(10))

if __name__ == "__main__":
    generate()
