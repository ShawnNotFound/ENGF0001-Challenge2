

import json

target = {
    "T": 37.0,        # target temperature in °C
    "pH": 7.20,       # target pH
    "RPM": 300.0      # target stirring speed
}

tolerance = {
    "T": 0.5,         # ±0.5°C acceptable range
    "pH": 0.10,       # ±0.10 pH units
    "RPM": 15.0       # ±15 RPM
}

config = {"target": target, "tolerance": tolerance}
with open("target_config.json", "w") as f:
    json.dump(config, f, indent=2)

print("✅ Saved target_config.json:")
print(json.dumps(config, indent=2))