# collect_normal_data.py
import json, paho.mqtt.client as mqtt, pandas as pd, time

BROKER = "engf0001.cs.ucl.ac.uk"
PORT = 1883
TOPIC = "bioreactor_sim/nofaults/telemetry/summary"

records = []

def on_connect(client, userdata, flags, rc):
    print("Connected, collecting data...")
    client.subscribe(TOPIC)

def on_message(client, userdata, msg):
    data = json.loads(msg.payload.decode())
    t = data["temperature_C"]["mean"]
    p = data["pH"]["mean"]
    r = data["rpm"]["mean"]
    sp = data["setpoints"]
    records.append({
        "T_mean": t, "pH_mean": p, "RPM_mean": r,
        "ΔT": t - sp["temperature_C"],
        "ΔpH": p - sp["pH"],
        "ΔRPM": r - sp["rpm"]
    })
    if len(records) % 50 == 0:
        print(f"Collected {len(records)} samples")
    if len(records) >= 2000:
        return client.disconnect()

client = mqtt.Client()
client.on_connect = on_connect
client.on_message = on_message
client.connect(BROKER, PORT, 60)

try:
    client.loop_start()
    time.sleep(1200)  # collect for ~2 minutes
finally:
    client.loop_stop()
    pd.DataFrame(records).to_csv("normal_data.csv", index=False)
    print("✅ Saved normal_data.csv")