# live_tolerance_detector.py
import json, joblib, paho.mqtt.client as mqtt

BROKER = "engf0001.cs.ucl.ac.uk"
PORT = 1883
TOPIC = "bioreactor_sim/three_faults/telemetry/summary"

# load learned tolerance model
model = joblib.load("tolerance_model.pkl")
Z_THRESHOLD = 2.0   # 3σ rule

def zscore(delta, key):
    mu, sigma = model[key]["mean"], model[key]["std"]
    return (delta - mu) / sigma if sigma > 1e-9 else 0

def on_connect(client, userdata, flags, rc):
    print("✅ Connected, monitoring...")
    client.subscribe(TOPIC)

def on_message(client, userdata, msg):
    try:
        data = json.loads(msg.payload.decode())
        sp = data["setpoints"]

        # compute deviations
        ΔT  = data["temperature_C"]["mean"] - sp["temperature_C"]
        ΔpH = data["pH"]["mean"] - sp["pH"]
        ΔR  = data["rpm"]["mean"] - sp["rpm"]

        print(f"ΔT={ΔT:+.2f}, ΔpH={ΔpH:+.2f}, ΔRPM={ΔR:+.2f} | ", end="")

        # compute z-scores
        zT, zH, zR = zscore(ΔT, "ΔT"), zscore(ΔpH, "ΔpH"), zscore(ΔR, "ΔRPM")

        # check anomalies per variable
        anomalies = []
        if abs(zT) > Z_THRESHOLD: anomalies.append("Temperature")
        if abs(zH) > Z_THRESHOLD: anomalies.append("pH")
        if abs(zR) > Z_THRESHOLD: anomalies.append("Stirring speed")

        # global classification
        if anomalies:
            cause_text = ", ".join(anomalies)
            print(f"⚠️  ANOMALY ({cause_text}) | "
                  f"zT={zT:+.2f}, zH={zH:+.2f}, zR={zR:+.2f}")
        else:
            print(f"OK | zT={zT:+.2f}, zH={zH:+.2f}, zR={zR:+.2f}")

    except Exception as e:
        print("❌ Error:", e)

client = mqtt.Client()
client.on_connect = on_connect
client.on_message = on_message
client.connect(BROKER, PORT, 60)
client.loop_forever()