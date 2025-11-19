# live_tolerance_detector.py
import json, joblib, paho.mqtt.client as mqtt, time

# BROKER = "engf0001.cs.ucl.ac.uk"
BROKER = "localhost"
PORT = 1883
# TOPIC = "bioreactor_sim/three_faults/telemetry/summary"
TOPIC = "bioreactor/anomaly"
ALERT_TOPIC = "bioreactor/anomaly/status"

# load learned tolerance model
model = joblib.load("tolerance_model.pkl")
Z_THRESHOLD = 3.0   # 3σ rule
Z_LOW_THRESHOLD = 2.0  # reset anomaly counter below this
TP = FP = TN = FN = 0
temp_anomaly_count = ph_anomaly_count = stirring_anomaly_count = 0

def zscore(delta, key):
    mu, sigma = model[key]["mean"], model[key]["std"]
    return (delta - mu) / sigma if sigma > 1e-9 else 0

def on_connect(client, userdata, flags, rc):
    print("✅ Connected, monitoring...")
    client.subscribe(TOPIC)

def update_confusion(fault_present, anomaly_detected):
    global TP, FP, TN, FN
    if fault_present and anomaly_detected:
        TP += 1
    elif fault_present and not anomaly_detected:
        FN += 1
    elif not fault_present and anomaly_detected:
        FP += 1
    else:
        TN += 1

def confusion_text():
    return f"Matrix → TP={TP}, FP={FP}, TN={TN}, FN={FN}"

def call_alarm():
    print("Anomaly Confirmed!")

def severity_level(z):
    val = abs(z)
    if val > Z_THRESHOLD:
        return 2  # critical / red
    if val > Z_LOW_THRESHOLD:
        return 1  # caution / yellow
    return 0

def publish_status(client, zT, zH, zR, delta_t, delta_h, delta_r, anomalies, confirmed):
    payload = {
        "timestamp": int(time.time() * 1000),
        "temperature": {"z": zT, "delta": delta_t, "level": severity_level(zT)},
        "ph": {"z": zH, "delta": delta_h, "level": severity_level(zH)},
        "stirring_pwm": {"z": zR, "delta": delta_r, "level": severity_level(zR)},
        "anomalies": anomalies,
        "confirmed": confirmed
    }
    message = ", ".join(anomalies) if anomalies else "No anomalies"
    payload["message"] = f"Status: {message}"
    try:
        client.publish(ALERT_TOPIC, json.dumps(payload), qos=0, retain=False)
    except Exception as exc:
        print("⚠️ Could not publish status:", exc)

def on_message(client, userdata, msg):
    global temp_anomaly_count, ph_anomaly_count, stirring_anomaly_count
    try:
        data = json.loads(msg.payload.decode())
        sp = data["setpoints"]

        faults = data.get("faults", {}).get("last_active", [])
        fault_present = len(faults) > 0
        if fault_present:
            print(f"⚠️⚠️  FAULTS ACTIVE: {faults} | ", end="")
        else:
            print("No faults | ", end="")

        # compute deviations
        ΔT  = data["temperature_C"]["mean"] - sp["temperature_C"]
        ΔpH = data["pH"]["mean"] - sp["pH"]
        ΔR  = data["rpm"]["mean"] - sp["rpm"]

        print(f"ΔT={ΔT:+.2f}, ΔpH={ΔpH:+.2f}, ΔRPM={ΔR:+.2f} | ", end="")

        # compute z-scores
        zT, zH, zR = zscore(ΔT, "ΔT"), zscore(ΔpH, "ΔpH"), zscore(ΔR, "ΔRPM")

        # check anomalies per variable
        anomalies = []
        if abs(zT) > Z_THRESHOLD: 
            temp_anomaly_count += 1
            anomalies.append("Temperature")
        if abs(zH) > Z_THRESHOLD: 
            ph_anomaly_count += 1
            anomalies.append("pH")
        if abs(zR) > Z_THRESHOLD: 
            stirring_anomaly_count += 1
            anomalies.append("Stirring speed")

        confirmed = temp_anomaly_count > 3 or ph_anomaly_count > 3 or stirring_anomaly_count > 3
        if confirmed:
            call_alarm()
        
        if abs(zT) <= Z_LOW_THRESHOLD:
            temp_anomaly_count = 0
        if abs(zH) <= Z_LOW_THRESHOLD:
            ph_anomaly_count = 0
        if abs(zR) <= Z_LOW_THRESHOLD:
            stirring_anomaly_count = 0

        # global classification
        anomaly_detected = bool(anomalies)
        update_confusion(fault_present, anomaly_detected)

        if anomaly_detected:
            cause_text = ", ".join(anomalies)
            print(f"⚠️  ANOMALY ({cause_text}) | "
                  f"zT={zT:+.2f}, zH={zH:+.2f}, zR={zR:+.2f} | {confusion_text()}")
        else:
            print(f"OK | zT={zT:+.2f}, zH={zH:+.2f}, zR={zR:+.2f} | {confusion_text()}")

        publish_status(client, zT, zH, zR, ΔT, ΔpH, ΔR, anomalies, confirmed)

    except Exception as e:
        print("❌ Error:", e)

client = mqtt.Client()
client.on_connect = on_connect
client.on_message = on_message
client.connect(BROKER, PORT, 60)
client.loop_forever()
