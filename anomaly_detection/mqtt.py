"""
mqtt_anomaly_detector.py
ENGF0001 Bioreactor Anomaly Detection in MQTT Streams
Author: Shawn + ChatGPT
Date: 2025-10
------------------------------------------------------
- Connects to UCL MQTT broker and subscribes to telemetry streams
- Applies band-based anomaly detection (start/stop region + 3 anomaly trigger)
- Prints live detection results with TP/FP statistics (if labels available)
------------------------------------------------------
Requires:
    pip install paho-mqtt
Run:
    python mqtt_anomaly_detector.py
"""

import json, time, statistics
import paho.mqtt.client as mqtt

# ========= CONFIGURATION =========
BROKER = "engf0001.cs.ucl.ac.uk"
PORT = 1883
TOPIC = "bioreactor_sim/single_fault/telemetry/summary"  # change to nofaults / three_faults / variable_setpoints

# Target setpoints and tolerances (from nofaults baseline)
TARGET = {"T": 37.0, "pH": 7.2, "RPM": 300.0}
TOL = {"T": 0.5, "pH": 0.10, "RPM": 15.0}

START_MULT = 2.0    # Start region: 2× tolerance
STOP_MULT = 1.0     # Stop region: 1× tolerance
ANOMALY_TRIGGER = 3 # must see anomaly 3 times in a row before acting

# Statistics
anomaly_count = 0
modifying = False
TP = FP = TN = FN = 0
TOTAL = 0

# ========= HELPER FUNCTIONS =========
def in_band(delta, mult):
    """Check if all deviations are within multiplier × tolerance."""
    return all(abs(delta[k]) <= mult * TOL[k] for k in delta)

def detect_anomaly(reading):
    """Core detection logic."""
    global anomaly_count, modifying

    # compute deviation from target
    delta = {k: reading[k] - TARGET[k] for k in TARGET}

    # check region bands
    in_stop = in_band(delta, STOP_MULT)
    in_start = in_band(delta, START_MULT)

    # anomaly persistence logic
    if not in_start:
        anomaly_count += 1
    else:
        anomaly_count = 0

    if anomaly_count >= ANOMALY_TRIGGER:
        modifying = True
    if in_stop:
        modifying = False
        anomaly_count = 0

    # control actions
    actions = {"heater": 0, "ph": 0, "stir": 0}
    if modifying:
        if delta["T"] >  TOL["T"]:  actions["heater"] = -1
        elif delta["T"] < -TOL["T"]: actions["heater"] = +1
        if delta["pH"] >  TOL["pH"]:  actions["ph"] = +1
        elif delta["pH"] < -TOL["pH"]: actions["ph"] = -1
        if delta["RPM"] >  TOL["RPM"]: actions["stir"] = -1
        elif delta["RPM"] < -TOL["RPM"]: actions["stir"] = +1

    return delta, modifying, actions

# ========= MQTT CALLBACKS =========
def on_connect(client, userdata, flags, rc):
    print(f"✅ Connected to MQTT broker {BROKER} with result code {rc}")
    client.subscribe(TOPIC)
    print(f"📡 Subscribed to topic: {TOPIC}\n")

def on_message(client, userdata, msg):
    print("📥 Message received")
    global TP, FP, TN, FN, TOTAL
    try:
        data = json.loads(msg.payload.decode())

        # expected keys: temperature, pH, rpm, maybe faults[]
        reading = {
            "T": float(data.get("temperature", 0)),
            "pH": float(data.get("pH", 0)),
            "RPM": float(data.get("rpm", 0))
        }

        delta, modifying, actions = detect_anomaly(reading)

        # print live state
        print(f"T={reading['T']:.2f}, pH={reading['pH']:.2f}, RPM={reading['RPM']:.1f} | "
              f"ΔT={delta['T']:+.2f}, ΔpH={delta['pH']:+.2f}, ΔRPM={delta['RPM']:+.1f} | "
              f"AnomCount={anomaly_count} | Modifying={modifying} | Actions={actions}")

        # scoring if fault labels are included in the stream
        faults = data.get("faults", [])
        fault_present = (len(faults) > 0)
        TOTAL += 1
        if fault_present and modifying: TP += 1
        elif not fault_present and not modifying: TN += 1
        elif fault_present and not modifying: FN += 1
        elif not fault_present and modifying: FP += 1

        if TOTAL % 30 == 0:  # show periodic summary
            precision = TP / (TP + FP + 1e-9)
            recall = TP / (TP + FN + 1e-9)
            print(f"\n--- Summary after {TOTAL} samples ---")
            print(f"TP={TP}, FP={FP}, TN={TN}, FN={FN}")
            print(f"Precision={precision:.3f}, Recall={recall:.3f}\n")

    except Exception as e:
        print("❌ Error processing message:", e)

# ========= MAIN =========
def main():
    print("Starting ENGF0001 Bioreactor Anomaly Detector...")
    print(f"Connecting to broker {BROKER}:{PORT}")
    client = mqtt.Client()
    client.on_connect = on_connect
    client.on_message = on_message

    try:
        client.connect(BROKER, PORT, 60)
    except Exception as e:
        print("❌ Could not connect to broker:", e)
        print("Make sure you are on UCL network or VPN.")
        return

    try:
        client.loop_forever()
    except KeyboardInterrupt:
        print("\n🛑 Stopped by user.")
        if TOTAL > 0:
            print(f"Final Score: TP={TP}, FP={FP}, TN={TN}, FN={FN}")

if __name__ == "__main__":
    main()