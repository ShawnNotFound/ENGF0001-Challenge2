import csv, json, time, statistics
from pathlib import Path
import paho.mqtt.client as mqtt

# ========= CONFIGURATION =========
BROKER = "engf0001.cs.ucl.ac.uk"
PORT = 1883
TOPIC = "bioreactor_sim/three_faults/telemetry/summary"  # change to nofaults / three_faults / variable_setpoints

# Target setpoints and tolerances (from nofaults baseline)
TARGET = {"T": 37.0, "pH": 7.2, "RPM": 300.0}
TOL = {"T": 0.5, "pH": 0.10, "RPM": 15.0}

START_MULT = 2.0    # Start region: 2× tolerance
STOP_MULT = 1.0     # Stop region: 1× tolerance
ANOMALY_TRIGGER = 3 # must see anomaly 3 times in a row before acting
CSV_PATH = (Path(__file__).resolve().parent.parent / "rip" / "normal_data.csv")
CSV_HEADERS = ("T", "pH", "RPM")

# Statistics
anomaly_count = 0
modifying = False
TP = FP = TN = FN = 0
TOTAL = 0
_csv_ready = False

# ========= HELPER FUNCTIONS =========
def _coerce_numeric(value):
    """Best-effort conversion of nested MQTT fields to float."""
    if value is None:
        return None
    if isinstance(value, (int, float)):
        return float(value)
    if isinstance(value, str):
        try:
            return float(value)
        except ValueError:
            return None
    if isinstance(value, dict):
        # look for common numeric subkeys first (value, avg, etc.)
        for key in ("value", "val", "avg", "mean", "current", "reading", "measurement"):
            if key in value:
                coerced = _coerce_numeric(value[key])
                if coerced is not None:
                    return coerced
        for subvalue in value.values():
            coerced = _coerce_numeric(subvalue)
            if coerced is not None:
                return coerced
        return None
    if isinstance(value, (list, tuple)):
        for item in value:
            coerced = _coerce_numeric(item)
            if coerced is not None:
                return coerced
        return None
    return None


def extract_numeric(payload, *keys, default=0.0):
    """Retrieve the first numeric-looking value for the given keys."""
    for key in keys:
        if key in payload:
            value = _coerce_numeric(payload[key])
            if value is not None:
                return value
    return default


def append_reading_to_csv(reading):
    """Persist readings to CSV for offline analysis."""
    global _csv_ready
    if not _csv_ready:
        CSV_PATH.parent.mkdir(parents=True, exist_ok=True)
        if not CSV_PATH.exists() or CSV_PATH.stat().st_size == 0:
            with CSV_PATH.open("w", newline="") as f:
                writer = csv.writer(f)
                writer.writerow(CSV_HEADERS)
        _csv_ready = True
    with CSV_PATH.open("a", newline="") as f:
        writer = csv.writer(f)
        writer.writerow([reading["T"], reading["pH"], reading["RPM"]])



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

        print("Raw data:", data)

        # expected keys: temperature, pH, rpm, maybe faults[]
        reading = {
            "T": extract_numeric(data, "temperature", "temp", "T"),
            "pH": extract_numeric(data, "pH", "ph"),
            "RPM": extract_numeric(data, "rpm", "RPM")
        }
        append_reading_to_csv(reading)

        delta, modifying, actions = detect_anomaly(reading)

        # # print live state
        print(f"T={reading['T']:.2f}, pH={reading['pH']:.2f}, RPM={reading['RPM']:.1f} ")
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

if __name__ == "__main__":
    main()
