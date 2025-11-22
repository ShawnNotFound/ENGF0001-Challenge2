#!/usr/bin/env python3
"""
Live anomaly detector that combines univariate z-scores with a correlated
Mahalanobis check and an optional LSTM sequence model. Works with the ENG0001
MQTT streams (`nofaults`, `single_fault`, `three_faults`) or the local
`bioreactor/anomaly` topic for offline testing.

Pass `--offline-csv data.csv ...` to replay captured telemetry (e.g. from the
collector) and evaluate the detector without connecting to MQTT. Use
`--sequence-model sequence_model.pt` to enable the temporal fault classifier.
"""

import argparse
import csv
import json
import time
from collections import deque
from pathlib import Path
from typing import Dict, List

import joblib
import numpy as np
import pandas as pd
import paho.mqtt.client as mqtt

try:
    import torch
except ImportError:  # pragma: no cover - optional dependency
    torch = None


def parse_args():
    parser = argparse.ArgumentParser(description="Monitor ENG0001 telemetry for anomalies.")
    parser.add_argument("--host", default="engf0001.cs.ucl.ac.uk", help="MQTT broker hostname.")
    parser.add_argument("--port", default=1883, type=int, help="MQTT broker port.")
    parser.add_argument(
        "--scenario",
        choices=["single_fault", "three_faults", "nofaults"],
        default="single_fault",
        help="Which ENG0001 stream to subscribe to when --topic is not provided.",
    )
    parser.add_argument(
        "--topic",
        default=None,
        help="Override MQTT topic. If omitted we subscribe to "
             "bioreactor_sim/<scenario>/telemetry/summary.",
    )
    parser.add_argument(
        "--alert-topic",
        default="bioreactor/anomaly/status",
        help="Topic to publish detector status to.",
    )
    parser.add_argument("--model", default="tolerance_model.pkl", help="Path to model file.")
    parser.add_argument(
        "--z-threshold", type=float, default=3.0,
        help="Per-signal z-score to flag local anomalies."
    )
    parser.add_argument(
        "--z-reset", type=float, default=2.0,
        help="Reset z-score counters once magnitudes fall below this level."
    )
    parser.add_argument(
        "--confirmation-window", type=int, default=3,
        help="Consecutive anomaly count before confirming alarm.",
    )
    parser.add_argument(
        "--mahal-threshold", type=float, default=None,
        help="Override Mahalanobis anomaly threshold.",
    )
    parser.add_argument(
        "--mahal-reset", type=float, default=None,
        help="Override Mahalanobis reset threshold.",
    )
    parser.add_argument(
        "--offline-csv",
        nargs="+",
        default=None,
        help="Replay detector against cached CSV(s) instead of live MQTT.",
    )
    parser.add_argument(
        "--offline-sleep",
        type=float,
        default=0.0,
        help="Seconds to wait between samples while replaying CSVs.",
    )
    parser.add_argument(
        "--no-publish",
        action="store_true",
        help="Skip publishing status packets (useful for offline evaluation).",
    )
    parser.add_argument(
        "--log-csv",
        default=None,
        help="Optional CSV to append evaluation results (fault labels + detected anomalies).",
    )
    parser.add_argument(
        "--sequence-model",
        default=None,
        help="Optional PyTorch LSTM checkpoint for temporal fault classification.",
    )
    parser.add_argument(
        "--sequence-threshold",
        type=float,
        default=None,
        help="Override probability threshold for the sequence model (default uses checkpoint value).",
    )
    return parser.parse_args()


args = parse_args()
TOPIC = args.topic or f"bioreactor_sim/{args.scenario}/telemetry/summary"
ALERT_TOPIC = args.alert_topic

model = joblib.load(args.model)
FEATURES = model["feature_names"]
MEAN_VEC = np.array(model["mean"])
INV_COV = np.array(model["inv_covariance"])
DELTA_STATS = model["delta_stats"]
MAHAL_THRESHOLD = args.mahal_threshold or model["mahal_threshold"]
MAHAL_RESET = args.mahal_reset or model["mahal_reset_threshold"]
FRIENDLY = {"ΔT": "Temperature", "ΔpH": "pH", "ΔRPM": "Stirring speed"}

Z_THRESHOLD = args.z_threshold
Z_LOW_THRESHOLD = args.z_reset

TP = FP = TN = FN = 0
feature_counters = {k: 0 for k in FEATURES}
mahal_counter = 0
PUBLISH_CLIENT = None
LOG_WRITER = None
LOG_FILE = None
LOG_FIELDS = [
    "timestamp",
    "source",
    "fault_active",
    "fault_labels",
    "anomaly_detected",
    "confirmed",
    "anomalies",
    "mahalanobis",
    "zT",
    "zPH",
    "zRPM",
    "ΔT",
    "ΔpH",
    "ΔRPM",
    "sequence_faults",
    "sequence_probs",
]
SEQUENCE_MODEL = None
SEQUENCE_CFG = None
SEQUENCE_BUFFER = None
SEQUENCE_THRESHOLD = None


def zscore(delta, key):
    stats = DELTA_STATS[key]
    sigma = stats["std"]
    return (delta - stats["mean"]) / sigma if sigma > 1e-9 else 0.0


def mahalanobis(vec):
    diff = vec - MEAN_VEC
    return float(np.sqrt(diff @ INV_COV @ diff.T))


def on_connect(client, _userdata, _flags, rc):
    if rc == 0:
        print(f"✅ Connected to {args.host}:{args.port}, monitoring {TOPIC}")
        client.subscribe(TOPIC)
    else:
        print(f"❌ Connection failed (rc={rc})")


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
    print("🚨 Multivariate anomaly confirmed!")


def severity_level(z):
    val = abs(z)
    if val > Z_THRESHOLD:
        return 2
    if val > Z_LOW_THRESHOLD:
        return 1
    return 0


def publish_status(payload):
    if args.no_publish or PUBLISH_CLIENT is None:
        return
    try:
        PUBLISH_CLIENT.publish(ALERT_TOPIC, json.dumps(payload), qos=0, retain=False)
    except Exception as exc:
        print("⚠️ Could not publish status:", exc)


def init_logger():
    global LOG_WRITER, LOG_FILE
    if not args.log_csv:
        return
    path = Path(args.log_csv)
    exists = path.exists()
    LOG_FILE = path.open("a", newline="")
    LOG_WRITER = csv.DictWriter(LOG_FILE, fieldnames=LOG_FIELDS)
    if not exists:
        LOG_WRITER.writeheader()


def log_sample(record: Dict):
    if not LOG_WRITER:
        return
    LOG_WRITER.writerow(record)
    LOG_FILE.flush()


def init_sequence_model():
    global SEQUENCE_MODEL, SEQUENCE_CFG, SEQUENCE_BUFFER, SEQUENCE_THRESHOLD
    if not args.sequence_model:
        return
    if torch is None:
        raise SystemExit(
            "PyTorch is required for --sequence-model but is not installed. "
            "Install torch (CPU build is fine) and retry."
        )
    from sequence_model import build_lstm_model  # imported lazily

    checkpoint = torch.load(args.sequence_model, map_location="cpu")
    config = checkpoint["config"]
    normalization = checkpoint["normalization"]
    SEQUENCE_MODEL = build_lstm_model(config)
    SEQUENCE_MODEL.load_state_dict(checkpoint["state_dict"])
    SEQUENCE_MODEL.eval()
    SEQUENCE_CFG = {
        "feature_columns": config["feature_columns"],
        "fault_types": config["fault_types"],
        "window": config["window"],
        "mean": np.array(normalization["mean"], dtype=np.float32),
        "std": np.array(normalization["std"], dtype=np.float32),
    }
    SEQUENCE_THRESHOLD = args.sequence_threshold or config.get("threshold", 0.5)
    SEQUENCE_BUFFER = deque(maxlen=SEQUENCE_CFG["window"])
    print(
        f"✅ Loaded sequence model ({args.sequence_model}) with window={SEQUENCE_CFG['window']} "
        f"and threshold={SEQUENCE_THRESHOLD}"
    )


def build_status_payload(z_scores, deltas, anomalies, confirmed, mah_score, fault_text):
    timestamp = int(time.time() * 1000)
    return {
        "timestamp": timestamp,
        "temperature": {
            "z": z_scores["ΔT"],
            "delta": deltas["ΔT"],
            "level": severity_level(z_scores["ΔT"]),
        },
        "ph": {
            "z": z_scores["ΔpH"],
            "delta": deltas["ΔpH"],
            "level": severity_level(z_scores["ΔpH"]),
        },
        "stirring_pwm": {
            "z": z_scores["ΔRPM"],
            "delta": deltas["ΔRPM"],
            "level": severity_level(z_scores["ΔRPM"]),
        },
        "mahalanobis": {
            "value": mah_score,
            "threshold": MAHAL_THRESHOLD,
            "level": int(mah_score > MAHAL_THRESHOLD),
        },
        "anomalies": anomalies,
        "confirmed": confirmed,
        "message": fault_text,
        "topic": TOPIC,
    }


def extract_sequence_features(data: Dict, deltas: Dict[str, float]):
    if not SEQUENCE_CFG:
        return None
    sp = data["setpoints"]
    mapping = {
        "ΔT": deltas["ΔT"],
        "ΔpH": deltas["ΔpH"],
        "ΔRPM": deltas["ΔRPM"],
        "T_mean": data["temperature_C"]["mean"],
        "pH_mean": data["pH"]["mean"],
        "RPM_mean": data["rpm"]["mean"],
        "set_T": sp["temperature_C"],
        "set_pH": sp["pH"],
        "set_RPM": sp["rpm"],
    }
    return np.array([mapping[col] for col in SEQUENCE_CFG["feature_columns"]], dtype=np.float32)


def run_sequence_inference(feature_vector: np.ndarray):
    global SEQUENCE_BUFFER
    if SEQUENCE_MODEL is None:
        return None
    normalized = (feature_vector - SEQUENCE_CFG["mean"]) / SEQUENCE_CFG["std"]
    SEQUENCE_BUFFER.append(normalized)
    if len(SEQUENCE_BUFFER) < SEQUENCE_CFG["window"]:
        return None
    window = np.stack(SEQUENCE_BUFFER, axis=0)
    tensor = torch.from_numpy(window).unsqueeze(0)
    with torch.no_grad():
        logits = SEQUENCE_MODEL(tensor)
        probs = torch.sigmoid(logits).cpu().numpy()[0]
    target_names = SEQUENCE_CFG["fault_types"]
    prob_map = {name: float(probs[idx]) for idx, name in enumerate(target_names)}
    detected = [
        name for idx, name in enumerate(target_names[:-1]) if probs[idx] >= SEQUENCE_THRESHOLD
    ]
    any_prob = prob_map.get("any_fault", None)
    sequence_flag = (any_prob is not None and any_prob >= SEQUENCE_THRESHOLD) or bool(detected)
    return {
        "probabilities": prob_map,
        "detected": detected,
        "any_prob": any_prob,
        "flagged": sequence_flag,
    }


def handle_sample(data: Dict, source: str):
    global feature_counters, mahal_counter
    sp = data["setpoints"]
    raw_faults = data.get("faults", {}).get("last_active", []) or []
    faults = []
    for entry in raw_faults:
        if isinstance(entry, str):
            faults.append(entry)
        elif isinstance(entry, dict):
            name = entry.get("name") or entry.get("fault") or entry.get("id")
            if name:
                faults.append(str(name))
        else:
            faults.append(str(entry))
    fault_present = len(faults) > 0
    fault_text = (
        f"[{source}] Faults active: {', '.join(faults)}"
        if faults else f"[{source}] No faults reported"
    )

    deltas = {
        "ΔT": data["temperature_C"]["mean"] - sp["temperature_C"],
        "ΔpH": data["pH"]["mean"] - sp["pH"],
        "ΔRPM": data["rpm"]["mean"] - sp["rpm"],
    }
    z_scores = {k: zscore(v, k) for k, v in deltas.items()}
    vector = np.array([deltas[k] for k in FEATURES])
    mah_score = mahalanobis(vector)

    anomalies = []
    for key, z in z_scores.items():
        if abs(z) > Z_THRESHOLD:
            feature_counters[key] += 1
            anomalies.append(FRIENDLY[key])
        elif abs(z) <= Z_LOW_THRESHOLD:
            feature_counters[key] = 0

    multivariate_flag = mah_score > MAHAL_THRESHOLD
    if multivariate_flag:
        mahal_counter += 1
        if "Multivariate" not in anomalies:
            anomalies.append("Multivariate")
    elif mah_score < MAHAL_RESET:
        mahal_counter = 0

    confirmed = (
        max(feature_counters.values(), default=0) >= args.confirmation_window
        or mahal_counter >= args.confirmation_window
    )
    if confirmed:
        call_alarm()

    anomaly_detected = bool(anomalies)
    sequence_info = None
    seq_features = extract_sequence_features(data, deltas)
    if seq_features is not None:
        sequence_info = run_sequence_inference(seq_features)
        if sequence_info and sequence_info["flagged"]:
            seq_faults = sequence_info["detected"]
            if seq_faults:
                anomalies.extend([f"LSTM:{fault}" for fault in seq_faults])
            else:
                anomalies.append("LSTM:any_fault")
            anomaly_detected = True

    update_confusion(fault_present, anomaly_detected)

    z_text = (
        f"ΔT={deltas['ΔT']:+.2f} (z={z_scores['ΔT']:+.2f}), "
        f"ΔpH={deltas['ΔpH']:+.2f} (z={z_scores['ΔpH']:+.2f}), "
        f"ΔRPM={deltas['ΔRPM']:+.2f} (z={z_scores['ΔRPM']:+.2f})"
    )
    mah_text = f"Mahalanobis={mah_score:.2f} / {MAHAL_THRESHOLD:.2f}"

    log_parts = [fault_text, z_text, mah_text]
    if sequence_info:
        probs_text = ", ".join(
            f"{name}:{sequence_info['probabilities'][name]:.2f}"
            for name in sequence_info["probabilities"]
        )
        log_parts.append(f"LSTM={probs_text}")
    log_prefix = "⚠️" if anomaly_detected else "✅"
    print(f"{log_prefix} " + " | ".join(log_parts) + f" | {confusion_text()}")

    status_payload = build_status_payload(
        z_scores,
        deltas,
        anomalies,
        confirmed,
        mah_score,
        fault_text,
    )
    if sequence_info:
        status_payload["sequence_model"] = {
            "threshold": SEQUENCE_THRESHOLD,
            "window": SEQUENCE_CFG["window"] if SEQUENCE_CFG else None,
            "probabilities": sequence_info["probabilities"],
            "detected": sequence_info["detected"],
        }
    publish_status(status_payload)
    record = {
        "timestamp": status_payload["timestamp"],
        "source": source,
        "fault_active": int(fault_present),
        "fault_labels": ";".join(faults),
        "anomaly_detected": int(anomaly_detected),
        "confirmed": int(confirmed),
        "anomalies": ";".join(anomalies),
        "mahalanobis": mah_score,
        "zT": z_scores["ΔT"],
        "zPH": z_scores["ΔpH"],
        "zRPM": z_scores["ΔRPM"],
        "ΔT": deltas["ΔT"],
        "ΔpH": deltas["ΔpH"],
        "ΔRPM": deltas["ΔRPM"],
        "sequence_faults": ";".join(sequence_info["detected"]) if sequence_info else "",
        "sequence_probs": json.dumps(sequence_info["probabilities"]) if sequence_info else "",
    }
    log_sample(record)
    return record


def on_message(_client, _userdata, msg):
    try:
        data = json.loads(msg.payload.decode())
        handle_sample(data, source=TOPIC)
    except Exception as exc:
        print("❌ Error:", exc)


def bool_from_value(value) -> bool:
    if value is None or (isinstance(value, float) and np.isnan(value)):
        return False
    if isinstance(value, (bool, np.bool_)):
        return bool(value)
    if isinstance(value, (int, np.integer, float, np.floating)):
        return bool(value)
    return str(value).strip().lower() in {"true", "1", "yes", "y"}


def safe_float(record: Dict, key: str, fallback: str = None):
    if key in record:
        val = record[key]
        if val is not None and not pd.isna(val):
            return float(val)
    if fallback and fallback in record:
        val = record[fallback]
        if val is not None and not pd.isna(val):
            return float(val)
    return None


def csv_record_to_payload(record: Dict, source: str) -> Dict:
    set_t = safe_float(record, "set_T")
    set_ph = safe_float(record, "set_pH")
    set_rpm = safe_float(record, "set_RPM")
    if any(x is None for x in (set_t, set_ph, set_rpm)):
        raise ValueError(f"{source}: missing setpoint columns.")

    payload = {
        "temperature_C": {
            "mean": safe_float(record, "T_mean"),
            "min": safe_float(record, "T_min", fallback="T_mean"),
            "max": safe_float(record, "T_max", fallback="T_mean"),
        },
        "pH": {
            "mean": safe_float(record, "pH_mean"),
            "min": safe_float(record, "pH_min", fallback="pH_mean"),
            "max": safe_float(record, "pH_max", fallback="pH_mean"),
        },
        "rpm": {
            "mean": safe_float(record, "RPM_mean"),
            "min": safe_float(record, "RPM_min", fallback="RPM_mean"),
            "max": safe_float(record, "RPM_max", fallback="RPM_mean"),
        },
        "setpoints": {
            "temperature_C": set_t,
            "pH": set_ph,
            "rpm": set_rpm,
        },
        "timestamp": int(record["timestamp"]) if "timestamp" in record and not pd.isna(record["timestamp"]) else int(time.time() * 1000),
    }

    fault_active = bool_from_value(record.get("fault_active"))
    labels = []
    if fault_active:
        raw = str(record.get("fault_labels") or "")
        labels = [label.strip() for label in raw.split(";") if label.strip()]
    payload["faults"] = {"last_active": labels}
    return payload


def replay_offline(paths: List[str]):
    total = 0
    for csv_path in paths:
        file_path = Path(csv_path)
        if not file_path.exists():
            print(f"⚠️ Offline CSV not found: {file_path}")
            continue
        df = pd.read_csv(file_path)
        if df.empty:
            print(f"⚠️ Offline CSV empty: {file_path}")
            continue
        print(f"▶️ Replaying {file_path} ({len(df)} rows)")
        for record in df.to_dict("records"):
            try:
                payload = csv_record_to_payload(record, str(file_path))
            except ValueError as exc:
                print(f"⚠️ {exc}")
                continue
            handle_sample(payload, source=file_path.name)
            total += 1
            if args.offline_sleep > 0:
                time.sleep(args.offline_sleep)
    print(f"Offline replay complete ({total} samples) | {confusion_text()}")


def main():
    global PUBLISH_CLIENT
    init_logger()
    init_sequence_model()
    if args.offline_csv:
        if not args.no_publish:
            PUBLISH_CLIENT = mqtt.Client()
            PUBLISH_CLIENT.connect(args.host, args.port, 60)
            PUBLISH_CLIENT.loop_start()
        replay_offline(args.offline_csv)
        if PUBLISH_CLIENT:
            PUBLISH_CLIENT.loop_stop()
            PUBLISH_CLIENT.disconnect()
        if LOG_FILE:
            LOG_FILE.close()
        return

    client = mqtt.Client()
    if not args.no_publish:
        PUBLISH_CLIENT = client
    client.on_connect = on_connect
    client.on_message = on_message
    client.connect(args.host, args.port, 60)
    try:
        client.loop_forever()
    finally:
        if LOG_FILE:
            LOG_FILE.close()


if __name__ == "__main__":
    main()
