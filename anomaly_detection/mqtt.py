#!/usr/bin/env python3
"""
collect_data.py

Utility to grab telemetry summaries from the ENG0001 MQTT broker. Use the
`nofaults` stream to build normal baselines and `single_fault` / `three_faults`
to gather labelled fault data for evaluation.
"""

import argparse
import json
import time
from pathlib import Path

import pandas as pd
import paho.mqtt.client as mqtt

DEFAULT_HOST = "engf0001.cs.ucl.ac.uk"
DEFAULT_PORT = 1883
SCENARIOS = ("nofaults", "single_fault", "three_faults", "variable_setpoints")


def parse_args():
    parser = argparse.ArgumentParser(
        description="Collect telemetry summaries from ENG0001 MQTT broker."
    )
    parser.add_argument("--host", default=DEFAULT_HOST, help="MQTT broker hostname.")
    parser.add_argument("--port", default=DEFAULT_PORT, type=int, help="MQTT broker port.")
    parser.add_argument(
        "--scenario",
        choices=SCENARIOS,
        default="nofaults",
        help="Simulation stream to follow.",
    )
    parser.add_argument(
        "--samples",
        type=int,
        default=2000,
        help="Number of samples to collect before stopping.",
    )
    parser.add_argument(
        "--max-seconds",
        type=int,
        default=12000,
        help="Safety timeout to stop collecting if not enough samples were received.",
    )
    parser.add_argument(
        "--output",
        default=None,
        help="CSV path. Defaults to normal_data.csv for nofaults, "
             "or <scenario>_data.csv otherwise.",
    )
    return parser.parse_args()


def parse_fault_labels(last_active):
    if not last_active:
        return []
    entries = last_active
    if isinstance(entries, dict):
        entries = [entries]
    labels = []
    for entry in entries:
        label = None
        if isinstance(entry, str):
            label = entry
        elif isinstance(entry, dict):
            label = entry.get("name") or entry.get("fault") or entry.get("id")
        else:
            label = str(entry)
        if label:
            labels.append(str(label))
    return labels


def build_record(data, scenario):
    sp = data["setpoints"]
    t = data["temperature_C"]["mean"]
    p = data["pH"]["mean"]
    r = data["rpm"]["mean"]
    faults_raw = data.get("faults", {}).get("last_active", [])
    fault_labels = parse_fault_labels(faults_raw)
    timestamp = data.get("timestamp") or data.get("ts") or int(time.time() * 1000)

    record = {
        "timestamp": timestamp,
        "scenario": scenario,
        "fault_active": bool(fault_labels),
        "fault_labels": ";".join(fault_labels),
        "T_mean": t,
        "T_min": data["temperature_C"].get("min"),
        "T_max": data["temperature_C"].get("max"),
        "pH_mean": p,
        "pH_min": data["pH"].get("min"),
        "pH_max": data["pH"].get("max"),
        "RPM_mean": r,
        "RPM_min": data["rpm"].get("min"),
        "RPM_max": data["rpm"].get("max"),
        "set_T": sp["temperature_C"],
        "set_pH": sp["pH"],
        "set_RPM": sp["rpm"],
        "ΔT": t - sp["temperature_C"],
        "ΔpH": p - sp["pH"],
        "ΔRPM": r - sp["rpm"],
    }
    return record


def main():
    args = parse_args()
    output = args.output
    if not output:
        output = "normal_data.csv" if args.scenario == "nofaults" else f"{args.scenario}_data.csv"
    output_path = Path(output)

    topic = f"bioreactor_sim/{args.scenario}/telemetry/summary"
    records = []

    def on_connect(client, _userdata, _flags, rc):
        if rc == 0:
            print(f"✅ Connected to {args.host}:{args.port}, subscribing to {topic}")
            client.subscribe(topic)
        else:
            print(f"❌ Connection failed (rc={rc})")

    def on_message(_client, _userdata, msg):
        try:
            data = json.loads(msg.payload.decode())
            records.append(build_record(data, args.scenario))
            if len(records) % 50 == 0:
                print(f"Collected {len(records)} samples...")
            if len(records) >= args.samples:
                _client.disconnect()
        except Exception as exc:
            print("⚠️ Failed to parse message:", exc)

    client = mqtt.Client()
    client.on_connect = on_connect
    client.on_message = on_message
    client.connect(args.host, args.port, keepalive=60)

    start = time.time()
    client.loop_start()
    try:
        while len(records) < args.samples and (time.time() - start) < args.max_seconds:
            time.sleep(0.5)
        if client.is_connected():
            client.disconnect()
    finally:
        client.loop_stop()

    if not records:
        print("⚠️ No data collected, skipping CSV export.")
        return

    pd.DataFrame(records).to_csv(output_path, index=False)
    print(f"✅ Saved {len(records)} samples to {output_path}")


if __name__ == "__main__":
    main()
