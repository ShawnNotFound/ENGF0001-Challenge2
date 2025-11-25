#!/usr/bin/env python3
"""
Quick precision/recall diagnostics for eval logs.
Falls back to binary metrics when per-fault labels are missing.
"""

import argparse
import json
from typing import Dict, Optional

import pandas as pd

DEFAULT_CLASSES = ["therm_voltage_bias", "ph_offset_bias", "heater_power_loss"]
EXPECTED_COLUMNS = [
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


def parse_args():
    parser = argparse.ArgumentParser(description="Evaluate per-fault metrics from eval.csv")
    parser.add_argument("--csv", default="eval.csv", help="Path to eval CSV (default: eval.csv)")
    parser.add_argument(
        "--threshold", type=float, default=0.7, help="Sequence probability threshold"
    )
    parser.add_argument(
        "--classes",
        default=",".join(DEFAULT_CLASSES),
        help="Comma-separated fault classes to evaluate.",
    )
    return parser.parse_args()


def parse_probs(cell: str) -> Dict:
    if not cell or not isinstance(cell, str) or not cell.strip():
        return {}
    try:
        return json.loads(cell)
    except Exception:
        return {}


def find_label_column(df: pd.DataFrame) -> Optional[str]:
    for candidate in ["fault_labels", "fault_label", "faults", "labels"]:
        if candidate in df.columns:
            return candidate
    return None


def binary_metrics(df: pd.DataFrame):
    if not {"fault_active", "anomaly_detected"} <= set(df.columns):
        print("⚠️ CSV lacks fault_active/anomaly_detected; cannot compute binary metrics.")
        return
    gt = df["fault_active"].astype(bool)
    pred = df["anomaly_detected"].astype(bool)
    tp = ((gt) & pred).sum()
    fp = ((~gt) & pred).sum()
    fn = ((gt) & (~pred)).sum()
    tn = ((~gt) & (~pred)).sum()
    prec = tp / (tp + fp + 1e-9)
    rec = tp / (tp + fn + 1e-9)
    print(
        f"Binary (any fault): prec={prec:.3f}, rec={rec:.3f}, "
        f"tp={tp}, fp={fp}, fn={fn}, tn={tn}"
    )


def main():
    args = parse_args()
    classes = [c.strip() for c in args.classes.split(",") if c.strip()]
    df = pd.read_csv(args.csv)
    if not set(EXPECTED_COLUMNS) <= set(df.columns):
        if len(df.columns) == len(EXPECTED_COLUMNS):
            df = pd.read_csv(args.csv, names=EXPECTED_COLUMNS, header=None)
        else:
            print("⚠️ No fault label column found. Available columns:", list(df.columns))
            binary_metrics(df)
            return
    label_col = find_label_column(df)
    if not label_col:
        print("⚠️ No fault label column found. Available columns:", list(df.columns))
        binary_metrics(df)
        return

    for cls in classes:
        gt = df[label_col].fillna("").astype(str).str.contains(cls)
        preds = df.get("sequence_probs", pd.Series([""] * len(df))).fillna("").apply(
            lambda s: parse_probs(s).get(cls, 0) >= args.threshold
        )
        tp = ((gt) & preds).sum()
        fp = ((~gt) & preds).sum()
        fn = ((gt) & (~preds)).sum()
        tn = ((~gt) & (~preds)).sum()
        prec = tp / (tp + fp + 1e-9)
        rec = tp / (tp + fn + 1e-9)
        print(
            f"{cls}: prec={prec:.3f}, rec={rec:.3f}, "
            f"tp={tp}, fp={fp}, fn={fn}, tn={tn}"
        )


if __name__ == "__main__":
    main()
