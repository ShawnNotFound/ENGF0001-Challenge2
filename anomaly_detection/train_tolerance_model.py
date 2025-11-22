#!/usr/bin/env python3
"""
Fit tolerance model with correlated deltas. We estimate a multivariate normal
model over [ΔT, ΔpH, ΔRPM] so the live detector can reason about each deviation
*and* their covariance (Mahalanobis distance).

New: we can mix multiple CSVs (e.g. nofault stream + fault streams with segments
where no fault is active). Only rows where `fault_active` evaluates to False are
used for training so recovery behaviour is captured without contaminating the
baseline with faulty samples.
"""

import argparse
import json
from pathlib import Path
from typing import Iterable, List

import joblib
import numpy as np
import pandas as pd

FEATURES = ["ΔT", "ΔpH", "ΔRPM"]


def parse_args():
    parser = argparse.ArgumentParser(description="Train correlated tolerance model.")
    parser.add_argument(
        "--data",
        nargs="+",
        default=["normal_data.csv"],
        help="CSV files to use for training. Rows with fault_active=True are ignored.",
    )
    parser.add_argument(
        "--min-samples",
        type=int,
        default=200,
        help="Minimum healthy samples required to fit the model.",
    )
    return parser.parse_args()


def normalize_fault_flags(series: pd.Series) -> pd.Series:
    """Return boolean fault flags (True = fault active)."""
    if series.dtype == bool:
        return series.fillna(False)
    if np.issubdtype(series.dtype, np.number):
        return series.fillna(0).astype(int).astype(bool)
    lowered = series.fillna("").astype(str).str.lower()
    return lowered.isin({"true", "1", "yes", "y"})


def load_healthy_rows(paths: Iterable[str]) -> pd.DataFrame:
    frames: List[pd.DataFrame] = []
    for path in paths:
        csv_path = Path(path)
        if not csv_path.exists():
            raise FileNotFoundError(f"CSV not found: {csv_path}")
        df = pd.read_csv(csv_path)
        if df.empty:
            print(f"⚠️ {csv_path} is empty, skipping.")
            continue
        if "fault_active" in df.columns:
            flags = normalize_fault_flags(df["fault_active"])
            healthy = df[~flags].copy()
            print(
                f"Loaded {csv_path}: {len(df)} rows "
                f"({len(healthy)} healthy, {len(df) - len(healthy)} with faults)."
            )
        else:
            healthy = df.copy()
            print(f"Loaded {csv_path}: {len(df)} rows (no fault labels present).")
        if healthy.empty:
            print(f"⚠️ No healthy rows found in {csv_path}, skipping.")
            continue
        frames.append(healthy)
    if not frames:
        raise SystemExit("No healthy samples available – collect more data.")
    combined = pd.concat(frames, ignore_index=True)
    print(f"Total healthy rows: {len(combined)}")
    return combined


def main():
    args = parse_args()
    df = load_healthy_rows(args.data)
    if len(df) < args.min_samples:
        raise SystemExit(
            f"Only {len(df)} healthy samples available, "
            f"need at least {args.min_samples}. Collect more data."
        )

    missing_features = [col for col in FEATURES if col not in df.columns]
    if missing_features:
        raise SystemExit(f"Missing required columns: {missing_features}")

    feature_matrix = df[FEATURES].to_numpy()
    mean_vec = feature_matrix.mean(axis=0)
    cov = np.cov(feature_matrix, rowvar=False)
    inv_cov = np.linalg.pinv(cov)  # pseudo-inverse keeps things stable

    # Mahalanobis distances from baseline
    deltas = feature_matrix - mean_vec
    mahal_sq = np.einsum("ij,jk,ik->i", deltas, inv_cov, deltas)
    mahal = np.sqrt(mahal_sq)
    threshold = float(np.percentile(mahal, 99.7))  # ~3σ equivalent
    reset_threshold = float(np.percentile(mahal, 95))

    delta_stats = {
        k: {"mean": float(df[k].mean()), "std": float(df[k].std())}
        for k in FEATURES
    }

    model = {
        "feature_names": FEATURES,
        "mean": mean_vec.tolist(),
        "covariance": cov.tolist(),
        "inv_covariance": inv_cov.tolist(),
        "mahal_threshold": threshold,
        "mahal_reset_threshold": reset_threshold,
        "delta_stats": delta_stats,
        "training_rows": int(len(df)),
        "source_files": args.data,
    }

    joblib.dump(model, "tolerance_model.pkl")
    print("✅ Learned tolerance model:")
    print(json.dumps(model, indent=2))


if __name__ == "__main__":
    main()
