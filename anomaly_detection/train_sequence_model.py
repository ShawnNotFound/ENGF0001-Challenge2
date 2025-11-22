#!/usr/bin/env python3
"""
Train an LSTM-based anomaly detector that captures temporal dependencies between
temperature, pH, RPM, and their setpoints. The model performs multi-label
classification to predict which fault(s) are active within a sliding time
window (and also outputs a general "any fault" score).
"""

import argparse
import json
import math
import random
from pathlib import Path
from typing import Dict, Iterable, List, Sequence, Tuple

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader, Dataset

from sequence_model import LSTMAnomalyDetector

FEATURE_COLUMNS = [
    "ΔT",
    "ΔpH",
    "ΔRPM",
    "T_mean",
    "pH_mean",
    "RPM_mean",
    "set_T",
    "set_pH",
    "set_RPM",
]

FAULT_TYPES = ["therm_voltage_bias", "ph_offset_bias", "heater_power_loss"]
TARGET_NAMES = FAULT_TYPES + ["any_fault"]


def parse_args():
    parser = argparse.ArgumentParser(description="Train LSTM anomaly detector.")
    parser.add_argument(
        "--data",
        nargs="+",
        required=True,
        help="CSV files (e.g., normal_data.csv, single_fault.csv, three_faults.csv).",
    )
    parser.add_argument("--window", type=int, default=60, help="Sequence length (samples).")
    parser.add_argument(
        "--stride",
        type=int,
        default=5,
        help="Step between windows (controls sample overlap).",
    )
    parser.add_argument("--epochs", type=int, default=25, help="Training epochs.")
    parser.add_argument("--batch-size", type=int, default=128, help="Mini-batch size.")
    parser.add_argument("--lr", type=float, default=1e-3, help="Adam learning rate.")
    parser.add_argument(
        "--hidden-size", type=int, default=128, help="LSTM hidden dimension per layer."
    )
    parser.add_argument("--layers", type=int, default=2, help="Number of LSTM layers.")
    parser.add_argument("--dropout", type=float, default=0.3, help="Dropout probability.")
    parser.add_argument("--val-ratio", type=float, default=0.2, help="Validation split.")
    parser.add_argument("--seed", type=int, default=42, help="Random seed.")
    parser.add_argument(
        "--output",
        default="sequence_model.pt",
        help="Path to save the trained PyTorch model.",
    )
    parser.add_argument(
        "--threshold",
        type=float,
        default=0.5,
        help="Default probability threshold for inference.",
    )
    return parser.parse_args()


def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def load_csv(path: str) -> pd.DataFrame:
    csv_path = Path(path)
    if not csv_path.exists():
        raise FileNotFoundError(f"CSV not found: {csv_path}")
    df = pd.read_csv(csv_path)
    if df.empty:
        raise ValueError(f"{csv_path} is empty.")
    return df.sort_values(by="timestamp").reset_index(drop=True)


def ensure_columns(df: pd.DataFrame):
    missing = [col for col in FEATURE_COLUMNS if col not in df.columns]
    if missing:
        raise ValueError(f"Missing columns in dataframe: {missing}")
    if "fault_labels" not in df.columns:
        df["fault_labels"] = ""


def parse_fault_labels(cell) -> List[str]:
    if isinstance(cell, str):
        if not cell.strip():
            return []
        return [label.strip() for label in cell.split(";") if label.strip()]
    if isinstance(cell, float) and math.isnan(cell):
        return []
    if isinstance(cell, (list, tuple)):
        return [str(x).strip() for x in cell if str(x).strip()]
    return [str(cell).strip()] if cell else []


def compute_normalization(dfs: Iterable[pd.DataFrame]) -> Tuple[np.ndarray, np.ndarray]:
    stacked = np.concatenate([df[FEATURE_COLUMNS].to_numpy(dtype=np.float32) for df in dfs], axis=0)
    mean = stacked.mean(axis=0)
    std = stacked.std(axis=0)
    std[std < 1e-6] = 1e-6
    return mean, std


def faults_to_vector(labels: Sequence[str]) -> np.ndarray:
    vec = np.zeros(len(TARGET_NAMES), dtype=np.float32)
    if not labels:
        return vec
    label_set = set(labels)
    for idx, name in enumerate(FAULT_TYPES):
        if name in label_set:
            vec[idx] = 1.0
    if label_set:
        vec[-1] = 1.0
    return vec


def window_sequences(
    df: pd.DataFrame,
    normalized_features: np.ndarray,
    window: int,
    stride: int,
) -> Tuple[List[np.ndarray], List[np.ndarray]]:
    sequences: List[np.ndarray] = []
    labels: List[np.ndarray] = []
    faults_series = df["fault_labels"].tolist()
    for start in range(0, len(df) - window + 1, stride):
        end = start + window
        seq = normalized_features[start:end]
        window_faults: List[str] = []
        for k in range(start, end):
            window_faults.extend(parse_fault_labels(faults_series[k]))
        label = faults_to_vector(window_faults)
        sequences.append(seq.astype(np.float32))
        labels.append(label)
    return sequences, labels


class SequenceDataset(Dataset):
    def __init__(self, sequences: List[np.ndarray], labels: List[np.ndarray]):
        self.sequences = sequences
        self.labels = labels

    def __len__(self):
        return len(self.sequences)

    def __getitem__(self, idx):
        seq = torch.from_numpy(self.sequences[idx])
        label = torch.from_numpy(self.labels[idx])
        return seq, label


def build_datasets(
    dfs: List[pd.DataFrame],
    mean: np.ndarray,
    std: np.ndarray,
    window: int,
    stride: int,
    val_ratio: float,
) -> Tuple[SequenceDataset, SequenceDataset]:
    sequences: List[np.ndarray] = []
    labels: List[np.ndarray] = []
    for df in dfs:
        normalized = (df[FEATURE_COLUMNS].to_numpy(dtype=np.float32) - mean) / std
        seqs, labs = window_sequences(df, normalized, window, stride)
        sequences.extend(seqs)
        labels.extend(labs)

    if not sequences:
        raise SystemExit("No sequences produced. Try reducing --window or collect more data.")

    indices = np.arange(len(sequences))
    np.random.shuffle(indices)
    split = int(len(indices) * (1 - val_ratio))
    split = min(max(split, 1), len(indices) - 1)
    train_idx, val_idx = indices[:split], indices[split:]

    def subset(idxs):
        return [sequences[i] for i in idxs], [labels[i] for i in idxs]

    train_sequences, train_labels = subset(train_idx)
    val_sequences, val_labels = subset(val_idx)

    print(f"Total sequences: {len(sequences)} | Train: {len(train_sequences)} | Val: {len(val_sequences)}")

    return SequenceDataset(train_sequences, train_labels), SequenceDataset(val_sequences, val_labels)


def run_epoch(model, loader, criterion, optimizer, device):
    model.train()
    total_loss = 0.0
    for batch_x, batch_y in loader:
        batch_x = batch_x.to(device)
        batch_y = batch_y.to(device)
        optimizer.zero_grad()
        logits = model(batch_x)
        loss = criterion(logits, batch_y)
        loss.backward()
        optimizer.step()
        total_loss += loss.item() * batch_x.size(0)
    return total_loss / len(loader.dataset)


def evaluate(model, loader, criterion, device, threshold):
    model.eval()
    total_loss = 0.0
    preds = []
    trues = []
    with torch.no_grad():
        for batch_x, batch_y in loader:
            batch_x = batch_x.to(device)
            batch_y = batch_y.to(device)
            logits = model(batch_x)
            loss = criterion(logits, batch_y)
            total_loss += loss.item() * batch_x.size(0)
            probs = torch.sigmoid(logits).cpu().numpy()
            preds.append((probs > threshold).astype(np.float32))
            trues.append(batch_y.cpu().numpy())
    if not preds:
        return {"loss": total_loss, "precision": 0, "recall": 0, "f1": 0}

    preds_arr = np.concatenate(preds, axis=0)
    trues_arr = np.concatenate(trues, axis=0)
    eps = 1e-9
    tp = (preds_arr * trues_arr).sum(axis=0)
    fp = (preds_arr * (1 - trues_arr)).sum(axis=0)
    fn = ((1 - preds_arr) * trues_arr).sum(axis=0)
    precision = (tp + eps) / (tp + fp + eps)
    recall = (tp + eps) / (tp + fn + eps)
    f1 = 2 * precision * recall / (precision + recall + eps)
    metrics = {
        "loss": total_loss / len(loader.dataset),
        "precision": precision.tolist(),
        "recall": recall.tolist(),
        "f1": f1.tolist(),
    }
    return metrics


if __name__ == "__main__":
    args = parse_args()
    set_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    dataframes = []
    for path in args.data:
        df = load_csv(path)
        ensure_columns(df)
        dataframes.append(df)
        print(f"Loaded {path}: {len(df)} rows")

    mean, std = compute_normalization(dataframes)
    train_dataset, val_dataset = build_datasets(
        dataframes, mean, std, args.window, args.stride, args.val_ratio
    )

    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, drop_last=False)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False, drop_last=False)

    model = LSTMAnomalyDetector(
        input_size=len(FEATURE_COLUMNS),
        hidden_size=args.hidden_size,
        num_layers=args.layers,
        dropout=args.dropout,
        output_dim=len(TARGET_NAMES),
    ).to(device)

    criterion = torch.nn.BCEWithLogitsLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)

    best_val = float("inf")
    best_state = None

    for epoch in range(1, args.epochs + 1):
        train_loss = run_epoch(model, train_loader, criterion, optimizer, device)
        val_metrics = evaluate(model, val_loader, criterion, device, args.threshold)
        val_loss = val_metrics["loss"]
        f1_any = val_metrics["f1"][-1]
        print(
            f"Epoch {epoch:02d} | Train loss={train_loss:.4f} | "
            f"Val loss={val_loss:.4f} | Any-fault F1={f1_any:.3f}"
        )
        if val_loss < best_val:
            best_val = val_loss
            best_state = model.state_dict()

    if best_state is not None:
        model.load_state_dict(best_state)

    checkpoint = {
        "state_dict": model.state_dict(),
        "config": {
            "input_size": len(FEATURE_COLUMNS),
            "hidden_size": args.hidden_size,
            "num_layers": args.layers,
            "dropout": args.dropout,
            "output_dim": len(TARGET_NAMES),
            "feature_columns": FEATURE_COLUMNS,
            "fault_types": TARGET_NAMES,
            "window": args.window,
            "threshold": args.threshold,
        },
        "normalization": {
            "mean": mean.tolist(),
            "std": std.tolist(),
        },
        "metadata": {
            "data_files": args.data,
            "window": args.window,
            "stride": args.stride,
            "epochs": args.epochs,
            "batch_size": args.batch_size,
            "val_ratio": args.val_ratio,
            "device": str(device),
        },
    }

    torch.save(checkpoint, args.output)
    print(f"✅ Saved sequence model to {args.output}")
    print("Thresholds / metrics (last epoch):")
    print(
        json.dumps(
            {
                "threshold": args.threshold,
                "precision": val_metrics["precision"],
                "recall": val_metrics["recall"],
                "f1": val_metrics["f1"],
            },
            indent=2,
        )
    )
