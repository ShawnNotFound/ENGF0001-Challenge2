#!/usr/bin/env python3
"""
Hybrid Ensemble Anomaly Detection Model
Combines LSTM and Random Forest for optimal per-fault detection:
- LSTM for Therm Voltage Bias (F1: 0.9279)
- Random Forest for pH Offset Bias (F1: 0.9617)
- Random Forest for Heater Power Loss (F1: 0.6147)
"""

import pandas as pd
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import confusion_matrix, precision_score, recall_score, f1_score, accuracy_score
import warnings
warnings.filterwarnings('ignore')


# ============================================================================
# LSTM Components
# ============================================================================

class LSTMAnomalyDetector(nn.Module):
    """LSTM-based anomaly detector"""
    def __init__(self, input_size: int, hidden_size: int = 128, num_layers: int = 2,
                 dropout: float = 0.3, output_dim: int = 4):
        super().__init__()
        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0.0,
        )
        self.head = nn.Sequential(
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_size, output_dim),
        )

    def forward(self, x):
        seq_out, _ = self.lstm(x)
        last_hidden = seq_out[:, -1, :]
        return self.head(last_hidden)


class SequenceDataset(Dataset):
    def __init__(self, sequences, labels):
        self.sequences = sequences
        self.labels = labels

    def __len__(self):
        return len(self.sequences)

    def __getitem__(self, idx):
        seq = torch.from_numpy(self.sequences[idx])
        label = torch.from_numpy(self.labels[idx])
        return seq, label


# ============================================================================
# Random Forest Components
# ============================================================================

class RFDetector:
    """Random Forest detector for a single fault type"""
    def __init__(self, fault_type, features, n_estimators=100):
        self.fault_type = fault_type
        self.features = features
        self.model = RandomForestClassifier(
            n_estimators=n_estimators,
            max_depth=10,
            min_samples_split=20,
            min_samples_leaf=10,
            class_weight='balanced',
            random_state=42
        )

    def fit(self, X_train, y_train):
        self.model.fit(X_train[self.features], y_train)
        return self

    def predict(self, X):
        return self.model.predict(X[self.features])


# ============================================================================
# Data Loading and Preparation
# ============================================================================

def load_and_prepare_data(filepath):
    """Load CSV and prepare all features for both models"""
    df = pd.read_csv(filepath)
    df = df.sort_values('timestamp').reset_index(drop=True)

    # Add temporal features for RF
    for col in ['T_mean', 'pH_mean', 'RPM_mean']:
        df[f'{col}_diff'] = df[col].diff().fillna(0)
        df[f'{col}_rolling_mean'] = df[col].rolling(window=5, min_periods=1).mean()
        df[f'{col}_rolling_std'] = df[col].rolling(window=5, min_periods=1).std().fillna(0)

    # Add interaction features for RF
    df['T_spread'] = df['T_max'] - df['T_min']
    df['pH_spread'] = df['pH_max'] - df['pH_min']
    df['RPM_spread'] = df['RPM_max'] - df['RPM_min']

    # Labels
    df['has_heater_power_loss'] = df['fault_labels'].fillna('').str.contains('heater_power_loss').astype(int)
    df['has_therm_voltage_bias'] = df['fault_labels'].fillna('').str.contains('therm_voltage_bias').astype(int)
    df['has_ph_offset_bias'] = df['fault_labels'].fillna('').str.contains('ph_offset_bias').astype(int)

    return df


def create_lstm_sequences(df, feature_columns, mean, std, window=60, stride=5):
    """Create sliding window sequences for LSTM"""
    normalized = (df[feature_columns].to_numpy(dtype=np.float32) - mean) / std

    sequences = []
    labels = []

    for start in range(0, len(df) - window + 1, stride):
        end = start + window
        seq = normalized[start:end].astype(np.float32)

        window_has_therm = df.iloc[start:end]['fault_labels'].fillna('').str.contains('therm_voltage_bias').any()
        window_has_ph = df.iloc[start:end]['fault_labels'].fillna('').str.contains('ph_offset_bias').any()
        window_has_heater = df.iloc[start:end]['fault_labels'].fillna('').str.contains('heater_power_loss').any()
        window_has_any = df.iloc[start:end]['fault_active'].any()

        label = np.array([
            1.0 if window_has_therm else 0.0,
            1.0 if window_has_ph else 0.0,
            1.0 if window_has_heater else 0.0,
            1.0 if window_has_any else 0.0
        ], dtype=np.float32)

        sequences.append(seq)
        labels.append(label)

    return np.array(sequences, dtype=np.float32), np.array(labels, dtype=np.float32)


# ============================================================================
# Training Functions
# ============================================================================

def train_lstm(model, loader, criterion, optimizer, device):
    """Train LSTM for one epoch"""
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


def evaluate_lstm(model, loader, device, threshold=0.5):
    """Evaluate LSTM model"""
    model.eval()
    preds = []
    with torch.no_grad():
        for batch_x, batch_y in loader:
            batch_x = batch_x.to(device)
            logits = model(batch_x)
            probs = torch.sigmoid(logits).cpu().numpy()
            preds.append((probs > threshold).astype(np.float32))
    return np.concatenate(preds, axis=0)


# ============================================================================
# Main Training and Evaluation
# ============================================================================

def main():
    print("="*70)
    print("HYBRID ENSEMBLE MODEL")
    print("="*70)
    print("\nStrategy:")
    print("  • LSTM for Therm Voltage Bias")
    print("  • Random Forest for pH Offset Bias")
    print("  • Random Forest for Heater Power Loss")

    # Configuration
    WINDOW = 60
    STRIDE = 5
    LSTM_EPOCHS = 25
    BATCH_SIZE = 128
    LSTM_LR = 1e-3
    VAL_RATIO = 0.2

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Load data
    print("\nLoading data...")
    df = load_and_prepare_data('full_data.csv')
    print(f"Total samples: {len(df)}")

    # Temporal split - 80% train, 20% validation
    split_idx = int(len(df) * (1 - VAL_RATIO))
    df_train = df.iloc[:split_idx].reset_index(drop=True)
    df_val = df.iloc[split_idx:].reset_index(drop=True)

    print(f"Train samples: {len(df_train)}")
    print(f"Val samples: {len(df_val)}")

    # ========================================================================
    # PART 1: Train LSTM for Therm Voltage Bias
    # ========================================================================
    print("\n" + "="*70)
    print("TRAINING LSTM FOR THERM VOLTAGE BIAS")
    print("="*70)

    lstm_features = ['ΔT', 'ΔpH', 'ΔRPM', 'T_mean', 'pH_mean', 'RPM_mean',
                     'set_T', 'set_pH', 'set_RPM']

    # Compute normalization
    feature_data = df_train[lstm_features].to_numpy(dtype=np.float32)
    mean = feature_data.mean(axis=0).astype(np.float32)
    std = feature_data.std(axis=0).astype(np.float32)
    std[std < 1e-6] = 1e-6

    # Create sequences
    train_seq, train_labels = create_lstm_sequences(df_train, lstm_features, mean, std, WINDOW, STRIDE)
    val_seq, val_labels = create_lstm_sequences(df_val, lstm_features, mean, std, WINDOW, STRIDE)

    print(f"Train sequences: {len(train_seq)}")
    print(f"Val sequences: {len(val_seq)}")

    # Create dataloaders
    train_dataset = SequenceDataset(train_seq, train_labels)
    val_dataset = SequenceDataset(val_seq, val_labels)
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False)

    # Train LSTM
    lstm_model = LSTMAnomalyDetector(
        input_size=len(lstm_features),
        hidden_size=128,
        num_layers=2,
        dropout=0.3,
        output_dim=4
    ).to(device)

    criterion = nn.BCEWithLogitsLoss()
    optimizer = torch.optim.Adam(lstm_model.parameters(), lr=LSTM_LR)

    best_lstm_state = None
    best_val_loss = float('inf')

    for epoch in range(1, LSTM_EPOCHS + 1):
        train_loss = train_lstm(lstm_model, train_loader, criterion, optimizer, device)
        val_pred = evaluate_lstm(lstm_model, val_loader, device)
        f1_therm = f1_score(val_labels[:, 0], val_pred[:, 0], zero_division=0)

        if epoch % 5 == 0 or epoch == LSTM_EPOCHS:
            print(f"Epoch {epoch:02d} | Train loss={train_loss:.4f} | Therm F1={f1_therm:.3f}")

        if train_loss < best_val_loss:
            best_val_loss = train_loss
            best_lstm_state = lstm_model.state_dict()

    if best_lstm_state:
        lstm_model.load_state_dict(best_lstm_state)

    # ========================================================================
    # PART 2: Train Random Forest for pH and Heater
    # ========================================================================
    print("\n" + "="*70)
    print("TRAINING RANDOM FOREST MODELS")
    print("="*70)

    rf_configs = {
        'heater_power_loss': {
            'features': ['T_mean', 'T_min', 'T_max', 'ΔT', 'T_spread',
                        'T_mean_diff', 'T_mean_rolling_mean', 'T_mean_rolling_std',
                        'RPM_mean', 'ΔRPM']
        },
        'ph_offset_bias': {
            'features': ['pH_mean', 'pH_min', 'pH_max', 'ΔpH', 'pH_spread',
                        'pH_mean_diff', 'pH_mean_rolling_mean', 'pH_mean_rolling_std',
                        'RPM_mean', 'RPM_spread', 'ΔRPM', 'T_mean', 'T_min']
        }
    }

    rf_models = {}
    for fault_type, config in rf_configs.items():
        print(f"\nTraining RF for {fault_type}...")
        y_train = df_train[f'has_{fault_type}']
        detector = RFDetector(fault_type, config['features'], n_estimators=100)
        detector.fit(df_train, y_train)
        rf_models[fault_type] = detector

    # ========================================================================
    # PART 3: Evaluate Hybrid Ensemble
    # ========================================================================
    print("\n" + "="*70)
    print("HYBRID ENSEMBLE EVALUATION")
    print("="*70)

    # Get LSTM predictions for therm
    val_pred_lstm = evaluate_lstm(lstm_model, val_loader, device)
    lstm_therm_pred = val_pred_lstm[:, 0]

    # Map LSTM sequence predictions to row-level predictions
    row_therm_pred = np.zeros(len(df_val))
    for seq_idx, start_idx in enumerate(range(0, len(df_val) - WINDOW + 1, STRIDE)):
        end_idx = start_idx + WINDOW
        for row in range(start_idx, min(end_idx, len(df_val))):
            row_therm_pred[row] = lstm_therm_pred[seq_idx]

    # Get RF predictions
    rf_ph_pred = rf_models['ph_offset_bias'].predict(df_val)
    rf_heater_pred = rf_models['heater_power_loss'].predict(df_val)

    # Combine predictions
    hybrid_pred_therm = (row_therm_pred > 0.5).astype(int)
    hybrid_pred_ph = rf_ph_pred
    hybrid_pred_heater = rf_heater_pred
    hybrid_pred_any = ((hybrid_pred_therm + hybrid_pred_ph + hybrid_pred_heater) > 0).astype(int)

    # Ground truth
    y_true_therm = df_val['has_therm_voltage_bias'].values
    y_true_ph = df_val['has_ph_offset_bias'].values
    y_true_heater = df_val['has_heater_power_loss'].values
    y_true_any = df_val['fault_active'].astype(int).values

    # Evaluate
    faults = [
        ('Therm Voltage Bias', hybrid_pred_therm, y_true_therm, 'LSTM'),
        ('pH Offset Bias', hybrid_pred_ph, y_true_ph, 'RF'),
        ('Heater Power Loss', hybrid_pred_heater, y_true_heater, 'RF'),
        ('ANY FAULT', hybrid_pred_any, y_true_any, 'Hybrid')
    ]

    for fault_name, y_pred, y_true, model_type in faults:
        tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
        precision = precision_score(y_true, y_pred, zero_division=0)
        recall = recall_score(y_true, y_pred, zero_division=0)
        f1 = f1_score(y_true, y_pred, zero_division=0)

        print(f"\n{fault_name} ({model_type}):")
        print(f"  F1: {f1:.4f} | Precision: {precision:.4f} | Recall: {recall:.4f}")
        print(f"  TP: {tp:>5} | FP: {fp:>5} | TN: {tn:>5} | FN: {fn:>5}")

    # Save models
    print("\n" + "="*70)
    print("SAVING MODELS")
    print("="*70)

    # Save LSTM
    lstm_checkpoint = {
        'state_dict': lstm_model.state_dict(),
        'config': {
            'input_size': len(lstm_features),
            'hidden_size': 128,
            'num_layers': 2,
            'dropout': 0.3,
            'output_dim': 4,
            'feature_columns': lstm_features,
            'window': WINDOW,
            'threshold': 0.5,
        },
        'normalization': {
            'mean': mean.tolist(),
            'std': std.tolist(),
        }
    }
    torch.save(lstm_checkpoint, 'lstm_therm_detector.pt')
    print("✅ Saved LSTM model to lstm_therm_detector.pt")

    # Save RF models (using pickle)
    import pickle
    with open('rf_detectors.pkl', 'wb') as f:
        pickle.dump({
            'ph_offset_bias': rf_models['ph_offset_bias'],
            'heater_power_loss': rf_models['heater_power_loss']
        }, f)
    print("✅ Saved RF models to rf_detectors.pkl")

    # Save hybrid config
    hybrid_config = {
        'model_type': 'hybrid_ensemble',
        'therm_model': 'lstm',
        'ph_model': 'rf',
        'heater_model': 'rf',
        'lstm_checkpoint': 'lstm_therm_detector.pt',
        'rf_checkpoint': 'rf_detectors.pkl',
        'lstm_features': lstm_features,
        'window': WINDOW,
        'stride': STRIDE
    }

    import json
    with open('hybrid_config.json', 'w') as f:
        json.dump(hybrid_config, f, indent=2)
    print("✅ Saved hybrid config to hybrid_config.json")

    print("\n" + "="*70)
    print("✅ Training complete!")
    print("="*70)


if __name__ == '__main__':
    main()
