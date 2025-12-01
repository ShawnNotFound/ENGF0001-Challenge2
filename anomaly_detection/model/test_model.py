#!/usr/bin/env python3
"""
Run the trained Hybrid Ensemble Model
Loads saved LSTM and RF models and makes predictions
"""

import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import pickle
import json
from sklearn.metrics import confusion_matrix, precision_score, recall_score, f1_score, accuracy_score
import warnings
warnings.filterwarnings('ignore')


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


class RFDetector:
    """Random Forest detector for a single fault type"""
    def __init__(self, fault_type, features, n_estimators=100):
        from sklearn.ensemble import RandomForestClassifier
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


def load_hybrid_model():
    """Load all components of the hybrid ensemble"""
    print("Loading hybrid ensemble model...")

    # Load config
    with open('hybrid_config.json', 'r') as f:
        config = json.load(f)
    print(f"  Config loaded: {config['model_type']}")

    # Load LSTM
    lstm_checkpoint = torch.load(config['lstm_checkpoint'], map_location='cpu')
    lstm_model = LSTMAnomalyDetector(
        input_size=lstm_checkpoint['config']['input_size'],
        hidden_size=lstm_checkpoint['config']['hidden_size'],
        num_layers=lstm_checkpoint['config']['num_layers'],
        dropout=lstm_checkpoint['config']['dropout'],
        output_dim=lstm_checkpoint['config']['output_dim']
    )
    lstm_model.load_state_dict(lstm_checkpoint['state_dict'])
    lstm_model.eval()
    print(f"  LSTM loaded from {config['lstm_checkpoint']}")

    # Load RF models
    with open(config['rf_checkpoint'], 'rb') as f:
        rf_models = pickle.load(f)
    print(f"  RF models loaded from {config['rf_checkpoint']}")

    return {
        'config': config,
        'lstm_model': lstm_model,
        'lstm_checkpoint': lstm_checkpoint,
        'rf_models': rf_models
    }


def load_and_prepare_data(filepath):
    """Load CSV and prepare all features"""
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


def predict_lstm_sequences(model, df, feature_columns, mean, std, window, stride, threshold=0.5):
    """Get LSTM predictions on sequences and map to rows"""
    # Normalize features
    normalized = (df[feature_columns].to_numpy(dtype=np.float32) - np.array(mean, dtype=np.float32)) / np.array(std, dtype=np.float32)

    # Create sequences
    sequences = []
    for start in range(0, len(df) - window + 1, stride):
        end = start + window
        seq = normalized[start:end].astype(np.float32)
        sequences.append(seq)

    if len(sequences) == 0:
        raise ValueError(f"No sequences created. DataFrame has {len(df)} rows, need at least {window}")

    sequences = np.array(sequences, dtype=np.float32)

    # Get predictions
    model.eval()
    with torch.no_grad():
        batch_size = 256
        all_probs = []
        for i in range(0, len(sequences), batch_size):
            batch = sequences[i:i+batch_size]
            batch_tensor = torch.from_numpy(batch)
            logits = model(batch_tensor)
            probs = torch.sigmoid(logits).cpu().numpy()
            all_probs.append(probs)

        all_probs = np.vstack(all_probs)

    # Map sequence predictions to row predictions
    row_probs = np.zeros((len(df), 4))
    for seq_idx, start_idx in enumerate(range(0, len(df) - window + 1, stride)):
        end_idx = start_idx + window
        for row in range(start_idx, min(end_idx, len(df))):
            row_probs[row] = all_probs[seq_idx]

    return (row_probs > threshold).astype(int)


def run_prediction(data_filepath, test_ratio=0.2):
    """Run hybrid ensemble prediction on data"""

    # Load model
    hybrid = load_hybrid_model()
    config = hybrid['config']
    lstm_model = hybrid['lstm_model']
    lstm_checkpoint = hybrid['lstm_checkpoint']
    rf_models = hybrid['rf_models']

    # Load data
    print(f"\nLoading data from {data_filepath}...")
    df = load_and_prepare_data(data_filepath)
    print(f"Total samples: {len(df)}")

    # Split data - temporal split
    split_idx = int(len(df) * (1 - test_ratio))
    df_test = df.iloc[split_idx:].reset_index(drop=True)
    print(f"Test samples: {len(df_test)} (last {test_ratio*100}%)")

    # Get LSTM predictions for therm voltage bias
    print("\nGetting LSTM predictions (Therm Voltage Bias)...")
    lstm_features = config['lstm_features']
    mean = lstm_checkpoint['normalization']['mean']
    std = lstm_checkpoint['normalization']['std']
    window = config['window']
    stride = config['stride']

    lstm_preds = predict_lstm_sequences(
        lstm_model, df_test, lstm_features, mean, std, window, stride
    )
    pred_therm = lstm_preds[:, 0]  # Therm voltage bias predictions

    # Get RF predictions
    print("Getting RF predictions (pH Offset Bias & Heater Power Loss)...")
    pred_ph = rf_models['ph_offset_bias'].predict(df_test)
    pred_heater = rf_models['heater_power_loss'].predict(df_test)

    # Combine predictions
    pred_any = ((pred_therm + pred_ph + pred_heater) > 0).astype(int)

    # Ground truth
    y_true_therm = df_test['has_therm_voltage_bias'].values
    y_true_ph = df_test['has_ph_offset_bias'].values
    y_true_heater = df_test['has_heater_power_loss'].values
    y_true_any = df_test['fault_active'].astype(int).values

    # Evaluate
    print("\n" + "="*70)
    print("HYBRID ENSEMBLE RESULTS")
    print("="*70)

    results = {}
    faults = [
        ('Therm Voltage Bias', pred_therm, y_true_therm, 'LSTM'),
        ('pH Offset Bias', pred_ph, y_true_ph, 'RF'),
        ('Heater Power Loss', pred_heater, y_true_heater, 'RF'),
        ('ANY FAULT', pred_any, y_true_any, 'Hybrid')
    ]

    for fault_name, y_pred, y_true, model_type in faults:
        tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
        precision = precision_score(y_true, y_pred, zero_division=0)
        recall = recall_score(y_true, y_pred, zero_division=0)
        f1 = f1_score(y_true, y_pred, zero_division=0)
        accuracy = accuracy_score(y_true, y_pred)

        results[fault_name] = {
            'model': model_type,
            'TP': tp, 'FP': fp, 'TN': tn, 'FN': fn,
            'precision': precision, 'recall': recall,
            'f1': f1, 'accuracy': accuracy
        }

        print(f"\n{fault_name} ({model_type}):")
        print(f"  F1 Score:   {f1:.4f}")
        print(f"  Precision:  {precision:.4f}")
        print(f"  Recall:     {recall:.4f}")
        print(f"  Accuracy:   {accuracy:.4f}")
        print(f"  TP: {tp:>5} | FP: {fp:>5} | TN: {tn:>5} | FN: {fn:>5}")

    # Summary comparison
    print("\n" + "="*70)
    print("MODEL COMPARISON")
    print("="*70)
    print(f"{'Model':<30} {'Overall F1':<15} {'Precision':<15} {'Recall':<15}")
    print("-"*70)
    print(f"{'Statistical (Hybrid)':<30} {'0.7169':<15} {'0.5756':<15} {'N/A':<15}")
    print(f"{'Random Forest':<30} {'0.8408':<15} {'0.7971':<15} {'0.8894':<15}")
    print(f"{'LSTM (all faults)':<30} {'0.8822':<15} {'0.9179':<15} {'0.8491':<15}")

    hybrid_f1 = results['ANY FAULT']['f1']
    hybrid_prec = results['ANY FAULT']['precision']
    hybrid_recall = results['ANY FAULT']['recall']
    print(f"{'Hybrid Ensemble':<30} {hybrid_f1:<15.4f} {hybrid_prec:<15.4f} {hybrid_recall:<15.4f}")
    print("="*70)

    print("\n✅ Prediction complete!")

    return results


def main():
    print("="*70)
    print("HYBRID ENSEMBLE MODEL - PREDICTION")
    print("="*70)

    # Run prediction on test data
    results = run_prediction('full_data.csv', test_ratio=0.2)

    # Optional: Save results to file
    output = {
        'model_type': 'hybrid_ensemble',
        'results': {
            fault: {
                'model': info['model'],
                'f1': float(info['f1']),
                'precision': float(info['precision']),
                'recall': float(info['recall']),
                'accuracy': float(info['accuracy']),
                'confusion_matrix': {
                    'TP': int(info['TP']),
                    'FP': int(info['FP']),
                    'TN': int(info['TN']),
                    'FN': int(info['FN'])
                }
            }
            for fault, info in results.items()
        }
    }

    with open('prediction_results.json', 'w') as f:
        json.dump(output, f, indent=2)
    print("\n✅ Results saved to prediction_results.json")


if __name__ == '__main__':
    main()
