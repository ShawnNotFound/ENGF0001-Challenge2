#!/usr/bin/env python3
"""
Machine Learning Anomaly Detection - Random Forest
Uses ML to find complex patterns that statistical methods miss
"""

import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, precision_score, recall_score, f1_score, accuracy_score
import joblib
import os
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

class MLFaultDetector:
    """Machine Learning fault detector using Random Forest"""

    def __init__(self, fault_type, features, n_estimators=100):
        self.fault_type = fault_type
        self.features = features
        self.model = RandomForestClassifier(
            n_estimators=n_estimators,
            max_depth=10,
            min_samples_split=20,
            min_samples_leaf=10,
            class_weight='balanced',  # Handle imbalanced data
            random_state=42
        )

    def fit(self, X_train, y_train):
        """Train Random Forest on ALL data (not just normal)"""
        self.model.fit(X_train[self.features], y_train)
        return self

    def predict(self, X):
        """Predict if fault is present"""
        return self.model.predict(X[self.features])

    def predict_proba(self, X):
        """Get prediction probabilities"""
        return self.model.predict_proba(X[self.features])[:, 1]

    def feature_importance(self):
        """Get feature importances"""
        return dict(zip(self.features, self.model.feature_importances_))


def load_and_prepare_data(filepath):
    """Load CSV and prepare labels"""
    df = pd.read_csv(filepath)
    df = df.sort_values('timestamp').reset_index(drop=True)

    # Add temporal features
    for col in ['T_mean', 'pH_mean', 'RPM_mean']:
        df[f'{col}_diff'] = df[col].diff().fillna(0)
        df[f'{col}_rolling_mean'] = df[col].rolling(window=5, min_periods=1).mean()
        df[f'{col}_rolling_std'] = df[col].rolling(window=5, min_periods=1).std().fillna(0)

    # Add interaction features
    df['T_spread'] = df['T_max'] - df['T_min']
    df['pH_spread'] = df['pH_max'] - df['pH_min']
    df['RPM_spread'] = df['RPM_max'] - df['RPM_min']

    # Labels
    df['has_heater_power_loss'] = df['fault_labels'].fillna('').str.contains('heater_power_loss').astype(int)
    df['has_therm_voltage_bias'] = df['fault_labels'].fillna('').str.contains('therm_voltage_bias').astype(int)
    df['has_ph_offset_bias'] = df['fault_labels'].fillna('').str.contains('ph_offset_bias').astype(int)

    return df


def train_detectors(df_train):
    """Train ML detectors"""
    print(f"\nTraining on {len(df_train)} samples (including faults!)")
    print("Using Random Forest with balanced class weights\n")

    # Feature sets with temporal and interaction features
    configs = {
        'heater_power_loss': {
            'features': [
                'T_mean', 'T_min', 'T_max', 'ΔT', 'T_spread',
                'T_mean_diff', 'T_mean_rolling_mean', 'T_mean_rolling_std',
                'RPM_mean', 'ΔRPM'
            ]
        },
        'therm_voltage_bias': {
            'features': [
                'T_mean', 'T_min', 'T_max', 'ΔT', 'T_spread',
                'T_mean_diff', 'T_mean_rolling_mean', 'T_mean_rolling_std'
            ]
        },
        'ph_offset_bias': {
            'features': [
                'pH_mean', 'pH_min', 'pH_max', 'ΔpH', 'pH_spread',
                'pH_mean_diff', 'pH_mean_rolling_mean', 'pH_mean_rolling_std',
                'RPM_mean', 'RPM_spread', 'ΔRPM',
                'T_mean', 'T_min'
            ]
        }
    }

    detectors = {}
    for fault_type, config in configs.items():
        print(f"  Training {fault_type} detector...")

        y_train = df_train[f'has_{fault_type}']
        pos_samples = y_train.sum()
        neg_samples = len(y_train) - pos_samples

        print(f"    Positive samples: {pos_samples}")
        print(f"    Negative samples: {neg_samples}")
        print(f"    Features: {len(config['features'])}")

        detector = MLFaultDetector(fault_type, config['features'], n_estimators=100)
        detector.fit(df_train, y_train)

        # Show feature importance
        importance = detector.feature_importance()
        top_features = sorted(importance.items(), key=lambda x: x[1], reverse=True)[:3]
        print(f"    Top features: {[f'{k}({v:.3f})' for k, v in top_features]}\n")

        detectors[fault_type] = detector

    return detectors


def evaluate_detector(detector, df_test, fault_type):
    """Evaluate a single detector"""
    y_true = df_test[f'has_{fault_type}'].values
    y_pred = detector.predict(df_test)

    tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
    precision = precision_score(y_true, y_pred, zero_division=0)
    recall = recall_score(y_true, y_pred, zero_division=0)
    f1 = f1_score(y_true, y_pred, zero_division=0)
    accuracy = accuracy_score(y_true, y_pred)

    return {
        'TP': tp, 'FP': fp, 'TN': tn, 'FN': fn,
        'Precision': precision, 'Recall': recall,
        'F1': f1, 'Accuracy': accuracy
    }


def save_models(detectors, metrics, save_dir='saved_models'):
    """Save trained models and metrics to disk"""
    # Create save directory if it doesn't exist
    os.makedirs(save_dir, exist_ok=True)

    # Generate timestamp for version control
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')

    # Save each detector
    for fault_type, detector in detectors.items():
        model_path = os.path.join(save_dir, f'{fault_type}_detector.pkl')
        joblib.dump(detector, model_path)
        print(f"  Saved {fault_type} detector to {model_path}")

    # Save metrics
    metrics_path = os.path.join(save_dir, f'metrics_{timestamp}.pkl')
    joblib.dump(metrics, metrics_path)
    print(f"  Saved metrics to {metrics_path}")

    # Save metadata
    metadata = {
        'timestamp': timestamp,
        'fault_types': list(detectors.keys()),
        'overall_metrics': {
            'avg_f1': np.mean([metrics[ft]['F1'] for ft in detectors.keys()]),
        }
    }
    metadata_path = os.path.join(save_dir, 'metadata.pkl')
    joblib.dump(metadata, metadata_path)
    print(f"  Saved metadata to {metadata_path}")

    return save_dir


def load_models(save_dir='saved_models'):
    """Load trained models from disk"""
    detectors = {}

    # Load each detector
    for fault_type in ['heater_power_loss', 'therm_voltage_bias', 'ph_offset_bias']:
        model_path = os.path.join(save_dir, f'{fault_type}_detector.pkl')
        if os.path.exists(model_path):
            detectors[fault_type] = joblib.load(model_path)
            print(f"  Loaded {fault_type} detector from {model_path}")
        else:
            print(f"  Warning: {model_path} not found")

    # Load metadata
    metadata_path = os.path.join(save_dir, 'metadata.pkl')
    metadata = None
    if os.path.exists(metadata_path):
        metadata = joblib.load(metadata_path)
        print(f"  Loaded metadata from {metadata_path}")

    return detectors, metadata


def main():
    print("="*70)
    print("MACHINE LEARNING DETECTOR - Random Forest")
    print("="*70)

    # Load data
    print("\nLoading data...")
    df = load_and_prepare_data('full_data.csv')
    print(f"Total samples: {len(df)}")
    print(f"  Fault-free: {(df['fault_active'] == False).sum()}")
    print(f"  Heater power loss: {df['has_heater_power_loss'].sum()}")
    print(f"  Therm voltage bias: {df['has_therm_voltage_bias'].sum()}")
    print(f"  pH offset bias: {df['has_ph_offset_bias'].sum()}")

    # Split data - keep temporal ordering
    df_temp, df_test = train_test_split(df, test_size=0.3, random_state=42, stratify=df['fault_active'])
    df_train = df_temp.sort_values('timestamp').reset_index(drop=True)
    df_test = df_test.sort_values('timestamp').reset_index(drop=True)

    print(f"\nSplit: {len(df_train)} train, {len(df_test)} test")

    # Train detectors
    print("\n" + "="*70)
    print("TRAINING ML DETECTORS")
    print("="*70)
    detectors = train_detectors(df_train)

    # Evaluate
    print("="*70)
    print("EVALUATION RESULTS ON TEST SET")
    print("="*70)

    all_metrics = {}
    for fault_type in ['heater_power_loss', 'therm_voltage_bias', 'ph_offset_bias']:
        print(f"\n{fault_type.upper()}")
        print("-" * 70)

        metrics = evaluate_detector(detectors[fault_type], df_test, fault_type)
        all_metrics[fault_type] = metrics

        print(f"  True Positives (TP):   {metrics['TP']}")
        print(f"  False Positives (FP):  {metrics['FP']}")
        print(f"  True Negatives (TN):   {metrics['TN']}")
        print(f"  False Negatives (FN):  {metrics['FN']}")
        print(f"  Precision:             {metrics['Precision']:.4f}")
        print(f"  Recall:                {metrics['Recall']:.4f}")
        print(f"  F1 Score:              {metrics['F1']:.4f}")
        print(f"  Accuracy:              {metrics['Accuracy']:.4f}")

    # Combined system
    print("\n" + "="*70)
    print("COMBINED SYSTEM EVALUATION")
    print("="*70)

    pred_heater = detectors['heater_power_loss'].predict(df_test)
    pred_therm = detectors['therm_voltage_bias'].predict(df_test)
    pred_ph = detectors['ph_offset_bias'].predict(df_test)

    y_pred_any = ((pred_heater + pred_therm + pred_ph) > 0).astype(int)
    y_true_any = df_test['fault_active'].astype(int)

    tn, fp, fn, tp = confusion_matrix(y_true_any, y_pred_any).ravel()
    precision = precision_score(y_true_any, y_pred_any, zero_division=0)
    recall = recall_score(y_true_any, y_pred_any, zero_division=0)
    f1 = f1_score(y_true_any, y_pred_any, zero_division=0)
    accuracy = accuracy_score(y_true_any, y_pred_any)

    print(f"\nOverall Fault Detection (Any Fault):")
    print(f"  True Positives (TP):   {tp}")
    print(f"  False Positives (FP):  {fp}")
    print(f"  True Negatives (TN):   {tn}")
    print(f"  False Negatives (FN):  {fn}")
    print(f"  Precision:             {precision:.4f}")
    print(f"  Recall:                {recall:.4f}")
    print(f"  F1 Score:              {f1:.4f}")
    print(f"  Accuracy:              {accuracy:.4f}")

    # Summary
    print("\n" + "="*70)
    print("COMPARISON WITH PREVIOUS MODELS")
    print("="*70)

    avg_f1 = np.mean([all_metrics[ft]['F1'] for ft in ['heater_power_loss', 'therm_voltage_bias', 'ph_offset_bias']])

    print("\nModel Comparison:")
    print("-" * 70)
    print("                          Avg F1    Overall F1   Accuracy   Precision")
    print("-" * 70)
    print(f"Statistical (hybrid)      0.5015      0.7169      69.92%     57.56%")
    print(f"Machine Learning (RF)     {avg_f1:.4f}      {f1:.4f}      {accuracy*100:.2f}%     {precision*100:.2f}%")
    print("-" * 70)

    if f1 > 0.7169:
        improvement = ((f1 - 0.7169) / 0.7169) * 100
        print(f"\n✓ BEST RESULT! F1 improved by {improvement:.1f}%!")
    else:
        decline = ((0.7169 - f1) / 0.7169) * 100
        print(f"\n⚠ F1 declined by {decline:.1f}%")

    print("\n" + "="*70)
    print("MODEL CHARACTERISTICS")
    print("="*70)
    print("✓ Random Forest with 100 trees")
    print("✓ Trained on ALL data (normal + faults)")
    print("✓ Balanced class weights for imbalanced data")
    print("✓ Temporal features (diff, rolling mean/std)")
    print("✓ Interaction features (spreads, combinations)")
    print("✓ Can learn complex non-linear patterns")

    # Save models
    print("\n" + "="*70)
    print("SAVING MODELS")
    print("="*70)
    save_dir = save_models(detectors, all_metrics)
    print(f"\n✓ Models saved successfully to '{save_dir}/' directory")
    print("  Use load_models() to reload them later")

    print("\n" + "="*70)
    print("Training complete! ML detectors ready.")
    print("="*70)


if __name__ == '__main__':
    main()
