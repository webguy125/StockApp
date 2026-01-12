"""
Options Feature Engineering
Extracts ~50+ features from labeled options data for ML training
"""

import pandas as pd
import numpy as np
import os
from sklearn.preprocessing import StandardScaler, LabelEncoder
import pickle

INPUT_FILE = os.path.join(os.path.dirname(__file__), '..', 'data', 'options_training', 'labeled_options_training_data.parquet')
OUTPUT_FILE = os.path.join(os.path.dirname(__file__), '..', 'data', 'options_training', 'training_features.parquet')
SCALER_FILE = os.path.join(os.path.dirname(__file__), '..', 'data', 'options_models', 'v1.0', 'feature_scaler.pkl')

print("Starting feature engineering...")
print("="*80)

def engineer_features(df):
    """Extract all features from raw data"""

    features_df = df.copy()

    # 1. Basic option characteristics (already have most)
    features_df['moneyness'] = features_df['strike'] / features_df['stock_price_entry']
    features_df['distance_to_atm'] = abs(features_df['strike'] - features_df['stock_price_entry'])
    features_df['bid_ask_spread_pct'] = 0.02  # Assume 2% spread (we don't have real bid/ask)

    # 2. Greek derivatives
    features_df['gamma_dollar'] = features_df['gamma'] * (features_df['stock_price_entry'] ** 2)
    features_df['theta_to_delta_ratio'] = abs(features_df['theta']) / (abs(features_df['delta']) + 0.01)
    features_df['vega_to_premium_ratio'] = features_df['vega'] / (features_df['entry_premium'] + 0.01)
    features_df['delta_adjusted_exposure'] = features_df['delta'] * features_df['entry_premium']

    # 3. Volatility features
    features_df['hv_iv_ratio'] = features_df['historical_vol_30d'] / (features_df['entry_iv'] + 0.01)

    # 4. TurboMode ML features
    features_df['signal_strength'] = features_df['confidence'] * features_df['expected_move_pct']
    features_df['distance_to_target_pct'] = (features_df['target_price'] - features_df['stock_price_entry']) / features_df['stock_price_entry'] * 100

    # 5. Rules-based scoring (calculate same way as options_api.py)
    # Delta score (0-40)
    delta_abs = abs(features_df['delta'])
    features_df['delta_score'] = np.where(
        (delta_abs >= 0.60) & (delta_abs <= 0.80), 40,
        np.where((delta_abs >= 0.50) & (delta_abs < 0.60), 30,
        np.where((delta_abs >= 0.40) & (delta_abs < 0.50), 20, 10))
    )

    # IV score (0-15)
    features_df['iv_score'] = np.where(
        features_df['entry_iv'] < 0.25, 15,
        np.where(features_df['entry_iv'] < 0.35, 12,
        np.where(features_df['entry_iv'] < 0.50, 8, 4))
    )

    # Alignment score (0-25) - how close strike is to target
    features_df['deviation_from_target'] = abs(features_df['strike'] - features_df['target_price']) / features_df['target_price']
    features_df['alignment_score'] = np.where(
        features_df['deviation_from_target'] < 0.02, 25,
        np.where(features_df['deviation_from_target'] < 0.05, 20,
        np.where(features_df['deviation_from_target'] < 0.10, 15, 5))
    )

    # Liquidity score (0-20) - we don't have real OI/volume, use estimated
    features_df['liquidity_score'] = 15  # Assume medium liquidity

    # Total rules score
    features_df['rules_total_score'] = (
        features_df['delta_score'] +
        features_df['iv_score'] +
        features_df['alignment_score'] +
        features_df['liquidity_score']
    )

    # 6. Time features
    features_df['entry_date'] = pd.to_datetime(features_df['entry_date'])
    features_df['day_of_week'] = features_df['entry_date'].dt.dayofweek
    features_df['month'] = features_df['entry_date'].dt.month
    features_df['dte_binned'] = pd.cut(features_df['dte'], bins=[0, 35, 40, 50], labels=[0, 1, 2]).astype(int)

    # 7. Encode categoricals
    # Option type (CALL=1, PUT=0)
    features_df['option_type_encoded'] = (features_df['option_type'] == 'CALL').astype(int)

    # Signal type (BUY=1, SELL=0)
    features_df['signal_type_encoded'] = (features_df['signal_type'] == 'BUY').astype(int)

    # 8. Log transforms for skewed features
    features_df['log_entry_premium'] = np.log1p(features_df['entry_premium'])
    features_df['log_strike'] = np.log1p(features_df['strike'])

    print(f"[OK] Engineered {len(features_df.columns)} total columns")

    return features_df

def select_features(df):
    """Select final feature set for training"""

    # Define feature columns (exclude target and metadata)
    feature_columns = [
        # Greeks
        'delta', 'gamma', 'theta', 'vega', 'rho',

        # Option characteristics
        'strike', 'stock_price_entry', 'moneyness', 'dte', 'entry_premium',
        'distance_to_atm', 'entry_iv', 'option_type_encoded',

        # Greek derivatives
        'gamma_dollar', 'theta_to_delta_ratio', 'vega_to_premium_ratio', 'delta_adjusted_exposure',

        # Volatility
        'historical_vol_30d', 'hv_iv_ratio',

        # TurboMode ML
        'signal_type_encoded', 'confidence', 'expected_move_pct',
        'signal_strength', 'distance_to_target_pct',

        # Rules-based scores
        'delta_score', 'iv_score', 'alignment_score', 'liquidity_score', 'rules_total_score',

        # Time features
        'day_of_week', 'month', 'dte_binned',

        # Log transforms
        'log_entry_premium', 'log_strike'
    ]

    # Target variable
    target_column = 'option_success'

    # Check all features exist
    missing = [f for f in feature_columns if f not in df.columns]
    if missing:
        print(f"[WARNING] Missing features: {missing}")
        feature_columns = [f for f in feature_columns if f in df.columns]

    X = df[feature_columns].copy()
    y = df[target_column].copy()

    print(f"[OK] Selected {len(feature_columns)} features for training")
    print(f"[OK] Feature names: {feature_columns[:10]}... (showing first 10)")

    return X, y, feature_columns

def preprocess_features(X, feature_names, is_training=True, scaler=None):
    """Handle missing values and scale features"""

    # Handle missing values (median imputation)
    X_filled = X.fillna(X.median())

    # Scale numerical features
    if is_training:
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X_filled)

        # Save scaler
        os.makedirs(os.path.dirname(SCALER_FILE), exist_ok=True)
        with open(SCALER_FILE, 'wb') as f:
            pickle.dump(scaler, f)
        print(f"[OK] Saved feature scaler to {SCALER_FILE}")
    else:
        if scaler is None:
            raise ValueError("Scaler required for inference")
        X_scaled = scaler.transform(X_filled)

    # Convert back to DataFrame
    X_processed = pd.DataFrame(X_scaled, columns=feature_names, index=X.index)

    return X_processed, scaler

def main():
    # Load data
    print(f"[INFO] Loading data from {INPUT_FILE}")
    df = pd.read_parquet(INPUT_FILE)
    print(f"[OK] Loaded {len(df)} examples")

    # Engineer features
    print("\n[INFO] Engineering features...")
    df_engineered = engineer_features(df)

    # Select features
    print("\n[INFO] Selecting feature set...")
    X, y, feature_names = select_features(df_engineered)

    # Preprocess
    print("\n[INFO] Preprocessing features...")
    X_processed, scaler = preprocess_features(X, feature_names, is_training=True)

    # Combine X and y
    final_df = X_processed.copy()
    final_df['target'] = y.values

    # Save
    print(f"\n[INFO] Saving to {OUTPUT_FILE}")
    final_df.to_parquet(OUTPUT_FILE, index=False)

    # Save feature names
    feature_names_file = os.path.join(os.path.dirname(SCALER_FILE), 'feature_names.json')
    import json
    with open(feature_names_file, 'w') as f:
        json.dump(feature_names, f)

    print(f"[OK] Saved feature names to {feature_names_file}")

    # Print summary
    print(f"\n{'='*80}")
    print("FEATURE ENGINEERING SUMMARY")
    print(f"{'='*80}")
    print(f"Total examples: {len(final_df)}")
    print(f"Total features: {len(feature_names)}")
    print(f"Positive examples (hit +10%): {y.sum()} ({y.mean()*100:.1f}%)")
    print(f"Negative examples (<10%): {len(y)-y.sum()} ({(1-y.mean())*100:.1f}%)")
    print(f"\nFeature list ({len(feature_names)} features):")
    for i, feat in enumerate(feature_names, 1):
        print(f"  {i}. {feat}")
    print(f"{'='*80}\n")

if __name__ == '__main__':
    main()
