"""
Training Script for Crypto Pivot Reliability Model

Separate model for cryptocurrency-specific pivot patterns.
Crypto behaves differently from stocks/ETFs due to:
- 24/7 trading (no gaps)
- Higher volatility
- Different market dynamics

This model will be used when analyzing crypto symbols (BTC-USD, ETH-USD, etc.)
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
import logging
from pathlib import Path

from data_storage import OHLCVDatabase, get_database
from data_fetcher import fetch_historical_data
from timeframe_aggregator import aggregate_ohlcv
from feature_engineering import compute_ml_features
from pivot_model import PivotClassifier, save_model

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


# Crypto-specific training configuration
CRYPTO_TRAINING_CONFIG = {
    # Crypto assets only
    'crypto_assets': [
        'BTC-USD',   # Bitcoin
        'ETH-USD',   # Ethereum
        'SOL-USD',   # Solana
        'BNB-USD',   # Binance Coin
        'XRP-USD',   # Ripple
        'ADA-USD',   # Cardano
        'DOGE-USD',  # Dogecoin
        # 'MATIC-USD', # Polygon - delisted from yfinance
    ],

    # Timeframes (using daily data from yfinance)
    'timeframes': ['1d'],

    # Data period (crypto: 365 days to get enough samples)
    'crypto_days': 365,

    # Pivot labeling (crypto is more volatile - adjust thresholds)
    'reversal_threshold': 0.07,  # 7% reversal for crypto (vs 5% for stocks)
    'reversal_bars': 15,  # Check 15 bars (vs 10 for stocks)

    # Training params
    'train_split': 0.7,
    'batch_size': 32,
    'epochs': 100,
    'learning_rate': 0.001,
    'early_stopping_patience': 10,

    # Model params (same architecture)
    'hidden1': 32,
    'hidden2': 16,
    'dropout1': 0.3,
    'dropout2': 0.2
}


def detect_pivots(candles, lookback=5):
    """
    Detect pivot highs and lows

    Args:
        candles: DataFrame with OHLCV data
        lookback: Bars to check on each side

    Returns:
        tuple: (pivot_high_indices, pivot_low_indices)
    """
    highs = candles['high'].values
    lows = candles['low'].values

    pivot_highs = []
    pivot_lows = []

    for i in range(lookback, len(candles) - lookback):
        # Pivot high: current high is highest in window
        window_highs = highs[i - lookback:i + lookback + 1]
        if highs[i] == np.max(window_highs):
            pivot_highs.append(i)

        # Pivot low: current low is lowest in window
        window_lows = lows[i - lookback:i + lookback + 1]
        if lows[i] == np.min(window_lows):
            pivot_lows.append(i)

    return pivot_highs, pivot_lows


def label_pivots(candles, pivot_indices, is_high, reversal_threshold=0.07, reversal_bars=15):
    """
    Label pivots as reliable (1) or unreliable (0)

    A pivot is reliable if price reverses by at least reversal_threshold
    within reversal_bars in the expected direction.

    Args:
        candles: DataFrame with OHLCV data
        pivot_indices: List of pivot indices
        is_high: True for pivot highs, False for pivot lows
        reversal_threshold: Minimum price reversal (e.g., 0.07 = 7%)
        reversal_bars: Window to check for reversal

    Returns:
        numpy array: Binary labels (1 = reliable, 0 = unreliable)
    """
    closes = candles['close'].values
    labels = []

    for idx in pivot_indices:
        pivot_price = closes[idx]

        # Look ahead for reversal
        end_idx = min(idx + reversal_bars + 1, len(closes))
        future_prices = closes[idx + 1:end_idx]

        if len(future_prices) == 0:
            continue

        if is_high:
            # For pivot high, expect price to drop
            min_future_price = np.min(future_prices)
            reversal = (pivot_price - min_future_price) / pivot_price

            # Reliable if price drops by threshold
            labels.append(1 if reversal >= reversal_threshold else 0)

        else:
            # For pivot low, expect price to rise
            max_future_price = np.max(future_prices)
            reversal = (max_future_price - pivot_price) / pivot_price

            # Reliable if price rises by threshold
            labels.append(1 if reversal >= reversal_threshold else 0)

    return np.array(labels)


def fetch_and_process_crypto(asset, timeframes, config):
    """
    Fetch data for one crypto asset and process all timeframes

    Args:
        asset: Crypto symbol (e.g., 'BTC-USD')
        timeframes: List of timeframes to process
        config: Training configuration dict

    Returns:
        dict: {timeframe: (features, labels)}
    """
    try:
        logger.info(f"\n{'='*60}")
        logger.info(f"📊 Processing {asset}...")
        logger.info(f"{'='*60}")

        # Fetch historical data using yfinance (use 1d as base - yfinance has better crypto support)
        # yfinance provides reliable crypto data for daily timeframes
        df = fetch_historical_data(asset, days=config['crypto_days'], timeframe='1d')

        if df.empty:
            logger.warning(f"⚠️ No data for {asset}")
            return {}

        results = {}

        # Process each timeframe
        for timeframe in timeframes:
            logger.info(f"\n  Processing {timeframe}...")

            # For daily data, skip aggregation
            if timeframe == '1d':
                df_agg = df.copy()
            else:
                # Skip non-daily timeframes since we're using daily base data
                logger.info(f"    ⚠️ Skipping {timeframe} (base data is daily)")
                continue

            if len(df_agg) < 100:
                logger.warning(f"    ⚠️ Not enough data ({len(df_agg)} bars)")
                continue

            # Compute features
            features_df = compute_ml_features(df_agg, timeframe=timeframe)

            # Detect pivots
            pivot_highs, pivot_lows = detect_pivots(df_agg, lookback=5)

            logger.info(f"    Found {len(pivot_highs)} pivot highs, {len(pivot_lows)} pivot lows")

            # Label pivots
            labels_high = label_pivots(
                df_agg, pivot_highs, is_high=True,
                reversal_threshold=config['reversal_threshold'],
                reversal_bars=config['reversal_bars']
            )

            labels_low = label_pivots(
                df_agg, pivot_lows, is_high=False,
                reversal_threshold=config['reversal_threshold'],
                reversal_bars=config['reversal_bars']
            )

            # Combine pivots
            all_pivot_indices = pivot_highs[:len(labels_high)] + pivot_lows[:len(labels_low)]
            all_labels = np.concatenate([labels_high, labels_low])

            # Extract features for pivots
            pivot_features = features_df.iloc[all_pivot_indices].values

            # Remove rows with NaN
            valid_mask = ~np.any(np.isnan(pivot_features), axis=1)
            pivot_features = pivot_features[valid_mask]
            all_labels = all_labels[valid_mask]

            if len(pivot_features) == 0:
                logger.warning(f"    ⚠️ No valid pivot features")
                continue

            logger.info(f"    ✅ {len(pivot_features)} valid pivots")
            logger.info(f"    Reliable: {np.sum(all_labels)} ({np.mean(all_labels)*100:.1f}%)")

            results[timeframe] = (pivot_features, all_labels)

        return results

    except Exception as e:
        logger.error(f"❌ Crypto processing failed for {asset}: {e}")
        return {}


def prepare_crypto_training_data(config):
    """
    Fetch and prepare training data for crypto assets

    Args:
        config: Training configuration dict

    Returns:
        tuple: (X_train, y_train, X_val, y_val)
    """
    logger.info("="*60)
    logger.info("PREPARING CRYPTO TRAINING DATA")
    logger.info("="*60)

    all_features = []
    all_labels = []

    # Process crypto assets
    for asset in config['crypto_assets']:
        results = fetch_and_process_crypto(
            asset,
            timeframes=config['timeframes'],
            config=config
        )

        for timeframe, (features, labels) in results.items():
            all_features.append(features)
            all_labels.append(labels)

    if len(all_features) == 0:
        logger.error("❌ No training data collected!")
        return None, None, None, None

    # Combine all data
    X = np.vstack(all_features)
    y = np.concatenate(all_labels)

    logger.info(f"\n{'='*60}")
    logger.info(f"CRYPTO DATASET SUMMARY")
    logger.info(f"{'='*60}")
    logger.info(f"Total samples: {len(X)}")
    logger.info(f"Reliable pivots: {np.sum(y)} ({np.mean(y)*100:.1f}%)")
    logger.info(f"Unreliable pivots: {len(y) - np.sum(y)} ({(1-np.mean(y))*100:.1f}%)")

    # Shuffle
    indices = np.random.permutation(len(X))
    X = X[indices]
    y = y[indices]

    # Split train/validation
    split_idx = int(len(X) * config['train_split'])
    X_train, X_val = X[:split_idx], X[split_idx:]
    y_train, y_val = y[:split_idx], y[split_idx:]

    logger.info(f"\nTrain: {len(X_train)} samples")
    logger.info(f"Validation: {len(X_val)} samples")
    logger.info(f"{'='*60}\n")

    return X_train, y_train, X_val, y_val


def train_model(X_train, y_train, X_val, y_val, config):
    """
    Train the PyTorch model

    Args:
        X_train: Training features
        y_train: Training labels
        X_val: Validation features
        y_val: Validation labels
        config: Training configuration

    Returns:
        PivotClassifier: Trained model
    """
    logger.info("="*60)
    logger.info("TRAINING CRYPTO MODEL")
    logger.info("="*60)

    # Convert to tensors
    X_train_t = torch.FloatTensor(X_train)
    y_train_t = torch.FloatTensor(y_train).reshape(-1, 1)
    X_val_t = torch.FloatTensor(X_val)
    y_val_t = torch.FloatTensor(y_val).reshape(-1, 1)

    # Create data loaders
    train_dataset = TensorDataset(X_train_t, y_train_t)
    train_loader = DataLoader(train_dataset, batch_size=config['batch_size'], shuffle=True)

    # Initialize model
    model = PivotClassifier(
        input_dim=9,
        hidden1=config['hidden1'],
        hidden2=config['hidden2']
    )

    # Loss and optimizer
    criterion = nn.BCELoss()
    optimizer = optim.Adam(model.parameters(), lr=config['learning_rate'])

    # Training loop
    best_val_loss = float('inf')
    patience_counter = 0
    best_model_state = None

    for epoch in range(config['epochs']):
        # Training phase
        model.train()
        train_loss = 0.0

        for batch_X, batch_y in train_loader:
            optimizer.zero_grad()

            outputs = model(batch_X)
            loss = criterion(outputs, batch_y)

            loss.backward()
            optimizer.step()

            train_loss += loss.item()

        train_loss /= len(train_loader)

        # Validation phase
        model.eval()
        with torch.no_grad():
            val_outputs = model(X_val_t)
            val_loss = criterion(val_outputs, y_val_t).item()

            # Calculate accuracy
            val_preds = (val_outputs >= 0.5).float()
            val_acc = (val_preds == y_val_t).float().mean().item()

        # Log progress
        if (epoch + 1) % 10 == 0 or epoch == 0:
            logger.info(f"Epoch {epoch+1}/{config['epochs']} - "
                       f"Train Loss: {train_loss:.4f}, "
                       f"Val Loss: {val_loss:.4f}, "
                       f"Val Acc: {val_acc*100:.2f}%")

        # Early stopping
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_model_state = model.state_dict().copy()
            patience_counter = 0
        else:
            patience_counter += 1

        if patience_counter >= config['early_stopping_patience']:
            logger.info(f"\n⚠️ Early stopping at epoch {epoch+1}")
            break

    # Restore best model
    model.load_state_dict(best_model_state)

    logger.info(f"\n✅ Training complete!")
    logger.info(f"Best validation loss: {best_val_loss:.4f}")
    logger.info(f"{'='*60}\n")

    return model


def evaluate_model(model, X_val, y_val):
    """
    Evaluate model performance

    Args:
        model: Trained PivotClassifier
        X_val: Validation features
        y_val: Validation labels
    """
    logger.info("="*60)
    logger.info("CRYPTO MODEL EVALUATION")
    logger.info("="*60)

    model.eval()

    with torch.no_grad():
        X_val_t = torch.FloatTensor(X_val)
        y_val_t = torch.FloatTensor(y_val)

        probs = model(X_val_t).numpy().flatten()

        # Test different thresholds
        thresholds = [0.5, 0.6, 0.7, 0.8, 0.9]

        for thresh in thresholds:
            preds = (probs >= thresh).astype(int)

            accuracy = np.mean(preds == y_val)
            precision = np.sum((preds == 1) & (y_val == 1)) / max(np.sum(preds == 1), 1)
            recall = np.sum((preds == 1) & (y_val == 1)) / max(np.sum(y_val == 1), 1)
            f1 = 2 * (precision * recall) / max(precision + recall, 1e-8)

            logger.info(f"\nThreshold: {thresh}")
            logger.info(f"  Accuracy:  {accuracy*100:.2f}%")
            logger.info(f"  Precision: {precision*100:.2f}%")
            logger.info(f"  Recall:    {recall*100:.2f}%")
            logger.info(f"  F1 Score:  {f1*100:.2f}%")

    logger.info(f"\n{'='*60}\n")


def save_crypto_model(model, epoch, loss):
    """
    Save the crypto-specific model

    Args:
        model: Trained PivotClassifier
        epoch: Final epoch number
        loss: Final loss value
    """
    model_dir = Path(__file__).parent / 'models'
    model_dir.mkdir(exist_ok=True)

    model_path = model_dir / 'pivot_model_crypto.pth'

    torch.save({
        'model_state_dict': model.state_dict(),
        'epoch': epoch,
        'loss': loss,
        'model_type': 'crypto'
    }, model_path)

    logger.info(f"✅ Crypto model saved to: {model_path}")


def main():
    """
    Main crypto training script
    """
    logger.info("\n" + "="*60)
    logger.info("CRYPTO PIVOT RELIABILITY MODEL TRAINING")
    logger.info("="*60 + "\n")

    # Prepare data
    X_train, y_train, X_val, y_val = prepare_crypto_training_data(CRYPTO_TRAINING_CONFIG)

    if X_train is None:
        logger.error("❌ Failed to prepare training data")
        return

    # Train model
    model = train_model(X_train, y_train, X_val, y_val, CRYPTO_TRAINING_CONFIG)

    # Evaluate
    evaluate_model(model, X_val, y_val)

    # Save model
    save_crypto_model(model, epoch=CRYPTO_TRAINING_CONFIG['epochs'], loss=None)

    logger.info("✅ Crypto training pipeline complete!")
    logger.info(f"Model saved to: {Path(__file__).parent / 'models' / 'pivot_model_crypto.pth'}\n")


if __name__ == "__main__":
    main()
