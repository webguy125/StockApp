"""
Training Script for TriadTrendPulse Signal Validation Model

This script trains an ML model specifically for validating TriadTrendPulse oscillator signals.

Workflow:
1. Fetch historical data for multiple stock assets
2. Calculate TriadTrendPulse indicator (Weighted Regression + Adaptive Oscillator)
3. Detect buy/sell signals using same logic as frontend
4. Label signals as "good" (1) or "bad" (0) based on price movement after signal
5. Extract 9 ML features at each signal point
6. Train PyTorch model with validation split
7. Save model to models/pivot_model.pth (replaces old pivot model)

Signal Labeling Logic:
- BUY signal is GOOD if price rises >= 3% within 10 bars
- SELL signal is GOOD if price drops >= 3% within 10 bars
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

from data_fetcher import fetch_historical_data
from feature_engineering import compute_ml_features
from pivot_model import PivotClassifier

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


# Training configuration
TRAINING_CONFIG = {
    # Stock assets to train on
    'stock_assets': [
        'AAPL', 'MSFT', 'GOOGL', 'AMZN', 'NVDA',  # Tech
        'TSLA', 'AMD', 'INTC', 'NFLX', 'META',    # Growth
        'SPY', 'QQQ', 'DIA', 'IWM',               # ETFs
        'JPM', 'BAC', 'WMT', 'JNJ', 'PG',         # Value
        'XLE', 'XLF', 'XLK', 'XLV', 'XLI'         # Sector ETFs
    ],

    # Data period
    'stock_days': 730,  # 2 years of data

    # TriadTrendPulse indicator settings (match frontend defaults)
    'weighted_length': 20,
    'weighted_source': 2,  # 0=close, 1=hl2, 2=hlc3, 3=ohlc4
    'adaptive_period': 50,

    # Signal detection settings (relaxed for training - need more samples)
    'overbought_threshold': 60,   # Relaxed from 80 to get more signals
    'oversold_threshold': -60,    # Relaxed from -80 to get more signals
    'min_separation': 3,          # Reduced from 5 to get more signals

    # Signal labeling
    'profit_target': 0.03,  # 3% price move to be considered "good"
    'lookforward_bars': 10,  # Check next 10 bars for price movement

    # Training params
    'train_split': 0.7,
    'batch_size': 32,
    'epochs': 100,
    'learning_rate': 0.001,
    'early_stopping_patience': 10,

    # Model params
    'hidden1': 32,
    'hidden2': 16,
    'dropout1': 0.3,
    'dropout2': 0.2
}


def calculate_weighted_regression(closes, length=20, source=2):
    """
    Calculate Weighted Linear Regression

    Args:
        closes: Array of close prices
        length: Regression period
        source: Price source (2 = hlc3, handled externally)

    Returns:
        Array of weighted regression values normalized to [-100, 100]
    """
    result = np.full(len(closes), np.nan)

    for i in range(length - 1, len(closes)):
        y = closes[i - length + 1:i + 1]
        x = np.arange(length)

        # Weighted linear regression
        weights = np.arange(1, length + 1)
        poly = np.polyfit(x, y, 1, w=weights)
        reg_value = np.polyval(poly, length - 1)

        # Normalize to [-100, 100] based on price range
        price_range = np.max(y) - np.min(y)
        if price_range > 0:
            deviation = (reg_value - np.mean(y)) / price_range
            result[i] = np.clip(deviation * 200, -100, 100)
        else:
            result[i] = 0

    return result


def calculate_adaptive_oscillator(closes, period=50):
    """
    Calculate Adaptive Oscillator

    Args:
        closes: Array of close prices
        period: Oscillator period

    Returns:
        Array of oscillator values normalized to [-100, 100]
    """
    result = np.full(len(closes), np.nan)

    for i in range(period - 1, len(closes)):
        window = closes[i - period + 1:i + 1]

        # Calculate adaptive components (simplified version)
        sma = np.mean(window)
        std = np.std(window)

        if std > 0:
            deviation = (closes[i] - sma) / std
            result[i] = np.clip(deviation * 20, -100, 100)
        else:
            result[i] = 0

    return result


def detect_triad_signals(df, config):
    """
    Detect TriadTrendPulse buy/sell signals

    Args:
        df: DataFrame with OHLCV data
        config: Configuration dict with indicator settings

    Returns:
        List of signal dicts: [{index, type, wr, ao}, ...]
    """
    # Calculate indicator components
    wr = calculate_weighted_regression(df['close'].values, config['weighted_length'])
    ao = calculate_adaptive_oscillator(df['close'].values, config['adaptive_period'])

    signals = []

    # Start from where both indicators are valid
    start_idx = max(config['weighted_length'], config['adaptive_period'])

    for i in range(start_idx, len(df)):
        if np.isnan(wr[i]) or np.isnan(ao[i]):
            continue

        # Check for overbought (both high)
        is_overbought = (wr[i] >= config['overbought_threshold'] and
                        ao[i] >= config['overbought_threshold'])

        # Check for oversold (both low)
        is_oversold = (wr[i] <= config['oversold_threshold'] and
                      ao[i] <= config['oversold_threshold'])

        if not (is_overbought or is_oversold):
            continue

        # Check separation
        separation = abs(wr[i] - ao[i])
        if separation < config['min_separation']:
            continue

        # Determine signal type
        if is_overbought and wr[i] < ao[i]:
            # Overbought, WR pulling away downward = SELL
            signals.append({
                'index': i,
                'type': 'sell',
                'wr': wr[i],
                'ao': ao[i]
            })
        elif is_oversold and wr[i] > ao[i]:
            # Oversold, WR pulling away upward = BUY
            signals.append({
                'index': i,
                'type': 'buy',
                'wr': wr[i],
                'ao': ao[i]
            })

    return signals


def label_signals(df, signals, config):
    """
    Label signals as good (1) or bad (0) based on price movement

    Args:
        df: DataFrame with OHLCV data
        signals: List of signal dicts
        config: Configuration with profit_target and lookforward_bars

    Returns:
        numpy array of labels (1 = good, 0 = bad)
    """
    labels = []
    closes = df['close'].values

    for signal in signals:
        idx = signal['index']
        signal_type = signal['type']

        # Get future price window
        end_idx = min(idx + config['lookforward_bars'] + 1, len(closes))
        future_prices = closes[idx + 1:end_idx]

        if len(future_prices) == 0:
            continue

        current_price = closes[idx]

        if signal_type == 'buy':
            # Good BUY if price rises >= profit_target
            max_future = np.max(future_prices)
            profit = (max_future - current_price) / current_price
            labels.append(1 if profit >= config['profit_target'] else 0)

        elif signal_type == 'sell':
            # Good SELL if price drops >= profit_target
            min_future = np.min(future_prices)
            profit = (current_price - min_future) / current_price
            labels.append(1 if profit >= config['profit_target'] else 0)

    return np.array(labels)


def process_stock_for_training(symbol, config):
    """
    Process one stock symbol for training

    Args:
        symbol: Stock ticker
        config: Training configuration

    Returns:
        tuple: (features, labels) or (None, None) on error
    """
    try:
        logger.info(f"\n{'='*60}")
        logger.info(f"Processing {symbol}...")
        logger.info(f"{'='*60}")

        # Fetch historical data
        df = fetch_historical_data(symbol, days=config['stock_days'], timeframe='1d')

        if df.empty or len(df) < 100:
            logger.warning(f"⚠️ Insufficient data for {symbol}")
            return None, None

        # Rename columns to lowercase
        df.columns = [col.lower() for col in df.columns]

        logger.info(f"  Fetched {len(df)} bars")

        # Detect signals
        signals = detect_triad_signals(df, config)
        logger.info(f"  Detected {len(signals)} signals")

        if len(signals) == 0:
            logger.warning(f"⚠️ No signals detected for {symbol}")
            return None, None

        # Label signals
        labels = label_signals(df, signals, config)

        # Filter out signals without labels
        valid_signals = [s for s, l in zip(signals, labels) if l is not None]
        valid_labels = labels[:len(valid_signals)]

        if len(valid_signals) == 0:
            logger.warning(f"⚠️ No valid labeled signals for {symbol}")
            return None, None

        logger.info(f"  Valid signals: {len(valid_signals)}")
        logger.info(f"  Good signals: {np.sum(valid_labels)} ({np.mean(valid_labels)*100:.1f}%)")

        # Calculate ML features for each signal
        features_df = compute_ml_features(df, timeframe='1d')

        signal_features = []
        for signal in valid_signals:
            idx = signal['index']
            if idx < len(features_df):
                feat = features_df.iloc[idx].values
                if not np.any(np.isnan(feat)):
                    signal_features.append(feat)
                else:
                    logger.warning(f"  ⚠️ NaN features at index {idx}")

        signal_features = np.array(signal_features)

        # Match labels to valid features
        final_labels = valid_labels[:len(signal_features)]

        if len(signal_features) == 0:
            logger.warning(f"⚠️ No valid features for {symbol}")
            return None, None

        logger.info(f"  ✅ Final: {len(signal_features)} samples")

        return signal_features, final_labels

    except Exception as e:
        logger.error(f"❌ Error processing {symbol}: {e}")
        return None, None


def prepare_training_data(config):
    """
    Prepare training dataset from all stocks

    Args:
        config: Training configuration

    Returns:
        tuple: (X_train, y_train, X_val, y_val)
    """
    logger.info("="*60)
    logger.info("PREPARING TRAINING DATA FOR SIGNAL VALIDATION")
    logger.info("="*60)

    all_features = []
    all_labels = []

    for symbol in config['stock_assets']:
        features, labels = process_stock_for_training(symbol, config)

        if features is not None and labels is not None:
            all_features.append(features)
            all_labels.append(labels)

    if len(all_features) == 0:
        logger.error("❌ No training data collected!")
        return None, None, None, None

    # Combine all data
    X = np.vstack(all_features)
    y = np.concatenate(all_labels)

    logger.info(f"\n{'='*60}")
    logger.info(f"DATASET SUMMARY")
    logger.info(f"{'='*60}")
    logger.info(f"Total signals: {len(X)}")
    logger.info(f"Good signals: {np.sum(y)} ({np.mean(y)*100:.1f}%)")
    logger.info(f"Bad signals: {len(y) - np.sum(y)} ({(1-np.mean(y))*100:.1f}%)")
    logger.info(f"Feature shape: {X.shape}")

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
        X_train, y_train: Training data
        X_val, y_val: Validation data
        config: Training configuration

    Returns:
        PivotClassifier: Trained model
    """
    logger.info("="*60)
    logger.info("TRAINING SIGNAL VALIDATION MODEL")
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
    logger.info("SIGNAL VALIDATION MODEL EVALUATION")
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


def save_signal_model(model, epoch, loss):
    """
    Save the signal validation model

    Args:
        model: Trained PivotClassifier
        epoch: Final epoch number
        loss: Final loss value
    """
    model_dir = Path(__file__).parent / 'models'
    model_dir.mkdir(exist_ok=True)

    model_path = model_dir / 'pivot_model.pth'

    torch.save({
        'model_state_dict': model.state_dict(),
        'epoch': epoch,
        'loss': loss,
        'model_type': 'stock',
        'trained_for': 'signal_validation',
        'timestamp': datetime.now().isoformat()
    }, model_path)

    logger.info(f"✅ Signal validation model saved to: {model_path}")


def main():
    """
    Main training script
    """
    logger.info("\n" + "="*60)
    logger.info("TRIAD TREND PULSE SIGNAL VALIDATION MODEL TRAINING")
    logger.info("="*60 + "\n")

    # Prepare data
    X_train, y_train, X_val, y_val = prepare_training_data(TRAINING_CONFIG)

    if X_train is None:
        logger.error("❌ Failed to prepare training data")
        return

    # Train model
    model = train_model(X_train, y_train, X_val, y_val, TRAINING_CONFIG)

    # Evaluate
    evaluate_model(model, X_val, y_val)

    # Save model
    save_signal_model(model, epoch=TRAINING_CONFIG['epochs'], loss=None)

    logger.info("✅ Signal validation training pipeline complete!")
    logger.info(f"Model saved to: {Path(__file__).parent / 'models' / 'pivot_model.pth'}\n")


if __name__ == "__main__":
    main()
