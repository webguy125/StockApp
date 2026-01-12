"""
Training Script for RSI Signal Validation Model

This script trains an ML model specifically for validating RSI indicator signals.

Workflow:
1. Fetch historical data for multiple stock assets
2. Calculate RSI indicator
3. Detect buy/sell signals (RSI crosses oversold/overbought levels)
4. Label signals as "good" (1) or "bad" (0) based on price movement after signal
5. Extract 9 ML features at each signal point
6. Train PyTorch model with validation split
7. Save model to models/rsi_model.pth

Signal Logic:
- BUY signal: RSI crosses above 30 (oversold)
- SELL signal: RSI crosses below 70 (overbought)

Signal Labeling:
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

    # RSI settings
    'rsi_period': 14,
    'oversold_level': 30,
    'overbought_level': 70,

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


def calculate_rsi(closes, period=14):
    """Calculate RSI indicator"""
    if len(closes) < period + 1:
        return np.full(len(closes), np.nan)

    result = np.full(len(closes), np.nan)
    changes = np.diff(closes)

    # Initial average gain/loss
    gains = np.where(changes > 0, changes, 0)
    losses = np.where(changes < 0, -changes, 0)

    avg_gain = np.mean(gains[:period])
    avg_loss = np.mean(losses[:period])

    if avg_loss == 0:
        result[period] = 100
    else:
        rs = avg_gain / avg_loss
        result[period] = 100 - (100 / (1 + rs))

    # Wilder's smoothing
    for i in range(period, len(changes)):
        avg_gain = (avg_gain * (period - 1) + gains[i]) / period
        avg_loss = (avg_loss * (period - 1) + losses[i]) / period

        if avg_loss == 0:
            result[i + 1] = 100
        else:
            rs = avg_gain / avg_loss
            result[i + 1] = 100 - (100 / (1 + rs))

    return result


def detect_rsi_signals(df, config):
    """Detect RSI buy/sell signals"""
    closes = df['close'].values
    rsi = calculate_rsi(closes, config['rsi_period'])

    signals = []

    for i in range(1, len(df)):
        if np.isnan(rsi[i]) or np.isnan(rsi[i - 1]):
            continue

        # Buy signal: RSI crosses above oversold
        if rsi[i - 1] <= config['oversold_level'] and rsi[i] > config['oversold_level']:
            signals.append({'index': i, 'type': 'buy', 'rsi': rsi[i]})

        # Sell signal: RSI crosses below overbought
        elif rsi[i - 1] >= config['overbought_level'] and rsi[i] < config['overbought_level']:
            signals.append({'index': i, 'type': 'sell', 'rsi': rsi[i]})

    return signals


def label_signals(df, signals, config):
    """Label signals as good (1) or bad (0) based on price movement"""
    labels = []
    closes = df['close'].values

    for signal in signals:
        idx = signal['index']
        signal_type = signal['type']

        end_idx = min(idx + config['lookforward_bars'] + 1, len(closes))
        future_prices = closes[idx + 1:end_idx]

        if len(future_prices) == 0:
            continue

        current_price = closes[idx]

        if signal_type == 'buy':
            max_future = np.max(future_prices)
            profit = (max_future - current_price) / current_price
            labels.append(1 if profit >= config['profit_target'] else 0)
        elif signal_type == 'sell':
            min_future = np.min(future_prices)
            profit = (current_price - min_future) / current_price
            labels.append(1 if profit >= config['profit_target'] else 0)

    return np.array(labels)


def process_stock_for_training(symbol, config):
    """Process one stock symbol for training"""
    try:
        logger.info(f"\n{'='*60}")
        logger.info(f"Processing {symbol}...")
        logger.info(f"{'='*60}")

        df = fetch_historical_data(symbol, days=config['stock_days'], timeframe='1d')

        if df.empty or len(df) < 100:
            logger.warning(f"⚠️ Insufficient data for {symbol}")
            return None, None

        df.columns = [col.lower() for col in df.columns]
        logger.info(f"  Fetched {len(df)} bars")

        signals = detect_rsi_signals(df, config)
        logger.info(f"  Detected {len(signals)} signals")

        if len(signals) == 0:
            logger.warning(f"⚠️ No signals detected for {symbol}")
            return None, None

        labels = label_signals(df, signals, config)
        valid_signals = [s for s, l in zip(signals, labels) if l is not None]
        valid_labels = labels[:len(valid_signals)]

        if len(valid_signals) == 0:
            logger.warning(f"⚠️ No valid labeled signals for {symbol}")
            return None, None

        logger.info(f"  Valid signals: {len(valid_signals)}")
        logger.info(f"  Good signals: {np.sum(valid_labels)} ({np.mean(valid_labels)*100:.1f}%)")

        features_df = compute_ml_features(df, timeframe='1d')

        signal_features = []
        for signal in valid_signals:
            idx = signal['index']
            if idx < len(features_df):
                feat = features_df.iloc[idx].values
                if not np.any(np.isnan(feat)):
                    signal_features.append(feat)

        signal_features = np.array(signal_features)
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
    """Prepare training dataset from all stock assets"""
    logger.info("="*60)
    logger.info("PREPARING RSI TRAINING DATA FOR SIGNAL VALIDATION")
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

    X = np.vstack(all_features)
    y = np.concatenate(all_labels)

    logger.info(f"\n{'='*60}")
    logger.info(f"RSI DATASET SUMMARY")
    logger.info(f"{'='*60}")
    logger.info(f"Total signals: {len(X)}")
    logger.info(f"Good signals: {np.sum(y)} ({np.mean(y)*100:.1f}%)")
    logger.info(f"Bad signals: {len(y) - np.sum(y)} ({(1-np.mean(y))*100:.1f}%)")
    logger.info(f"Feature shape: {X.shape}")

    indices = np.random.permutation(len(X))
    X = X[indices]
    y = y[indices]

    split_idx = int(len(X) * config['train_split'])
    X_train, X_val = X[:split_idx], X[split_idx:]
    y_train, y_val = y[:split_idx], y[split_idx:]

    logger.info(f"\nTrain: {len(X_train)} samples")
    logger.info(f"Validation: {len(X_val)} samples")
    logger.info(f"{'='*60}\n")

    return X_train, y_train, X_val, y_val


def train_model(X_train, y_train, X_val, y_val, config):
    """Train the PyTorch model"""
    logger.info("="*60)
    logger.info("TRAINING RSI SIGNAL VALIDATION MODEL")
    logger.info("="*60)

    X_train_t = torch.FloatTensor(X_train)
    y_train_t = torch.FloatTensor(y_train).reshape(-1, 1)
    X_val_t = torch.FloatTensor(X_val)
    y_val_t = torch.FloatTensor(y_val).reshape(-1, 1)

    train_dataset = TensorDataset(X_train_t, y_train_t)
    train_loader = DataLoader(train_dataset, batch_size=config['batch_size'], shuffle=True)

    model = PivotClassifier(
        input_dim=9,
        hidden1=config['hidden1'],
        hidden2=config['hidden2']
    )

    criterion = nn.BCELoss()
    optimizer = optim.Adam(model.parameters(), lr=config['learning_rate'])

    best_val_loss = float('inf')
    patience_counter = 0
    best_model_state = None

    for epoch in range(config['epochs']):
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

        model.eval()
        with torch.no_grad():
            val_outputs = model(X_val_t)
            val_loss = criterion(val_outputs, y_val_t).item()
            val_preds = (val_outputs >= 0.5).float()
            val_acc = (val_preds == y_val_t).float().mean().item()

        if (epoch + 1) % 10 == 0 or epoch == 0:
            logger.info(f"Epoch {epoch+1}/{config['epochs']} - "
                       f"Train Loss: {train_loss:.4f}, "
                       f"Val Loss: {val_loss:.4f}, "
                       f"Val Acc: {val_acc*100:.2f}%")

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_model_state = model.state_dict().copy()
            patience_counter = 0
        else:
            patience_counter += 1

        if patience_counter >= config['early_stopping_patience']:
            logger.info(f"\n⚠️ Early stopping at epoch {epoch+1}")
            break

    model.load_state_dict(best_model_state)

    logger.info(f"\n✅ Training complete!")
    logger.info(f"Best validation loss: {best_val_loss:.4f}")
    logger.info(f"{'='*60}\n")

    return model


def evaluate_model(model, X_val, y_val):
    """Evaluate model performance"""
    logger.info("="*60)
    logger.info("RSI SIGNAL VALIDATION MODEL EVALUATION")
    logger.info("="*60)

    model.eval()

    with torch.no_grad():
        X_val_t = torch.FloatTensor(X_val)
        y_val_t = torch.FloatTensor(y_val)
        probs = model(X_val_t).numpy().flatten()

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


def save_rsi_model(model, epoch, loss):
    """Save the RSI signal validation model"""
    model_dir = Path(__file__).parent / 'models'
    model_dir.mkdir(exist_ok=True)

    model_path = model_dir / 'rsi_model.pth'

    torch.save({
        'model_state_dict': model.state_dict(),
        'epoch': epoch,
        'loss': loss,
        'model_type': 'rsi',
        'trained_for': 'signal_validation',
        'timestamp': datetime.now().isoformat()
    }, model_path)

    logger.info(f"✅ RSI signal validation model saved to: {model_path}")


def main():
    """Main RSI training script"""
    logger.info("\n" + "="*60)
    logger.info("RSI SIGNAL VALIDATION MODEL TRAINING")
    logger.info("="*60 + "\n")

    X_train, y_train, X_val, y_val = prepare_training_data(TRAINING_CONFIG)

    if X_train is None:
        logger.error("❌ Failed to prepare training data")
        return

    model = train_model(X_train, y_train, X_val, y_val, TRAINING_CONFIG)
    evaluate_model(model, X_val, y_val)
    save_rsi_model(model, epoch=TRAINING_CONFIG['epochs'], loss=None)

    logger.info("✅ RSI signal validation training pipeline complete!")
    logger.info(f"Model saved to: {Path(__file__).parent / 'models' / 'rsi_model.pth'}\n")


if __name__ == "__main__":
    main()
