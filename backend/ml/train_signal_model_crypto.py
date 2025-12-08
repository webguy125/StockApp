"""
Training Script for TriadTrendPulse Signal Validation Model (Crypto)

Crypto-specific model for cryptocurrency signal validation.
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

from data_fetcher import fetch_historical_data
from feature_engineering import compute_ml_features
from pivot_model import PivotClassifier

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
        'AVAX-USD',  # Avalanche
        'DOT-USD',   # Polkadot
        'LINK-USD',  # Chainlink
    ],

    # Data period (crypto: 1 year)
    'crypto_days': 365,

    # TriadTrendPulse indicator settings (match frontend defaults)
    'weighted_length': 20,
    'weighted_source': 2,
    'adaptive_period': 50,

    # Signal detection settings (relaxed for training - need more samples)
    'overbought_threshold': 60,   # Relaxed from 80 to get more signals
    'oversold_threshold': -60,    # Relaxed from -80 to get more signals
    'min_separation': 3,          # Reduced from 5 to get more signals

    # Signal labeling (crypto is more volatile - adjust targets)
    'profit_target': 0.05,  # 5% price move for crypto (vs 3% for stocks)
    'lookforward_bars': 10,

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


def calculate_weighted_regression(closes, length=20, source=2):
    """Calculate Weighted Linear Regression"""
    result = np.full(len(closes), np.nan)

    for i in range(length - 1, len(closes)):
        y = closes[i - length + 1:i + 1]
        x = np.arange(length)

        weights = np.arange(1, length + 1)
        poly = np.polyfit(x, y, 1, w=weights)
        reg_value = np.polyval(poly, length - 1)

        price_range = np.max(y) - np.min(y)
        if price_range > 0:
            deviation = (reg_value - np.mean(y)) / price_range
            result[i] = np.clip(deviation * 200, -100, 100)
        else:
            result[i] = 0

    return result


def calculate_adaptive_oscillator(closes, period=50):
    """Calculate Adaptive Oscillator"""
    result = np.full(len(closes), np.nan)

    for i in range(period - 1, len(closes)):
        window = closes[i - period + 1:i + 1]
        sma = np.mean(window)
        std = np.std(window)

        if std > 0:
            deviation = (closes[i] - sma) / std
            result[i] = np.clip(deviation * 20, -100, 100)
        else:
            result[i] = 0

    return result


def detect_triad_signals(df, config):
    """Detect TriadTrendPulse buy/sell signals"""
    wr = calculate_weighted_regression(df['close'].values, config['weighted_length'])
    ao = calculate_adaptive_oscillator(df['close'].values, config['adaptive_period'])

    signals = []
    start_idx = max(config['weighted_length'], config['adaptive_period'])

    for i in range(start_idx, len(df)):
        if np.isnan(wr[i]) or np.isnan(ao[i]):
            continue

        is_overbought = (wr[i] >= config['overbought_threshold'] and
                        ao[i] >= config['overbought_threshold'])
        is_oversold = (wr[i] <= config['oversold_threshold'] and
                      ao[i] <= config['oversold_threshold'])

        if not (is_overbought or is_oversold):
            continue

        separation = abs(wr[i] - ao[i])
        if separation < config['min_separation']:
            continue

        if is_overbought and wr[i] < ao[i]:
            signals.append({'index': i, 'type': 'sell', 'wr': wr[i], 'ao': ao[i]})
        elif is_oversold and wr[i] > ao[i]:
            signals.append({'index': i, 'type': 'buy', 'wr': wr[i], 'ao': ao[i]})

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


def process_crypto_for_training(symbol, config):
    """Process one crypto symbol for training"""
    try:
        logger.info(f"\n{'='*60}")
        logger.info(f"Processing {symbol}...")
        logger.info(f"{'='*60}")

        df = fetch_historical_data(symbol, days=config['crypto_days'], timeframe='1d')

        if df.empty or len(df) < 100:
            logger.warning(f"⚠️ Insufficient data for {symbol}")
            return None, None

        df.columns = [col.lower() for col in df.columns]
        logger.info(f"  Fetched {len(df)} bars")

        signals = detect_triad_signals(df, config)
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


def prepare_crypto_training_data(config):
    """Prepare training dataset from all crypto assets"""
    logger.info("="*60)
    logger.info("PREPARING CRYPTO TRAINING DATA FOR SIGNAL VALIDATION")
    logger.info("="*60)

    all_features = []
    all_labels = []

    for symbol in config['crypto_assets']:
        features, labels = process_crypto_for_training(symbol, config)

        if features is not None and labels is not None:
            all_features.append(features)
            all_labels.append(labels)

    if len(all_features) == 0:
        logger.error("❌ No training data collected!")
        return None, None, None, None

    X = np.vstack(all_features)
    y = np.concatenate(all_labels)

    logger.info(f"\n{'='*60}")
    logger.info(f"CRYPTO DATASET SUMMARY")
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
    logger.info("TRAINING CRYPTO SIGNAL VALIDATION MODEL")
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
    logger.info("CRYPTO SIGNAL VALIDATION MODEL EVALUATION")
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


def save_crypto_signal_model(model, epoch, loss):
    """Save the crypto signal validation model"""
    model_dir = Path(__file__).parent / 'models'
    model_dir.mkdir(exist_ok=True)

    model_path = model_dir / 'pivot_model_crypto.pth'

    torch.save({
        'model_state_dict': model.state_dict(),
        'epoch': epoch,
        'loss': loss,
        'model_type': 'crypto',
        'trained_for': 'signal_validation',
        'timestamp': datetime.now().isoformat()
    }, model_path)

    logger.info(f"✅ Crypto signal validation model saved to: {model_path}")


def main():
    """Main crypto training script"""
    logger.info("\n" + "="*60)
    logger.info("CRYPTO SIGNAL VALIDATION MODEL TRAINING")
    logger.info("="*60 + "\n")

    X_train, y_train, X_val, y_val = prepare_crypto_training_data(CRYPTO_TRAINING_CONFIG)

    if X_train is None:
        logger.error("❌ Failed to prepare training data")
        return

    model = train_model(X_train, y_train, X_val, y_val, CRYPTO_TRAINING_CONFIG)
    evaluate_model(model, X_val, y_val)
    save_crypto_signal_model(model, epoch=CRYPTO_TRAINING_CONFIG['epochs'], loss=None)

    logger.info("✅ Crypto signal validation training pipeline complete!")
    logger.info(f"Model saved to: {Path(__file__).parent / 'models' / 'pivot_model_crypto.pth'}\n")


if __name__ == "__main__":
    main()
