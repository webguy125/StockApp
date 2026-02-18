import os
import sys
import json
import sqlite3
import logging
from datetime import datetime, timedelta
from typing import Dict, List, Tuple, Optional
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
import lightgbm as lgb
import pickle

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
OPTIONS_DIR = os.path.abspath(os.path.join(SCRIPT_DIR, '..'))
TURBOMODE_DIR = os.path.abspath(os.path.join(OPTIONS_DIR, '..'))
BACKEND_DIR = os.path.abspath(os.path.join(TURBOMODE_DIR, '..'))
STOCKAPP_DIR = os.path.abspath(os.path.join(BACKEND_DIR, '..'))

# Use master_market_data.db for price data (not turbomode.db)
MASTER_MARKET_DB = os.path.join(STOCKAPP_DIR, 'master_market_data', 'master_market_data.db')
OPTIONS_UNIVERSE_DB = os.path.join(OPTIONS_DIR, 'Data', 'options_universe.db')
MODELS_DIR = os.path.join(SCRIPT_DIR, 'models')

os.makedirs(MODELS_DIR, exist_ok=True)

logging.basicConfig(level=logging.INFO, format='[%(asctime)s] %(message)s')
logger = logging.getLogger(__name__)


def load_full_dataset():
    logger.info("Loading full combined dataset from master_market_data.db")

    query_underlying = """
        SELECT symbol, timestamp as date, close, volume
        FROM ohlcv
        ORDER BY symbol, timestamp
    """

    conn = sqlite3.connect(MASTER_MARKET_DB)
    df_underlying = pd.read_sql_query(query_underlying, conn)
    conn.close()

    return df_underlying

def load_options_features():
    conn = sqlite3.connect(OPTIONS_UNIVERSE_DB)
    df_opt = pd.read_sql_query("SELECT * FROM option_features_daily", conn)
    conn.close()
    return df_opt


def build_features(df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame]:
    df = df.copy()
    df = df.sort_values(['symbol', 'date']).reset_index(drop=True)

    df['return_1d'] = df.groupby('symbol')['close'].pct_change(1)
    df['return_3d'] = df.groupby('symbol')['close'].pct_change(3)
    df['return_5d'] = df.groupby('symbol')['close'].pct_change(5)
    df['return_10d'] = df.groupby('symbol')['close'].pct_change(10)

    df['realized_vol_5d'] = df.groupby('symbol')['return_1d'].transform(lambda x: x.rolling(5).std() * np.sqrt(252))
    df['realized_vol_10d'] = df.groupby('symbol')['return_1d'].transform(lambda x: x.rolling(10).std() * np.sqrt(252))
    df['realized_vol_20d'] = df.groupby('symbol')['return_1d'].transform(lambda x: x.rolling(20).std() * np.sqrt(252))

    for col in ['iv_atm_call', 'iv_atm_put', 'iv_otm_call_1', 'iv_otm_put_1', 'iv_7d', 'iv_14d', 'iv_30d', 'iv_60d', 'total_oi', 'total_vol']:
        if col not in df.columns:
            df[col] = 0

    df['skew_put_call'] = df['iv_atm_put'] - df['iv_atm_call']
    df['skew_otm_atm_call'] = df['iv_otm_call_1'] - df['iv_atm_call']
    df['skew_otm_atm_put'] = df['iv_otm_put_1'] - df['iv_atm_put']

    df['term_slope_7_14'] = df['iv_14d'] - df['iv_7d']
    df['term_slope_14_30'] = df['iv_30d'] - df['iv_14d']
    df['term_slope_30_60'] = df['iv_60d'] - df['iv_30d']

    df['oi_vol_ratio'] = df['total_oi'] / (df['total_vol'] + 1)
    df['liquidity_score'] = np.log1p(df['total_vol']) * np.log1p(df['total_oi'])

    df['vol_regime'] = (df['realized_vol_20d'] > df['realized_vol_20d'].rolling(60).mean()).astype(int)
    df['trend_regime'] = (df['return_10d'] > 0).astype(int)

    feature_cols = [
        'return_1d', 'return_3d', 'return_5d', 'return_10d',
        'realized_vol_5d', 'realized_vol_10d', 'realized_vol_20d',
        'iv_atm_call', 'iv_atm_put', 'iv_otm_call_1', 'iv_otm_put_1',
        'iv_7d', 'iv_14d', 'iv_30d', 'iv_60d',
        'skew_put_call', 'skew_otm_atm_call', 'skew_otm_atm_put',
        'term_slope_7_14', 'term_slope_14_30', 'term_slope_30_60',
        'oi_vol_ratio', 'liquidity_score',
        'vol_regime', 'trend_regime',
        'close', 'volume'
    ]

    X = df[feature_cols].copy()
    X = X.fillna(0)

    return X, df


def compute_direction_labels(df: pd.DataFrame, horizons: Dict[str, int]) -> pd.DataFrame:
    df = df.copy()

    for horizon_name, horizon_days in horizons.items():
        col_name = f'direction_{horizon_name}'
        # Calculate returns, handling division by zero
        future_price = df.groupby('symbol')['close'].shift(-horizon_days)
        current_price = df['close']

        # Only calculate where current price > 0 to avoid division by zero
        df[col_name] = np.where(
            current_price > 0,
            (future_price / current_price) - 1,
            np.nan
        )

        # Replace inf/-inf with NaN
        df[col_name] = df[col_name].replace([np.inf, -np.inf], np.nan)

    return df


def compute_volatility_labels(df: pd.DataFrame, horizons: Dict[str, int]) -> pd.DataFrame:
    df = df.copy()

    for horizon_name, horizon_days in horizons.items():
        col_name = f'volatility_{horizon_name}'
        returns = df.groupby('symbol')['close'].pct_change()
        df[col_name] = returns.groupby(df['symbol']).transform(
            lambda x: x.shift(-horizon_days).rolling(horizon_days).std() * np.sqrt(252)
        )

    return df


def compute_strategy_label(row: pd.Series, horizons: Dict[str, int]) -> str:
    direction_h1 = row.get('direction_H1', 0)
    direction_h2 = row.get('direction_H2', 0)
    vol_h1 = row.get('volatility_H1', 0)
    vol_h2 = row.get('volatility_H2', 0)

    avg_vol = (vol_h1 + vol_h2) / 2 if (vol_h1 > 0 and vol_h2 > 0) else 0.3
    avg_direction = (direction_h1 + direction_h2) / 2

    strategies = {}

    if avg_direction > 0.03:
        call_pnl = max(0, avg_direction - 0.02) * 100
        strategies['CALL'] = call_pnl
        call_spread_pnl = min(avg_direction * 50, 5)
        strategies['CALL_SPREAD'] = call_spread_pnl
    elif avg_direction < -0.03:
        put_pnl = max(0, -avg_direction - 0.02) * 100
        strategies['PUT'] = put_pnl
        put_spread_pnl = min(-avg_direction * 50, 5)
        strategies['PUT_SPREAD'] = put_spread_pnl

    if abs(avg_direction) < 0.02 and avg_vol < 0.25:
        ic_pnl = 3.0
        strategies['IRON_CONDOR'] = ic_pnl

    if avg_vol > 0.35:
        cal_pnl = (avg_vol - 0.35) * 10
        strategies['CALENDAR'] = cal_pnl

    liquidity = row.get('liquidity_score', 0)
    if liquidity < 5 or len(strategies) == 0:
        strategies['NO_TRADE'] = 5.0

    if len(strategies) == 0:
        return 'NO_TRADE'

    best_strategy = max(strategies, key=strategies.get)
    return best_strategy


def split_dataset(df: pd.DataFrame, train_ratio: float, val_ratio: float) -> Tuple:
    df = df.sort_values('date').reset_index(drop=True)

    n = len(df)
    train_end = int(n * train_ratio)
    val_end = int(n * (train_ratio + val_ratio))

    train_df = df.iloc[:train_end]
    val_df = df.iloc[train_end:val_end]
    test_df = df.iloc[val_end:]

    # Dynamically select feature columns (exclude metadata and labels)
    exclude_cols = ['symbol', 'date', 'direction_H1', 'direction_H2',
                    'volatility_H1', 'volatility_H2', 'strategy_label', 'id', 'created_at']
    feature_cols = [col for col in df.columns if col not in exclude_cols]

    logger.info(f"Using {len(feature_cols)} features for training: {feature_cols}")

    X_train = train_df[feature_cols].fillna(0).values
    X_val = val_df[feature_cols].fillna(0).values
    X_test = test_df[feature_cols].fillna(0).values

    y_train_direction = train_df[['direction_H1', 'direction_H2']].fillna(0).values
    y_val_direction = val_df[['direction_H1', 'direction_H2']].fillna(0).values
    y_test_direction = test_df[['direction_H1', 'direction_H2']].fillna(0).values

    y_train_volatility = train_df[['volatility_H1', 'volatility_H2']].fillna(0).values
    y_val_volatility = val_df[['volatility_H1', 'volatility_H2']].fillna(0).values
    y_test_volatility = test_df[['volatility_H1', 'volatility_H2']].fillna(0).values

    strategy_map = {
        'CALL': 0, 'PUT': 1, 'CALL_SPREAD': 2, 'PUT_SPREAD': 3,
        'IRON_CONDOR': 4, 'CALENDAR': 5, 'NO_TRADE': 6
    }

    y_train_strategy = train_df['strategy_label'].map(strategy_map).fillna(6).values
    y_val_strategy = val_df['strategy_label'].map(strategy_map).fillna(6).values
    y_test_strategy = test_df['strategy_label'].map(strategy_map).fillna(6).values

    return (X_train, y_train_direction, y_train_volatility, y_train_strategy,
            X_val, y_val_direction, y_val_volatility, y_val_strategy,
            X_test, y_test_direction, y_test_volatility, y_test_strategy)


def train_direction_model(X_train: np.ndarray, y_train: np.ndarray) -> Tuple:
    model_h1 = lgb.LGBMRegressor(n_estimators=200, learning_rate=0.05, max_depth=6, random_state=42)
    model_h2 = lgb.LGBMRegressor(n_estimators=200, learning_rate=0.05, max_depth=6, random_state=42)

    model_h1.fit(X_train, y_train[:, 0])
    model_h2.fit(X_train, y_train[:, 1])

    logger.info(f'Direction model H1 trained')
    logger.info(f'Direction model H2 trained')

    return model_h1, model_h2


def train_volatility_model(X_train: np.ndarray, y_train: np.ndarray) -> Tuple:
    model_h1 = lgb.LGBMRegressor(n_estimators=200, learning_rate=0.05, max_depth=6, random_state=42)
    model_h2 = lgb.LGBMRegressor(n_estimators=200, learning_rate=0.05, max_depth=6, random_state=42)

    model_h1.fit(X_train, y_train[:, 0])
    model_h2.fit(X_train, y_train[:, 1])

    logger.info(f'Volatility model H1 trained')
    logger.info(f'Volatility model H2 trained')

    return model_h1, model_h2


def train_strategy_model(X_train: np.ndarray, y_train: np.ndarray) -> lgb.LGBMClassifier:
    model = lgb.LGBMClassifier(
        n_estimators=200,
        learning_rate=0.05,
        max_depth=6,
        num_class=7,
        random_state=42
    )

    model.fit(X_train, y_train)

    logger.info(f'Strategy model trained')

    return model


def save_model(model, path: str):
    with open(path, 'wb') as f:
        pickle.dump(model, f)
    logger.info(f'Model saved to {path}')


def save_metadata(metadata: Dict, path: str):
    with open(path, 'w') as f:
        json.dump(metadata, f, indent=2)
    logger.info(f'Metadata saved to {path}')


def run_training_pipeline(hyperparams):
    horizons = hyperparams
    logger.info(f'Loading full dataset')
    df = load_full_dataset()
    logger.info(f'Loaded {len(df)} rows')

    logger.info(f'Building features')
    X, df_features = build_features(df)
    logger.info(f'Built features: {len(df_features)} rows')

    logger.info(f'Computing direction labels')
    df_features = compute_direction_labels(df_features, horizons)

    logger.info(f'Computing volatility labels')
    df_features = compute_volatility_labels(df_features, horizons)

    logger.info(f'Computing strategy labels')
    df_features['strategy_label'] = df_features.apply(lambda row: compute_strategy_label(row, horizons), axis=1)
    logger.info(f'Features after labels: {len(df_features)} rows, columns: {list(df_features.columns)}')

    # ---------------------------------------------------------
    # MERGE OPTIONS FEATURES
    # ---------------------------------------------------------
    logger.info("Loading and merging options features")

    df_opt = load_options_features()

    # Ensure date alignment
    df_opt['date'] = pd.to_datetime(df_opt['date'])
    df_features['date'] = pd.to_datetime(df_features['date'])

    # Merge on symbol + date
    df_features = df_features.merge(
        df_opt,
        on=['symbol', 'date'],
        how='left'
    )

    logger.info(f"After merging options features: {len(df_features)} rows, columns: {len(df_features.columns)}")

    # Replace inf with NaN globally
    df_features = df_features.replace([np.inf, -np.inf], np.nan)

    # Drop rows with invalid labels
    required_labels = ['direction_H1','direction_H2','volatility_H1','volatility_H2','strategy_label']
    df_features = df_features.dropna(subset=required_labels)

    logger.info(f"After cleaning invalid labels: {len(df_features)} rows")
    # ---------------------------------------------------------

    logger.info(f'Splitting dataset')
    split_result = split_dataset(df_features, train_ratio=0.7, val_ratio=0.15)
    (X_train, y_train_dir, y_train_vol, y_train_strat,
     X_val, y_val_dir, y_val_vol, y_val_strat,
     X_test, y_test_dir, y_test_vol, y_test_strat) = split_result
    logger.info(f'Split sizes - Train: {len(X_train)}, Val: {len(X_val)}, Test: {len(X_test)}')

    logger.info(f'Training direction models')
    dir_model_h1, dir_model_h2 = train_direction_model(X_train, y_train_dir)

    logger.info(f'Training volatility models')
    vol_model_h1, vol_model_h2 = train_volatility_model(X_train, y_train_vol)

    logger.info(f'Training strategy model')
    strat_model = train_strategy_model(X_train, y_train_strat)

    logger.info(f'Saving models')
    save_model(dir_model_h1, os.path.join(MODELS_DIR, 'direction_h1.pkl'))
    save_model(dir_model_h2, os.path.join(MODELS_DIR, 'direction_h2.pkl'))
    save_model(vol_model_h1, os.path.join(MODELS_DIR, 'volatility_h1.pkl'))
    save_model(vol_model_h2, os.path.join(MODELS_DIR, 'volatility_h2.pkl'))
    save_model(strat_model, os.path.join(MODELS_DIR, 'strategy.pkl'))

    metadata = {
        'train_window': 'full_dataset',
        'horizons': horizons,
        'feature_list': [
            'return_1d', 'return_3d', 'return_5d', 'return_10d',
            'realized_vol_5d', 'realized_vol_10d', 'realized_vol_20d',
            'iv_atm_call', 'iv_atm_put', 'iv_otm_call_1', 'iv_otm_put_1',
            'iv_7d', 'iv_14d', 'iv_30d', 'iv_60d',
            'skew_put_call', 'skew_otm_atm_call', 'skew_otm_atm_put',
            'term_slope_7_14', 'term_slope_14_30', 'term_slope_30_60',
            'oi_vol_ratio', 'liquidity_score',
            'vol_regime', 'trend_regime',
            'close', 'volume'
        ],
        'metrics': {
            'train_samples': len(X_train),
            'val_samples': len(X_val),
            'test_samples': len(X_test)
        },
        'timestamp': datetime.now().isoformat()
    }

    save_metadata(metadata, os.path.join(MODELS_DIR, 'metadata.json'))

    logger.info(f'Training pipeline complete')


if __name__ == '__main__':
    run_training_pipeline({'H1': 2, 'H2': 10})
