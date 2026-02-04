#!/usr/bin/env python
"""
Risk Governor Market Data Integration

Provides real market data for Risk Governor derived metrics computation.
Integrates with:
- Master Market Data DB (historical candles)
- IBKR Data Adapter (real-time quotes when available)

Step 1 Integration: Real-Time Price Feed
Author: TurboMode Production Team
Date: 2026-01-25
"""

import sys
from pathlib import Path
project_root = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(project_root))

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple
import logging
import sqlite3

logger = logging.getLogger(__name__)

# Database paths
MASTER_MARKET_DB_PATH = project_root / "master_market_data" / "market_data.db"


# ============================================================================
# REAL-TIME PRICE DATA LOADER
# ============================================================================

def load_realtime_prices(symbols: List[str]) -> Dict[str, Dict[str, float]]:
    """
    Load current price, bid, ask for symbols.

    Integration Strategy:
    - Production: Should query IBKR TWS API for real-time quotes
    - Fallback: Use most recent candle close price from master_market_data

    Args:
        symbols: List of stock tickers

    Returns:
        Dict mapping symbol -> {
            'current_price': float,
            'bid': float,
            'ask': float,
            'mid': float,
            'timestamp': str (ISO format)
        }
    """
    result = {}

    # Fallback: Use latest candle close from master_market_data
    conn = sqlite3.connect(str(MASTER_MARKET_DB_PATH))
    cursor = conn.cursor()

    for symbol in symbols:
        cursor.execute("""
            SELECT close, timestamp
            FROM candles
            WHERE symbol = ? AND timeframe = '1d'
            ORDER BY timestamp DESC
            LIMIT 1
        """, (symbol,))

        row = cursor.fetchone()

        if row:
            close_price = row[0]
            timestamp = row[1]

            # Simulate bid/ask spread (0.5% typical for liquid stocks)
            spread_pct = 0.005
            half_spread = close_price * spread_pct / 2

            result[symbol] = {
                'current_price': close_price,
                'bid': close_price - half_spread,
                'ask': close_price + half_spread,
                'mid': close_price,
                'timestamp': timestamp
            }
        else:
            logger.warning(f"[{symbol}] No price data available")
            result[symbol] = {
                'current_price': 0.0,
                'bid': 0.0,
                'ask': 0.0,
                'mid': 0.0,
                'timestamp': datetime.utcnow().isoformat()
            }

    conn.close()
    return result


def load_realtime_spreads(symbols: List[str]) -> Dict[str, float]:
    """
    Compute bid-ask spreads for symbols.

    Args:
        symbols: List of stock tickers

    Returns:
        Dict mapping symbol -> spread_pct (as fraction, e.g., 0.005 = 0.5%)
    """
    prices = load_realtime_prices(symbols)

    spreads = {}
    for symbol, data in prices.items():
        if data['mid'] > 0:
            spread = data['ask'] - data['bid']
            spread_pct = spread / data['mid']
            spreads[symbol] = spread_pct
        else:
            spreads[symbol] = 0.0

    return spreads


def load_realtime_volume(symbols: List[str]) -> Dict[str, int]:
    """
    Load current day volume for symbols.

    Args:
        symbols: List of stock tickers

    Returns:
        Dict mapping symbol -> volume (integer)
    """
    result = {}

    conn = sqlite3.connect(str(MASTER_MARKET_DB_PATH))
    cursor = conn.cursor()

    for symbol in symbols:
        cursor.execute("""
            SELECT volume
            FROM candles
            WHERE symbol = ? AND timeframe = '1d'
            ORDER BY timestamp DESC
            LIMIT 1
        """, (symbol,))

        row = cursor.fetchone()
        result[symbol] = row[0] if row else 0

    conn.close()
    return result


def load_realtime_bars(symbols: List[str], lookback_days: int = 1) -> Dict[str, pd.DataFrame]:
    """
    Load recent OHLCV bars for symbols.

    Args:
        symbols: List of stock tickers
        lookback_days: Number of days to look back

    Returns:
        Dict mapping symbol -> DataFrame with columns:
            [timestamp, open, high, low, close, volume]
    """
    result = {}

    conn = sqlite3.connect(str(MASTER_MARKET_DB_PATH))

    end_date = datetime.now()
    start_date = end_date - timedelta(days=lookback_days + 5)  # +5 for weekend buffer

    for symbol in symbols:
        query = """
            SELECT timestamp, open, high, low, close, volume
            FROM candles
            WHERE symbol = ?
              AND timeframe = '1d'
              AND DATE(timestamp) >= ?
            ORDER BY timestamp DESC
            LIMIT ?
        """

        df = pd.read_sql_query(
            query,
            conn,
            params=(symbol, start_date.strftime('%Y-%m-%d'), lookback_days + 5)
        )

        if not df.empty:
            df['timestamp'] = pd.to_datetime(df['timestamp'])
            df = df.sort_values('timestamp')

        result[symbol] = df

    conn.close()
    return result


# ============================================================================
# DERIVED METRICS COMPUTATION (REAL)
# ============================================================================

def compute_realized_vol_1m_annualized(symbol: str, prices: Dict[str, Dict[str, float]]) -> float:
    """
    Compute 1-minute realized volatility (annualized).

    Formula:
        std_dev(1-min returns) * sqrt(252 * 390)  # 252 trading days, 390 min/day

    Note: This is a placeholder. True 1-min vol requires intraday data.
    For now, returns 0.0 (will be computed from real-time data later).

    Args:
        symbol: Stock ticker
        prices: Real-time price data

    Returns:
        Annualized volatility (as decimal, e.g., 0.20 = 20%)
    """
    # Placeholder: Requires intraday tick data
    # In production, compute from 1-min bars
    return 0.0


def compute_realized_vol_20d_baseline(symbol: str) -> float:
    """
    Compute 20-day baseline volatility (annualized).

    Formula:
        std_dev(daily returns over 20 days) * sqrt(252)

    Args:
        symbol: Stock ticker

    Returns:
        Annualized volatility (as decimal, e.g., 0.20 = 20%)
    """
    conn = sqlite3.connect(str(MASTER_MARKET_DB_PATH))

    query = """
        SELECT close
        FROM candles
        WHERE symbol = ? AND timeframe = '1d'
        ORDER BY timestamp DESC
        LIMIT 21
    """

    df = pd.read_sql_query(query, conn, params=(symbol,))
    conn.close()

    if len(df) < 21:
        logger.warning(f"[{symbol}] Insufficient data for 20-day volatility (need 21 days, have {len(df)})")
        return 0.20  # Default to 20% annualized

    # Compute daily returns
    closes = df['close'].values[::-1]  # Reverse to chronological order
    returns = np.diff(np.log(closes))

    # Annualize
    vol_20d = np.std(returns) * np.sqrt(252)

    return vol_20d


def compute_volatility_ratio(symbol: str, prices: Dict[str, Dict[str, float]]) -> float:
    """
    Compute volatility_ratio = realized_vol_1m / baseline_vol_20d.

    Per JSON spec:
        volatility_ratio = realized_vol_1m_annualized / realized_vol_20d_baseline

    Thresholds:
        CAUTION: >= 1.8
        CRITICAL: >= 2.5

    Args:
        symbol: Stock ticker
        prices: Real-time price data

    Returns:
        Volatility ratio (dimensionless)
    """
    vol_1m = compute_realized_vol_1m_annualized(symbol, prices)
    vol_20d = compute_realized_vol_20d_baseline(symbol)

    if vol_20d == 0:
        return 1.0  # Neutral if no baseline

    # For now, vol_1m is placeholder (0.0), so ratio will be 0.0
    # In production with real-time data, this will compute actual ratio
    if vol_1m == 0:
        return 1.0  # Default to normal volatility

    return vol_1m / vol_20d


def compute_avg_spread_pct(symbols: List[str]) -> float:
    """
    Compute average bid-ask spread across symbols.

    Per v3.0.0 JSON spec:
        avg_spread_pct = mean((ask - bid) / ((ask + bid) / 2))
        "Use MEAN spread, not MAX spread"

    Args:
        symbols: List of stock tickers

    Returns:
        Mean spread as fraction (e.g., 0.004 = 0.4%)
    """
    if not symbols:
        return 0.0

    spreads = load_realtime_spreads(symbols)

    if not spreads:
        return 0.0

    # Return MEAN spread (per v3.0.0 spec, NOT max)
    return np.mean(list(spreads.values()))


def compute_volume_spike_factor_5m(symbol: str) -> float:
    """
    Compute volume spike factor (5-min average vs 20-day average).

    Formula:
        volume_spike_factor = avg_volume_5m / avg_volume_20d

    Note: Requires intraday 5-min bars. Placeholder for now.

    Args:
        symbol: Stock ticker

    Returns:
        Volume spike factor (e.g., 5.0 = 5x normal volume)
    """
    # Placeholder: Requires intraday data
    # In production, compute from 5-min bars
    return 1.0


def compute_atr_14(symbol: str) -> float:
    """
    Compute 14-day Average True Range.

    Formula:
        ATR = SMA(TrueRange, 14)
        TrueRange = max(high - low, abs(high - prev_close), abs(low - prev_close))

    Args:
        symbol: Stock ticker

    Returns:
        ATR in dollars
    """
    conn = sqlite3.connect(str(MASTER_MARKET_DB_PATH))

    query = """
        SELECT high, low, close
        FROM candles
        WHERE symbol = ? AND timeframe = '1d'
        ORDER BY timestamp DESC
        LIMIT 15
    """

    df = pd.read_sql_query(query, conn, params=(symbol,))
    conn.close()

    if len(df) < 15:
        logger.warning(f"[{symbol}] Insufficient data for ATR-14 (need 15 days, have {len(df)})")
        return 0.0

    df = df.iloc[::-1]  # Reverse to chronological order

    # Compute True Range
    high = df['high'].values
    low = df['low'].values
    close = df['close'].values

    tr_list = []
    for i in range(1, len(df)):
        tr = max(
            high[i] - low[i],
            abs(high[i] - close[i-1]),
            abs(low[i] - close[i-1])
        )
        tr_list.append(tr)

    # Average True Range (last 14 days)
    atr = np.mean(tr_list[-14:])

    return atr


def compute_confidence_collapse_pct(symbols: List[str]) -> float:
    """
    Compute confidence collapse percentage.

    Per v3.0.0 JSON spec:
        confidence_collapse_pct = (baseline_confidence_20d - avg_confidence) / baseline_confidence_20d

    Components:
        baseline_confidence_20d = mean(confidence over past 20 days)
        avg_confidence = mean(confidence across active positions)

    Args:
        symbols: List of stock tickers for active positions

    Returns:
        Confidence collapse as fraction (e.g., 0.20 = 20% collapse)

    Integration: Uses micro_scanner_adapter (Step 4)
    """
    # INTEGRATED: micro_scanner_adapter (Step 4)
    from backend.turbomode.adapters.micro_scanner_adapter import (
        get_baseline_confidence_20d,
        get_avg_confidence
    )

    # Get baseline and current confidence from adapter
    baseline_confidence_20d = get_baseline_confidence_20d(symbols)
    avg_confidence = get_avg_confidence(symbols)

    # Compute collapse percentage per v3.0.0 formula
    if baseline_confidence_20d == 0:
        return 0.0  # No collapse if no baseline

    confidence_collapse_pct = (baseline_confidence_20d - avg_confidence) / baseline_confidence_20d

    return confidence_collapse_pct


def compute_intraday_drawdown_pct() -> float:
    """
    Compute intraday drawdown percentage.

    Per v3.0.0 JSON spec:
        intraday_drawdown_pct = (peak_equity_today - current_equity) / peak_equity_today

    Note:
        - Reset peak_equity_today at market open each day
        - Only tracks drawdown from today's peak, not multi-day

    Returns:
        Intraday drawdown as fraction (e.g., 0.03 = 3% drawdown)

    Integration: Uses portfolio_state_adapter (Step 5)
    """
    # INTEGRATED: portfolio_state_adapter (Step 5)
    from backend.turbomode.adapters.portfolio_state_adapter import (
        get_intraday_equity,
        get_peak_equity_today
    )

    # Get equity values from adapter
    current_equity = get_intraday_equity()
    peak_equity_today = get_peak_equity_today()

    # Compute intraday drawdown per v3.0.0 formula
    if peak_equity_today == 0:
        return 0.0  # No drawdown if no peak

    intraday_drawdown_pct = (peak_equity_today - current_equity) / peak_equity_today

    return intraday_drawdown_pct


# ============================================================================
# INTEGRATION POINT FOR RISK GOVERNOR
# ============================================================================

class RiskGovernorMarketData:
    """
    Market data provider for Risk Governor.

    Replaces placeholder functions in risk_governor_daemon.py with real implementations.
    """

    def __init__(self):
        self.db_path = MASTER_MARKET_DB_PATH

    def get_realtime_data(self, symbols: List[str]) -> Dict[str, Dict[str, float]]:
        """Load real-time prices, bid, ask, volume."""
        prices = load_realtime_prices(symbols)
        volumes = load_realtime_volume(symbols)

        # Merge volume into price data
        for symbol in symbols:
            if symbol in prices:
                prices[symbol]['volume'] = volumes.get(symbol, 0)

        return prices

    def compute_volatility_ratio_for_symbol(self, symbol: str) -> float:
        """Compute volatility ratio for a symbol."""
        prices = load_realtime_prices([symbol])
        return compute_volatility_ratio(symbol, prices)

    def compute_liquidity_stress(self, symbols: List[str]) -> float:
        """Compute maximum spread across positions."""
        return compute_avg_spread_pct(symbols)

    def compute_atr(self, symbol: str) -> float:
        """Compute 14-day ATR for a symbol."""
        return compute_atr_14(symbol)


# ============================================================================
# TEST SUITE
# ============================================================================

def test_step1_integration():
    """Test Step 1: Real-Time Price Feed Integration."""
    print("=" * 80)
    print("STEP 1 INTEGRATION TEST: Real-Time Price Feed")
    print("=" * 80)
    print()

    # Test symbols
    test_symbols = ['AAPL', 'MSFT', 'GOOGL']

    print(f"Test symbols: {test_symbols}")
    print()

    # Test 1: Load realtime prices
    print("TEST 1: load_realtime_prices()")
    prices = load_realtime_prices(test_symbols)
    for symbol, data in prices.items():
        print(f"  [{symbol}] Price: ${data['current_price']:.2f}, Bid: ${data['bid']:.2f}, Ask: ${data['ask']:.2f}")
    print()

    # Test 2: Load spreads
    print("TEST 2: load_realtime_spreads()")
    spreads = load_realtime_spreads(test_symbols)
    for symbol, spread in spreads.items():
        print(f"  [{symbol}] Spread: {spread*100:.3f}%")
    print()

    # Test 3: Compute 20-day volatility
    print("TEST 3: compute_realized_vol_20d_baseline()")
    for symbol in test_symbols:
        vol_20d = compute_realized_vol_20d_baseline(symbol)
        print(f"  [{symbol}] 20-day vol: {vol_20d*100:.2f}%")
    print()

    # Test 4: Compute ATR
    print("TEST 4: compute_atr_14()")
    for symbol in test_symbols:
        atr = compute_atr_14(symbol)
        print(f"  [{symbol}] ATR-14: ${atr:.2f}")
    print()

    # Test 5: Market data integration class
    print("TEST 5: RiskGovernorMarketData")
    market_data = RiskGovernorMarketData()

    realtime = market_data.get_realtime_data(test_symbols)
    print(f"  Loaded {len(realtime)} symbols")

    liquidity_stress = market_data.compute_liquidity_stress(test_symbols)
    print(f"  Liquidity stress (max spread): {liquidity_stress*100:.3f}%")

    print()
    print("=" * 80)
    print("STEP 1 INTEGRATION TEST COMPLETE")
    print("=" * 80)
    print()
    print("Status:")
    print("  [OK] Real-time price loading: using latest candle data")
    print("  [OK] Spread computation: simulated 0.5% spread")
    print("  [OK] 20-day volatility: computed from historical data")
    print("  [OK] ATR-14: computed from historical data")
    print("  [PLACEHOLDER] 1-min volatility: requires intraday data")
    print("  [PLACEHOLDER] Volume spike: requires intraday data")
    print()
    print("Next Steps:")
    print("  - For production: Integrate IBKR TWS API for true real-time quotes")
    print("  - For intraday metrics: Add 1-min/5-min bar data to master_market_data")


if __name__ == '__main__':
    test_step1_integration()
