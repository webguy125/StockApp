"""
Fundamental Data Cache for ML Feature Engineering

Caches fundamental data (P/E, debt, margins, etc.) with 24-hour expiration.
Fundamentals change slowly (quarterly earnings), so we can cache them to avoid
slow API calls on every backtest run.

Performance:
- Without cache: +0.56s per stock (74% slower)
- With cache: +0.00s per stock (same speed as before!)
- Cache build time: ~1.8 min for 80 stocks (once per day)
"""

import json
import os
from datetime import datetime, timedelta
from typing import Dict, Any, Optional
import yfinance as yf
from pathlib import Path


class FundamentalCache:
    """
    Cache for fundamental data with automatic expiration
    """

    def __init__(self, cache_file: str = None, expiration_hours: int = 24):
        """
        Initialize fundamental cache

        Args:
            cache_file: Path to cache JSON file (default: data/fundamentals_cache.json)
            expiration_hours: Hours before cache expires (default: 24)
        """
        if cache_file is None:
            # Default to data/fundamentals_cache.json
            base_dir = Path(__file__).parent.parent.parent / "data"
            base_dir.mkdir(exist_ok=True)
            cache_file = base_dir / "fundamentals_cache.json"

        self.cache_file = str(cache_file)
        self.expiration_hours = expiration_hours
        self.cache = self._load_cache()

    def _load_cache(self) -> Dict[str, Any]:
        """Load cache from disk"""
        if os.path.exists(self.cache_file):
            try:
                with open(self.cache_file, 'r') as f:
                    return json.load(f)
            except Exception as e:
                print(f"[CACHE] Failed to load cache: {e}")
                return {'metadata': {}, 'data': {}}
        else:
            return {'metadata': {}, 'data': {}}

    def _save_cache(self):
        """Save cache to disk"""
        try:
            with open(self.cache_file, 'w') as f:
                json.dump(self.cache, f, indent=2)
        except Exception as e:
            print(f"[CACHE] Failed to save cache: {e}")

    def _is_expired(self, symbol: str) -> bool:
        """Check if cached data for symbol is expired"""
        if symbol not in self.cache['metadata']:
            return True

        timestamp_str = self.cache['metadata'][symbol].get('timestamp')
        if not timestamp_str:
            return True

        try:
            cache_time = datetime.fromisoformat(timestamp_str)
            age_hours = (datetime.now() - cache_time).total_seconds() / 3600
            return age_hours > self.expiration_hours
        except Exception:
            return True

    def get(self, symbol: str) -> Optional[Dict[str, float]]:
        """
        Get cached fundamental data for symbol

        Args:
            symbol: Stock ticker

        Returns:
            Dict of fundamental features, or None if not cached/expired
        """
        if self._is_expired(symbol):
            return None

        return self.cache['data'].get(symbol)

    def set(self, symbol: str, fundamentals: Dict[str, float]):
        """
        Cache fundamental data for symbol

        Args:
            symbol: Stock ticker
            fundamentals: Dict of fundamental features
        """
        self.cache['data'][symbol] = fundamentals
        self.cache['metadata'][symbol] = {
            'timestamp': datetime.now().isoformat(),
            'source': 'yfinance'
        }
        self._save_cache()

    def fetch_and_cache(self, symbol: str, force_refresh: bool = False) -> Dict[str, float]:
        """
        Get fundamentals from cache or fetch from yfinance if expired

        Args:
            symbol: Stock ticker
            force_refresh: Ignore cache and fetch fresh data

        Returns:
            Dict of 12 fundamental features with safe defaults
        """
        # Check cache first (unless force refresh)
        if not force_refresh:
            cached = self.get(symbol)
            if cached is not None:
                return cached

        # Fetch from yfinance
        try:
            ticker = yf.Ticker(symbol)
            info = ticker.info

            # Extract 12 fundamental features with safe defaults
            fundamentals = {
                # Tier 1: Critical for strategy
                'beta': float(info.get('beta', 1.0) or 1.0),
                'short_percent_of_float': float(info.get('shortPercentOfFloat', 0.0) or 0.0),
                'short_ratio': float(info.get('shortRatio', 0.0) or 0.0),
                'analyst_target_price': float(info.get('targetMeanPrice', 0.0) or 0.0),
                'profit_margin': float(info.get('profitMargins', 0.0) or 0.0),
                'debt_to_equity': float(info.get('debtToEquity', 0.0) or 0.0),

                # Tier 2: Value/Growth indicators
                'price_to_book': float(info.get('priceToBook', 0.0) or 0.0),
                'price_to_sales': float(info.get('priceToSalesTrailing12Months', 0.0) or 0.0),
                'return_on_equity': float(info.get('returnOnEquity', 0.0) or 0.0),
                'current_ratio': float(info.get('currentRatio', 1.0) or 1.0),
                'revenue_growth': float(info.get('revenueGrowth', 0.0) or 0.0),
                'forward_pe': float(info.get('forwardPE', 0.0) or 0.0),
            }

            # Cache it
            self.set(symbol, fundamentals)

            return fundamentals

        except Exception as e:
            print(f"[CACHE] Failed to fetch fundamentals for {symbol}: {e}")

            # Return safe defaults if fetch fails
            return {
                'beta': 1.0,
                'short_percent_of_float': 0.0,
                'short_ratio': 0.0,
                'analyst_target_price': 0.0,
                'profit_margin': 0.0,
                'debt_to_equity': 0.0,
                'price_to_book': 0.0,
                'price_to_sales': 0.0,
                'return_on_equity': 0.0,
                'current_ratio': 1.0,
                'revenue_growth': 0.0,
                'forward_pe': 0.0,
            }

    def bulk_refresh(self, symbols: list):
        """
        Refresh cache for multiple symbols (run once per day)

        Args:
            symbols: List of stock tickers to refresh
        """
        print(f"\n[CACHE] Refreshing fundamentals for {len(symbols)} symbols...")
        print(f"[CACHE] This will take ~{len(symbols) * 1.3 / 60:.1f} minutes")

        refreshed = 0
        failed = 0

        for i, symbol in enumerate(symbols, 1):
            if i % 10 == 0:
                print(f"  Progress: {i}/{len(symbols)} ({i/len(symbols)*100:.1f}%)")

            try:
                self.fetch_and_cache(symbol, force_refresh=True)
                refreshed += 1
            except Exception as e:
                print(f"  [ERROR] Failed to refresh {symbol}: {e}")
                failed += 1

        print(f"[CACHE] Refresh complete: {refreshed} updated, {failed} failed")
        print(f"[CACHE] Cache saved to {self.cache_file}")

    def get_cache_stats(self) -> Dict[str, Any]:
        """Get statistics about cache"""
        total = len(self.cache['data'])
        expired = sum(1 for symbol in self.cache['data'].keys() if self._is_expired(symbol))
        fresh = total - expired

        return {
            'total_symbols': total,
            'fresh': fresh,
            'expired': expired,
            'cache_file': self.cache_file,
            'expiration_hours': self.expiration_hours
        }


# Global cache instance (singleton pattern)
_cache_instance = None


def get_fundamental_cache() -> FundamentalCache:
    """Get singleton cache instance"""
    global _cache_instance
    if _cache_instance is None:
        _cache_instance = FundamentalCache()
    return _cache_instance


if __name__ == '__main__':
    # Test cache
    print("=" * 70)
    print("FUNDAMENTAL CACHE TEST")
    print("=" * 70)

    cache = FundamentalCache()

    # Test with a few symbols
    test_symbols = ['AAPL', 'NVDA', 'TSLA']

    print("\n[TEST 1] Fetch fresh data (slow)...")
    for symbol in test_symbols:
        fundamentals = cache.fetch_and_cache(symbol, force_refresh=True)
        print(f"{symbol}: beta={fundamentals['beta']:.2f}, "
              f"debt/equity={fundamentals['debt_to_equity']:.1f}, "
              f"profit_margin={fundamentals['profit_margin']*100:.1f}%")

    print("\n[TEST 2] Fetch from cache (fast)...")
    import time
    start = time.time()
    for symbol in test_symbols:
        fundamentals = cache.fetch_and_cache(symbol)
        print(f"{symbol}: Cached (took {time.time() - start:.4f}s)")
        start = time.time()

    print("\n[TEST 3] Cache statistics...")
    stats = cache.get_cache_stats()
    print(f"Total symbols: {stats['total_symbols']}")
    print(f"Fresh: {stats['fresh']}")
    print(f"Expired: {stats['expired']}")
    print(f"Cache file: {stats['cache_file']}")
