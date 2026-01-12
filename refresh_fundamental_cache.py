"""
Refresh Fundamental Cache for All 80 Stocks

Run this once per day (or before retraining models) to populate the cache
with fresh fundamental data. Subsequent operations will use the cache.

Runtime: ~1.8 minutes for 80 stocks
"""

import sys
import os

backend_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "backend")
if backend_path not in sys.path:
    sys.path.insert(0, backend_path)

from advanced_ml.config.core_symbols import get_all_core_symbols
from advanced_ml.features.fundamental_cache import FundamentalCache
from datetime import datetime


def main():
    print("=" * 70)
    print("FUNDAMENTAL CACHE REFRESH")
    print("=" * 70)
    print(f"Started: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print()

    # Get all 80 curated symbols
    symbols = get_all_core_symbols()
    print(f"Refreshing fundamentals for {len(symbols)} stocks...")
    print(f"Estimated time: ~{len(symbols) * 1.3 / 60:.1f} minutes")
    print()

    # Initialize cache
    cache = FundamentalCache()

    # Show current cache stats
    stats = cache.get_cache_stats()
    print(f"Cache before refresh:")
    print(f"  Total symbols: {stats['total_symbols']}")
    print(f"  Fresh: {stats['fresh']}")
    print(f"  Expired: {stats['expired']}")
    print()

    # Bulk refresh
    cache.bulk_refresh(symbols)

    # Show updated stats
    stats = cache.get_cache_stats()
    print()
    print(f"Cache after refresh:")
    print(f"  Total symbols: {stats['total_symbols']}")
    print(f"  Fresh: {stats['fresh']}")
    print(f"  Expired: {stats['expired']}")
    print()

    print("=" * 70)
    print(f"Completed: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("=" * 70)
    print()
    print("Next steps:")
    print("  1. Cache is now populated for 24 hours")
    print("  2. Run backtest training: python backend/advanced_ml/train_with_fundamentals.py")
    print("  3. All feature extractions will use cached data (fast!)")
    print()


if __name__ == '__main__':
    main()
