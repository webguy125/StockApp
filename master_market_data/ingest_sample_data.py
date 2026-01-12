"""
Quick test ingestion for Master Market Data DB
Ingests a small sample of symbols to verify the system works
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

from master_market_data.ingest_market_data import MarketDataIngestion

print("=" * 80)
print("MASTER MARKET DATA DB - SAMPLE INGESTION")
print("=" * 80)

# Test with a few symbols
test_symbols = ['AAPL', 'MSFT', 'GOOGL', 'NVDA', 'TSLA']

print(f"\nIngesting {len(test_symbols)} symbols for testing...")
print(f"Symbols: {', '.join(test_symbols)}\n")

ingestion = MarketDataIngestion()
results = ingestion.ingest_multiple_symbols(test_symbols, period='2y', timeframe='1d')

print("\n[OK] Sample ingestion complete!")
print(f"Database ready for testing with {results['total_candles']:,} candles")
