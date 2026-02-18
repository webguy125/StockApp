"""
Test EODHD Historical Options Ingestion

Tests the ingestion system with a small dataset:
- 1 symbol (AAPL)
- 7 days of historical data
"""
import sys
sys.path.insert(0, 'C:\\StockApp\\backend')

from backend.turbomode.Options.Ingestion.ingest_historical_options import run_ingestion

print("\n" + "="*80)
print("TESTING EODHD INGESTION")
print("="*80)
print("\nThis will:")
print("  - Test with 1 symbol (AAPL)")
print("  - Download last 7 days of options data")
print("  - Store in options_universe.db")
print("\n" + "="*80 + "\n")

# Run with test parameters - using HISTORICAL data (30-60 days ago)
# This avoids recent dates that may not be available yet
run_ingestion(
    lookback_days=30,    # 30 days of HISTORICAL data (starting 30 days ago)
    symbols_limit=1      # Just 1 symbol (AAPL)
)
