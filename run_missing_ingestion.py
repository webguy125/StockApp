"""
Ingest the 63 missing training symbols into master_market_data DB
Uses the MarketDataIngestion class to fetch 10 years of daily data
"""
import sys
import os

# Add master_market_data to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'master_market_data'))

from ingest_market_data import MarketDataIngestion

print("=" * 80)
print("INGEST MISSING TRAINING SYMBOLS")
print("=" * 80)
print()

# Read missing symbols from file
symbols_file = 'missing_symbols_to_ingest.txt'
if not os.path.exists(symbols_file):
    print(f"ERROR: {symbols_file} not found!")
    print("Run 'python ingest_missing_training_symbols.py' first to generate the file.")
    sys.exit(1)

with open(symbols_file, 'r') as f:
    missing_symbols = [line.strip() for line in f if line.strip()]

print(f"Symbols to ingest: {len(missing_symbols)}")
print(f"Period: 10 years")
print(f"Timeframe: 1 day")
print()
print("Estimated time: 15-20 minutes (63 symbols × ~15 seconds each)")
print()
print("Missing symbols:")
for i, symbol in enumerate(missing_symbols, 1):
    print(f"  {i:2d}. {symbol}")
print()
print("=" * 80)
print()

response = input("Proceed with ingestion? (yes/no): ")
if response.lower() not in ['yes', 'y']:
    print("Cancelled.")
    sys.exit(0)

print()
print("Starting ingestion...")
print()

# Initialize ingestion
ingestion = MarketDataIngestion()

# Ingest all missing symbols
results = ingestion.ingest_multiple_symbols(
    symbols=missing_symbols,
    period='10y',
    timeframe='1d'
)

print()
print("=" * 80)
print("INGESTION COMPLETE!")
print("=" * 80)
print(f"Successful: {results['successful']}/{results['total_symbols']}")
print(f"Failed:     {results['failed']}/{results['total_symbols']}")
print(f"Total candles ingested: {results['total_candles']:,}")
print()

if results['failed'] > 0:
    print("WARNING: Some symbols failed to ingest.")
    print("Check the logs above for details.")
else:
    print("SUCCESS: All symbols ingested successfully!")
    print()
    print("Next step: Regenerate training data")
    print("  python backend/turbomode/generate_backtest_data.py")
