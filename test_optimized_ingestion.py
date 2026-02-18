"""
Test Optimized EODHD Ingestion

Tests the optimized ingestion system with 3 symbols
"""
import sys
sys.path.insert(0, 'C:\\StockApp\\backend')

from backend.turbomode.Options.Ingestion.ingest_historical_options_optimized import run_optimized_ingestion

if __name__ == '__main__':
    print("\n" + "="*80)
    print("TESTING OPTIMIZED EODHD INGESTION")
    print("="*80)
    print("\nThis will:")
    print("  - Test with 3 symbols (A, AA, AAPL)")
    print("  - Use 4 parallel workers")
    print("  - Download last 90 days of options data")
    print("  - Use expiration-driven batched approach")
    print("  - Store in options_universe.db with WAL mode")
    print("\n" + "="*80 + "\n")

    # Run with test parameters
    run_optimized_ingestion(
        lookback_days=90,    # 3 months
        symbols_limit=3,     # First 3 symbols (A, AA, AAPL in sorted order)
        num_workers=4        # 4 parallel workers
    )
