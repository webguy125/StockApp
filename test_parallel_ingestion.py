"""
Test Parallel EODHD Ingestion

Tests the parallel ingestion system with 3 symbols
"""
import sys
sys.path.insert(0, 'C:\\StockApp\\backend')

from backend.turbomode.Options.Ingestion.ingest_historical_options_parallel import run_parallel_ingestion

if __name__ == '__main__':
    print("\n" + "="*80)
    print("TESTING PARALLEL EODHD INGESTION")
    print("="*80)
    print("\nThis will:")
    print("  - Test with 3 symbols (A, AA, AAPL)")
    print("  - Use 4 parallel workers")
    print("  - Download last 90 days of options data")
    print("  - Use date-driven batched approach")
    print("  - Store in options_universe.db with WAL mode")
    print("\n" + "="*80 + "\n")

    # Run with test parameters
    run_parallel_ingestion(
        lookback_days=90,    # 3 months
        symbols_limit=3,     # First 3 symbols (A, AA, AAPL in sorted order)
        num_workers=4        # 4 parallel workers
    )
