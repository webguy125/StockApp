"""
TOP 10 INTRADAY SCANNER
=======================
Fast scanner optimized for the Top 10 stocks from adaptive_stock_ranker.py

Purpose:
- Scans only Top 10 stocks (vs 82 for overnight scanner)
- Designed for multiple intraday runs (market open, mid-day, pre-close)
- Updates all_predictions.json with fresh signals
- Typically completes in ~5-8 minutes (vs ~60 minutes for full scan)

Usage:
  python top10_scanner.py                    # Auto-loads Top 10 from stock_rankings.json
  python top10_scanner.py --symbols AAPL,TSLA,NVDA  # Override with custom symbols
  python top10_scanner.py --force            # Force rescan even if active signals exist

Output:
  - Updates backend/data/all_predictions.json with fresh predictions
  - Replaces predictions for Top 10 stocks only (preserves other 72 stocks)
  - Options page reads from all_predictions.json
"""

import os
import sys
import json
import time
import argparse
from datetime import datetime
from pathlib import Path

# Add backend to path for imports
sys.path.append(str(Path(__file__).parent.parent))

# Import the overnight scanner class (reuse all its logic)
from turbomode.overnight_scanner import OvernightScanner


def load_top_10_from_rankings(rankings_file: str = None) -> list:
    """
    Load Top 10 symbols from stock_rankings.json

    Args:
        rankings_file: Path to stock_rankings.json (default: backend/data/stock_rankings.json)

    Returns:
        List of Top 10 stock symbols
    """
    if rankings_file is None:
        # Default path relative to this script
        rankings_file = os.path.join(
            os.path.dirname(__file__),
            '../data/stock_rankings.json'
        )

    try:
        with open(rankings_file, 'r') as f:
            data = json.load(f)

        top_10 = data.get('top_10', [])
        symbols = [stock['symbol'] for stock in top_10]

        if not symbols:
            raise ValueError("No Top 10 stocks found in stock_rankings.json")

        print(f"[TOP10] Loaded {len(symbols)} symbols from rankings: {', '.join(symbols)}")
        return symbols

    except FileNotFoundError:
        print(f"[ERROR] Rankings file not found: {rankings_file}")
        print("[ERROR] Run adaptive_stock_ranker.py first to generate stock_rankings.json")
        sys.exit(1)
    except Exception as e:
        print(f"[ERROR] Failed to load Top 10: {e}")
        sys.exit(1)


def merge_top10_predictions(scanner, top10_symbols: list) -> dict:
    """
    Generate predictions for Top 10 and merge into existing all_predictions.json

    Args:
        scanner: OvernightScanner instance
        top10_symbols: List of Top 10 symbols to scan

    Returns:
        Complete predictions dict with merged data
    """
    output_path = os.path.join(
        os.path.dirname(__file__),
        '../data/all_predictions.json'
    )

    # Load existing predictions
    existing_data = {"predictions": [], "timestamp": None}
    if os.path.exists(output_path):
        try:
            with open(output_path, 'r') as f:
                existing_data = json.load(f)
            print(f"[MERGE] Loaded {len(existing_data.get('predictions', []))} existing predictions")
        except Exception as e:
            print(f"[WARN] Could not load existing predictions: {e}")

    # Generate fresh predictions for Top 10
    print(f"\n[SCAN] Generating predictions for {len(top10_symbols)} stocks...")
    new_predictions = []

    for i, symbol in enumerate(top10_symbols, 1):
        print(f"  [{i}/{len(top10_symbols)}] Scanning {symbol}...", end=" ")
        pred = scanner.get_prediction_for_symbol(symbol)

        if pred:
            new_predictions.append(pred)
            signal_str = f"{pred['prediction'].upper()} ({pred['confidence']:.1%})"
            print(f"{signal_str}")
        else:
            print("SKIPPED")

    # Create symbol set for Top 10
    top10_set = set(top10_symbols)

    # Keep all non-Top10 predictions from existing file
    merged_predictions = [
        pred for pred in existing_data.get('predictions', [])
        if pred['symbol'] not in top10_set
    ]

    # Add new Top 10 predictions
    merged_predictions.extend(new_predictions)

    # Sort by symbol for readability
    merged_predictions.sort(key=lambda x: x['symbol'])

    # Count signals
    buy_count = sum(1 for p in new_predictions if p['prediction'] == 'buy')
    sell_count = sum(1 for p in new_predictions if p['prediction'] == 'sell')
    hold_count = sum(1 for p in new_predictions if p['prediction'] == 'hold')

    # Create final data structure
    final_data = {
        "timestamp": datetime.now().isoformat(),
        "predictions": merged_predictions,
        "metadata": {
            "total_symbols": len(merged_predictions),
            "top10_updated": len(new_predictions),
            "top10_symbols": sorted(list(top10_set)),
            "scanner_type": "top10_intraday",
            "signals": {
                "buy_count": buy_count,
                "sell_count": sell_count,
                "hold_count": hold_count
            }
        }
    }

    print(f"\n[MERGE] Combined {len(merged_predictions)} total predictions")
    print(f"[MERGE] Updated {len(new_predictions)} Top 10 stocks (BUY: {buy_count}, SELL: {sell_count}, HOLD: {hold_count})")
    print(f"[MERGE] Preserved {len(merged_predictions) - len(new_predictions)} other stocks")

    return final_data


def run_top10_scanner(symbols: list = None):
    """
    Run scanner on Top 10 stocks only

    Args:
        symbols: List of symbols to scan (default: load from stock_rankings.json)
    """
    start_time = time.time()

    print("\n" + "="*70)
    print("TOP 10 INTRADAY SCANNER")
    print("="*70)
    print(f"Start time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

    # Load symbols
    if symbols is None:
        symbols = load_top_10_from_rankings()
    else:
        print(f"[TOP10] Using provided symbols: {', '.join(symbols)}")

    print(f"[TOP10] Scanning {len(symbols)} stocks\n")

    # Initialize scanner (loads all ML models)
    print("[INIT] Initializing scanner...")
    scanner = OvernightScanner()

    # Generate and merge predictions
    final_data = merge_top10_predictions(scanner, symbols)

    # Save to file
    output_file = os.path.join(
        os.path.dirname(__file__),
        '../data/all_predictions.json'
    )

    with open(output_file, 'w') as f:
        json.dump(final_data, f, indent=2)

    print(f"\n[SAVE] Saved to {output_file}")

    # Print summary
    elapsed = time.time() - start_time
    minutes = int(elapsed // 60)
    seconds = int(elapsed % 60)

    print("\n" + "="*70)
    print("SCAN COMPLETE")
    print("="*70)
    print(f"Duration: {minutes}m {seconds}s")
    print(f"Stocks scanned: {len(symbols)}")
    print(f"Average time per stock: {elapsed/len(symbols):.1f}s")
    print(f"Output: {output_file}")
    print("="*70 + "\n")


def main():
    parser = argparse.ArgumentParser(
        description='Fast scanner for Top 10 stocks (intraday use)'
    )
    parser.add_argument(
        '--symbols',
        type=str,
        help='Comma-separated symbols to scan (default: auto-load Top 10 from rankings)'
    )

    args = parser.parse_args()

    # Parse symbols if provided
    symbols = None
    if args.symbols:
        symbols = [s.strip().upper() for s in args.symbols.split(',')]

    # Run scanner
    run_top10_scanner(symbols=symbols)


if __name__ == '__main__':
    main()
