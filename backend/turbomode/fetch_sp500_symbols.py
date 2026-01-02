"""
Fetch current S&P 500 symbols using yfinance
This validates each symbol by checking if data is available
"""

import yfinance as yf
import pandas as pd
from datetime import datetime, timedelta

def fetch_sp500_symbols_from_yfinance():
    """
    Fetch S&P 500 constituent symbols using yfinance
    Returns list of valid ticker symbols
    """
    print("\n" + "=" * 70)
    print("FETCHING S&P 500 SYMBOLS FROM YFINANCE")
    print("=" * 70)

    try:
        # Method 1: Use yfinance to get S&P 500 index constituents
        # The S&P 500 index symbol is ^GSPC
        sp500 = yf.Ticker("^GSPC")

        # Try to get holdings if available
        # Note: This might not work, so we'll use pandas to read from Wikipedia as fallback

        print("\n[STEP 1] Fetching S&P 500 list from Wikipedia (via pandas)...")
        table = pd.read_html('https://en.wikipedia.org/wiki/List_of_S%26P_500_companies')
        df = table[0]

        # Get symbols and convert from DataFrame to list
        all_symbols = df['Symbol'].str.replace('.', '-').tolist()
        print(f"[INFO] Found {len(all_symbols)} symbols from Wikipedia")

        # Validate each symbol using yfinance
        print(f"\n[STEP 2] Validating symbols with yfinance...")
        print(f"[INFO] Testing each symbol to ensure it has available data...")

        valid_symbols = []
        invalid_symbols = []

        for i, symbol in enumerate(all_symbols, 1):
            if i % 50 == 0:
                print(f"  Progress: {i}/{len(all_symbols)} ({i/len(all_symbols)*100:.1f}%)")

            try:
                # Try to fetch recent data to validate symbol
                ticker = yf.Ticker(symbol)
                hist = ticker.history(period="5d")

                if not hist.empty:
                    valid_symbols.append(symbol)
                else:
                    invalid_symbols.append(symbol)

            except Exception as e:
                invalid_symbols.append(symbol)

        print(f"\n[STEP 3] Validation complete!")
        print(f"  Valid symbols: {len(valid_symbols)}")
        print(f"  Invalid/delisted: {len(invalid_symbols)}")

        if invalid_symbols:
            print(f"\n[INFO] Delisted/invalid symbols found: {len(invalid_symbols)}")
            print(f"  Examples: {invalid_symbols[:10]}")

        return valid_symbols

    except Exception as e:
        print(f"\n[ERROR] Failed to fetch S&P 500 symbols: {e}")
        return []


if __name__ == "__main__":
    symbols = fetch_sp500_symbols_from_yfinance()

    print("\n" + "=" * 70)
    print("RESULTS")
    print("=" * 70)
    print(f"Total valid S&P 500 symbols: {len(symbols)}")
    print(f"\nFirst 20 symbols:")
    print(symbols[:20])
    print(f"\nLast 20 symbols:")
    print(symbols[-20:])

    # Save to file
    output_file = "sp500_valid_symbols.txt"
    with open(output_file, 'w') as f:
        for symbol in symbols:
            f.write(f"{symbol}\n")

    print(f"\n[SAVED] Valid symbols written to: {output_file}")
    print("=" * 70)
