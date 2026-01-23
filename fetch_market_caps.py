"""
Fetch market cap data for all scanning symbols
Classifies into small/mid/large cap tiers
"""
import yfinance as yf
import json
from backend.turbomode.scanning_symbols import get_scanning_symbols

def classify_market_cap(market_cap):
    """
    Classify market cap into tiers

    Args:
        market_cap: Market cap in dollars

    Returns:
        Tuple of (tier_name, tier_code)
    """
    if market_cap is None or market_cap == 0:
        return ('unknown', 3)

    # Convert to billions
    cap_billions = market_cap / 1_000_000_000

    if cap_billions < 2:
        return ('small_cap', 0)
    elif cap_billions < 10:
        return ('mid_cap', 1)
    else:
        return ('large_cap', 2)


def fetch_all_market_caps():
    """Fetch market caps for all scanning symbols"""
    symbols = get_scanning_symbols()

    print(f"Fetching market caps for {len(symbols)} symbols...")
    print("This will take 2-3 minutes...")
    print()

    results = {}
    failed = []

    for i, symbol in enumerate(symbols, 1):
        try:
            ticker = yf.Ticker(symbol)
            info = ticker.info

            # Get market cap
            market_cap = info.get('marketCap', None)

            # Classify
            tier_name, tier_code = classify_market_cap(market_cap)

            results[symbol] = {
                'market_cap': market_cap,
                'market_cap_billions': round(market_cap / 1_000_000_000, 2) if market_cap else None,
                'tier_name': tier_name,
                'tier_code': tier_code
            }

            if i % 10 == 0:
                print(f"Progress: {i}/{len(symbols)} ({i/len(symbols)*100:.1f}%)")

        except Exception as e:
            print(f"  [ERROR] {symbol}: {e}")
            failed.append(symbol)
            results[symbol] = {
                'market_cap': None,
                'market_cap_billions': None,
                'tier_name': 'unknown',
                'tier_code': 3,
                'error': str(e)
            }

    # Save results
    output_file = 'market_caps_data.json'
    with open(output_file, 'w') as f:
        json.dump(results, f, indent=2)

    print()
    print(f"✓ Complete! Saved to {output_file}")
    print()

    # Summary
    small = sum(1 for r in results.values() if r['tier_code'] == 0)
    mid = sum(1 for r in results.values() if r['tier_code'] == 1)
    large = sum(1 for r in results.values() if r['tier_code'] == 2)
    unknown = sum(1 for r in results.values() if r['tier_code'] == 3)

    print("Summary:")
    print(f"  Small Cap (<$2B):     {small}")
    print(f"  Mid Cap ($2B-$10B):   {mid}")
    print(f"  Large Cap (>$10B):    {large}")
    print(f"  Unknown/Failed:       {unknown}")
    print()

    if failed:
        print(f"Failed symbols: {', '.join(failed)}")

    return results


if __name__ == '__main__':
    fetch_all_market_caps()
