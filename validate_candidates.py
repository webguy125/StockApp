"""
Validate curated mid/small cap candidates
Check: market cap tier, options availability, liquidity
"""
import yfinance as yf
import json
import time
from curated_midsmall_caps import CURATED_STOCKS

def classify_market_cap(market_cap):
    """Classify market cap into tiers"""
    if market_cap is None or market_cap == 0:
        return ('unknown', 3)

    cap_billions = market_cap / 1_000_000_000

    if cap_billions < 2:
        return ('small_cap', 0)
    elif cap_billions < 10:
        return ('mid_cap', 1)
    else:
        return ('large_cap', 2)


def validate_stock(symbol):
    """Validate a single stock"""
    try:
        ticker = yf.Ticker(symbol)
        info = ticker.info

        # Get market cap
        market_cap = info.get('marketCap', None)
        tier_name, tier_code = classify_market_cap(market_cap)
        cap_billions = round(market_cap / 1_000_000_000, 2) if market_cap else None

        # Check average volume
        avg_volume = info.get('averageVolume', 0)

        # Try to get options data (if this succeeds, options exist)
        has_options = False
        try:
            options_dates = ticker.options
            has_options = len(options_dates) > 0
        except:
            has_options = False

        return {
            'symbol': symbol,
            'market_cap': market_cap,
            'market_cap_billions': cap_billions,
            'tier_name': tier_name,
            'tier_code': tier_code,
            'avg_volume': avg_volume,
            'has_options': has_options,
            'valid': market_cap is not None and avg_volume > 500000,
            'error': None
        }

    except Exception as e:
        return {
            'symbol': symbol,
            'market_cap': None,
            'market_cap_billions': None,
            'tier_name': 'unknown',
            'tier_code': 3,
            'avg_volume': 0,
            'has_options': False,
            'valid': False,
            'error': str(e)
        }


def validate_all_candidates():
    """Validate all curated candidates"""
    print("=" * 80)
    print("VALIDATING CURATED CANDIDATES")
    print("=" * 80)
    print()
    print("This will take 5-10 minutes to fetch data for ~150 stocks...")
    print()

    validated_results = {}
    total_count = 0

    # Count total stocks
    for sector in CURATED_STOCKS:
        total_count += len(CURATED_STOCKS[sector]['mid_cap'])
        total_count += len(CURATED_STOCKS[sector]['small_cap'])

    progress = 0

    for sector in sorted(CURATED_STOCKS.keys()):
        print(f"\nValidating {sector}...")
        validated_results[sector] = {
            'mid_cap': [],
            'small_cap': []
        }

        # Validate mid caps
        for symbol in CURATED_STOCKS[sector]['mid_cap']:
            result = validate_stock(symbol)
            validated_results[sector]['mid_cap'].append(result)
            progress += 1

            if progress % 10 == 0:
                print(f"  Progress: {progress}/{total_count} ({progress/total_count*100:.1f}%)")

            time.sleep(0.5)  # Rate limiting

        # Validate small caps
        for symbol in CURATED_STOCKS[sector]['small_cap']:
            result = validate_stock(symbol)
            validated_results[sector]['small_cap'].append(result)
            progress += 1

            if progress % 10 == 0:
                print(f"  Progress: {progress}/{total_count} ({progress/total_count*100:.1f}%)")

            time.sleep(0.5)  # Rate limiting

    # Save results
    with open('validated_candidates.json', 'w') as f:
        json.dump(validated_results, f, indent=2)

    print()
    print("=" * 80)
    print("VALIDATION COMPLETE")
    print("=" * 80)
    print()

    # Analyze results
    analyze_validation_results(validated_results)

    return validated_results


def analyze_validation_results(results):
    """Analyze validation results"""
    print("\nVALIDATION SUMMARY BY SECTOR:")
    print("=" * 80)

    total_valid = 0
    total_invalid = 0
    total_with_options = 0
    total_without_options = 0

    by_sector = {}

    for sector in sorted(results.keys()):
        mid_valid = sum(1 for s in results[sector]['mid_cap'] if s['valid'])
        mid_options = sum(1 for s in results[sector]['mid_cap'] if s['has_options'])
        mid_total = len(results[sector]['mid_cap'])

        small_valid = sum(1 for s in results[sector]['small_cap'] if s['valid'])
        small_options = sum(1 for s in results[sector]['small_cap'] if s['has_options'])
        small_total = len(results[sector]['small_cap'])

        total_valid += mid_valid + small_valid
        total_invalid += (mid_total - mid_valid) + (small_total - small_valid)
        total_with_options += mid_options + small_options
        total_without_options += (mid_total - mid_options) + (small_total - small_options)

        by_sector[sector] = {
            'mid_valid': mid_valid,
            'mid_options': mid_options,
            'small_valid': small_valid,
            'small_options': small_options
        }

        print(f"{sector:30s}")
        print(f"  Mid:   {mid_valid}/{mid_total} valid, {mid_options}/{mid_total} with options")
        print(f"  Small: {small_valid}/{small_total} valid, {small_options}/{small_total} with options")

    print()
    print(f"Overall:")
    print(f"  Valid stocks: {total_valid}")
    print(f"  Invalid stocks: {total_invalid}")
    print(f"  With options: {total_with_options}")
    print(f"  Without options: {total_without_options}")
    print()

    # Show stocks without options
    print("\nSTOCKS WITHOUT OPTIONS (need to exclude):")
    print("=" * 80)
    for sector in sorted(results.keys()):
        no_options = [s['symbol'] for s in results[sector]['mid_cap'] + results[sector]['small_cap']
                     if not s['has_options']]
        if no_options:
            print(f"{sector}: {', '.join(no_options)}")
    print()

    # Show tier mismatches (expected mid but is large, etc)
    print("\nTIER MISMATCHES:")
    print("=" * 80)
    for sector in sorted(results.keys()):
        for expected, stocks in [('mid_cap', results[sector]['mid_cap']),
                                  ('small_cap', results[sector]['small_cap'])]:
            for s in stocks:
                if s['valid'] and s['tier_name'] != expected:
                    print(f"{s['symbol']:6s} Expected {expected:10s}, Got {s['tier_name']:10s} (${s['market_cap_billions']}B)")
    print()

    print("Results saved to: validated_candidates.json")


if __name__ == '__main__':
    validate_all_candidates()
