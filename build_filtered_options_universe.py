"""
Step 17B: Build filtered options universe
Created per task specification version 3.0.2
"""
import json

# File paths
CORE_230_PATH = 'C:/StockApp/config/symbols/CORE_230.json'
VALIDATION_REPORT_PATH = 'C:/StockApp/backend/turbomode/Options/Reports/historical_validation_report.json'
OUTPUT_PATH = 'C:/StockApp/backend/turbomode/Options/options_universe_filtered.json'

# Symbols to exclude (from validation report)
SYMBOLS_TO_EXCLUDE = [
    'BTC-USD',
    'ETH-USD',
    'SOL-USD',
    'BTAI',
    'IRBT',
    'OMI',
    'VTLE'
]


def build_filtered_universe():
    """
    Create filtered options universe excluding symbols that failed validation
    """
    print('=' * 80)
    print('STEP 17B: BUILD FILTERED OPTIONS UNIVERSE')
    print('=' * 80)

    # Read core_230.json
    print('[READ] Loading CORE_230.json...')
    with open(CORE_230_PATH, 'r') as f:
        core_230 = json.load(f)

    print(f'[READ] Loaded {len(core_230)} symbols from CORE_230.json')

    # Read validation report
    print('[READ] Loading historical_validation_report.json...')
    with open(VALIDATION_REPORT_PATH, 'r') as f:
        validation_report = json.load(f)

    # Verify symbols_missing from report match our exclusion list
    symbols_missing_from_report = set(validation_report['symbols_missing'])
    symbols_to_exclude_set = set(SYMBOLS_TO_EXCLUDE)

    print(f'[VERIFY] Symbols missing from report: {sorted(symbols_missing_from_report)}')
    print(f'[VERIFY] Symbols to exclude: {sorted(SYMBOLS_TO_EXCLUDE)}')

    if symbols_missing_from_report != symbols_to_exclude_set:
        print('[WARNING] Mismatch between validation report and exclusion list!')
        print(f'[WARNING] Using exclusion list as specified: {sorted(SYMBOLS_TO_EXCLUDE)}')

    # Filter symbols
    print('[FILTER] Creating filtered universe...')
    filtered_universe = []

    for symbol_entry in core_230:
        ticker = symbol_entry['ticker']

        if ticker not in SYMBOLS_TO_EXCLUDE:
            filtered_universe.append(symbol_entry)
        else:
            print(f'[FILTER] Excluding: {ticker}')

    print(f'[FILTER] Filtered universe size: {len(filtered_universe)} symbols')
    print(f'[FILTER] Excluded: {len(core_230) - len(filtered_universe)} symbols')

    # Write filtered universe
    print('[WRITE] Writing options_universe_filtered.json...')
    with open(OUTPUT_PATH, 'w') as f:
        json.dump(filtered_universe, f, indent=2)

    print(f'[WRITE] Written to: {OUTPUT_PATH}')

    # Verify output
    print('[VERIFY] Verifying output file...')
    with open(OUTPUT_PATH, 'r') as f:
        verification = json.load(f)

    print(f'[VERIFY] Output file contains {len(verification)} symbols')
    print(f'[VERIFY] Output file is valid JSON: Yes')

    # Summary
    print('=' * 80)
    print('STEP 17B: COMPLETE')
    print('=' * 80)
    print(f'Input symbols (CORE_230): {len(core_230)}')
    print(f'Excluded symbols: {len(SYMBOLS_TO_EXCLUDE)}')
    print(f'Output symbols (filtered): {len(filtered_universe)}')
    print('')
    print('Excluded symbols:')
    for symbol in sorted(SYMBOLS_TO_EXCLUDE):
        print(f'  - {symbol}')
    print('=' * 80)


if __name__ == '__main__':
    build_filtered_universe()
