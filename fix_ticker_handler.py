#!/usr/bin/env python
"""
Fix all timeframe files to ignore ticker updates for stock symbols
"""

import os
import re
import glob

def fix_ticker_handler(filepath):
    """Fix the handleTickerUpdate function in a timeframe file"""

    # Get the timeframe label from the filename
    filename = os.path.basename(filepath)
    match = re.match(r'(\d+[mhwd]|1mo|3mo|1range)', filename.replace('.js', ''))
    if not match:
        print(f"  ⚠️  Could not extract timeframe label from {filename}")
        return False

    label = match.group(1).upper()
    if label == '1MO':
        label = '1MO'
    elif label == '3MO':
        label = '3MO'
    elif label == '1RANGE':
        label = '1RANGE'
    elif label.endswith('M'):
        label = label  # Keep as is (1M, 5M, 15M, etc.)
    elif label.endswith('H'):
        label = label  # Keep as is (1H, 2H, etc.)
    elif label.endswith('D'):
        label = label  # Keep as is (1D)
    elif label.endswith('W'):
        label = label  # Keep as is (1W)

    with open(filepath, 'r', encoding='utf-8') as f:
        content = f.read()

    # Check if already fixed
    if 'Ignore ticker updates for stock symbols' in content:
        print(f"  ✅ {filename} already fixed")
        return True

    # Check if file has handleTickerUpdate function
    if 'handleTickerUpdate(data)' not in content:
        print(f"  ⏭️  {filename} doesn't have handleTickerUpdate")
        return True

    # Pattern to match the beginning of handleTickerUpdate function
    pattern = r'(handleTickerUpdate\(data\) \{\s*\n)(.*?)(\n\s*// Check if this ticker is for our symbol)'

    replacement = r'\1\2    // Ignore ticker updates for stock symbols (they don\'t have real-time data from Coinbase)\n' + \
                  r'    const cryptoSymbols = [\'BTC\', \'ETH\', \'SOL\', \'XRP\', \'DOGE\', \'ADA\', \'AVAX\', \'DOT\', \'LINK\', \'LTC\'];\n' + \
                  r'    const isCrypto = cryptoSymbols.includes(this.symbol) || this.symbol.endsWith(\'-USD\');\n\n' + \
                  r'    if (!isCrypto) {\n' + \
                  f'      console.log(`📊 [{label}] Ignoring ticker update for stock symbol: ${{this.symbol}}`);\n' + \
                  r'      return;\n' + \
                  r'    }\n\3'

    # Apply the fix
    new_content = re.sub(pattern, replacement, content, flags=re.DOTALL)

    if new_content == content:
        print(f"  ⚠️  {filename} - pattern not matched, trying alternative approach")
        # Try a simpler pattern
        simple_pattern = r'(handleTickerUpdate\(data\) \{[^\n]*\n)'
        simple_replacement = r'\1' + \
                           r'    // Ignore ticker updates for stock symbols (they don\'t have real-time data from Coinbase)\n' + \
                           r'    const cryptoSymbols = [\'BTC\', \'ETH\', \'SOL\', \'XRP\', \'DOGE\', \'ADA\', \'AVAX\', \'DOT\', \'LINK\', \'LTC\'];\n' + \
                           r'    const isCrypto = cryptoSymbols.includes(this.symbol) || this.symbol.endsWith(\'-USD\');\n\n' + \
                           r'    if (!isCrypto) {\n' + \
                           f'      console.log(`📊 [{label}] Ignoring ticker update for stock symbol: ${{this.symbol}}`);\n' + \
                           r'      return;\n' + \
                           r'    }\n\n'

        new_content = re.sub(simple_pattern, simple_replacement, content)

        if new_content == content:
            print(f"  ❌ {filename} - could not apply fix")
            return False

    # Write the fixed content back
    with open(filepath, 'w', encoding='utf-8') as f:
        f.write(new_content)

    print(f"  ✅ {filename} fixed successfully")
    return True

def main():
    print("🔧 Fixing handleTickerUpdate in all timeframe files...")
    print()

    # Find all timeframe files
    patterns = [
        'frontend/js/timeframes/days/*.js',
        'frontend/js/timeframes/hours/*.js',
        'frontend/js/timeframes/minutes/*.js',
        'frontend/js/timeframes/ranges/*.js'
    ]

    total_files = 0
    fixed_files = 0

    for pattern in patterns:
        files = glob.glob(pattern)
        for filepath in files:
            # Skip backup files and BaseTimeframe
            if '.backup' in filepath or 'BaseTimeframe' in filepath:
                continue

            print(f"Processing {filepath}...")
            total_files += 1
            if fix_ticker_handler(filepath):
                fixed_files += 1
            print()

    print(f"✅ Fixed {fixed_files}/{total_files} files")

if __name__ == '__main__':
    main()