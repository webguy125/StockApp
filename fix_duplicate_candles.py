"""
Script to apply duplicate candle detection fix to all timeframe files
Prevents flat candles by removing duplicates with the same timestamp
"""

import os
import re

# Define all timeframes that need the fix
TIMEFRAMES = {
    'minutes': ['10m', '15m', '30m', '45m'],  # Already did: 1m, 2m, 3m, 5m
    'hours': ['1h', '2h', '3h', '4h', '6h'],
    'days': ['1d', '1w', '1mo', '3mo']
}

def apply_fix_to_file(file_path, interval_id, interval_multiplier):
    """
    Apply the duplicate candle detection fix to a timeframe file

    Args:
        file_path: Path to the timeframe .js file
        interval_id: The interval ID (e.g., '10m', '1h', '1d')
        interval_multiplier: The multiplier for calculating candle time (e.g., '10 * 60000' for 10m)
    """
    print(f"\nProcessing {file_path}...")

    with open(file_path, 'r', encoding='utf-8') as f:
        content = f.read()

    # Check if fix is already applied
    if 'CRITICAL FIX: Check if last candle already has this timestamp' in content:
        print(f"   [OK] Already fixed - skipping")
        return

    # Build the interval label for logging (e.g., '10M', '1H', '1D')
    label = interval_id.upper().replace('M', 'M').replace('H', 'H').replace('D', 'D').replace('W', 'W').replace('MO', 'MO')

    # Find the pattern: after candleTime calculation, before newCandle creation
    pattern = rf"(const candleTime = new Date\(Math\.floor\(now\.getTime\(\) / \({re.escape(interval_multiplier)}\)\) \* \({re.escape(interval_multiplier)}\)\);)\s*\n(\s*)(// Create new candle|const newCandle)"

    # The fix to insert
    fix_code = f"""
          // CRITICAL FIX: Check if last candle already has this timestamp (duplicate detection)
          const lastCandleTime = new Date(lastCandle.Date.includes('Z') ? lastCandle.Date : lastCandle.Date + 'Z');

          if (lastCandleTime.getTime() === candleTime.getTime()) {{
            // Duplicate detected! Remove the flat candle and add the correct one
            console.log(`🗑️ [{label}] Removing duplicate flat candle at ${{candleTime.toLocaleTimeString()}}`);
            this.data.pop(); // Remove the flat candle
          }}

          """

    # Replace the pattern
    replacement = rf"\1\n{fix_code}\2\3"
    new_content = re.sub(pattern, replacement, content)

    if new_content == content:
        print(f"   [WARN] Pattern not found - manual fix needed")
        return False

    # Also update the log message from "adding to data array" to "checking data array"
    new_content = new_content.replace(
        f'console.log(`🕐 [{label}] New candle detected - adding to data array`);',
        f'console.log(`🕐 [{label}] New candle detected - checking data array`);'
    )

    # Write the updated content
    with open(file_path, 'w', encoding='utf-8') as f:
        f.write(new_content)

    print(f"   [FIXED] Successfully applied fix")
    return True

def get_interval_multiplier(interval_id):
    """Get the JavaScript expression for calculating candle time boundaries"""
    if interval_id.endswith('m'):
        minutes = interval_id[:-1]
        return f"{minutes} * 60000"
    elif interval_id.endswith('h'):
        hours = interval_id[:-1]
        return f"{hours} * 60 * 60000"
    elif interval_id == '1d':
        return "24 * 60 * 60000"
    elif interval_id == '1w':
        return "7 * 24 * 60 * 60000"
    elif interval_id == '1mo':
        return "30 * 24 * 60 * 60000"  # Approximate
    elif interval_id == '3mo':
        return "90 * 24 * 60 * 60000"  # Approximate
    else:
        return "60000"  # Default to 1 minute

def main():
    print("=" * 60)
    print("DUPLICATE CANDLE FIX SCRIPT")
    print("=" * 60)

    base_path = r"C:\StockApp\frontend\js\timeframes"
    fixed_count = 0
    skipped_count = 0
    failed_count = 0

    for category, intervals in TIMEFRAMES.items():
        print(f"\nProcessing {category}...")

        for interval_id in intervals:
            file_path = os.path.join(base_path, category, f"{interval_id}.js")

            if not os.path.exists(file_path):
                print(f"   [ERROR] File not found: {file_path}")
                failed_count += 1
                continue

            multiplier = get_interval_multiplier(interval_id)
            result = apply_fix_to_file(file_path, interval_id, multiplier)

            if result is True:
                fixed_count += 1
            elif result is False:
                failed_count += 1
            else:
                skipped_count += 1

    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)
    print(f"Fixed: {fixed_count}")
    print(f"Skipped (already fixed): {skipped_count}")
    print(f"Failed: {failed_count}")
    print("=" * 60)

if __name__ == "__main__":
    main()
