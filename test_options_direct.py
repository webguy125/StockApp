"""
Direct test of options API without Flask
Run this to test the options analyzer directly
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'backend'))

from turbomode.options_api import OptionsAnalyzer
import json

print("Testing Options Analyzer...")
print("=" * 80)

analyzer = OptionsAnalyzer()

# Test with AAPL
symbol = 'AAPL'
print(f"\nAnalyzing options for {symbol}...")

result = analyzer.analyze_options_chain(symbol)

if 'error' in result:
    print(f"\nERROR: {result['error']}")
else:
    print("\n✓ SUCCESS!")
    print("\n" + "=" * 80)
    print("RESULTS:")
    print("=" * 80)
    print(json.dumps(result, indent=2))

print("\n" + "=" * 80)
print("Test complete!")
