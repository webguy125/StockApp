"""
Quick test of scanner functions with Tradier integration
"""
import sys
sys.path.insert(0, 'C:\\StockApp\\backend')

from backend.turbomode.Options.Scanner.options_scanner import get_current_price, get_option_chain

# Test symbol
symbol = 'AAPL'

print(f"\n{'='*60}")
print(f"Testing Scanner Functions for {symbol}")
print(f"{'='*60}\n")

# Test price fetch
print("[1] Testing get_current_price()...")
price = get_current_price(symbol)
if price:
    print(f"    [OK] Price: ${price:.2f}")
else:
    print(f"    [FAIL] Price fetch failed")

# Test chain fetch
print("\n[2] Testing get_option_chain()...")
chain = get_option_chain(symbol)
if chain:
    print(f"    [OK] Chain fetched: {len(chain['expirations'])} expirations")
    print(f"    [OK] First expiration: {chain['expirations'][0]}")

    # Check first expiration data
    first_exp = chain['expirations'][0]
    if first_exp in chain['chains']:
        calls = chain['chains'][first_exp]['calls']
        puts = chain['chains'][first_exp]['puts']
        print(f"    [OK] Calls: {len(calls)} strikes")
        print(f"    [OK] Puts: {len(puts)} strikes")
else:
    print(f"    [FAIL] Chain fetch failed")

print(f"\n{'='*60}")
print("Test complete")
print(f"{'='*60}\n")
