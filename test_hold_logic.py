"""
Test to understand why HOLD signals aren't being generated.
"""

import numpy as np

# Symbol A example (after biasing)
prob_buy = 0.114
prob_sell = 0.033
prob_hold = 0.853

# Calculate neutrality band (current logic)
model_std = np.std([prob_buy, prob_sell, prob_hold])
neutrality_band = 0.5 * model_std

prob_diff = abs(prob_buy - prob_sell)

print("="*80)
print("HOLD SIGNAL LOGIC TEST - Symbol A")
print("="*80)
print()
print(f"Probabilities after biasing:")
print(f"  BUY:  {prob_buy:.3f}")
print(f"  SELL: {prob_sell:.3f}")
print(f"  HOLD: {prob_hold:.3f}")
print()
print(f"Neutrality band calculation:")
print(f"  model_std = np.std([{prob_buy:.3f}, {prob_sell:.3f}, {prob_hold:.3f}])")
print(f"  model_std = {model_std:.3f}")
print(f"  neutrality_band = 0.5 * {model_std:.3f} = {neutrality_band:.3f}")
print()
print(f"Current logic check:")
print(f"  |BUY - SELL| < neutrality_band?")
print(f"  |{prob_buy:.3f} - {prob_sell:.3f}| < {neutrality_band:.3f}?")
print(f"  {prob_diff:.3f} < {neutrality_band:.3f}?")
print(f"  Result: {prob_diff < neutrality_band}")
print()

if prob_diff < neutrality_band:
    signal = "HOLD"
elif prob_buy > prob_sell:
    signal = "BUY"
else:
    signal = "SELL"

print(f"Final signal: {signal}")
print()
print("="*80)
print("THE PROBLEM")
print("="*80)
print()
print("Symbol A has HOLD=0.853 (85.3% confidence for neutral regime)")
print("But current logic converts it to BUY because:")
print(f"  - BUY ({prob_buy:.3f}) > SELL ({prob_sell:.3f})")
print(f"  - Difference ({prob_diff:.3f}) exceeds band ({neutrality_band:.3f})")
print()
print("The neutrality band is too narrow!")
print(f"  With HOLD=0.853, std=[{prob_buy:.3f}, {prob_sell:.3f}, {prob_hold:.3f}]")
print(f"  std = {model_std:.3f}")
print(f"  band = {neutrality_band:.3f}")
print()
print("SOLUTION: Use argmax(BUY, SELL, HOLD) instead of neutrality band")
print("  If HOLD is highest → signal = HOLD")
print("  If BUY is highest → signal = BUY")
print("  If SELL is highest → signal = SELL")
print()
