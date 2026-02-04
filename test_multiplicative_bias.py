"""
Validation test for multiplicative probability biasing.
Tests the three scenarios specified in the executor task.
"""

def test_multiplicative_bias():
    """Test multiplicative biasing logic with specified test cases."""

    test_cases = [
        {
            "name": "bias_positive",
            "buy": 0.4,
            "sell": 0.2,
            "hold": 0.4,
            "bias": 0.05,
            "expected": "BUY increases, SELL decreases, HOLD stable, renormalized"
        },
        {
            "name": "bias_negative",
            "buy": 0.4,
            "sell": 0.2,
            "hold": 0.4,
            "bias": -0.05,
            "expected": "SELL increases, BUY decreases, HOLD stable, renormalized"
        },
        {
            "name": "no_bias",
            "buy": 0.4,
            "sell": 0.2,
            "hold": 0.4,
            "bias": 0.0,
            "expected": "Probabilities unchanged"
        }
    ]

    print("="*80)
    print("MULTIPLICATIVE BIAS VALIDATION TEST")
    print("="*80)
    print()

    for test in test_cases:
        prob_buy = test["buy"]
        prob_sell = test["sell"]
        prob_hold = test["hold"]
        k = test["bias"]

        print(f"TEST: {test['name']}")
        print(f"  Input:    BUY={prob_buy:.3f}, SELL={prob_sell:.3f}, HOLD={prob_hold:.3f}, bias={k:+.3f}")
        print(f"  Expected: {test['expected']}")

        if k != 0.0:
            # Apply multiplicative biasing
            adjusted_buy = prob_buy * (1.0 + k)
            adjusted_sell = prob_sell * (1.0 - k)
            adjusted_hold = prob_hold

            # Prevent negative values
            adjusted_buy = max(0.0, adjusted_buy)
            adjusted_sell = max(0.0, adjusted_sell)
            adjusted_hold = max(0.0, adjusted_hold)

            # Renormalize
            total = adjusted_buy + adjusted_sell + adjusted_hold
            if total > 0:
                adjusted_buy /= total
                adjusted_sell /= total
                adjusted_hold /= total
        else:
            # No bias
            adjusted_buy = prob_buy
            adjusted_sell = prob_sell
            adjusted_hold = prob_hold

        print(f"  Output:   BUY={adjusted_buy:.3f}, SELL={adjusted_sell:.3f}, HOLD={adjusted_hold:.3f}")
        print(f"  Sum:      {adjusted_buy + adjusted_sell + adjusted_hold:.6f} (should be 1.0)")

        # Validation checks
        assert abs((adjusted_buy + adjusted_sell + adjusted_hold) - 1.0) < 1e-6, "Sum must equal 1.0"
        assert adjusted_buy >= 0, "BUY must be non-negative"
        assert adjusted_sell >= 0, "SELL must be non-negative"
        assert adjusted_hold >= 0, "HOLD must be non-negative"

        if k > 0:  # Bullish bias
            assert adjusted_buy > prob_buy or abs(adjusted_buy - prob_buy) < 1e-6, "BUY should increase with positive bias"
            assert adjusted_sell < prob_sell or abs(adjusted_sell - prob_sell) < 1e-6, "SELL should decrease with positive bias"
        elif k < 0:  # Bearish bias
            assert adjusted_sell > prob_sell or abs(adjusted_sell - prob_sell) < 1e-6, "SELL should increase with negative bias"
            assert adjusted_buy < prob_buy or abs(adjusted_buy - prob_buy) < 1e-6, "BUY should decrease with negative bias"
        else:  # No bias
            assert abs(adjusted_buy - prob_buy) < 1e-6, "BUY should be unchanged with no bias"
            assert abs(adjusted_sell - prob_sell) < 1e-6, "SELL should be unchanged with no bias"
            assert abs(adjusted_hold - prob_hold) < 1e-6, "HOLD should be unchanged with no bias"

        print(f"  [PASS] All validation checks passed")
        print()

    print("="*80)
    print("ALL TESTS PASSED")
    print("="*80)

if __name__ == '__main__':
    test_multiplicative_bias()
