"""
Directional Override Meta-Learner
Intermediate layer between base models and final meta-learner
Discourages HOLD overuse by applying directional override logic
"""

from typing import Dict, List
import numpy as np


def adjust_probabilities_with_override(probs: Dict[str, float]) -> Dict[str, float]:
    """
    Adjust probabilities based on directional override logic.

    If HOLD dominates but there's strong directional asymmetry,
    redistribute probability mass from HOLD to the stronger direction.

    Args:
        probs: Dictionary with keys 'buy', 'sell', 'hold'

    Returns:
        Adjusted probabilities (still sum to 1.0)
    """
    prob_buy = probs['buy']
    prob_sell = probs['sell']
    prob_hold = probs['hold']

    # Find max class
    max_class = max(probs, key=probs.get)

    # If HOLD is max and override conditions are met
    if max_class == 'hold':
        buy_sell_diff = abs(prob_buy - prob_sell)
        max_directional = max(prob_buy, prob_sell)

        # Override conditions: asymmetry >5% and max directional >15%
        # (Relaxed thresholds to work at individual model level)
        if buy_sell_diff > 0.05 and max_directional > 0.15:
            # Redistribute HOLD probability to the stronger direction
            if prob_buy > prob_sell:
                # Boost BUY by transferring from HOLD
                boost = min(prob_hold * 0.5, 0.3)  # Transfer up to 50% of HOLD or 30% absolute
                return {
                    'buy': prob_buy + boost,
                    'sell': prob_sell,
                    'hold': prob_hold - boost
                }
            else:
                # Boost SELL by transferring from HOLD
                boost = min(prob_hold * 0.5, 0.3)
                return {
                    'buy': prob_buy,
                    'sell': prob_sell + boost,
                    'hold': prob_hold - boost
                }

    # No adjustment needed
    return probs


def directional_override(probs: Dict[str, float]) -> str:
    """
    Apply directional override logic to discourage HOLD overuse.

    If HOLD has the highest probability, but there's strong asymmetric confidence
    between BUY and SELL, override to the directional prediction.

    Args:
        probs: Dictionary with keys 'buy', 'sell', 'hold' (probabilities sum to 1.0)

    Returns:
        Prediction string: 'buy', 'sell', or 'hold'

    Logic:
        - If 'hold' is highest, but |buy - sell| > 0.15 AND max(buy, sell) > 0.4,
          override to 'buy' or 'sell' based on which is higher
        - Otherwise, return the class with highest probability
    """
    # Get probabilities
    prob_buy = probs['buy']
    prob_sell = probs['sell']
    prob_hold = probs['hold']

    # Find class with highest probability
    max_class = max(probs, key=probs.get)

    # Check if HOLD is the current winner
    if max_class == 'hold':
        # Calculate asymmetry between buy and sell
        buy_sell_diff = abs(prob_buy - prob_sell)
        max_directional = max(prob_buy, prob_sell)

        # Override conditions (relaxed for individual model level):
        # 1. Asymmetry is significant (>5% difference)
        # 2. At least one direction shows moderate confidence (>15%)
        if buy_sell_diff > 0.05 and max_directional > 0.15:
            # Override to the stronger direction
            if prob_buy > prob_sell:
                return 'buy'
            else:
                return 'sell'

    # No override needed - return class with highest probability
    return max_class


def apply_override_to_each_model(ensemble_outputs: List[Dict]) -> Dict:
    """
    Apply directional override to each of the 8 base models individually.
    Keeps 24 features (8 models × 3 probabilities) for final meta-learner.

    Args:
        ensemble_outputs: List of 8 dictionaries with prob_up, prob_down, prob_neutral

    Returns:
        Dictionary with:
            - 'adjusted_outputs': List of 8 adjusted probability dictionaries
            - 'override_count': Number of models where override was applied
    """
    adjusted_outputs = []
    override_count = 0

    for output in ensemble_outputs:
        # Convert to format expected by adjust function
        probs = {
            'buy': output['prob_up'],
            'sell': output['prob_down'],
            'hold': output['prob_neutral']
        }

        # Store original for comparison
        original_probs = probs.copy()

        # Apply adjustment
        adjusted_probs = adjust_probabilities_with_override(probs)

        # Check if adjustment occurred
        if adjusted_probs != original_probs:
            override_count += 1

        # Convert back to model output format
        adjusted_outputs.append({
            'prob_up': adjusted_probs['buy'],
            'prob_down': adjusted_probs['sell'],
            'prob_neutral': adjusted_probs['hold']
        })

    return {
        'adjusted_outputs': adjusted_outputs,
        'override_count': override_count
    }


def apply_override_to_ensemble_outputs(ensemble_outputs: List[Dict]) -> Dict:
    """
    Average 8 base model outputs and apply directional override logic.

    This is the intermediate meta-learner that sits between base models
    and the final meta-learner.

    Args:
        ensemble_outputs: List of 8 dictionaries, each containing:
            - 'prob_up': probability of UP/BUY class
            - 'prob_down': probability of DOWN/SELL class
            - 'prob_neutral': probability of NEUTRAL/HOLD class

    Returns:
        Dictionary with:
            - 'prob_up': averaged UP probability
            - 'prob_down': averaged DOWN probability
            - 'prob_neutral': averaged NEUTRAL probability
            - 'prediction': final prediction after override ('buy', 'sell', or 'hold')
            - 'overridden': boolean flag indicating if override logic was applied
            - 'original_prediction': what the prediction would have been without override
    """
    # Step 1: Average the 8 base model probabilities
    prob_up_values = [output['prob_up'] for output in ensemble_outputs]
    prob_down_values = [output['prob_down'] for output in ensemble_outputs]
    prob_neutral_values = [output['prob_neutral'] for output in ensemble_outputs]

    avg_prob_up = float(np.mean(prob_up_values))
    avg_prob_down = float(np.mean(prob_down_values))
    avg_prob_neutral = float(np.mean(prob_neutral_values))

    # Step 2: Normalize to ensure probabilities sum to 1.0
    total = avg_prob_up + avg_prob_down + avg_prob_neutral
    avg_prob_up /= total
    avg_prob_down /= total
    avg_prob_neutral /= total

    # Step 3: Create probability dictionary for override function
    probs = {
        'buy': avg_prob_up,
        'sell': avg_prob_down,
        'hold': avg_prob_neutral
    }

    # Step 4: Determine original prediction (without override)
    original_prediction = max(probs, key=probs.get)

    # Step 5: Apply directional override logic
    final_prediction = directional_override(probs)

    # Step 6: Check if override was applied
    overridden = (final_prediction != original_prediction)

    # Step 7: Return results
    return {
        'prob_up': avg_prob_up,
        'prob_down': avg_prob_down,
        'prob_neutral': avg_prob_neutral,
        'prediction': final_prediction,
        'overridden': overridden,
        'original_prediction': original_prediction
    }


def test_directional_override():
    """Test cases for directional override logic"""

    # Test 1: Strong neutral, no override
    test1 = {'buy': 0.1, 'sell': 0.05, 'hold': 0.85}
    result1 = directional_override(test1)
    print(f"Test 1 (strong neutral): {result1}")
    assert result1 == 'hold', "Should remain hold"

    # Test 2: Neutral highest, but strong buy signal - OVERRIDE
    test2 = {'buy': 0.45, 'sell': 0.05, 'hold': 0.50}
    result2 = directional_override(test2)
    print(f"Test 2 (neutral but strong buy asymmetry): {result2}")
    assert result2 == 'buy', "Should override to buy"

    # Test 3: Neutral highest, but strong sell signal - OVERRIDE
    test3 = {'buy': 0.05, 'sell': 0.48, 'hold': 0.47}
    result3 = directional_override(test3)
    print(f"Test 3 (neutral but strong sell asymmetry): {result3}")
    assert result3 == 'sell', "Should override to sell"

    # Test 4: Buy is already highest - no override needed
    test4 = {'buy': 0.60, 'sell': 0.20, 'hold': 0.20}
    result4 = directional_override(test4)
    print(f"Test 4 (buy already highest): {result4}")
    assert result4 == 'buy', "Should remain buy"

    # Test 5: Neutral highest, moderate asymmetry but low directional - NO OVERRIDE
    test5 = {'buy': 0.35, 'sell': 0.15, 'hold': 0.50}
    result5 = directional_override(test5)
    print(f"Test 5 (asymmetry but low directional): {result5}")
    assert result5 == 'hold', "Should remain hold (max directional < 0.4)"

    print("\n[OK] All tests passed")


if __name__ == '__main__':
    # Run test cases
    test_directional_override()

    # Test ensemble averaging
    print("\nTesting ensemble averaging...")

    # Simulate 8 base model outputs (models agree on neutral bias)
    ensemble_outputs = [
        {'prob_up': 0.20, 'prob_down': 0.05, 'prob_neutral': 0.75},
        {'prob_up': 0.18, 'prob_down': 0.07, 'prob_neutral': 0.75},
        {'prob_up': 0.22, 'prob_down': 0.03, 'prob_neutral': 0.75},
        {'prob_up': 0.25, 'prob_down': 0.05, 'prob_neutral': 0.70},
        {'prob_up': 0.19, 'prob_down': 0.06, 'prob_neutral': 0.75},
        {'prob_up': 0.21, 'prob_down': 0.04, 'prob_neutral': 0.75},
        {'prob_up': 0.23, 'prob_down': 0.02, 'prob_neutral': 0.75},
        {'prob_up': 0.20, 'prob_down': 0.05, 'prob_neutral': 0.75},
    ]

    result = apply_override_to_ensemble_outputs(ensemble_outputs)

    print(f"Averaged probabilities:")
    print(f"  prob_up: {result['prob_up']:.4f}")
    print(f"  prob_down: {result['prob_down']:.4f}")
    print(f"  prob_neutral: {result['prob_neutral']:.4f}")
    print(f"Original prediction: {result['original_prediction']}")
    print(f"Final prediction: {result['prediction']}")
    print(f"Override applied: {result['overridden']}")
