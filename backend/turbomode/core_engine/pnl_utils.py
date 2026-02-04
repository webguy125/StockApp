"""
Unified P&L Calculation Engine

Single source of truth for profit/loss calculations across the entire TurboMode system.
All modules must import and use calculate_pnl_pct() to ensure consistency.

This eliminates the SELL position P&L inversion bug that was causing:
- Short winners to appear as losers
- Short losers to appear as winners
- Corrupted adaptive ranking scores
- Inverted backtest results
"""


def calculate_pnl_pct(entry_price: float, exit_price: float, signal_type: str) -> float:
    """
    Calculate profit/loss percentage for a trade.

    Args:
        entry_price: Price at which position was entered
        exit_price: Price at which position was exited
        signal_type: 'BUY', 'SELL', or 'HOLD'

    Returns:
        Profit/loss as a percentage (positive = profit, negative = loss)

    Logic:
        - BUY (long): profit when exit_price > entry_price
          Formula: ((exit_price - entry_price) / entry_price) * 100
          Example: Entry $100, Exit $110 → +10%

        - SELL (short): profit when exit_price < entry_price
          Formula: ((entry_price - exit_price) / entry_price) * 100
          Example: Entry $100, Exit $90 → +10%
          Example: Entry $100, Exit $110 → -10%

        - HOLD (neutral): same as BUY for tracking purposes
          Formula: ((exit_price - entry_price) / entry_price) * 100
    """
    if signal_type == 'SELL':
        # Short position: profit when price decreases
        pnl_pct = ((entry_price - exit_price) / entry_price) * 100.0
    else:
        # Long position (BUY/HOLD): profit when price increases
        pnl_pct = ((exit_price - entry_price) / entry_price) * 100.0

    return pnl_pct


def validate_pnl_calculation():
    """
    Unit tests to verify P&L calculation correctness.
    Run this to confirm the module is working properly.
    """
    print("P&L CALCULATION VALIDATION")
    print("=" * 60)

    # Test BUY positions
    print("\nBUY (Long) Positions:")
    buy_profit = calculate_pnl_pct(100.0, 110.0, 'BUY')
    print(f"  Entry $100, Exit $110 -> {buy_profit:+.2f}% (expected +10.00%)")
    assert abs(buy_profit - 10.0) < 0.01, "BUY profit test failed"

    buy_loss = calculate_pnl_pct(100.0, 90.0, 'BUY')
    print(f"  Entry $100, Exit $90  -> {buy_loss:+.2f}% (expected -10.00%)")
    assert abs(buy_loss - (-10.0)) < 0.01, "BUY loss test failed"

    # Test SELL positions
    print("\nSELL (Short) Positions:")
    sell_profit = calculate_pnl_pct(100.0, 90.0, 'SELL')
    print(f"  Entry $100, Exit $90  -> {sell_profit:+.2f}% (expected +10.00%)")
    assert abs(sell_profit - 10.0) < 0.01, "SELL profit test failed"

    sell_loss = calculate_pnl_pct(100.0, 110.0, 'SELL')
    print(f"  Entry $100, Exit $110 -> {sell_loss:+.2f}% (expected -10.00%)")
    assert abs(sell_loss - (-10.0)) < 0.01, "SELL loss test failed"

    # Test HOLD positions
    print("\nHOLD (Neutral) Positions:")
    hold_profit = calculate_pnl_pct(100.0, 105.0, 'HOLD')
    print(f"  Entry $100, Exit $105 -> {hold_profit:+.2f}% (expected +5.00%)")
    assert abs(hold_profit - 5.0) < 0.01, "HOLD profit test failed"

    hold_loss = calculate_pnl_pct(100.0, 95.0, 'HOLD')
    print(f"  Entry $100, Exit $95  -> {hold_loss:+.2f}% (expected -5.00%)")
    assert abs(hold_loss - (-5.0)) < 0.01, "HOLD loss test failed"

    # Test real-world example that was broken
    print("\nReal-World Bug Example (INTU SELL):")
    intu_pnl = calculate_pnl_pct(528.95, 563.97, 'SELL')
    print(f"  Entry $528.95, Exit $563.97 -> {intu_pnl:+.2f}%")
    print(f"  (Price went UP by 6.62%, so short position LOST 6.62%)")
    assert intu_pnl < 0, "INTU SELL should show loss when price rises"

    print("\n" + "=" * 60)
    print("ALL TESTS PASSED ✓")
    print("P&L calculation engine is working correctly.")


if __name__ == '__main__':
    validate_pnl_calculation()
