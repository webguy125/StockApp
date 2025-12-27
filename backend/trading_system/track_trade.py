"""
Trade Tracking Tool
Record trades and their outcomes for ML learning
"""

import sys
import os
from datetime import datetime

# Add parent directory to path
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
if parent_dir not in sys.path:
    sys.path.insert(0, parent_dir)

from trading_system.core.trade_tracker import TradeTracker


def record_trade():
    """Interactive trade recording"""
    tracker = TradeTracker()

    print("\n" + "=" * 60)
    print("RECORD A TRADE")
    print("=" * 60 + "\n")

    # Get trade details
    symbol = input("Symbol: ").strip().upper()
    entry_price = float(input("Entry Price: $"))
    position_size = float(input("Position Size (shares): "))

    # Record trade
    trade_id = tracker.save_trade(
        symbol=symbol,
        entry_price=entry_price,
        position_size=position_size
    )

    print(f"\n✅ Trade recorded! Trade ID: {trade_id}")
    print(f"Symbol: {symbol}")
    print(f"Entry: ${entry_price:.2f}")
    print(f"Size: {position_size} shares")
    print(f"Total Cost: ${entry_price * position_size:.2f}")
    print("\nUse this Trade ID to close the trade later.")


def close_trade():
    """Close an existing trade"""
    tracker = TradeTracker()

    print("\n" + "=" * 60)
    print("CLOSE A TRADE")
    print("=" * 60 + "\n")

    # Show open trades
    open_trades = tracker.get_trades(status='open')

    if not open_trades:
        print("No open trades found.")
        return

    print("Open Trades:")
    print("-" * 60)
    for i, trade in enumerate(open_trades, 1):
        print(f"{i}. {trade['symbol']} - Entry: ${trade['entry_price']:.2f} - ID: {trade['id'][:8]}")

    # Get trade to close
    choice = int(input("\nSelect trade number to close (0 to cancel): "))
    if choice == 0 or choice > len(open_trades):
        print("Cancelled.")
        return

    trade = open_trades[choice - 1]

    # Get exit details
    print(f"\nClosing trade: {trade['symbol']}")
    exit_price = float(input("Exit Price: $"))

    exit_reason_choices = {
        '1': 'target',
        '2': 'stop',
        '3': 'time',
        '4': 'manual'
    }

    print("\nExit Reason:")
    print("1. Hit target")
    print("2. Hit stop loss")
    print("3. Time exit")
    print("4. Manual exit")
    reason_choice = input("Select (1-4): ")
    exit_reason = exit_reason_choices.get(reason_choice, 'manual')

    # Close trade
    tracker.close_trade(trade['id'], exit_price, exit_reason)

    # Calculate P&L
    entry_price = trade['entry_price']
    position_size = trade['position_size']
    profit_loss = (exit_price - entry_price) * position_size
    profit_loss_pct = ((exit_price - entry_price) / entry_price) * 100

    print("\n" + "=" * 60)
    print("TRADE CLOSED")
    print("=" * 60)
    print(f"Symbol: {trade['symbol']}")
    print(f"Entry: ${entry_price:.2f}")
    print(f"Exit: ${exit_price:.2f}")
    print(f"P/L: ${profit_loss:.2f} ({profit_loss_pct:+.2f}%)")
    print(f"Outcome: {'WIN ✅' if profit_loss > 0 else 'LOSS ❌'}")
    print(f"Exit Reason: {exit_reason}")
    print("=" * 60 + "\n")


def view_stats():
    """View performance statistics"""
    tracker = TradeTracker()
    stats = tracker.get_performance_stats()

    print("\n" + "=" * 60)
    print("TRADING PERFORMANCE STATISTICS")
    print("=" * 60 + "\n")

    print(f"Total Trades:     {stats['total_trades']}")
    print(f"Wins:             {stats['wins']}")
    print(f"Losses:           {stats['losses']}")
    print(f"Win Rate:         {stats['win_rate']:.2f}%")
    print(f"Avg P/L:          ${stats['avg_profit_loss']:.2f}")
    print(f"Total P/L:        ${stats['total_profit_loss']:.2f}")
    print(f"Gross Profit:     ${stats['gross_profit']:.2f}")
    print(f"Gross Loss:       ${stats['gross_loss']:.2f}")
    print(f"Profit Factor:    {stats['profit_factor']:.2f}")

    print("\n" + "=" * 60 + "\n")


def view_recent_trades():
    """View recent trades"""
    tracker = TradeTracker()

    print("\n" + "=" * 60)
    print("RECENT TRADES")
    print("=" * 60 + "\n")

    trades = tracker.get_trades(limit=10)

    if not trades:
        print("No trades found.")
        return

    print(f"{'Symbol':<8} {'Entry':<10} {'Exit':<10} {'P/L %':<10} {'Status':<10}")
    print("-" * 60)

    for trade in trades:
        symbol = trade['symbol']
        entry = f"${trade['entry_price']:.2f}"
        exit_val = f"${trade['exit_price']:.2f}" if trade['exit_price'] else "Open"
        pl_pct = f"{trade['profit_loss_pct']:+.2f}%" if trade['profit_loss_pct'] else "-"
        status = trade['outcome'].upper()

        print(f"{symbol:<8} {entry:<10} {exit_val:<10} {pl_pct:<10} {status:<10}")

    print("")


def main_menu():
    """Main menu"""
    while True:
        print("\n" + "=" * 60)
        print("ML TRADING SYSTEM - TRADE TRACKER")
        print("=" * 60)
        print("\n1. Record New Trade")
        print("2. Close Trade")
        print("3. View Performance Stats")
        print("4. View Recent Trades")
        print("5. Exit")

        choice = input("\nSelect option (1-5): ")

        if choice == '1':
            record_trade()
        elif choice == '2':
            close_trade()
        elif choice == '3':
            view_stats()
        elif choice == '4':
            view_recent_trades()
        elif choice == '5':
            print("\nExiting...")
            break
        else:
            print("Invalid choice. Try again.")


if __name__ == '__main__':
    try:
        main_menu()
    except KeyboardInterrupt:
        print("\n\nExiting...")
    except Exception as e:
        print(f"\n❌ Error: {e}")
        import traceback
        traceback.print_exc()
