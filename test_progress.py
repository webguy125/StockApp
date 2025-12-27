"""Test training progress endpoint"""
import sys
import os

# Add backend to path
backend_dir = os.path.join(os.path.dirname(__file__), 'backend')
sys.path.insert(0, backend_dir)

from trading_system.core.trade_tracker import TradeTracker

tracker = TradeTracker()
stats = tracker.get_performance_stats()

print('=== TRAINING PROGRESS TEST ===')
print(f'Total Trades: {stats.get("total_trades", 0)}')
print(f'Wins: {stats.get("wins", 0)}')
print(f'Losses: {stats.get("losses", 0)}')
print(f'Win Rate: {stats.get("win_rate", 0):.1f}%')
print(f'Total P/L: ${stats.get("total_pl", 0):.2f}')
print()
print('Full stats:', stats)
