"""
Iron Condor P&L Engine for HOLD Signals Only
Phase 1: Real-Time Option Pricing via IBKR

RESTRICTIONS:
- ONLY affects HOLD row P&L display
- NO modifications to BUY/SELL logic
- NO modifications to prediction engine
- NO modifications to existing P&L calculations
- Market hours ONLY (08:30-15:00 CST)
"""

from .hold_condor_engine import compute_hold_condor_pnl

__all__ = ['compute_hold_condor_pnl']
