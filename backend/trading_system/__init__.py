"""
Modular ML Trading System
Zero-cost, plugin-based stock analysis and learning system
"""

__version__ = "1.0.0"

from .core.trading_system import TradingSystem
from .core.analyzer_registry import AnalyzerRegistry
from .core.stock_scanner import StockScanner
from .core.trade_tracker import TradeTracker

__all__ = [
    'TradingSystem',
    'AnalyzerRegistry',
    'StockScanner',
    'TradeTracker'
]
