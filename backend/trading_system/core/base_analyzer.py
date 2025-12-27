"""
Base Analyzer Interface
All indicator analyzers must inherit from this class
"""

from abc import ABC, abstractmethod
from typing import Dict, Any
from datetime import datetime


class BaseAnalyzer(ABC):
    """
    Base class for all analyzers (indicators)

    Each analyzer implements the analyze() method which returns:
    - signal_strength: Float 0.0 to 1.0
    - direction: 'bullish', 'bearish', or 'neutral'
    - confidence: Float 0.0 to 1.0
    - metadata: Dict with analyzer-specific details
    """

    def __init__(self, name: str):
        self.name = name
        self.enabled = True
        self.weight = 1.0  # Can be adjusted by learning engine

    @abstractmethod
    def analyze(self, symbol: str, start_date: datetime, end_date: datetime) -> Dict[str, Any]:
        """
        Analyze a stock symbol over a date range

        Args:
            symbol: Stock ticker (e.g., 'AAPL', 'BTC-USD')
            start_date: Analysis start date
            end_date: Analysis end date

        Returns:
            {
                'signal_strength': float (0.0 to 1.0),
                'direction': str ('bullish', 'bearish', 'neutral'),
                'confidence': float (0.0 to 1.0),
                'metadata': dict (optional analyzer-specific data)
            }
        """
        pass

    def validate_result(self, result: Dict[str, Any]) -> bool:
        """Validate analyzer output format"""
        required_keys = ['signal_strength', 'direction', 'confidence']

        if not all(key in result for key in required_keys):
            return False

        if not 0.0 <= result['signal_strength'] <= 1.0:
            return False

        if not 0.0 <= result['confidence'] <= 1.0:
            return False

        if result['direction'] not in ['bullish', 'bearish', 'neutral']:
            return False

        return True

    def set_weight(self, weight: float):
        """Update analyzer weight (from learning engine)"""
        self.weight = max(0.1, min(3.0, weight))  # Clamp between 0.1 and 3.0

    def enable(self):
        """Enable this analyzer"""
        self.enabled = True

    def disable(self):
        """Disable this analyzer"""
        self.enabled = False

    def __repr__(self):
        return f"<{self.__class__.__name__} name='{self.name}' weight={self.weight:.2f} enabled={self.enabled}>"
