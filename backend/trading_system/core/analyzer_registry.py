"""
Analyzer Registry
Plugin system for managing all analyzers
"""

from typing import Dict, List, Any
from datetime import datetime
from .base_analyzer import BaseAnalyzer


class AnalyzerRegistry:
    """
    Registry for all indicator analyzers
    Provides plugin-like system for adding/removing analyzers
    """

    def __init__(self):
        self.analyzers: Dict[str, BaseAnalyzer] = {}

    def register(self, analyzer: BaseAnalyzer):
        """Register a new analyzer"""
        if not isinstance(analyzer, BaseAnalyzer):
            raise TypeError(f"Analyzer must inherit from BaseAnalyzer, got {type(analyzer)}")

        self.analyzers[analyzer.name] = analyzer
        print(f"[OK] Registered analyzer: {analyzer.name}")

    def unregister(self, name: str):
        """Unregister an analyzer by name"""
        if name in self.analyzers:
            del self.analyzers[name]
            print(f"[REMOVED] Unregistered analyzer: {name}")

    def get(self, name: str) -> BaseAnalyzer:
        """Get analyzer by name"""
        return self.analyzers.get(name)

    def list_all(self) -> List[str]:
        """List all registered analyzer names"""
        return list(self.analyzers.keys())

    def analyze_symbol(self, symbol: str, start_date: datetime, end_date: datetime) -> Dict[str, Any]:
        """
        Run ALL enabled analyzers on a symbol

        Returns:
            {
                'analyzer_name': {
                    'signal_strength': float,
                    'direction': str,
                    'confidence': float,
                    'metadata': dict
                },
                ...
            }
        """
        results = {}

        for name, analyzer in self.analyzers.items():
            if not analyzer.enabled:
                continue

            try:
                result = analyzer.analyze(symbol, start_date, end_date)

                if analyzer.validate_result(result):
                    results[name] = result
                else:
                    print(f"[WARNING] Invalid result from {name}, skipping")

            except Exception as e:
                print(f"[ERROR] Error in {name} for {symbol}: {e}")
                continue

        return results

    def get_enabled_count(self) -> int:
        """Count enabled analyzers"""
        return sum(1 for a in self.analyzers.values() if a.enabled)

    def enable_all(self):
        """Enable all analyzers"""
        for analyzer in self.analyzers.values():
            analyzer.enable()

    def disable_all(self):
        """Disable all analyzers"""
        for analyzer in self.analyzers.values():
            analyzer.disable()

    def set_weights(self, weights: Dict[str, float]):
        """
        Update analyzer weights (from learning engine)

        Args:
            weights: Dict mapping analyzer name to weight
        """
        for name, weight in weights.items():
            if name in self.analyzers:
                self.analyzers[name].set_weight(weight)

    def get_weights(self) -> Dict[str, float]:
        """Get current weights for all analyzers"""
        return {name: analyzer.weight for name, analyzer in self.analyzers.items()}

    def __repr__(self):
        return f"<AnalyzerRegistry total={len(self.analyzers)} enabled={self.get_enabled_count()}>"
