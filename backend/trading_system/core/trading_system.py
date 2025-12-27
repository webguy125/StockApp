"""
Trading System Orchestrator
Main class that coordinates all components
"""

import json
import os
from datetime import datetime, timedelta
from typing import List, Dict, Any
import sys

# Add parent directory to path
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
if parent_dir not in sys.path:
    sys.path.insert(0, parent_dir)

from trading_system.core.stock_scanner import StockScanner
from trading_system.core.analyzer_registry import AnalyzerRegistry
from trading_system.core.feature_extractor import FeatureExtractor
from trading_system.core.trade_tracker import TradeTracker
from trading_system.models.simple_trading_model import SimpleTradingModel
from trading_system.ml_model_manager import MLModelManager


class TradingSystem:
    """
    Main orchestrator for the ML trading system

    Workflow:
    1. Scan S&P 500 for candidates
    2. Run all analyzers on each candidate
    3. Extract features and make predictions
    4. Rank and score signals
    5. Save top signals to JSON
    """

    def __init__(self):
        self.scanner = StockScanner()
        self.registry = AnalyzerRegistry()
        self.extractor = FeatureExtractor()
        self.tracker = TradeTracker()
        self.model = SimpleTradingModel()
        self.model_manager = MLModelManager()

        # Load active model configuration
        self.active_config = self.model_manager.get_active_model()
        self.active_model_name = self.active_config['name'] if self.active_config else None

        # Register analyzers based on active configuration
        self._register_analyzers_from_config()

    def _register_analyzers_from_config(self):
        """Register analyzers based on active model configuration"""
        if not self.active_config:
            # No active model - use default analyzers
            print("[WARNING] No active model configuration found - using default analyzers")
            self._register_default_analyzers()
            return

        analysis_type = self.active_config.get('analysis_type', 'price_action')

        print(f"[OK] Loading analyzers for model: {self.active_model_name}")
        print(f"   Analysis type: {analysis_type}")

        # Load appropriate analyzer based on configuration
        if analysis_type == 'price_action':
            from trading_system.analyzers.price_action_analyzer import PriceActionAnalyzer
            self.registry.register(PriceActionAnalyzer())
            print("   - Price Action Analyzer")

        elif analysis_type == 'volume_profile':
            from trading_system.analyzers.volume_profile_analyzer import VolumeProfileAnalyzer
            self.registry.register(VolumeProfileAnalyzer())
            print("   - Volume Profile Analyzer")

        elif analysis_type == 'raw_ohlcv':
            from trading_system.analyzers.raw_ohlcv_analyzer import RawOHLCVAnalyzer
            self.registry.register(RawOHLCVAnalyzer())
            print("   - Raw OHLCV Analyzer")

        elif analysis_type == 'market_structure':
            from trading_system.analyzers.market_structure_analyzer import MarketStructureAnalyzer
            self.registry.register(MarketStructureAnalyzer())
            print("   - Market Structure Analyzer")

        else:
            print(f"[WARNING] Unknown analysis type: {analysis_type} - using default")
            self._register_default_analyzers()

        print(f"[OK] Registered {self.registry.get_enabled_count()} analyzers")

    def _register_default_analyzers(self):
        """Register default built-in analyzers (fallback)"""
        from trading_system.analyzers.rsi_analyzer import RSIAnalyzer
        from trading_system.analyzers.macd_analyzer import MACDAnalyzer
        from trading_system.analyzers.volume_analyzer import VolumeAnalyzer
        from trading_system.analyzers.trend_analyzer import TrendAnalyzer

        self.registry.register(RSIAnalyzer())
        self.registry.register(MACDAnalyzer())
        self.registry.register(VolumeAnalyzer())
        self.registry.register(TrendAnalyzer())

    def run_daily_scan(self, max_stocks: int = 500, include_crypto: bool = True,
                      progress_callback=None) -> List[Dict[str, Any]]:
        """
        Run complete daily scan - SCANS ENTIRE S&P 500!

        Args:
            max_stocks: Maximum stocks to analyze (default 500 = entire S&P 500)
            include_crypto: Whether to include crypto
            progress_callback: Optional callback function(current, total, symbol)

        Returns:
            List of signals sorted by score
        """
        print("\n" + "=" * 60)
        print("ML TRADING SYSTEM - DAILY SCAN")
        print(f"Started: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        print("=" * 60 + "\n")

        # Step 1: Scan for candidates (ENTIRE S&P 500 + top 100 cryptos)
        candidates = self.scanner.scan(max_results=max_stocks)

        if include_crypto:
            crypto_candidates = self.scanner.scan_crypto(top_n=100)
            candidates.extend(crypto_candidates)

        print(f"\n[ANALYZE] Analyzing {len(candidates)} candidates (S&P 500 + top 100 cryptos)...\n")

        # Step 2: Analyze each candidate
        signals = []
        start_date = datetime.now() - timedelta(days=120)  # 4 months for more accurate indicators
        end_date = datetime.now()

        for i, candidate in enumerate(candidates):
            symbol = candidate['symbol']

            # Report progress via callback
            if progress_callback and (i + 1) % 10 == 0:
                progress_callback(current=i + 1, total=len(candidates), symbol=symbol)

            if (i + 1) % 10 == 0:
                print(f"Progress: {i + 1}/{len(candidates)}")

            try:
                # Run all analyzers
                analyzer_results = self.registry.analyze_symbol(symbol, start_date, end_date)

                if not analyzer_results:
                    continue

                # Extract features
                features = self.extractor.extract_features(analyzer_results)

                # Get ML prediction
                ml_signal = self.model.get_signal(features)

                # Calculate combined score
                # Weight: 60% from analyzers, 40% from ML model
                analyzer_avg = sum(r['signal_strength'] for r in analyzer_results.values()) / len(analyzer_results)
                ml_score = ml_signal['probabilities']['buy']  # Use buy probability as score

                combined_score = (analyzer_avg * 0.6) + (ml_score * 0.4)

                # Determine direction
                if combined_score > 0.6:
                    direction = 'bullish'
                elif combined_score < 0.4:
                    direction = 'bearish'
                else:
                    direction = 'neutral'

                # Average confidence
                avg_confidence = sum(r['confidence'] for r in analyzer_results.values()) / len(analyzer_results)
                final_confidence = (avg_confidence * 0.5) + (ml_signal['confidence'] * 0.5)

                signal = {
                    'symbol': symbol,
                    'score': combined_score,
                    'confidence': final_confidence,
                    'direction': direction,
                    'ml_prediction': ml_signal['prediction'],
                    'ml_confidence': ml_signal['confidence'],
                    'ml_probabilities': ml_signal['probabilities'],
                    'analyzers': analyzer_results,
                    'price': candidate.get('price', 0.0),
                    'volatility': candidate.get('volatility', 0.0),
                    'timestamp': datetime.now().isoformat()
                }

                signals.append(signal)

                # Save signal to database
                self.tracker.save_signal(
                    symbol=symbol,
                    score=combined_score,
                    confidence=final_confidence,
                    direction=direction,
                    analyzers=analyzer_results,
                    model_predictions=ml_signal,
                    model_name=self.active_model_name
                )

            except Exception as e:
                print(f"[ERROR] Error analyzing {symbol}: {e}")
                continue

        # Step 3: Rank ALL signals
        signals.sort(key=lambda x: x['score'], reverse=True)

        # Track total scanned
        total_scanned = len(candidates)
        total_signals_generated = len(signals)

        # Step 4: Filter to TOP signals only (confidence >= 50% OR top 100 by score)
        MIN_CONFIDENCE = 0.50  # 50% minimum confidence
        MAX_SIGNALS = 100      # Keep max 100 top signals

        # Filter: confidence >= 50% OR in top 100
        filtered_signals = [
            sig for sig in signals
            if sig['confidence'] >= MIN_CONFIDENCE or signals.index(sig) < MAX_SIGNALS
        ]

        # Ensure we don't exceed MAX_SIGNALS
        filtered_signals = filtered_signals[:MAX_SIGNALS]

        # Add ranks to filtered signals
        for i, signal in enumerate(filtered_signals):
            signal['rank'] = i + 1

        # Step 5: Save to JSON with scan metadata
        output_file = "backend/data/ml_trading_signals.json"
        self._save_signals(
            filtered_signals,
            output_file,
            total_scanned=total_scanned,
            total_generated=total_signals_generated
        )

        # Print summary
        print("\n" + "=" * 60)
        print("SCAN COMPLETE")
        print("=" * 60)
        print(f"Total symbols scanned: {total_scanned}")
        print(f"Total signals generated: {total_signals_generated}")
        print(f"Top signals displayed: {len(filtered_signals)}")
        if filtered_signals:
            print(f"\nTop 5 signals:")
            for i, sig in enumerate(filtered_signals[:5]):
                print(f"  {i + 1}. {sig['symbol']:8s} - Score: {sig['score']:.3f} - Confidence: {sig['confidence']:.3f} - {sig['direction']:8s}")
        print(f"\nResults saved to: {output_file}")
        print("=" * 60 + "\n")

        return filtered_signals

    def _save_signals(self, signals: List[Dict[str, Any]], output_file: str,
                     total_scanned: int = None, total_generated: int = None):
        """Save signals to JSON file with scan metadata"""
        # Ensure directory exists
        output_dir = os.path.dirname(output_file)
        if output_dir and not os.path.exists(output_dir):
            os.makedirs(output_dir, exist_ok=True)

        # Categorize signals
        categorized = {
            'intraday': [],
            'daily': [],
            'monthly': [],
            'all_signals': signals,
            'timestamp': datetime.now().isoformat(),
            'total_count': len(signals),
            'scan_metadata': {
                'total_scanned': total_scanned or len(signals),
                'total_generated': total_generated or len(signals),
                'top_displayed': len(signals),
                'min_confidence_threshold': 0.50,
                'max_signals_limit': 100
            }
        }

        for signal in signals:
            # Categorize based on confidence and score thresholds
            score = signal['score']
            confidence = signal['confidence']

            if confidence >= 0.7 and score >= 0.65:
                categorized['intraday'].append(signal)
            if confidence >= 0.55 and score >= 0.55:
                categorized['daily'].append(signal)
            if confidence >= 0.4 and score >= 0.45:
                categorized['monthly'].append(signal)

        # Save to file
        with open(output_file, 'w') as f:
            json.dump(categorized, f, indent=2)

        print(f"[OK] Saved {len(signals)} top signals ({len(categorized['intraday'])} intraday, {len(categorized['daily'])} daily, {len(categorized['monthly'])} monthly)")

    def get_performance_stats(self) -> Dict[str, Any]:
        """Get overall system performance"""
        return self.tracker.get_performance_stats()

    def add_analyzer(self, analyzer):
        """Add a custom analyzer to the system"""
        self.registry.register(analyzer)

    def __repr__(self):
        return f"<TradingSystem analyzers={self.registry.get_enabled_count()} model={'trained' if self.model.is_trained else 'untrained'}>"
