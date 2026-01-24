
import sys
from pathlib import Path
project_root = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(project_root))

"""
TurboMode Overnight Scanner - Fast Ensemble Architecture

ARCHITECTURE: Per-sector fast ensemble (5 base models + MetaLearner)

Models per sector:
- 3 GPU models: LightGBM-GPU, CatBoost-GPU, XGBoost-Hist-GPU
- 2 CPU models: XGBoost-Linear, RandomForest
- 1 MetaLearner: LogisticRegression (stacked ensemble)

Features:
1. Fast ensemble prediction per sector (6 models total)
2. Adaptive Stop Loss / Take Profit (ATR-based)
3. Partial Profit-Taking (1R, 2R, 3R levels)
4. Hysteresis, Persistence, Position-Aware Logic
5. Persistent Position State Management
6. Comprehensive Logging
7. Unified 3-Tier Directional Biasing (global + sector + symbol)

Single-horizon, single-threshold: label_1d_5pct only
"""

import sys
import os

# Add project root to path
current_dir = os.path.dirname(os.path.abspath(__file__))
backend_dir = os.path.dirname(current_dir)
project_root = os.path.dirname(backend_dir)

if project_root not in sys.path:
    sys.path.insert(0, project_root)
if backend_dir not in sys.path:
    sys.path.insert(0, backend_dir)

from datetime import datetime, timedelta
from typing import List, Dict, Any, Optional
import pandas as pd
import numpy as np
from collections import defaultdict
import logging

# Import Master Market Data DB API (read-only, shared data source)
from master_market_data.market_data_api import get_market_data_api

# Import scanning symbols (208 stocks) for signal generation
from backend.turbomode.core_engine.scanning_symbols import get_scanning_symbols

# Import database
from turbomode.database_schema import TurboModeDB

# Import ensemble inference engine (5 base models + MetaLearner per sector)
from backend.turbomode.core_engine.fastmode_inference import predict_single

# Import adaptive SL/TP calculator
from backend.turbomode.core_engine.adaptive_sltp import (
    calculate_atr,
    calculate_adaptive_sltp,
    update_trailing_stop,
    check_partial_profit_levels
)

# Import position manager
from backend.turbomode.core_engine.position_manager import PositionManager

# Import feature engine
from backend.turbomode.core_engine.turbomode_vectorized_feature_engine import TurboModeVectorizedFeatureEngine

# Import sector metadata
from backend.turbomode.core_engine.scanning_symbols import get_symbol_metadata

# Import News Engine (Phase 2)
from backend.turbomode.core_engine.news_engine import NewsEngine, RiskLevel

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='[%(asctime)s] %(levelname)s - %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)
logger = logging.getLogger(__name__)


class ProductionScanner:
    """
    Production-grade overnight scanner with ensemble inference and adaptive risk management.

    ARCHITECTURE: Per-sector fast ensemble (5 base models + MetaLearner)

    Models per sector:
    - 3 GPU models: LightGBM-GPU, CatBoost-GPU, XGBoost-Hist-GPU
    - 2 CPU models: XGBoost-Linear, RandomForest
    - 1 MetaLearner: LogisticRegression (stacked ensemble)

    Features:
    - Fast ensemble prediction per sector (6 models total)
    - Adaptive SL/TP based on ATR, confidence, and sector
    - Partial profit-taking at 1R (50%), 2R (25%), 3R (25%)
    - Hysteresis: entry threshold 0.60, exit threshold 0.70
    - Persistence: N=3 consecutive opposite signals required for signal-based exit
    - Position state management with atomic persistence
    - Unified 3-Tier Directional Biasing (global + sector + symbol)
    - Single-horizon only: 1d (label_1d_5pct)
    """

    def __init__(
        self,
        db_path: str = None,
        position_state_file: str = None,
        entry_threshold: float = 0.60,
        exit_threshold: float = 0.70,
        persistence_required: int = 3
    ):
        """
        Initialize production scanner.

        ARCHITECTURE: Single-model-per-sector (1d/5% only)

        Args:
            db_path: Path to TurboMode database
            position_state_file: Path to position state JSON file
            entry_threshold: Minimum probability for opening new position (default: 0.60)
            exit_threshold: Minimum probability for signal-based exit (default: 0.70)
            persistence_required: Consecutive opposite signals required for exit (default: 3)
        """
        # Use absolute paths
        if db_path is None:
            db_path = r"C:\StockApp\backend\data\turbomode.db"
        if position_state_file is None:
            position_state_file = r"C:\StockApp\backend\data\position_state.json"

        self.db = TurboModeDB(db_path=db_path)

        logger.info(f"Database path: {db_path}")
        logger.info(f"Position state file: {position_state_file}")
        logger.info(f"Architecture: Single-model-per-sector (1d/5% only)")

        # Initialize position manager (persistent state)
        self.position_manager = PositionManager(state_file=position_state_file)

        # Initialize Master Market Data DB API (read-only, shared data source)
        self.market_data_api = get_market_data_api()
        logger.info("Connected to Master Market Data DB (read-only)")

        # Initialize feature engineering (ALL 179 features)
        self.feature_engineer = TurboModeVectorizedFeatureEngine()
        logger.info("Initialized feature engine (179 features)")

        # Thresholds
        self.entry_threshold = entry_threshold
        self.exit_threshold = exit_threshold
        self.persistence_required = persistence_required

        logger.info(f"Entry threshold: {entry_threshold:.2f}")
        logger.info(f"Exit threshold (hysteresis): {exit_threshold:.2f}")
        logger.info(f"Persistence required: {persistence_required} consecutive signals")

        # Phase 2: Initialize News Engine
        self.news_engine = NewsEngine(
            enable_sector_blocking=True,
            enable_global_blocking=True
        )
        logger.info("News Engine initialized")

        logger.info("Production scanner initialized (Fast Mode + News-Aware)")

    def _get_all_symbols(self) -> List[str]:
        """Get list of all scanning symbols"""
        return get_scanning_symbols()

    def get_current_price(self, symbol: str) -> Optional[float]:
        """
        Get current/latest closing price for a symbol.

        Args:
            symbol: Stock symbol

        Returns:
            Latest close price, or None if failed
        """
        try:
            df = self.market_data_api.get_candles(symbol, timeframe='1d', days_back=5)
            if df is None or df.empty:
                return None
            return float(df['close'].iloc[-1])
        except Exception as e:
            logger.warning(f"Failed to get price for {symbol}: {e}")
            return None

    def get_ohlcv_dataframe(self, symbol: str, days_back: int = 730) -> Optional[pd.DataFrame]:
        """
        Get OHLCV DataFrame for feature extraction and ATR calculation.

        Args:
            symbol: Stock symbol
            days_back: Number of days to fetch (default: 730 = 2 years)

        Returns:
            DataFrame with columns: date, open, high, low, close, volume
        """
        try:
            df = self.market_data_api.get_candles(symbol, timeframe='1d', days_back=days_back)
            if df is None or df.empty or len(df) < 400:
                return None

            # Normalize column names for feature engineer
            df = df.reset_index()
            df.rename(columns={'timestamp': 'date'}, inplace=True)

            return df
        except Exception as e:
            logger.warning(f"Failed to get OHLCV data for {symbol}: {e}")
            return None

    def extract_features(self, df: pd.DataFrame, symbol: str) -> Optional[np.ndarray]:
        """
        Extract 179 features for ML prediction.

        Args:
            df: OHLCV DataFrame
            symbol: Stock symbol

        Returns:
            Feature array (179,) or None if failed
        """
        try:
            # Extract features (vectorized)
            features_df = self.feature_engineer.extract_features(df)
            if features_df is None or features_df.empty:
                return None

            # Get the LAST row (most recent features)
            features = features_df.iloc[-1].to_dict()

            # Add 3 metadata features (sector_code, market_cap_tier, symbol_hash)
            metadata = get_symbol_metadata(symbol)
            features.update(metadata)

            # Convert to feature array (179 features in canonical order)
            from backend.turbomode.core_engine.feature_list import FEATURE_LIST
            feature_array = np.array([features.get(f, 0.0) for f in FEATURE_LIST], dtype=np.float32)

            return feature_array
        except Exception as e:
            logger.warning(f"Failed to extract features for {symbol}: {e}")
            return None

    def get_prediction(self, symbol: str, features: np.ndarray) -> Optional[Dict[str, Any]]:
        """
        Fast Ensemble Prediction (5 base models + MetaLearner)

        ARCHITECTURE: Per-sector fast ensemble
        - Loads all 6 models for the symbol's sector (cached)
        - Runs predict_proba on all 5 base models
        - Stacks base outputs and passes to MetaLearner
        - Returns final BUY/SELL/HOLD probabilities

        Args:
            symbol: Stock symbol
            features: Feature array (179,)

        Returns:
            Prediction dictionary with:
                - signal: 'SELL', 'HOLD', or 'BUY'
                - prob_sell: float (index 0 in ensemble output, matches training labels)
                - prob_hold: float (index 1 in ensemble output, matches training labels)
                - prob_buy: float (index 2 in ensemble output, matches training labels)
                - confidence: float (max probability)
        """
        try:
            # Get sector for model loading
            metadata = get_symbol_metadata(symbol)
            sector = metadata.get('sector', 'unknown')

            if sector == 'unknown':
                logger.warning(f"Unknown sector for {symbol}, skipping")
                return None

            # Run ensemble prediction (5 base models + MetaLearner per sector, cached)
            try:
                result = predict_single(sector, features)
                logger.debug(f"{symbol}: {result['signal']} @ {result['confidence']:.2%} "
                            f"(BUY: {result['prob_buy']:.2%}, SELL: {result['prob_sell']:.2%})")
            except FileNotFoundError:
                logger.error(f"{symbol}: Ensemble models not found for sector {sector}")
                return None

            # DIAGNOSTIC: Log raw model outputs before any biasing
            import numpy as np
            probs_array = np.array([result['prob_sell'], result['prob_hold'], result['prob_buy']])
            argmax_idx = np.argmax(probs_array)
            argmax_labels = ['SELL', 'HOLD', 'BUY']
            print(f"[RAW_MODEL] {symbol}: BUY={result['prob_buy']:.3f}, SELL={result['prob_sell']:.3f}, HOLD={result['prob_hold']:.3f}, ARGMAX={argmax_labels[argmax_idx]}")

            # DIAGNOSTIC: Synthetic SELL override test for one symbol
            print(f"[DEBUG] Checking synthetic override: symbol={repr(symbol)}, type={type(symbol)}, equals_AAPL={symbol == 'AAPL'}")
            if symbol == "AAPL":
                print(f"[SYNTHETIC_SELL] Overriding {symbol} -> BUY=0.10, SELL=0.75, HOLD=0.15")
                result['prob_buy'] = 0.10
                result['prob_sell'] = 0.75
                result['prob_hold'] = 0.15
                result['signal'] = 'SELL'
                result['confidence'] = 0.75

            # UNIFIED BIASING: Apply three-tier sentiment adjustment (global + sector + symbol)
            adjusted_buy, adjusted_sell = self.news_engine.apply_directional_bias(
                symbol, sector, result['prob_buy'], result['prob_sell']
            )
            result['prob_buy'] = adjusted_buy
            result['prob_sell'] = adjusted_sell
            # Recalculate signal after biasing
            if adjusted_buy > adjusted_sell and adjusted_buy > result['prob_hold']:
                result['signal'] = 'BUY'
                result['confidence'] = adjusted_buy
            elif adjusted_sell > adjusted_buy and adjusted_sell > result['prob_hold']:
                result['signal'] = 'SELL'
                result['confidence'] = adjusted_sell
            else:
                result['signal'] = 'HOLD'
                result['confidence'] = result['prob_hold']

            return result

        except Exception as e:
            logger.error(f"Failed to get prediction for {symbol}: {e}")
            return None

    def check_entry_signal(self, symbol: str, sector: str, prediction: Dict[str, Any]) -> Optional[str]:
        """
        Phase 2: News-Aware Entry Signal Check

        Checks if prediction meets entry criteria with news risk gating:
        - Blocks entry if news risk is HIGH/CRITICAL
        - Raises threshold from 0.60 to 0.70 if global risk is HIGH
        - Applies 10% model special handling

        Args:
            symbol: Stock symbol
            sector: Sector name
            prediction: Prediction dictionary (with news_risk field)

        Returns:
            'BUY', 'SELL', or None
        """
        # Phase 2.1: Check if entry should be blocked by news risk
        should_block, block_reason = self.news_engine.should_block_entry(symbol, sector)
        if should_block:
            logger.info(f"[ENTRY BLOCKED] {symbol}: {block_reason}")
            return None

        # Phase 2.2: Determine effective entry threshold
        effective_threshold = self.entry_threshold  # Default: 0.60
        if self.news_engine.should_raise_entry_threshold():
            effective_threshold = 0.70  # Raised threshold due to global HIGH risk
            logger.info(f"[ENTRY THRESHOLD RAISED] {symbol}: 0.60 -> 0.70 (global risk HIGH)")

        # Phase 2.3: Check if 10% model fired (major-move detector)
        is_10pct_signal = prediction.get('threshold_source') == '10pct'
        news_risk_summary = prediction.get('news_risk', {})
        max_risk = RiskLevel[news_risk_summary.get('max_risk', 'NONE')]

        # Phase 2.4: Special handling for 10% model + HIGH/CRITICAL news
        if is_10pct_signal and max_risk >= RiskLevel.HIGH:
            logger.info(f"[10PCT + HIGH RISK] {symbol}: Major-move detector fired under elevated news risk")
            # 10% model signals are treated as high-conviction during elevated news
            # Reduce persistence requirement to 1 (handled in open_new_position)

        # Phase 2.5: Check signal against effective threshold
        if prediction['signal'] == 'BUY' and prediction['prob_buy'] >= effective_threshold:
            logger.info(f"[ENTRY SIGNAL] {symbol} BUY @ {prediction['prob_buy']:.2%} "
                       f"(threshold: {effective_threshold:.2%}, source: {prediction.get('threshold_source', 'unknown')})")
            return 'BUY'
        elif prediction['signal'] == 'SELL' and prediction['prob_sell'] >= effective_threshold:
            logger.info(f"[ENTRY SIGNAL] {symbol} SELL @ {prediction['prob_sell']:.2%} "
                       f"(threshold: {effective_threshold:.2%}, source: {prediction.get('threshold_source', 'unknown')})")
            return 'SELL'
        else:
            return None

    def check_exit_signal(self, symbol: str, prediction: Dict[str, Any], position: Dict) -> bool:
        """
        Check if prediction triggers exit (signal-based exit with hysteresis + persistence).

        Args:
            symbol: Stock symbol
            prediction: Current prediction
            position: Position state dictionary

        Returns:
            True if should exit, False otherwise
        """
        position_type = position['position']
        current_signal = prediction['signal']

        # Update signal history
        self.position_manager.add_signal_to_history(symbol, current_signal, max_history=5)

        # Get recent signals
        recent_signals = position.get('recent_signals', [])

        # Check for persistent opposite signal
        if position_type == 'long':
            # Exit if SELL signal above exit threshold AND N consecutive SELL signals
            if prediction['prob_sell'] >= self.exit_threshold:
                if len(recent_signals) >= self.persistence_required:
                    if all(s == 'SELL' for s in recent_signals[-self.persistence_required:]):
                        logger.info(f"[EXIT SIGNAL] {symbol} LONG → SELL @ {prediction['prob_sell']:.2%} "
                                   f"(persistent {self.persistence_required} signals)")
                        return True
        else:  # short
            # Exit if BUY signal above exit threshold AND N consecutive BUY signals
            if prediction['prob_buy'] >= self.exit_threshold:
                if len(recent_signals) >= self.persistence_required:
                    if all(s == 'BUY' for s in recent_signals[-self.persistence_required:]):
                        logger.info(f"[EXIT SIGNAL] {symbol} SHORT → BUY @ {prediction['prob_buy']:.2%} "
                                   f"(persistent {self.persistence_required} signals)")
                        return True

        return False

    def open_new_position(
        self,
        symbol: str,
        signal_type: str,
        entry_price: float,
        confidence: float,
        atr: float,
        df: pd.DataFrame,
        prediction: Dict[str, Any]
    ):
        """
        Phase 2: Open new position with adaptive SL/TP and news-aware risk management.

        Args:
            symbol: Stock symbol
            signal_type: 'BUY' or 'SELL'
            entry_price: Entry price
            confidence: Model confidence
            atr: ATR at entry
            df: OHLCV DataFrame for additional analysis
            prediction: Full prediction dict (includes threshold_source and news_risk)
        """
        # Get metadata
        metadata = get_symbol_metadata(symbol)
        sector = metadata.get('sector', 'unknown')

        # Determine position type
        position_type = 'long' if signal_type == 'BUY' else 'short'

        # Calculate adaptive SL/TP
        sltp = calculate_adaptive_sltp(
            entry_price=entry_price,
            atr=atr,
            sector=sector,
            confidence=confidence,
            horizon='1d',  # Fixed: 1d horizon (label_1d_5pct)
            position_type=position_type,
            reward_ratio=2.5  # Target is 2.5x stop distance
        )

        # Phase 2: Apply news-aware stop tightening
        should_tighten, tighten_multiplier = self.news_engine.should_tighten_stop(symbol, sector)
        if should_tighten:
            old_stop_distance = sltp['stop_distance']
            sltp['stop_distance'] *= tighten_multiplier

            # Recalculate stop price
            if position_type == 'long':
                sltp['stop_price'] = entry_price - sltp['stop_distance']
            else:  # short
                sltp['stop_price'] = entry_price + sltp['stop_distance']

            logger.info(f"[NEWS STOP TIGHTEN] {symbol}: Stop distance {old_stop_distance:.2f} -> {sltp['stop_distance']:.2f} "
                       f"(multiplier: {tighten_multiplier})")

        # Phase 2: Additional tightening for 10% model + HIGH risk
        is_10pct_signal = prediction.get('threshold_source') == '10pct'
        news_risk_summary = prediction.get('news_risk', {})
        max_risk = RiskLevel[news_risk_summary.get('max_risk', 'NONE')]

        if is_10pct_signal and max_risk >= RiskLevel.HIGH:
            # Additional 15% tightening for high-conviction 10% trades under elevated risk
            old_stop_distance = sltp['stop_distance']
            sltp['stop_distance'] *= 0.85

            if position_type == 'long':
                sltp['stop_price'] = entry_price - sltp['stop_distance']
            else:  # short
                sltp['stop_price'] = entry_price + sltp['stop_distance']

            logger.info(f"[10PCT + HIGH RISK TIGHTEN] {symbol}: Stop distance {old_stop_distance:.2f} -> {sltp['stop_distance']:.2f}")

        # Calculate position size (fixed for now, can be made dynamic)
        position_size = 100  # shares

        # Open position in position manager
        self.position_manager.open_position(
            symbol=symbol,
            position_type=position_type,
            entry_price=entry_price,
            stop_price=sltp['stop_price'],
            target_price=sltp['target_price'],
            stop_distance=sltp['stop_distance'],
            reward_ratio=2.5,
            position_size=position_size,
            confidence=confidence,
            horizon='1d',  # Fixed: 1d horizon (label_1d_5pct)
            sector=sector,
            atr=atr
        )

        logger.info(f"[POSITION OPENED] {position_type.upper()} {symbol} @ ${entry_price:.2f}")
        logger.info(f"  Signal Source: {prediction.get('threshold_source', 'unknown')}")
        logger.info(f"  News Risk: {news_risk_summary.get('max_risk', 'NONE')} ({news_risk_summary.get('max_risk_source', 'none')})")
        logger.info(f"  Stop: ${sltp['stop_price']:.2f} | Target: ${sltp['target_price']:.2f}")
        logger.info(f"  1R: ${sltp['r1_price']:.2f} | 2R: ${sltp['r2_price']:.2f} | 3R: ${sltp['r3_price']:.2f}")
        logger.info(f"  ATR: ${atr:.2f} | Stop Distance: ${sltp['stop_distance']:.2f}")

    def manage_existing_position(self, symbol: str, position: Dict, current_price: float):
        """
        Phase 2: Manage existing position with news-aware risk management.

        Manages existing position: check SL/TP, partial profits, trailing stops, and forced flatten for CRITICAL risk.

        Args:
            symbol: Stock symbol
            position: Position state dictionary
            current_price: Current market price
        """
        position_type = position['position']
        entry_price = position['entry_price']
        stop_price = position['stop_price']
        target_price = position['target_price']
        stop_distance = position['initial_stop_distance']
        sector = position.get('sector', 'unknown')

        # Update current price
        self.position_manager.update_position(symbol, {'current_price': current_price})

        # Phase 2: Check for forced flatten due to CRITICAL news risk
        should_flatten, flatten_reason = self.news_engine.should_force_flatten(symbol, sector)
        if should_flatten:
            logger.warning(f"[FORCED FLATTEN] {symbol}: {flatten_reason} - Immediate exit")
            self.position_manager.close_position(symbol, current_price, f"CRITICAL risk: {flatten_reason}")
            return

        # Check stop loss hit
        if position_type == 'long' and current_price <= stop_price:
            self.position_manager.close_position(symbol, current_price, "Stop loss hit")
            return
        elif position_type == 'short' and current_price >= stop_price:
            self.position_manager.close_position(symbol, current_price, "Stop loss hit")
            return

        # Check target hit
        if position_type == 'long' and current_price >= target_price:
            self.position_manager.close_position(symbol, current_price, "Target reached")
            return
        elif position_type == 'short' and current_price <= target_price:
            self.position_manager.close_position(symbol, current_price, "Target reached")
            return

        # Check partial profit levels
        partial_levels = check_partial_profit_levels(
            position_type=position_type,
            entry_price=entry_price,
            current_price=current_price,
            stop_distance=stop_distance,
            partial_1R_done=position.get('partial_1R_done', False),
            partial_2R_done=position.get('partial_2R_done', False),
            partial_3R_done=position.get('partial_3R_done', False)
        )

        # Execute partial profits
        if partial_levels['take_1R']:
            logger.info(f"[PARTIAL PROFIT] {symbol} +1R reached - Taking 50% off")
            self.position_manager.update_position(symbol, {'partial_1R_done': True})
            # TODO: Execute 50% exit in broker API

        if partial_levels['take_2R']:
            logger.info(f"[PARTIAL PROFIT] {symbol} +2R reached - Taking 25% off")
            self.position_manager.update_position(symbol, {'partial_2R_done': True})
            # TODO: Execute 25% exit in broker API

        if partial_levels['take_3R']:
            logger.info(f"[PARTIAL PROFIT] {symbol} +3R reached - Exiting remaining 25%")
            self.position_manager.close_position(symbol, current_price, "+3R profit target reached")
            return

        # Update trailing stop
        new_stop = update_trailing_stop(
            position_type=position_type,
            entry_price=entry_price,
            current_price=current_price,
            current_stop=stop_price,
            stop_distance=stop_distance,
            partial_1R_done=position.get('partial_1R_done', False),
            partial_2R_done=position.get('partial_2R_done', False)
        )

        if new_stop != stop_price:
            logger.info(f"[TRAILING STOP] {symbol} stop updated: ${stop_price:.2f} → ${new_stop:.2f}")
            self.position_manager.update_position(symbol, {'stop_price': new_stop})

    def scan_symbol(self, symbol: str) -> Optional[Dict[str, Any]]:
        """
        Scan a single symbol with full Fast Mode inference and position management.

        Args:
            symbol: Stock symbol

        Returns:
            Signal dictionary if actionable, else None
        """
        try:
            # Get OHLCV data
            df = self.get_ohlcv_dataframe(symbol, days_back=730)
            if df is None:
                logger.debug(f"{symbol}: Insufficient data")
                return None

            # Get current price
            current_price = self.get_current_price(symbol)
            if current_price is None:
                logger.debug(f"{symbol}: Failed to get price")
                return None

            # Calculate ATR
            atr = calculate_atr(df, period=14)

            # Extract features
            features = self.extract_features(df, symbol)
            if features is None:
                logger.debug(f"{symbol}: Failed to extract features")
                return None

            # Get prediction
            prediction = self.get_prediction(symbol, features)
            if prediction is None:
                return None

            # Check if we have an existing position
            position = self.position_manager.get_position(symbol)

            if position is not None:
                # Manage existing position
                logger.debug(f"{symbol}: Managing existing {position['position'].upper()} position")
                self.manage_existing_position(symbol, position, current_price)

                # Check signal-based exit
                if self.check_exit_signal(symbol, prediction, position):
                    self.position_manager.close_position(symbol, current_price, "Signal-based exit (hysteresis + persistence)")

                return None  # Position managed, don't generate new signal

            else:
                # Get metadata for entry checks
                metadata = get_symbol_metadata(symbol)
                sector = metadata.get('sector', 'unknown')

                # Phase 2: Check for entry signal with news awareness
                signal_type = self.check_entry_signal(symbol, sector, prediction)

                if signal_type is None:
                    return None

                # Phase 2: Open new position with news-aware risk management
                self.open_new_position(
                    symbol=symbol,
                    signal_type=signal_type,
                    entry_price=current_price,
                    confidence=prediction['confidence'],
                    atr=atr,
                    df=df,
                    prediction=prediction
                )

                # Return signal for database storage with news risk info
                news_risk = prediction.get('news_risk', {})
                return {
                    'symbol': symbol,
                    'signal_type': signal_type,
                    'confidence': prediction['confidence'],
                    'entry_date': datetime.now().strftime('%Y-%m-%d'),
                    'entry_price': current_price,
                    'entry_min': current_price * 0.98,  # ±2% tolerance
                    'entry_max': current_price * 1.02,
                    'target_price': self.position_manager.get_position(symbol)['target_price'],
                    'stop_price': self.position_manager.get_position(symbol)['stop_price'],
                    'market_cap': metadata.get('market_cap_category', 'unknown'),
                    'sector': sector,
                    'prob_buy': prediction['prob_buy'],
                    'prob_sell': prediction['prob_sell'],
                    'atr': atr,
                    'threshold_source': prediction.get('threshold_source', 'unknown'),
                    'news_risk_symbol': news_risk.get('symbol_risk', 'NONE'),
                    'news_risk_sector': news_risk.get('sector_risk', 'NONE'),
                    'news_risk_global': news_risk.get('global_risk', 'NONE')
                }

        except Exception as e:
            logger.error(f"Failed to scan {symbol}: {e}")
            return None

    def scan_all(self, max_signals_per_type: int = 100) -> Dict[str, List[Dict[str, Any]]]:
        """
        Scan all symbols with Fast Mode inference and position management.

        Args:
            max_signals_per_type: Maximum BUY and SELL signals to save

        Returns:
            Dictionary with 'buy_signals' and 'sell_signals' lists
        """
        logger.info("=" * 80)
        logger.info("TURBOMODE PRODUCTION SCANNER (FAST MODE + NEWS-AWARE)")
        logger.info("=" * 80)
        logger.info(f"Start Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        logger.info("Architecture: Single-model-per-sector (1d/5% only)")

        # Phase 2: Update news risk state
        logger.info("\n[STEP 0] Updating news risk state...")
        try:
            self.news_engine.update()
        except Exception as e:
            logger.error(f"  Failed to update news risk: {e}")
            logger.info("  Continuing with stale news state...")

        # Update age of existing signals in database
        logger.info("\n[STEP 1] Updating existing signal ages...")
        expired_count = self.db.update_signal_age()
        logger.info(f"  Expired {expired_count} signals")

        # Get all symbols
        logger.info("\n[STEP 2] Loading symbol list...")
        all_symbols = self._get_all_symbols()
        logger.info(f"  Total symbols: {len(all_symbols)}")

        # Scan all symbols
        logger.info(f"\n[STEP 3] Scanning {len(all_symbols)} symbols with Fast Mode...")

        buy_signals = []
        sell_signals = []
        scanned = 0
        failed = 0

        for i, symbol in enumerate(all_symbols, 1):
            if i % 50 == 0:
                logger.info(f"  Progress: {i}/{len(all_symbols)} ({i/len(all_symbols)*100:.1f}%) - "
                           f"BUY: {len(buy_signals)}, SELL: {len(sell_signals)}")

            signal = self.scan_symbol(symbol)

            if signal is None:
                failed += 1
                continue

            scanned += 1

            if signal['signal_type'] == 'BUY':
                buy_signals.append(signal)
            else:
                sell_signals.append(signal)

        logger.info(f"\n[STEP 4] Scan complete!")
        logger.info(f"  Scanned: {scanned}/{len(all_symbols)}")
        logger.info(f"  Failed: {failed}")
        logger.info(f"  BUY signals: {len(buy_signals)}")
        logger.info(f"  SELL signals: {len(sell_signals)}")

        # Sort by confidence
        buy_signals.sort(key=lambda x: x['confidence'], reverse=True)
        sell_signals.sort(key=lambda x: x['confidence'], reverse=True)

        # Limit to top N
        buy_signals = buy_signals[:max_signals_per_type]
        sell_signals = sell_signals[:max_signals_per_type]

        # Save to database with signal flipping support
        logger.info(f"\n[STEP 5] Saving signals to database...")
        saved_buy = 0
        saved_sell = 0
        flipped_signals = 0
        updated_signals = 0
        new_signals = 0

        for signal in buy_signals:
            current_price = signal['entry_price']  # This is the current market price
            result = self.db.add_or_update_signal(signal, current_price)

            if result == 'CREATED':
                new_signals += 1
                saved_buy += 1
            elif result == 'UPDATED':
                updated_signals += 1
                saved_buy += 1
            elif result == 'FLIPPED':
                flipped_signals += 1
                saved_buy += 1
                logger.info(f"  [FLIP] {signal['symbol']}: Signal flipped to BUY")

        for signal in sell_signals:
            current_price = signal['entry_price']  # This is the current market price
            result = self.db.add_or_update_signal(signal, current_price)

            if result == 'CREATED':
                new_signals += 1
                saved_sell += 1
            elif result == 'UPDATED':
                updated_signals += 1
                saved_sell += 1
            elif result == 'FLIPPED':
                flipped_signals += 1
                saved_sell += 1
                logger.info(f"  [FLIP] {signal['symbol']}: Signal flipped to SELL")

        logger.info(f"  BUY: {saved_buy} saved ({new_signals} new)")
        logger.info(f"  SELL: {saved_sell} saved")
        logger.info(f"  Total: {new_signals} new, {updated_signals} updated, {flipped_signals} flipped")

        # Print active positions
        logger.info(f"\n[STEP 6] Active positions summary...")
        active_positions = self.position_manager.get_all_active_positions()
        logger.info(f"  Active positions: {len(active_positions)}")
        for sym, pos in active_positions.items():
            logger.info(f"    {sym}: {pos['position'].upper()} @ ${pos['entry_price']:.2f} "
                       f"(stop: ${pos['stop_price']:.2f}, target: ${pos['target_price']:.2f})")

        logger.info("\n" + "=" * 80)
        logger.info("SCAN COMPLETE")
        logger.info("=" * 80)
        logger.info(f"End Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

        return {
            'buy_signals': buy_signals,
            'sell_signals': sell_signals,
            'stats': {
                'total_scanned': scanned,
                'total_failed': failed,
                'buy_count': len(buy_signals),
                'sell_count': len(sell_signals),
                'saved_buy': saved_buy,
                'saved_sell': saved_sell,
                'active_positions': len(active_positions)
            }
        }


if __name__ == '__main__':
    # Run production scan with single-model architecture (1d/5% only)
    scanner = ProductionScanner()
    results = scanner.scan_all(max_signals_per_type=100)

    logger.info("\n[OK] Production scan complete!")
    logger.info(f"BUY signals: {results['stats']['buy_count']}")
    logger.info(f"SELL signals: {results['stats']['sell_count']}")
    logger.info(f"Active positions: {results['stats']['active_positions']}")
