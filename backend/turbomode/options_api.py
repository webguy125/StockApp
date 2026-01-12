"""
TurboMode Options Trading API
Provides intelligent options recommendations with Greeks, ML scoring, and profit targets
"""

import sys
import os

# Add backend to path
if 'backend' not in [os.path.basename(p) for p in sys.path]:
    backend_path = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    if backend_path not in sys.path:
        sys.path.insert(0, backend_path)

from flask import Blueprint, jsonify, request
import yfinance as yf
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple
import math
import pickle
import json

# Greeks calculation
from py_vollib.black_scholes import black_scholes as bs
from py_vollib.black_scholes.greeks import analytical as greeks

# Import TurboMode database for TurboOptions predictions
from turbomode.database_schema import TurboModeDB

# TurboOptions Models
import xgboost as xgb
import lightgbm as lgb
import catboost as cb
from sklearn.preprocessing import StandardScaler

# Import hybrid data fetcher (IBKR + yfinance)
try:
    from turbomode.hybrid_data_fetcher import HybridDataFetcher
    HYBRID_FETCHER = HybridDataFetcher()
    print("[OPTIONS API] Using hybrid data fetcher (IBKR + yfinance) - 300x faster options data!")
except ImportError as e:
    print(f"[OPTIONS API] hybrid_data_fetcher not available, using yfinance only: {e}")
    HYBRID_FETCHER = None

# Create Blueprint
options_bp = Blueprint('options', __name__, url_prefix='/api/options')

# Constants from OPTIONS_TRADING_SETTINGS.md
HOLD_PERIOD_DAYS = 14
WIN_THRESHOLD_PCT = 0.10  # +10%
LOSS_THRESHOLD_PCT = -0.05  # -5%
PORTFOLIO_VALUE = 10000
MAX_POSITIONS = 30
POSITION_SIZE = PORTFOLIO_VALUE / MAX_POSITIONS  # $333 per position

# Risk-free rate (approximate US 10-year treasury yield)
RISK_FREE_RATE = 0.04

# Options predictions database
import sqlite3
OPTIONS_DB_PATH = os.path.join(os.path.dirname(__file__), '..', 'data', 'options_logs', 'options_predictions.db')

# TurboOptions Models paths
MODEL_DIR = os.path.join(os.path.dirname(__file__), '..', 'data', 'options_models', 'v1.0')
XGBOOST_MODEL_PATH = os.path.join(MODEL_DIR, 'xgboost_model.pkl')
LIGHTGBM_MODEL_PATH = os.path.join(MODEL_DIR, 'lightgbm_model.pkl')
CATBOOST_MODEL_PATH = os.path.join(MODEL_DIR, 'catboost_model.pkl')
META_LEARNER_PATH = os.path.join(MODEL_DIR, 'meta_learner.pkl')
SCALER_PATH = os.path.join(MODEL_DIR, 'feature_scaler.pkl')
FEATURE_NAMES_PATH = os.path.join(MODEL_DIR, 'feature_names.json')


def log_prediction_to_database(prediction_data: Dict):
    """Log options prediction to database for tracking"""
    try:
        conn = sqlite3.connect(OPTIONS_DB_PATH)
        cursor = conn.cursor()

        cursor.execute("""
            INSERT INTO options_predictions_log (
                symbol, stock_price_entry, turbomode_signal, turbomode_confidence, turbo_target_price, historical_vol_30d,
                option_type, strike, expiration_date, dte, entry_premium, entry_bid, entry_ask,
                entry_delta, entry_gamma, entry_theta, entry_vega, entry_rho, entry_iv,
                entry_open_interest, entry_volume,
                rules_score, turbooptions_score, hybrid_score
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """, (
            prediction_data['symbol'],
            prediction_data['stock_price'],
            prediction_data['turbomode_signal'],
            prediction_data['turbomode_confidence'],
            prediction_data['turbo_target_price'],
            prediction_data['historical_vol'],
            prediction_data['option_type'],
            prediction_data['strike'],
            prediction_data['expiration_date'],
            prediction_data['dte'],
            prediction_data['entry_premium'],
            prediction_data['entry_bid'],
            prediction_data['entry_ask'],
            prediction_data['delta'],
            prediction_data['gamma'],
            prediction_data['theta'],
            prediction_data['vega'],
            prediction_data['rho'],
            prediction_data['iv'],
            prediction_data['open_interest'],
            prediction_data['volume'],
            prediction_data['rules_score'],
            prediction_data.get('turbooptions_score', 0),
            prediction_data.get('hybrid_score', prediction_data['rules_score'])
        ))

        conn.commit()
        conn.close()
        print(f"[LOGGING] Saved prediction for {prediction_data['symbol']} to database")

    except Exception as e:
        print(f"[ERROR] Failed to log prediction: {e}")


class OptionsAnalyzer:
    """Analyzes options chains and recommends optimal strikes using TurboOptions predictions"""

    def __init__(self):
        self.db = TurboModeDB()

        # Load TurboOptions models
        self.models_loaded = False
        self.xgboost_model = None
        self.lightgbm_model = None
        self.catboost_model = None
        self.meta_learner = None
        self.scaler = None
        self.feature_names = None

        try:
            # Check if models exist
            if all(os.path.exists(p) for p in [XGBOOST_MODEL_PATH, LIGHTGBM_MODEL_PATH,
                                                CATBOOST_MODEL_PATH, META_LEARNER_PATH,
                                                SCALER_PATH, FEATURE_NAMES_PATH]):

                # Load models
                with open(XGBOOST_MODEL_PATH, 'rb') as f:
                    self.xgboost_model = pickle.load(f)

                with open(LIGHTGBM_MODEL_PATH, 'rb') as f:
                    self.lightgbm_model = pickle.load(f)

                with open(CATBOOST_MODEL_PATH, 'rb') as f:
                    self.catboost_model = pickle.load(f)

                with open(META_LEARNER_PATH, 'rb') as f:
                    self.meta_learner = pickle.load(f)

                with open(SCALER_PATH, 'rb') as f:
                    self.scaler = pickle.load(f)

                with open(FEATURE_NAMES_PATH, 'r') as f:
                    self.feature_names = json.load(f)

                self.models_loaded = True
                print("[TURBO-OPTIONS] Models loaded successfully")
                print(f"  - XGBoost, LightGBM, CatBoost base models")
                print(f"  - Meta-learner (stacking ensemble)")
                print(f"  - Feature scaler ({len(self.feature_names)} features)")

            else:
                print("[TURBO-OPTIONS] Models not found - using rules-based scoring only")
                print("  Run options_data_collector.py, options_feature_engineer.py, and train_options_ml_model.py to train models")

        except Exception as e:
            print(f"[ERROR] Failed to load TurboOptions models: {e}")
            print("  Falling back to rules-based scoring only")

    def get_turbomode_prediction(self, symbol: str) -> Optional[Dict]:
        """Get TurboMode prediction from all_predictions.json (source of truth)"""
        try:
            # Read from all_predictions.json (updated by scanner)
            import json
            predictions_file = os.path.join(os.path.dirname(__file__), '../data/all_predictions.json')

            if os.path.exists(predictions_file):
                with open(predictions_file, 'r') as f:
                    data = json.load(f)
                    for pred in data.get('predictions', []):
                        if pred['symbol'] == symbol:
                            return {
                                'signal': pred['prediction'].upper(),
                                'confidence': pred['confidence'],
                                'entry_price': pred['current_price'],
                                'target_price': pred['current_price'] * 1.05  # 5% target
                            }

            # If not in predictions file, return None (no fallback)
            print(f"[OPTIONS] {symbol} not found in all_predictions.json")
            return None

        except Exception as e:
            print(f"[ERROR] Failed to get TurboMode prediction for {symbol}: {e}")
            return None

    def generate_fallback_signal(self, symbol: str, ticker: yf.Ticker, current_price: float) -> Optional[Dict]:
        """Generate momentum-based signal for stocks without TurboMode predictions"""
        try:
            print(f"[OPTIONS] Generating fallback signal for {symbol} (not in Top 30)")

            # Get 30-day history
            hist = ticker.history(period="30d")
            if len(hist) < 20:
                return None

            # Calculate momentum indicators
            sma_20 = hist['Close'].rolling(20).mean().iloc[-1]
            sma_5 = hist['Close'].rolling(5).mean().iloc[-1]

            # Calculate recent price changes
            pct_change_5d = ((current_price - hist['Close'].iloc[-6]) / hist['Close'].iloc[-6]) * 100
            pct_change_20d = ((current_price - hist['Close'].iloc[-21]) / hist['Close'].iloc[-21]) * 100

            # Determine bullish score (0-4)
            bullish_score = 0
            if current_price > sma_20: bullish_score += 1
            if sma_5 > sma_20: bullish_score += 1
            if pct_change_5d > 0: bullish_score += 1
            if pct_change_20d > 0: bullish_score += 1

            # Generate signal
            if bullish_score >= 3:
                signal_type = 'BUY'
                confidence = 0.55 + (bullish_score - 3) * 0.05
                expected_move_pct = min(pct_change_20d * 0.3, 5.0)  # Conservative estimate
            elif bullish_score <= 1:
                signal_type = 'SELL'
                confidence = 0.55 + (1 - bullish_score) * 0.05
                expected_move_pct = max(pct_change_20d * 0.3, -5.0)  # Conservative estimate
            else:
                # Neutral - slight bullish bias
                signal_type = 'BUY'
                confidence = 0.50
                expected_move_pct = 2.0

            target_price = current_price * (1 + expected_move_pct / 100)

            return {
                'signal': signal_type,
                'confidence': confidence,
                'entry_price': current_price,
                'target_price': target_price,
                'fallback': True  # Flag to indicate this is a generated signal
            }

        except Exception as e:
            print(f"[ERROR] Failed to generate fallback signal for {symbol}: {e}")
            return None

    def calculate_historical_volatility(self, ticker: yf.Ticker, days: int = 30) -> float:
        """Calculate historical volatility (annualized)"""
        try:
            hist = ticker.history(period=f"{days}d")
            if len(hist) < 2:
                return 0.30  # Default 30% if not enough data

            # Calculate log returns
            log_returns = np.log(hist['Close'] / hist['Close'].shift(1))

            # Annualized volatility
            volatility = log_returns.std() * np.sqrt(252)
            return float(volatility)
        except Exception as e:
            print(f"[ERROR] HV calculation failed: {e}")
            return 0.30

    def predict_option_success(self, option_features: Dict) -> Tuple[float, Dict]:
        """
        Predict probability of option hitting +10% using trained TurboOptions ensemble

        Returns: (probability, breakdown_dict)
        """
        if not self.models_loaded:
            # Fallback: use confidence heuristic
            return 0.5, {'ensemble_unavailable': True}

        try:
            # Engineer features matching training pipeline
            features_dict = {}

            # Greeks
            features_dict['delta'] = option_features.get('delta', 0)
            features_dict['gamma'] = option_features.get('gamma', 0)
            features_dict['theta'] = option_features.get('theta', 0)
            features_dict['vega'] = option_features.get('vega', 0)
            features_dict['rho'] = option_features.get('rho', 0)

            # Option characteristics
            features_dict['strike'] = option_features.get('strike', 0)
            features_dict['stock_price_entry'] = option_features.get('stock_price_entry', 0)
            features_dict['moneyness'] = option_features['strike'] / option_features['stock_price_entry']
            features_dict['dte'] = option_features.get('dte', 35)
            features_dict['entry_premium'] = option_features.get('entry_premium', 0)
            features_dict['distance_to_atm'] = abs(option_features['strike'] - option_features['stock_price_entry'])
            features_dict['entry_iv'] = option_features.get('entry_iv', 0.30)
            features_dict['option_type_encoded'] = 1 if option_features.get('option_type') == 'CALL' else 0

            # Greek derivatives
            features_dict['gamma_dollar'] = features_dict['gamma'] * (features_dict['stock_price_entry'] ** 2)
            features_dict['theta_to_delta_ratio'] = abs(features_dict['theta']) / (abs(features_dict['delta']) + 0.01)
            features_dict['vega_to_premium_ratio'] = features_dict['vega'] / (features_dict['entry_premium'] + 0.01)
            features_dict['delta_adjusted_exposure'] = features_dict['delta'] * features_dict['entry_premium']

            # Volatility
            features_dict['historical_vol_30d'] = option_features.get('historical_vol_30d', 0.30)
            features_dict['hv_iv_ratio'] = features_dict['historical_vol_30d'] / (features_dict['entry_iv'] + 0.01)

            # TurboMode ML
            features_dict['signal_type_encoded'] = 1 if option_features.get('signal_type') == 'BUY' else 0
            features_dict['confidence'] = option_features.get('confidence', 0.5)
            features_dict['expected_move_pct'] = option_features.get('expected_move_pct', 0)
            features_dict['signal_strength'] = features_dict['confidence'] * features_dict['expected_move_pct']
            features_dict['distance_to_target_pct'] = option_features.get('distance_to_target_pct', 0)

            # Rules-based scores
            features_dict['delta_score'] = option_features.get('delta_score', 0)
            features_dict['iv_score'] = option_features.get('iv_score', 0)
            features_dict['alignment_score'] = option_features.get('alignment_score', 0)
            features_dict['liquidity_score'] = option_features.get('liquidity_score', 15)
            features_dict['rules_total_score'] = (features_dict['delta_score'] + features_dict['iv_score'] +
                                                   features_dict['alignment_score'] + features_dict['liquidity_score'])

            # Time features
            features_dict['day_of_week'] = datetime.now().weekday()
            features_dict['month'] = datetime.now().month
            features_dict['dte_binned'] = 1 if 35 <= features_dict['dte'] <= 40 else (2 if features_dict['dte'] > 40 else 0)

            # Log transforms
            features_dict['log_entry_premium'] = np.log1p(features_dict['entry_premium'])
            features_dict['log_strike'] = np.log1p(features_dict['strike'])

            # Create feature vector in correct order
            feature_vector = []
            for feature_name in self.feature_names:
                feature_vector.append(features_dict.get(feature_name, 0))

            # Convert to DataFrame
            X = pd.DataFrame([feature_vector], columns=self.feature_names)

            # Fill missing values with median (use 0 for simplicity)
            X = X.fillna(0)

            # Scale features
            X_scaled = self.scaler.transform(X)

            # Get base model predictions
            xgb_pred = self.xgboost_model.predict_proba(X_scaled)[0, 1]
            lgb_pred = self.lightgbm_model.predict_proba(X_scaled)[0, 1]
            cat_pred = self.catboost_model.predict_proba(X_scaled)[0, 1]

            # Stack predictions
            base_preds = np.array([[xgb_pred, lgb_pred, cat_pred]])

            # Meta-learner prediction
            ensemble_prob = self.meta_learner.predict_proba(base_preds)[0, 1]

            # Breakdown for transparency
            breakdown = {
                'xgboost_prob': float(xgb_pred),
                'lightgbm_prob': float(lgb_pred),
                'catboost_prob': float(cat_pred),
                'ensemble_prob': float(ensemble_prob)
            }

            return ensemble_prob, breakdown

        except Exception as e:
            print(f"[ERROR] TurboOptions prediction failed: {e}")
            import traceback
            traceback.print_exc()
            return 0.5, {'error': str(e)}

    def calculate_expected_move(self, current_price: float, turbomode_confidence: float,
                               hv: float, iv: float) -> float:
        """Calculate expected 14-day price move based on ML + volatility"""
        # Volatility factor for 14 days
        vol_blend = np.sqrt((hv**2 + iv**2) / 2)  # Blend HV and IV
        vol_14d = vol_blend * np.sqrt(14 / 252)  # Scale to 14 days

        # ML adjustment multiplier (higher confidence = higher expected move)
        turbo_multiplier = (turbomode_confidence - 0.5) * 2  # Range: 0 to 1 for 50-100% confidence

        # Expected move
        expected_move_pct = vol_14d * turbo_multiplier
        expected_move_dollars = current_price * expected_move_pct

        return expected_move_dollars

    def calculate_greeks(self, S: float, K: float, T: float, r: float, sigma: float,
                        option_type: str) -> Dict:
        """Calculate Black-Scholes Greeks"""
        try:
            flag = 'c' if option_type.upper() == 'CALL' else 'p'

            # Ensure T > 0 (at least 1 day)
            T = max(T, 1/365)

            # Calculate Greeks
            delta = greeks.delta(flag, S, K, T, r, sigma)
            gamma = greeks.gamma(flag, S, K, T, r, sigma)
            theta = greeks.theta(flag, S, K, T, r, sigma) / 365  # Per day
            vega = greeks.vega(flag, S, K, T, r, sigma) / 100  # Per 1% IV change
            rho = greeks.rho(flag, S, K, T, r, sigma) / 100  # Per 1% rate change

            return {
                'delta': round(delta, 3),
                'gamma': round(gamma, 4),
                'theta': round(theta, 3),
                'vega': round(vega, 3),
                'rho': round(rho, 3)
            }
        except Exception as e:
            print(f"[ERROR] Greeks calculation failed: {e}")
            return {'delta': 0, 'gamma': 0, 'theta': 0, 'vega': 0, 'rho': 0}

    def calculate_rules_score(self, option_data: Dict, target_strike: float,
                             turbomode_signal: str) -> Tuple[int, Dict]:
        """
        Calculate rules-based score (0-100) with breakdown

        Returns: (total_score, breakdown_dict)
        """
        # Component 1: Delta Score (40 points) - Prefer 0.60-0.80 for calls
        delta = abs(option_data.get('delta', 0))
        if 0.60 <= delta <= 0.80:
            delta_score = 40
        elif 0.50 <= delta < 0.60 or 0.80 < delta <= 0.90:
            delta_score = 30
        elif 0.40 <= delta < 0.50 or 0.90 < delta <= 1.0:
            delta_score = 20
        else:
            delta_score = 10

        # Component 2: Liquidity Score (20 points)
        oi = option_data.get('openInterest', 0)
        volume = option_data.get('volume', 0)

        if oi > 1000 and volume > 100:
            liquidity_score = 20
        elif oi > 100 and volume > 10:
            liquidity_score = 15
        elif oi > 50:
            liquidity_score = 10
        else:
            liquidity_score = 5

        # Component 3: IV Score (15 points) - Prefer lower IV (better value)
        iv = option_data.get('impliedVolatility', 0.30)
        if iv < 0.25:
            iv_score = 15
        elif iv < 0.35:
            iv_score = 12
        elif iv < 0.50:
            iv_score = 8
        else:
            iv_score = 4

        # Component 4: Alignment Score (25 points) - How close to predicted target
        strike = option_data.get('strike', 0)
        if target_strike > 0:
            deviation = abs(strike - target_strike) / target_strike
            if deviation < 0.02:  # Within 2%
                alignment_score = 25
            elif deviation < 0.05:  # Within 5%
                alignment_score = 20
            elif deviation < 0.10:  # Within 10%
                alignment_score = 15
            else:
                alignment_score = 5
        else:
            alignment_score = 10  # Neutral if no target

        total_score = min(delta_score + liquidity_score + iv_score + alignment_score, 100)

        breakdown = {
            'delta_score': delta_score,
            'liquidity_score': liquidity_score,
            'iv_score': iv_score,
            'alignment_score': alignment_score,
            'total_rules_score': total_score
        }

        return total_score, breakdown

    def calculate_hybrid_score(self, option_data: Dict, target_strike: float,
                               turbomode_signal: str, turbo_context: Dict) -> Tuple[float, Dict]:
        """
        Calculate hybrid score combining rules (40%) and TurboOptions ensemble (60%)

        Hybrid Score = (0.4 × Rules Score) + (0.6 × ML Probability × 100)

        Returns: (hybrid_score, breakdown_dict)
        """
        # Calculate rules-based score
        rules_score, rules_breakdown = self.calculate_rules_score(option_data, target_strike, turbomode_signal)

        # Get TurboOptions ensemble probability
        if self.models_loaded:
            # Prepare features for TurboOptions prediction
            turbo_features = {
                'delta': option_data.get('delta', 0),
                'gamma': option_data.get('gamma', 0),
                'theta': option_data.get('theta', 0),
                'vega': option_data.get('vega', 0),
                'rho': option_data.get('rho', 0),
                'strike': option_data.get('strike', 0),
                'stock_price_entry': turbo_context.get('stock_price', 0),
                'dte': turbo_context.get('dte', 35),
                'entry_premium': (option_data.get('bid', 0) + option_data.get('ask', 0)) / 2,
                'entry_iv': option_data.get('impliedVolatility', 0.30),
                'option_type': turbo_context.get('option_type', 'CALL'),
                'historical_vol_30d': turbo_context.get('historical_vol', 0.30),
                'signal_type': turbomode_signal,
                'confidence': turbo_context.get('confidence', 0.5),
                'expected_move_pct': turbo_context.get('expected_move_pct', 0),
                'distance_to_target_pct': turbo_context.get('distance_to_target_pct', 0),
                'delta_score': rules_breakdown['delta_score'],
                'iv_score': rules_breakdown['iv_score'],
                'alignment_score': rules_breakdown['alignment_score'],
                'liquidity_score': rules_breakdown['liquidity_score']
            }

            ml_prob, ml_breakdown = self.predict_option_success(turbo_features)
            turbooptions_score = ml_prob * 100  # Convert probability to 0-100 scale

        else:
            # Fallback: use confidence as proxy
            ml_prob = turbo_context.get('confidence', 0.5)
            turbooptions_score = ml_prob * 100
            ml_breakdown = {'ensemble_unavailable': True, 'confidence_proxy': ml_prob}

        # Calculate hybrid score
        hybrid_score = (0.4 * rules_score) + (0.6 * turbooptions_score)

        # Full breakdown
        breakdown = {
            'rules_component': {
                'score': rules_score,
                'weight': 0.4,
                'weighted_contribution': 0.4 * rules_score,
                **rules_breakdown
            },
            'ml_component': {
                'probability': ml_prob,
                'score': turbooptions_score,
                'weight': 0.6,
                'weighted_contribution': 0.6 * turbooptions_score,
                **ml_breakdown
            },
            'hybrid_score': hybrid_score
        }

        return hybrid_score, breakdown

    def analyze_options_chain(self, symbol: str) -> Dict:
        """Main function to analyze options and recommend best strikes"""
        try:
            print(f"[OPTIONS] Analyzing options for {symbol}...")

            # Use hybrid fetcher if available
            if HYBRID_FETCHER:
                # Get current price (fast)
                current_price = HYBRID_FETCHER.get_current_price(symbol)
                if not current_price:
                    return {'error': 'Failed to fetch stock price'}

                # Get options chain (300x faster via IBKR!)
                chain_info = HYBRID_FETCHER.get_options_chain(symbol)
                if not chain_info:
                    return {'error': 'No options available for this symbol'}

                expirations_raw = chain_info['expirations']

                # Convert YYYYMMDD to YYYY-MM-DD if needed
                expirations = []
                for exp in expirations_raw:
                    if '-' in exp:
                        expirations.append(exp)
                    else:
                        expirations.append(f"{exp[:4]}-{exp[4:6]}-{exp[6:8]}")

                # Get historical data for volatility calculation
                hist_data = HYBRID_FETCHER.get_stock_data(symbol, period='1mo', interval='1d')
                if hist_data is None or len(hist_data) < 20:
                    return {'error': 'Insufficient historical data'}

                # Create ticker for fallback signal generation
                ticker = yf.Ticker(symbol)

            else:
                # Fallback to yfinance only
                ticker = yf.Ticker(symbol)

                # Get current price
                hist = ticker.history(period="1d")
                if hist.empty:
                    return {'error': 'Failed to fetch stock price'}
                current_price = float(hist['Close'].iloc[-1])

                # Get options expiration dates
                expirations = ticker.options
                if not expirations:
                    return {'error': 'No options available for this symbol'}

                hist_data = ticker.history(period='1mo')

            # Get TurboOptions prediction (or generate fallback for non-Top-30 stocks)
            turbomode_pred = self.get_turbomode_prediction(symbol)
            if not turbomode_pred:
                # Generate momentum-based signal for stocks without TurboMode predictions
                turbomode_pred = self.generate_fallback_signal(symbol, ticker, current_price)
                if not turbomode_pred:
                    return {'error': f'Unable to generate prediction for {symbol}'}

            # Calculate volatility from historical data
            if HYBRID_FETCHER:
                returns = hist_data['Close'].pct_change().dropna()
                hv = returns.std() * np.sqrt(252)  # Annualized volatility
            else:
                hv = self.calculate_historical_volatility(ticker)

            # Find expiration in 30-45 DTE range
            target_date = datetime.now() + timedelta(days=37)  # Middle of range
            best_expiration = min(expirations,
                                 key=lambda d: abs((datetime.strptime(d, '%Y-%m-%d') - target_date).days))

            dte = (datetime.strptime(best_expiration, '%Y-%m-%d') - datetime.now()).days

            # Get options chain (yfinance for detailed bid/ask/IV data)
            chain = ticker.option_chain(best_expiration)

            # Determine option type based on TurboMode signal
            if turbomode_pred['signal'] == 'BUY':
                option_type = 'CALL'
                options_df = chain.calls
            else:
                option_type = 'PUT'
                options_df = chain.puts

            # Get average IV from chain
            avg_iv = options_df['impliedVolatility'].mean()

            # Calculate expected move
            expected_move = self.calculate_expected_move(
                current_price, turbomode_pred['confidence'], hv, avg_iv
            )

            # Calculate target strike (slightly ITM)
            if option_type == 'CALL':
                target_strike = current_price - (0.5 * expected_move)
            else:
                target_strike = current_price + (0.5 * expected_move)

            # Find target price
            expected_target = turbomode_pred.get('target_price', current_price * (1 + expected_move / current_price))

            # Prepare ML context for hybrid scoring
            turbo_context = {
                'stock_price': current_price,
                'dte': dte,
                'option_type': option_type,
                'historical_vol': hv,
                'confidence': turbomode_pred['confidence'],
                'expected_move_pct': (expected_move / current_price),
                'distance_to_target_pct': ((expected_target - current_price) / current_price) * 100
            }

            # Process options chain
            options_list = []
            for _, row in options_df.iterrows():
                strike = row['strike']

                # Filter by liquidity (skip OI check when markets are closed - OI=0 for all)
                # Only filter if OI data is available AND below threshold
                oi = row.get('openInterest', 0)
                if oi > 0 and oi < 5:
                    continue  # Skip only if we have OI data and it's too low

                # Calculate time to expiration (in years)
                T = dte / 365.0

                # Get IV for this option
                iv = row.get('impliedVolatility', avg_iv)
                if iv <= 0 or np.isnan(iv):
                    iv = avg_iv

                # Calculate Greeks
                greeks_data = self.calculate_greeks(
                    S=current_price,
                    K=strike,
                    T=T,
                    r=RISK_FREE_RATE,
                    sigma=iv,
                    option_type=option_type
                )

                # Build option data (handle NaN values from yfinance)
                volume_val = row.get('volume', 0)
                oi_val = row.get('openInterest', 0)

                option_data = {
                    'strike': float(strike),
                    'last': float(row.get('lastPrice', 0)),
                    'bid': float(row.get('bid', 0)),
                    'ask': float(row.get('ask', 0)),
                    'volume': int(volume_val) if not np.isnan(volume_val) else 0,
                    'openInterest': int(oi_val) if not np.isnan(oi_val) else 0,
                    'impliedVolatility': float(iv),
                    'delta': greeks_data['delta'],
                    'gamma': greeks_data['gamma'],
                    'theta': greeks_data['theta'],
                    'vega': greeks_data['vega'],
                    'rho': greeks_data['rho']
                }

                # Calculate Hybrid Score (Rules 40% + ML 60%)
                hybrid_score, score_breakdown = self.calculate_hybrid_score(
                    option_data, target_strike, turbomode_pred['signal'], turbo_context
                )
                option_data['hybrid_score'] = hybrid_score
                option_data['score_breakdown'] = score_breakdown

                options_list.append(option_data)

            # Sort by Hybrid Score
            options_list.sort(key=lambda x: x['hybrid_score'], reverse=True)

            if not options_list:
                print(f"[DEBUG] No options passed filters for {symbol}")
                print(f"[DEBUG] Total options in chain: {len(options_df)}")
                print(f"[DEBUG] Option type: {option_type}, Expiration: {best_expiration}")
                return {'error': 'No liquid options found'}

            # Best option is highest Hybrid Score
            recommended = options_list[0]
            recommended['recommended'] = True

            # Extract score breakdown for best option
            best_score_breakdown = recommended.get('score_breakdown', {})

            # Calculate profit targets
            entry_price = (recommended['bid'] + recommended['ask']) / 2
            if entry_price <= 0:
                entry_price = recommended['last']

            take_profit_price = entry_price * (1 + WIN_THRESHOLD_PCT)
            stop_loss_price = entry_price * (1 + LOSS_THRESHOLD_PCT)

            # Position sizing
            contracts_to_buy = int(POSITION_SIZE / (entry_price * 100))
            if contracts_to_buy < 1:
                contracts_to_buy = 1

            total_premium = contracts_to_buy * entry_price * 100

            # Profit/Loss calculations
            take_profit_pnl = contracts_to_buy * (take_profit_price - entry_price) * 100
            stop_loss_pnl = contracts_to_buy * (stop_loss_price - entry_price) * 100

            # Breakeven stock price (simplified)
            if option_type == 'CALL':
                breakeven = recommended['strike'] + entry_price
            else:
                breakeven = recommended['strike'] - entry_price

            # Build response
            response = {
                'symbol': symbol,
                'stock_data': {
                    'current_price': round(current_price, 2),
                    'historical_volatility_30d': round(hv, 3),
                    'implied_volatility': round(avg_iv, 3)
                },
                'turbomode_prediction': {
                    'signal': turbomode_pred['signal'],
                    'confidence': round(turbomode_pred['confidence'], 3),
                    'expected_14d_move_pct': round((expected_move / current_price) * 100, 2),
                    'expected_target_price': round(expected_target, 2)
                },
                'recommended_option': {
                    'type': option_type,
                    'strike': recommended['strike'],
                    'expiration': best_expiration,
                    'dte': dte,
                    'premium': round(entry_price, 2),
                    'bid': recommended['bid'],
                    'ask': recommended['ask'],
                    'greeks': {
                        'delta': recommended['delta'],
                        'gamma': recommended['gamma'],
                        'theta': recommended['theta'],
                        'vega': recommended['vega'],
                        'rho': recommended['rho']
                    },
                    'implied_volatility': round(recommended['impliedVolatility'], 3),
                    'open_interest': recommended['openInterest'],
                    'volume': recommended['volume'],
                    'hybrid_score': round(recommended['hybrid_score'], 1),
                    'score_breakdown': best_score_breakdown
                },
                'profit_targets': {
                    'entry_price': round(entry_price, 2),
                    'take_profit_price': round(take_profit_price, 2),
                    'take_profit_pnl': round(take_profit_pnl, 2),
                    'stop_loss_price': round(stop_loss_price, 2),
                    'stop_loss_pnl': round(stop_loss_pnl, 2),
                    'risk_reward_ratio': round(abs(take_profit_pnl / stop_loss_pnl), 2) if stop_loss_pnl != 0 else 0,
                    'breakeven_stock_price': round(breakeven, 2)
                },
                'position_sizing': {
                    'portfolio_value': PORTFOLIO_VALUE,
                    'max_positions': MAX_POSITIONS,
                    'position_size_dollars': POSITION_SIZE,
                    'contracts_to_buy': contracts_to_buy,
                    'total_premium': round(total_premium, 2),
                    'max_loss': round(total_premium, 2)
                },
                'full_options_chain': {
                    'calls' if option_type == 'CALL' else 'puts': options_list[:20]  # Top 20
                }
            }

            print(f"[OPTIONS] [OK] Analysis complete for {symbol}")
            print(f"  Recommended: {option_type} ${recommended['strike']} exp {best_expiration}")
            print(f"  Hybrid Score: {recommended['hybrid_score']:.1f}/100 (Rules: {best_score_breakdown.get('rules_component', {}).get('score', 0):.1f}, ML: {best_score_breakdown.get('ml_component', {}).get('score', 0):.1f})")
            print(f"  Entry: ${entry_price:.2f}, Target: ${take_profit_price:.2f}, Stop: ${stop_loss_price:.2f}")

            # Log prediction to database for tracking
            log_prediction_to_database({
                'symbol': symbol,
                'stock_price': current_price,
                'turbomode_signal': turbomode_pred['signal'],
                'turbomode_confidence': turbomode_pred['confidence'],
                'turbo_target_price': expected_target,
                'historical_vol': hv,
                'option_type': option_type,
                'strike': recommended['strike'],
                'expiration_date': best_expiration,
                'dte': dte,
                'entry_premium': entry_price,
                'entry_bid': recommended['bid'],
                'entry_ask': recommended['ask'],
                'delta': recommended['delta'],
                'gamma': recommended['gamma'],
                'theta': recommended['theta'],
                'vega': recommended['vega'],
                'rho': recommended['rho'],
                'iv': recommended['impliedVolatility'],
                'open_interest': recommended['openInterest'],
                'volume': recommended['volume'],
                'rules_score': best_score_breakdown.get('rules_component', {}).get('score', 0),
                'turbooptions_score': best_score_breakdown.get('ml_component', {}).get('score', 0),
                'hybrid_score': recommended['hybrid_score']
            })

            return response

        except Exception as e:
            print(f"[ERROR] Options analysis failed for {symbol}: {e}")
            import traceback
            traceback.print_exc()
            return {'error': str(e)}


# Create analyzer instance
analyzer = OptionsAnalyzer()


@options_bp.route('/<symbol>', methods=['GET'])
def get_options_analysis(symbol: str):
    """
    Get options analysis for a symbol

    Usage: GET /api/options/AAPL
    """
    symbol = symbol.upper()
    result = analyzer.analyze_options_chain(symbol)

    if 'error' in result:
        return jsonify(result), 400

    return jsonify(result), 200


@options_bp.route('/brief/<symbol>', methods=['GET'])
def get_brief_options(symbol: str):
    """
    Get brief options data for tooltips (lightweight, fast response)

    Usage: GET /api/options/brief/AAPL
    Returns: option_type, strike, delta, turbooptions_score
    """
    symbol = symbol.upper()

    try:
        # Get TurboOptions prediction
        turbomode_pred = analyzer.get_turbomode_prediction(symbol)
        if not turbomode_pred:
            return jsonify({'error': 'No TurboOptions prediction found'}), 404

        # Get ticker
        ticker = yf.Ticker(symbol)

        # Get current price
        hist = ticker.history(period="1d")
        if hist.empty:
            return jsonify({'error': 'Failed to fetch price'}), 400
        current_price = float(hist['Close'].iloc[-1])

        # Get options expirations
        expirations = ticker.options
        if not expirations:
            return jsonify({'error': 'No options available'}), 400

        # Find nearest 30-45 DTE expiration
        from datetime import datetime, timedelta
        target_date = datetime.now() + timedelta(days=37)
        best_expiration = min(expirations,
                             key=lambda d: abs((datetime.strptime(d, '%Y-%m-%d') - target_date).days))

        dte = (datetime.strptime(best_expiration, '%Y-%m-%d') - datetime.now()).days

        # Get options chain
        chain = ticker.option_chain(best_expiration)

        # Determine option type
        if turbomode_pred['signal'] == 'BUY':
            option_type = 'CALL'
            options_df = chain.calls
        else:
            option_type = 'PUT'
            options_df = chain.puts

        # Get average IV
        avg_iv = options_df['impliedVolatility'].mean()

        # Calculate expected move (simplified)
        hv = analyzer.calculate_historical_volatility(ticker)
        expected_move = analyzer.calculate_expected_move(
            current_price, turbomode_pred['confidence'], hv, avg_iv
        )

        # Calculate target strike
        if option_type == 'CALL':
            target_strike = current_price - (0.5 * expected_move)
        else:
            target_strike = current_price + (0.5 * expected_move)

        # Find best option (simplified scoring)
        best_option = None
        best_score = 0

        for _, row in options_df.iterrows():
            strike = row['strike']

            # Filter by liquidity
            if row.get('openInterest', 0) < 50:
                continue

            # Calculate time to expiration
            T = dte / 365.0

            # Get IV
            iv = row.get('impliedVolatility', avg_iv)
            if iv <= 0 or np.isnan(iv):
                iv = avg_iv

            # Calculate Greeks
            greeks_data = analyzer.calculate_greeks(
                S=current_price,
                K=strike,
                T=T,
                r=RISK_FREE_RATE,
                sigma=iv,
                option_type=option_type
            )

            # Build option data
            option_data = {
                'strike': float(strike),
                'delta': greeks_data['delta'],
                'openInterest': int(row.get('openInterest', 0)),
                'volume': int(row.get('volume', 0)),
                'impliedVolatility': float(iv)
            }

            # Calculate TurboOptions Score
            turbooptions_score = analyzer.calculate_turbooptions_score(option_data, target_strike, turbomode_pred['signal'])

            if turbooptions_score > best_score:
                best_score = turbooptions_score
                best_option = {
                    'option_type': option_type,
                    'strike': round(strike, 2),
                    'delta': greeks_data['delta'],
                    'turbooptions_score': turbooptions_score
                }

        if not best_option:
            return jsonify({'error': 'No suitable options found'}), 404

        return jsonify(best_option), 200

    except Exception as e:
        print(f"[ERROR] Brief options failed for {symbol}: {e}")
        return jsonify({'error': str(e)}), 500


@options_bp.route('/performance', methods=['GET'])
def get_performance_dashboard():
    """
    Get performance dashboard data

    Usage: GET /api/options/performance
    Returns: Statistics, charts data, recent predictions
    """
    try:
        conn = sqlite3.connect(OPTIONS_DB_PATH)
        cursor = conn.cursor()

        # Overall statistics
        cursor.execute("""
            SELECT
                COUNT(*) as total,
                SUM(CASE WHEN tracking_complete = 1 THEN 1 ELSE 0 END) as completed,
                SUM(CASE WHEN hit_10pct_target = 1 THEN 1 ELSE 0 END) as hits,
                AVG(CASE WHEN hit_10pct_target = 1 THEN max_profit_pct ELSE NULL END) as avg_win,
                AVG(CASE WHEN hit_10pct_target = 0 AND tracking_complete = 1 THEN max_profit_pct ELSE NULL END) as avg_loss,
                AVG(CASE WHEN hit_10pct_target = 1 THEN days_to_target ELSE NULL END) as avg_days,
                MAX(max_profit_pct) as best,
                AVG(hybrid_score) as avg_hybrid
            FROM options_predictions_log
        """)

        stats_row = cursor.fetchone()

        total = stats_row[0] or 0
        completed = stats_row[1] or 0
        hits = stats_row[2] or 0
        win_rate = (hits / completed * 100) if completed > 0 else 0

        stats = {
            'total_predictions': total,
            'completed_predictions': completed,
            'win_rate': win_rate,
            'avg_win_pct': stats_row[3] or 0,
            'avg_loss_pct': stats_row[4] or 0,
            'avg_days_to_target': stats_row[5] or 0,
            'best_prediction_pct': stats_row[6] or 0,
            'avg_hybrid_score': stats_row[7] or 0
        }

        # Win rate over time (by date)
        cursor.execute("""
            SELECT
                DATE(created_at) as date,
                SUM(CASE WHEN hit_10pct_target = 1 THEN 1 ELSE 0 END) * 100.0 / COUNT(*) as win_rate
            FROM options_predictions_log
            WHERE tracking_complete = 1
            GROUP BY DATE(created_at)
            ORDER BY date
        """)

        win_rate_data = cursor.fetchall()
        win_rate_over_time = {
            'dates': [row[0] for row in win_rate_data],
            'win_rates': [row[1] for row in win_rate_data]
        }

        # Profit distribution
        cursor.execute("""
            SELECT max_profit_pct
            FROM options_predictions_log
            WHERE tracking_complete = 1
        """)

        profits = [row[0] for row in cursor.fetchall() if row[0] is not None]

        # Create bins for histogram
        if profits:
            bins = list(range(-20, 60, 5))  # -20% to +55% in 5% increments
            hist_counts = [0] * (len(bins) - 1)

            for profit in profits:
                for i in range(len(bins) - 1):
                    if bins[i] <= profit < bins[i+1]:
                        hist_counts[i] += 1
                        break

            profit_distribution = {
                'bins': [f"{bins[i]}% to {bins[i+1]}%" for i in range(len(bins)-1)],
                'counts': hist_counts
            }
        else:
            profit_distribution = {'bins': [], 'counts': []}

        # Score vs success (scatter plot data)
        cursor.execute("""
            SELECT hybrid_score, max_profit_pct, hit_10pct_target
            FROM options_predictions_log
            WHERE tracking_complete = 1 AND hybrid_score IS NOT NULL
        """)

        scatter_data = cursor.fetchall()
        score_vs_success = {
            'successes': [{'x': row[0], 'y': row[1]} for row in scatter_data if row[2] == 1],
            'failures': [{'x': row[0], 'y': row[1]} for row in scatter_data if row[2] == 0]
        }

        # Recent predictions
        cursor.execute("""
            SELECT
                created_at, symbol, option_type, strike, entry_premium,
                hybrid_score, max_profit_pct, days_to_target,
                hit_10pct_target, tracking_complete
            FROM options_predictions_log
            ORDER BY created_at DESC
            LIMIT 50
        """)

        predictions = cursor.fetchall()
        recent_predictions = []

        for pred in predictions:
            recent_predictions.append({
                'created_at': pred[0],
                'symbol': pred[1],
                'option_type': pred[2],
                'strike': pred[3],
                'entry_premium': pred[4],
                'hybrid_score': pred[5] or 0,
                'max_profit_pct': pred[6] or 0,
                'days_to_target': pred[7],
                'hit_target': pred[8] == 1,
                'tracking_complete': pred[9] == 1
            })

        conn.close()

        return jsonify({
            'stats': stats,
            'win_rate_over_time': win_rate_over_time,
            'profit_distribution': profit_distribution,
            'score_vs_success': score_vs_success,
            'recent_predictions': recent_predictions
        }), 200

    except Exception as e:
        print(f"[ERROR] Performance dashboard failed: {e}")
        import traceback
        traceback.print_exc()
        return jsonify({'error': str(e)}), 500


@options_bp.route('/health', methods=['GET'])
def health_check():
    """Health check endpoint"""
    return jsonify({'status': 'ok', 'service': 'options_api'}), 200


if __name__ == '__main__':
    # Test
    print("Testing Options API...")
    result = analyzer.analyze_options_chain('AAPL')
    print(result)
