"""
Automated Learning System
Simulates trades, tracks outcomes, and retrains model automatically
NO MANUAL INTERVENTION REQUIRED!
"""

import sys
import os
import json
import yfinance as yf
from datetime import datetime, timedelta
from typing import List, Dict, Any
import time

# Add parent directory to path
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
if parent_dir not in sys.path:
    sys.path.insert(0, parent_dir)

from trading_system.core.trade_tracker import TradeTracker
from trading_system.models.simple_trading_model import SimpleTradingModel
from trading_system.ml_model_manager import MLModelManager


class AutomatedLearner:
    """
    Fully automated learning system

    Workflow:
    1. Reads signals from last scan
    2. Simulates taking trades on top signals
    3. Waits N days and checks outcomes
    4. Marks wins/losses automatically
    5. Retrains model when enough data accumulated
    """

    def __init__(self,
                 hold_period_days: int = 3,
                 win_threshold_pct: float = 2.0,
                 loss_threshold_pct: float = -2.0,
                 max_simulated_positions: int = 10):
        """
        Initialize automated learner

        Args:
            hold_period_days: How many days to hold simulated trades
            win_threshold_pct: % gain to mark as WIN (default: +2%)
            loss_threshold_pct: % loss to mark as LOSS (default: -2%)
            max_simulated_positions: Max simulated trades per scan
        """
        self.tracker = TradeTracker()
        self.model = SimpleTradingModel()
        self.model_manager = MLModelManager()

        # Get active model configuration
        active_config = self.model_manager.get_active_model()
        if active_config:
            # Use configuration from active model
            self.hold_period_days = active_config.get('hold_period_days', hold_period_days)
            self.win_threshold_pct = active_config.get('win_threshold_pct', win_threshold_pct)
            self.loss_threshold_pct = active_config.get('loss_threshold_pct', loss_threshold_pct)
            self.active_model_name = active_config.get('name')
            print(f"[OK] Using active model: {self.active_model_name}")
            print(f"   Hold period: {self.hold_period_days} days")
            print(f"   Win/Loss thresholds: +{self.win_threshold_pct}% / {self.loss_threshold_pct}%")
        else:
            # Use defaults if no active model
            self.hold_period_days = hold_period_days
            self.win_threshold_pct = win_threshold_pct
            self.loss_threshold_pct = loss_threshold_pct
            self.active_model_name = None
            print("[WARNING] No active model configuration - using defaults")

        self.max_positions = max_simulated_positions

        self.signals_file = os.path.join(parent_dir, 'data', 'ml_trading_signals.json')
        self.state_file = os.path.join(parent_dir, 'data', 'automated_learner_state.json')

    def load_signals(self) -> List[Dict[str, Any]]:
        """Load signals from latest scan"""
        if not os.path.exists(self.signals_file):
            print("[WARNING] No signals file found. Run a scan first.")
            return []

        with open(self.signals_file, 'r') as f:
            data = json.load(f)

        return data.get('all_signals', [])

    def load_state(self) -> Dict[str, Any]:
        """Load learner state (simulated positions)"""
        if os.path.exists(self.state_file):
            with open(self.state_file, 'r') as f:
                return json.load(f)
        return {'simulated_positions': []}

    def save_state(self, state: Dict[str, Any]):
        """Save learner state"""
        os.makedirs(os.path.dirname(self.state_file), exist_ok=True)
        with open(self.state_file, 'w') as f:
            json.dump(state, f, indent=2)

    def simulate_new_trades(self):
        """
        Simulate taking trades from latest signals
        Selects top signals and creates simulated positions
        """
        print("\n" + "=" * 60)
        print("AUTOMATED LEARNER - Simulating New Trades")
        print("=" * 60 + "\n")

        # Load signals
        signals = self.load_signals()
        if not signals:
            print("No signals to process.")
            return

        print(f"Found {len(signals)} signals from last scan")

        # Load current state
        state = self.load_state()
        current_positions = state.get('simulated_positions', [])

        print(f"Current simulated positions: {len(current_positions)}")

        # Filter to bullish signals with decent scores
        bullish_signals = [
            s for s in signals
            if s.get('direction') == 'bullish' and s.get('score', 0) > 0.5
        ]

        # Sort by score (best first)
        bullish_signals.sort(key=lambda x: x.get('score', 0), reverse=True)

        print(f"Bullish signals with score >50%: {len(bullish_signals)}")

        # Calculate how many new positions we can take
        available_slots = self.max_positions - len(current_positions)
        if available_slots <= 0:
            print(f"[WARNING] Already at max positions ({self.max_positions})")
            return

        # Take top N signals
        new_signals = bullish_signals[:available_slots]
        print(f"\n[DATA] Simulating {len(new_signals)} new trades:")

        for signal in new_signals:
            symbol = signal['symbol']
            entry_price = signal.get('price', 0)

            if entry_price == 0:
                print(f"[WARNING] Skipping {symbol} - no price data")
                continue

            # Create simulated position
            position = {
                'symbol': symbol,
                'entry_price': entry_price,
                'entry_date': datetime.now().isoformat(),
                'entry_signal': signal,
                'position_size': 1.0,  # Simulated (1 share for simplicity)
                'status': 'open'
            }

            current_positions.append(position)
            print(f"  [OK] {symbol} @ ${entry_price:.2f} (Score: {signal['score']:.1%})")

        # Save state
        state['simulated_positions'] = current_positions
        self.save_state(state)

        print(f"\n[OK] Total simulated positions: {len(current_positions)}/{self.max_positions}")

    def check_outcomes(self):
        """
        Check all open simulated positions
        Mark as win/loss if hold period elapsed
        """
        print("\n" + "=" * 60)
        print("AUTOMATED LEARNER - Checking Outcomes")
        print("=" * 60 + "\n")

        state = self.load_state()
        positions = state.get('simulated_positions', [])

        if not positions:
            print("No simulated positions to check.")
            return

        print(f"Checking {len(positions)} simulated positions...\n")

        now = datetime.now()
        updated_positions = []
        trades_to_record = []

        for pos in positions:
            symbol = pos['symbol']
            entry_price = pos['entry_price']
            entry_date = datetime.fromisoformat(pos['entry_date'])

            # Check if hold period elapsed
            days_held = (now - entry_date).days

            if days_held < self.hold_period_days:
                # Not ready to evaluate yet
                updated_positions.append(pos)
                print(f"  [WAIT] {symbol}: Held {days_held}/{self.hold_period_days} days")
                continue

            # Get current price
            try:
                ticker = yf.Ticker(symbol)
                current_price = ticker.info.get('currentPrice') or ticker.info.get('regularMarketPrice')

                if not current_price:
                    # Try getting last close from history
                    hist = ticker.history(period='1d')
                    if not hist.empty:
                        current_price = hist['Close'].iloc[-1]

                if not current_price:
                    print(f"  [WARNING] {symbol}: Could not get current price, skipping")
                    updated_positions.append(pos)
                    continue

                # Calculate return
                price_change_pct = ((current_price - entry_price) / entry_price) * 100

                # Determine outcome
                if price_change_pct >= self.win_threshold_pct:
                    outcome = 'win'
                    status = '[OK] WIN'
                elif price_change_pct <= self.loss_threshold_pct:
                    outcome = 'loss'
                    status = '[ERROR] LOSS'
                else:
                    outcome = 'loss'  # Didn't hit target = loss
                    status = '[WARNING] NO TARGET'

                print(f"  {status} {symbol}: ${entry_price:.2f} → ${current_price:.2f} ({price_change_pct:+.2f}%)")

                # Record trade for model training
                trades_to_record.append({
                    'symbol': symbol,
                    'entry_price': entry_price,
                    'exit_price': current_price,
                    'outcome': outcome,
                    'signal': pos.get('entry_signal', {})
                })

                # Don't keep in positions (trade closed)

            except Exception as e:
                print(f"  [ERROR] Error checking {symbol}: {e}")
                updated_positions.append(pos)  # Keep it for next check

        # Record all completed trades in database
        if trades_to_record:
            print(f"\n[DATA] Recording {len(trades_to_record)} completed trades...")

            for trade in trades_to_record:
                try:
                    # Save trade to database
                    trade_id = self.tracker.save_trade(
                        symbol=trade['symbol'],
                        entry_price=trade['entry_price'],
                        position_size=1.0,
                        analysis_snapshot=trade.get('signal', {}),
                        features=trade['signal'].get('analyzers', {}),
                        model_name=self.active_model_name
                    )

                    # Close trade
                    self.tracker.close_trade(
                        trade_id=trade_id,
                        exit_price=trade['exit_price'],
                        exit_reason='automated_learning'
                    )

                except Exception as e:
                    print(f"  [WARNING] Error recording {trade['symbol']}: {e}")

        # Update state
        state['simulated_positions'] = updated_positions
        self.save_state(state)

        print(f"\n[OK] Remaining simulated positions: {len(updated_positions)}")

        # Check if we should retrain
        if len(trades_to_record) > 0:
            self.maybe_retrain()

    def maybe_retrain(self):
        """Check if we have enough data to retrain, and retrain if so"""
        stats = self.tracker.get_performance_stats()
        total_trades = stats['total_trades']

        print(f"\n[DATA] Total trades in database: {total_trades}")

        # Retrain if we have at least 10 trades and it's a multiple of 10
        if total_trades >= 10 and total_trades % 10 == 0:
            print(f"\n[TRAIN] RETRAINING MODEL (reached {total_trades} trades)")
            self.retrain_model()

    def retrain_model(self):
        """Retrain the ML model using all completed trades"""
        import numpy as np

        print("\n" + "=" * 60)
        print("AUTOMATED RETRAINING")
        print("=" * 60 + "\n")

        # Get all completed trades
        wins = self.tracker.get_trades(status='win', limit=1000)
        losses = self.tracker.get_trades(status='loss', limit=1000)
        all_trades = wins + losses

        if len(all_trades) < 10:
            print(f"[WARNING] Not enough trades for retraining (need 10, have {len(all_trades)})")
            return

        print(f"Loading {len(all_trades)} trades ({len(wins)} wins, {len(losses)} losses)")

        # Extract features and labels
        features_list = []
        labels_list = []

        for trade in all_trades:
            # Get analyzer data from analysis snapshot
            try:
                import json
                if trade.get('analysis_snapshot'):
                    analysis = json.loads(trade['analysis_snapshot'])
                    analyzers = analysis.get('analyzers', {})

                    # Extract feature vector from analyzers
                    feature_vector = []
                    for analyzer_name in sorted(analyzers.keys()):
                        data = analyzers[analyzer_name]
                        feature_vector.append(data.get('signal_strength', 0.5))
                        feature_vector.append(data.get('confidence', 0.5))
                        direction = data.get('direction', 'neutral')
                        direction_val = 1.0 if direction == 'bullish' else -1.0 if direction == 'bearish' else 0.0
                        feature_vector.append(direction_val)

                    if len(feature_vector) > 0:
                        features_list.append(feature_vector)
                        # Label: 2=BUY (win), 0=SELL (loss)
                        label = 2 if trade['outcome'] == 'win' else 0
                        labels_list.append(label)

            except Exception as e:
                print(f"  [WARNING] Skipping trade {trade['symbol']}: {e}")
                continue

        if len(features_list) < 10:
            print(f"[WARNING] Not enough valid features (need 10, have {len(features_list)})")
            return

        # Convert to numpy
        X = np.array(features_list)
        y = np.array(labels_list)

        print(f"Training on {len(X)} samples")
        print(f"Feature dimensions: {X.shape}")

        # Train
        metrics = self.model.train(X, y)

        print("\n" + "=" * 60)
        print("RETRAINING COMPLETE")
        print("=" * 60)
        print(f"Training Accuracy:   {metrics['train_accuracy']:.2%}")
        print(f"Validation Accuracy: {metrics['val_accuracy']:.2%}")
        print("=" * 60 + "\n")

        print("[OK] Model updated! Next scan will use improved predictions.")

    def run_cycle(self):
        """Run one complete learning cycle"""
        print("\n" + "=" * 80)
        print("AUTOMATED LEARNING SYSTEM - DAILY CYCLE")
        print("=" * 80)
        print(f"Started: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        print("=" * 80)

        # Step 1: Check outcomes of existing positions
        self.check_outcomes()

        # Step 2: Simulate new trades from latest signals
        self.simulate_new_trades()

        # Step 3: Show stats
        stats = self.tracker.get_performance_stats()
        print("\n" + "=" * 60)
        print("LEARNING SYSTEM STATISTICS")
        print("=" * 60)
        print(f"Total Trades:     {stats['total_trades']}")
        print(f"Wins:             {stats['wins']}")
        print(f"Losses:           {stats['losses']}")
        print(f"Win Rate:         {stats['win_rate']:.1f}%")
        print(f"Total P/L:        ${stats['total_profit_loss']:.2f}")
        print(f"Profit Factor:    {stats['profit_factor']:.2f}")
        print(f"Model Status:     {'Trained' if self.model.is_trained else 'Untrained'}")
        print("=" * 60 + "\n")


def main():
    """Run automated learner"""
    import argparse

    parser = argparse.ArgumentParser(description='Automated ML Trading System Learner')
    parser.add_argument('--hold-days', type=int, default=3,
                       help='Days to hold simulated trades (default: 3)')
    parser.add_argument('--win-threshold', type=float, default=2.0,
                       help='Win threshold %% (default: 2.0)')
    parser.add_argument('--loss-threshold', type=float, default=-2.0,
                       help='Loss threshold %% (default: -2.0)')
    parser.add_argument('--max-positions', type=int, default=10,
                       help='Max simulated positions (default: 10)')

    args = parser.parse_args()

    learner = AutomatedLearner(
        hold_period_days=args.hold_days,
        win_threshold_pct=args.win_threshold,
        loss_threshold_pct=args.loss_threshold,
        max_simulated_positions=args.max_positions
    )

    learner.run_cycle()


if __name__ == '__main__':
    try:
        main()
    except KeyboardInterrupt:
        print("\n\n[WARNING] Interrupted by user")
    except Exception as e:
        print(f"\n[ERROR] Error: {e}")
        import traceback
        traceback.print_exc()
