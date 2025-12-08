"""
Supreme Leader - Governance and Position Management

Purpose: Apply risk management and position constraints
Input: Fusion agent recommendations
Output: Approved trades with position sizing and risk parameters

Constraints:
- Max positions: 5
- Max position size: 20% of portfolio
- Min score: 65 for new positions
- Min confidence: 0.6
- Risk/reward ratio: minimum 2:1
"""

import json
import os
import sys
import io
from datetime import datetime, timedelta
from pathlib import Path
from typing import List, Dict, Any, Optional
import random

# Fix Windows console encoding for emojis
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')

# Add parent directory to path to import from backend
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'language'))

from language.emoji_codec import EmojiCodec


class SupremeLeader:
    """
    The Supreme Leader enforces governance rules and risk management.
    Final authority on all trading decisions.
    """

    def __init__(self, portfolio_value: float = 100000):
        self.emoji_codec = EmojiCodec()
        self.repository_path = Path("C:/StockApp/agents/repository")
        self.supreme_path = self.repository_path / "supreme_leader"
        self.supreme_path.mkdir(parents=True, exist_ok=True)

        # Portfolio and risk parameters
        self.portfolio_value = portfolio_value
        self.constraints = {
            'max_positions': 5,
            'max_position_size_pct': 20,  # 20% of portfolio
            'min_position_size_pct': 2,   # 2% minimum
            'max_portfolio_risk_pct': 10,  # Max 10% portfolio at risk
            'min_score': 65,              # Minimum score for new positions
            'min_confidence': 0.6,         # Minimum confidence
            'min_risk_reward': 2.0,        # Minimum 2:1 risk/reward
            'max_correlated_positions': 3  # Max positions in same sector/type
        }

        # Current positions (would be loaded from database in production)
        self.current_positions = self.load_current_positions()

    def load_current_positions(self) -> List[Dict[str, Any]]:
        """Load current open positions from file or database"""
        positions_file = self.supreme_path / "current_positions.json"

        if positions_file.exists():
            try:
                with open(positions_file, 'r', encoding='utf-8') as f:
                    return json.load(f)
            except:
                pass

        return []  # No positions

    def save_current_positions(self):
        """Save current positions to file"""
        positions_file = self.supreme_path / "current_positions.json"
        with open(positions_file, 'w', encoding='utf-8') as f:
            json.dump(self.current_positions, f, indent=2, ensure_ascii=False)

    def load_fusion_output(self) -> Optional[Dict[str, Any]]:
        """Load fusion agent recommendations"""
        fusion_file = self.repository_path / "fusion_output.json"

        if not fusion_file.exists():
            print("❌ No fusion output found. Run fusion_agent.py first.")
            return None

        try:
            with open(fusion_file, 'r', encoding='utf-8') as f:
                return json.load(f)
        except Exception as e:
            print(f"❌ Error loading fusion output: {e}")
            return None

    def calculate_position_size(self, symbol: str, score: float, confidence: float) -> Dict[str, Any]:
        """Calculate position size based on Kelly Criterion and constraints"""
        # Kelly fraction (simplified)
        win_probability = confidence
        win_loss_ratio = 2.0  # Assume 2:1 risk/reward
        kelly_fraction = (win_probability * win_loss_ratio - (1 - win_probability)) / win_loss_ratio

        # Apply constraints
        kelly_fraction = max(0, min(kelly_fraction, 0.25))  # Cap at 25%

        # Scale by score
        score_multiplier = (score - 50) / 50  # -1 to 1 scale
        adjusted_fraction = kelly_fraction * (1 + score_multiplier * 0.5)

        # Apply min/max constraints
        position_size_pct = min(
            self.constraints['max_position_size_pct'],
            max(self.constraints['min_position_size_pct'], adjusted_fraction * 100)
        )

        position_size_usd = self.portfolio_value * (position_size_pct / 100)

        return {
            'position_size_pct': position_size_pct,
            'position_size_usd': position_size_usd,
            'kelly_fraction': kelly_fraction,
            'adjusted_fraction': adjusted_fraction
        }

    def calculate_risk_parameters(self, symbol: str, price: float,
                                 position_size: float, is_crypto: bool) -> Dict[str, Any]:
        """Calculate stop loss and take profit levels"""
        # Different risk parameters for stocks vs crypto
        if is_crypto:
            stop_loss_pct = 5.0   # 5% stop loss for crypto
            take_profit_pct = 10.0  # 10% take profit
        else:
            stop_loss_pct = 3.0   # 3% stop loss for stocks
            take_profit_pct = 6.0   # 6% take profit

        stop_loss_price = price * (1 - stop_loss_pct / 100)
        take_profit_price = price * (1 + take_profit_pct / 100)

        # Calculate risk amounts
        risk_amount = position_size * (stop_loss_pct / 100)
        reward_amount = position_size * (take_profit_pct / 100)
        risk_reward_ratio = reward_amount / risk_amount if risk_amount > 0 else 0

        return {
            'stop_loss_price': stop_loss_price,
            'stop_loss_pct': stop_loss_pct,
            'take_profit_price': take_profit_price,
            'take_profit_pct': take_profit_pct,
            'risk_amount': risk_amount,
            'reward_amount': reward_amount,
            'risk_reward_ratio': risk_reward_ratio
        }

    def check_correlation(self, symbol: str) -> bool:
        """Check if adding this position would exceed correlation limits"""
        is_crypto = symbol.endswith('-USD')

        # Count similar positions
        crypto_positions = sum(1 for pos in self.current_positions if pos['symbol'].endswith('-USD'))
        stock_positions = len(self.current_positions) - crypto_positions

        if is_crypto and crypto_positions >= self.constraints['max_correlated_positions']:
            return False
        elif not is_crypto and stock_positions >= self.constraints['max_correlated_positions']:
            return False

        return True

    def evaluate_recommendation(self, recommendation: Dict[str, Any]) -> Dict[str, Any]:
        """Evaluate a single recommendation against constraints"""
        symbol = recommendation['symbol']
        score = recommendation['total_score']
        confidence = recommendation['confidence']
        rec_type = recommendation['recommendation']

        # Initialize evaluation result
        evaluation = {
            'symbol': symbol,
            'approved': False,
            'reason': None,
            'action': None,
            'position_details': None
        }

        # Check if we already have a position
        existing_position = next((p for p in self.current_positions if p['symbol'] == symbol), None)

        if existing_position:
            # Evaluate existing position
            if rec_type in ['strong_sell', 'sell']:
                evaluation['approved'] = True
                evaluation['action'] = 'close'
                evaluation['reason'] = f"Exit signal: {rec_type}"
            elif rec_type == 'hold':
                evaluation['approved'] = True
                evaluation['action'] = 'hold'
                evaluation['reason'] = "Hold existing position"
            else:
                evaluation['approved'] = False
                evaluation['action'] = 'hold'
                evaluation['reason'] = "Already in position"
        else:
            # Evaluate new position
            # Check constraints
            if len(self.current_positions) >= self.constraints['max_positions']:
                evaluation['reason'] = "Max positions reached"
                return evaluation

            if score < self.constraints['min_score']:
                evaluation['reason'] = f"Score too low: {score:.1f} < {self.constraints['min_score']}"
                return evaluation

            if confidence < self.constraints['min_confidence']:
                evaluation['reason'] = f"Confidence too low: {confidence:.2f} < {self.constraints['min_confidence']}"
                return evaluation

            if rec_type not in ['strong_buy', 'buy']:
                evaluation['reason'] = f"Not a buy signal: {rec_type}"
                return evaluation

            if not self.check_correlation(symbol):
                evaluation['reason'] = "Too many correlated positions"
                return evaluation

            # Calculate position details
            is_crypto = symbol.endswith('-USD')

            # Simulate current price (in production, fetch real price)
            price = 100  # Placeholder

            position_sizing = self.calculate_position_size(symbol, score, confidence)
            risk_params = self.calculate_risk_parameters(
                symbol, price, position_sizing['position_size_usd'], is_crypto
            )

            # Check risk/reward ratio
            if risk_params['risk_reward_ratio'] < self.constraints['min_risk_reward']:
                evaluation['reason'] = f"Risk/reward too low: {risk_params['risk_reward_ratio']:.1f}"
                return evaluation

            # Approved!
            evaluation['approved'] = True
            evaluation['action'] = 'open'
            evaluation['reason'] = f"Approved: Score={score:.1f}, Confidence={confidence:.2f}"
            evaluation['position_details'] = {
                'entry_price': price,
                'position_size_pct': position_sizing['position_size_pct'],
                'position_size_usd': position_sizing['position_size_usd'],
                'stop_loss': risk_params['stop_loss_price'],
                'take_profit': risk_params['take_profit_price'],
                'risk_amount': risk_params['risk_amount'],
                'reward_amount': risk_params['reward_amount'],
                'risk_reward_ratio': risk_params['risk_reward_ratio']
            }

        return evaluation

    def govern(self) -> Dict[str, Any]:
        """Main governance function - evaluate all recommendations"""
        print("👑 Supreme Leader reviewing recommendations...")

        # Load fusion output
        fusion_data = self.load_fusion_output()
        if not fusion_data:
            return {'error': 'No fusion data'}

        # Get recommendations
        all_recommendations = fusion_data.get('all_fusions', [])

        if not all_recommendations:
            print("❌ No recommendations to review")
            return {'error': 'No recommendations'}

        print(f"   Reviewing {len(all_recommendations)} recommendations...")
        print(f"   Current positions: {len(self.current_positions)}/{self.constraints['max_positions']}")

        approved_trades = []
        rejected_trades = []

        for recommendation in all_recommendations:
            evaluation = self.evaluate_recommendation(recommendation)

            if evaluation['approved']:
                approved_trades.append({
                    'symbol': evaluation['symbol'],
                    'action': evaluation['action'],
                    'reason': evaluation['reason'],
                    'position_details': evaluation['position_details'],
                    'original_recommendation': recommendation
                })

                # If opening new position, add to current positions
                if evaluation['action'] == 'open':
                    self.current_positions.append({
                        'symbol': evaluation['symbol'],
                        'entry_price': evaluation['position_details']['entry_price'],
                        'position_size': evaluation['position_details']['position_size_usd'],
                        'entry_time': datetime.now().isoformat()
                    })

                print(f"   ✅ {evaluation['symbol']}: {evaluation['action'].upper()} - {evaluation['reason']}")
            else:
                rejected_trades.append({
                    'symbol': evaluation['symbol'],
                    'reason': evaluation['reason'],
                    'score': recommendation['total_score'],
                    'confidence': recommendation['confidence']
                })
                print(f"   ❌ {evaluation['symbol']}: REJECTED - {evaluation['reason']}")

        # Save updated positions
        self.save_current_positions()

        # Calculate portfolio metrics
        total_position_value = sum(p['position_size'] for p in self.current_positions)
        portfolio_utilization = (total_position_value / self.portfolio_value) * 100

        # Prepare output
        governance_output = {
            'timestamp': datetime.now().isoformat(),
            'agent': 'supreme_leader',
            'portfolio_value': self.portfolio_value,
            'current_positions_count': len(self.current_positions),
            'portfolio_utilization_pct': portfolio_utilization,
            'approved_trades': approved_trades,
            'rejected_trades': rejected_trades,
            'constraints_applied': self.constraints,
            'summary': {
                'total_reviewed': len(all_recommendations),
                'approved': len(approved_trades),
                'rejected': len(rejected_trades),
                'open_signals': sum(1 for t in approved_trades if t['action'] == 'open'),
                'close_signals': sum(1 for t in approved_trades if t['action'] == 'close'),
                'hold_signals': sum(1 for t in approved_trades if t['action'] == 'hold')
            }
        }

        # Save governance output
        output_file = self.repository_path / "supreme_leader_output.json"
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(governance_output, f, indent=2, ensure_ascii=False)

        print(f"\n👑 Supreme Leader Verdict:")
        print(f"   Portfolio Value: ${self.portfolio_value:,.2f}")
        print(f"   Utilization: {portfolio_utilization:.1f}%")
        print(f"   Approved: {len(approved_trades)} trades")
        print(f"   Rejected: {len(rejected_trades)} trades")

        if approved_trades:
            print("\n🎯 Approved Actions:")
            for trade in approved_trades[:5]:  # Show top 5
                symbol = trade['symbol']
                action = trade['action']
                if action == 'open' and trade['position_details']:
                    size = trade['position_details']['position_size_pct']
                    print(f"   {symbol}: OPEN {size:.1f}% position")
                else:
                    print(f"   {symbol}: {action.upper()}")

        print(f"\n📁 Governance saved to: {output_file}")

        return governance_output


if __name__ == "__main__":
    # Initialize with portfolio value
    supreme = SupremeLeader(portfolio_value=100000)
    results = supreme.govern()

    if 'error' not in results:
        print("\n👑 Supreme Leader has spoken!")
        print("Next step: Run worker_bee.py to execute approved trades")