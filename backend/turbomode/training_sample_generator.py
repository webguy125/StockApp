"""
TurboMode Training Sample Generator
Converts tracked signal outcomes into training samples
Adds real-world prediction results to the training dataset

This runs weekly (Sunday 3 AM) to convert accumulated outcomes into training data
"""

import os
import sys
import sqlite3
import json
import uuid
from datetime import datetime, timedelta
from typing import List, Dict, Any
import logging

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger('training_sample_generator')

# Add parent directory to path
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
if parent_dir not in sys.path:
    sys.path.insert(0, parent_dir)


class TrainingSampleGenerator:
    """
    Convert tracked outcomes into training samples
    Bridges the gap between real-world predictions and model training
    """

    def __init__(self, db_path: str = None):
        """
        Initialize training sample generator

        Args:
            db_path: Path to turbomode.db
        """
        if db_path is None:
            db_path = os.path.join(parent_dir, 'data', 'turbomode.db')

        self.db_path = db_path

    def get_unprocessed_outcomes(self) -> List[Dict[str, Any]]:
        """
        Get outcomes from signal_history that haven't been converted to training samples yet

        Returns:
            List of outcome dictionaries
        """
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()

        # Get outcomes that haven't been processed yet
        # We'll mark them as processed by setting a flag
        cursor.execute("""
            SELECT
                id, signal_id, symbol, signal_type, confidence,
                entry_date, entry_price, exit_date, exit_price,
                return_pct, outcome, is_correct,
                sector, market_cap, hold_days, created_at
            FROM signal_history
            WHERE processed_for_training = 0
            ORDER BY created_at ASC
        """)

        rows = cursor.fetchall()
        conn.close()

        outcomes = []
        for row in rows:
            outcomes.append({
                'id': row[0],
                'signal_id': row[1],
                'symbol': row[2],
                'signal_type': row[3],
                'confidence': row[4],
                'entry_date': row[5],
                'entry_price': row[6],
                'exit_date': row[7],
                'exit_price': row[8],
                'return_pct': row[9],
                'outcome': row[10],
                'is_correct': row[11],
                'sector': row[12],
                'market_cap': row[13],
                'hold_days': row[14],
                'created_at': row[15]
            })

        return outcomes

    def get_features_from_signal(self, signal_id: str) -> Dict[str, Any]:
        """
        Retrieve the original features that were used for this prediction
        These features are stored in the active_signals table (now in signal_history)

        Args:
            signal_id: Original signal ID

        Returns:
            Features dictionary or None
        """
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()

        # Try to get features from signal_history (if they were saved)
        # Note: We need to add a features_json column to signal_history in the future
        # For now, we'll need to regenerate features from historical data

        conn.close()
        return None  # TODO: Implement feature retrieval or regeneration

    def convert_outcome_to_training_sample(self, outcome: Dict[str, Any]) -> Dict[str, Any]:
        """
        Convert a tracked outcome into a training sample format

        Args:
            outcome: Outcome from signal_history

        Returns:
            Training sample ready for trades table
        """
        # CORRECTED LABEL LOGIC - Based on ACTUAL price movement, NOT prediction correctness
        # This fixes MISMATCH #2 from TURBOMODE_MISMATCH_AUDIT.md
        #
        # Labels based on actual price movement:
        # - return_pct >= +5%  → label = 'buy'  (price went UP significantly)
        # - return_pct <= -5%  → label = 'sell' (price went DOWN significantly)
        # - -5% < return_pct < +5% → label = 'hold' (no strong directional movement)
        #
        # This ensures model learns:
        # "These features → price went UP" (label = buy)
        # "These features → price went DOWN" (label = sell)
        # "These features → price stayed FLAT" (label = hold)

        return_pct = outcome['return_pct']

        if return_pct >= 0.05:  # Price went UP ≥5%
            label = 'buy'
        elif return_pct <= -0.05:  # Price went DOWN ≤-5%
            label = 'sell'
        else:  # Price moved between -5% and +5% (flat/sideways)
            label = 'hold'

        # Create training sample
        training_sample = {
            'id': str(uuid.uuid4()),
            'symbol': outcome['symbol'],
            'entry_date': outcome['entry_date'],
            'entry_price': outcome['entry_price'],
            'exit_date': outcome['exit_date'],
            'exit_price': outcome['exit_price'],
            'position_size': 1.0,
            'outcome': label,  # 'buy', 'sell', or 'hold' based on ACTUAL movement
            'profit_loss': outcome['exit_price'] - outcome['entry_price'],
            'profit_loss_pct': outcome['return_pct'],
            'exit_reason': 'outcome_tracking',
            'trade_type': 'real_prediction',  # Mark as real prediction vs backtest
            'strategy': 'turbomode_live',
            'notes': f"From live prediction (signal: {outcome['signal_type']}, return: {return_pct:.1%})"
        }

        return training_sample

    def save_training_sample(self, sample: Dict[str, Any], features_json: str = None):
        """
        Save training sample to trades table

        Args:
            sample: Training sample dictionary
            features_json: Features in JSON format (optional for now)
        """
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()

        try:
            cursor.execute("""
                INSERT INTO trades (
                    id, symbol, entry_date, entry_price,
                    exit_date, exit_price, position_size,
                    outcome, profit_loss, profit_loss_pct,
                    exit_reason, entry_features_json,
                    trade_type, strategy, notes, created_at
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, (
                sample['id'],
                sample['symbol'],
                sample['entry_date'],
                sample['entry_price'],
                sample['exit_date'],
                sample['exit_price'],
                sample['position_size'],
                sample['outcome'],
                sample['profit_loss'],
                sample['profit_loss_pct'],
                sample['exit_reason'],
                features_json,  # Will be NULL for now
                sample['trade_type'],
                sample['strategy'],
                sample['notes'],
                datetime.now().strftime('%Y-%m-%d %H:%M:%S')
            ))

            conn.commit()
            logger.info(f"[OK] {sample['symbol']}: Added training sample ({sample['outcome'].upper()})")

        except Exception as e:
            conn.rollback()
            logger.error(f"[ERROR] Failed to save training sample for {sample['symbol']}: {e}")
            raise
        finally:
            conn.close()

    def mark_outcome_as_processed(self, outcome_id: int):
        """
        Mark an outcome as processed so it won't be converted again

        Args:
            outcome_id: Outcome ID in signal_history table
        """
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()

        try:
            cursor.execute("""
                UPDATE signal_history
                SET processed_for_training = 1,
                    updated_at = ?
                WHERE id = ?
            """, (datetime.now().strftime('%Y-%m-%d %H:%M:%S'), outcome_id))

            conn.commit()
        except Exception as e:
            conn.rollback()
            logger.error(f"[ERROR] Failed to mark outcome {outcome_id} as processed: {e}")
            raise
        finally:
            conn.close()

    def generate_training_samples(self) -> Dict[str, Any]:
        """
        Main function: Convert all unprocessed outcomes into training samples

        Returns:
            Summary statistics
        """
        logger.info("=" * 80)
        logger.info("TURBOMODE TRAINING SAMPLE GENERATOR")
        logger.info(f"Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        logger.info("=" * 80)

        # Get unprocessed outcomes
        outcomes = self.get_unprocessed_outcomes()

        if not outcomes:
            logger.info("[INFO] No new outcomes to process")
            return {
                'total_processed': 0,
                'samples_added': 0,
                'errors': 0
            }

        logger.info(f"[INFO] Found {len(outcomes)} outcomes to convert\n")

        stats = {
            'total_processed': len(outcomes),
            'samples_added': 0,
            'errors': 0,
            'buy_labels': 0,
            'sell_labels': 0
        }

        for outcome in outcomes:
            try:
                # Convert outcome to training sample
                sample = self.convert_outcome_to_training_sample(outcome)

                # Save to trades table
                # Note: features_json is None for now - we'll add feature extraction later
                self.save_training_sample(sample, features_json=None)

                # Mark as processed
                self.mark_outcome_as_processed(outcome['id'])

                stats['samples_added'] += 1

                if sample['outcome'] == 'buy':
                    stats['buy_labels'] += 1
                else:
                    stats['sell_labels'] += 1

            except Exception as e:
                stats['errors'] += 1
                logger.error(f"[ERROR] Failed to process outcome for {outcome['symbol']}: {e}")

        # Print summary
        logger.info("\n" + "=" * 80)
        logger.info("SAMPLE GENERATION SUMMARY")
        logger.info("=" * 80)
        logger.info(f"Total Processed:    {stats['total_processed']}")
        logger.info(f"Samples Added:      {stats['samples_added']}")
        logger.info(f"Errors:             {stats['errors']}")
        logger.info(f"\nBUY Labels:         {stats['buy_labels']}")
        logger.info(f"SELL Labels:        {stats['sell_labels']}")
        logger.info("=" * 80)

        return stats


def generate_training_samples_from_outcomes():
    """
    Main entry point for scheduled job
    Called weekly by Flask scheduler (Sunday 3 AM)
    """
    generator = TrainingSampleGenerator()
    return generator.generate_training_samples()


if __name__ == '__main__':
    # Test the training sample generator
    print("Testing Training Sample Generator...")
    print("=" * 80)

    generator = TrainingSampleGenerator()
    results = generator.generate_training_samples()

    print("\n[OK] Sample generation complete!")
    print(f"Processed: {results['total_processed']}")
    print(f"Added: {results['samples_added']}")
