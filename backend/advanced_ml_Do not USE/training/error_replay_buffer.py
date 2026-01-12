"""
Error Replay Buffer Module

Stores worst predictions for targeted retraining (experience replay for ML).

Strategy:
- During validation, track predictions with highest error scores
- Error score = confidence * (1 if wrong, 0 if correct)
- Store top N worst errors (high confidence but wrong)
- During training, replay these samples 3x more frequently
- Helps model learn from mistakes
"""

import numpy as np
import sqlite3
import json
from typing import Dict, List, Any, Tuple, Optional
from datetime import datetime


class ErrorReplayBuffer:
    """
    Experience replay buffer for ML model errors

    Similar to RL experience replay but for supervised learning:
    - Stores high-confidence wrong predictions
    - Prioritizes errors with highest confidence
    - Allows sampling for targeted retraining
    """

    def __init__(
        self,
        max_size: int = 1000,
        db_path: str = "backend/data/advanced_ml_system.db"
    ):
        """
        Initialize error replay buffer

        Args:
            max_size: Maximum errors to store
            db_path: Path to database
        """
        self.max_size = max_size
        self.db_path = db_path
        self._init_table()

    def _init_table(self):
        """Create error_replay_buffer table if not exists"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()

        cursor.execute('''
            CREATE TABLE IF NOT EXISTS error_replay_buffer (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                symbol TEXT NOT NULL,
                date TEXT NOT NULL,
                regime TEXT,
                features_json TEXT NOT NULL,
                true_label INTEGER NOT NULL,
                predicted_label INTEGER NOT NULL,
                confidence REAL NOT NULL,
                error_score REAL NOT NULL,
                added_at TEXT DEFAULT CURRENT_TIMESTAMP,
                replayed_count INTEGER DEFAULT 0
            )
        ''')

        # Index for efficient querying
        cursor.execute('''
            CREATE INDEX IF NOT EXISTS idx_error_score
            ON error_replay_buffer(error_score DESC)
        ''')

        cursor.execute('''
            CREATE INDEX IF NOT EXISTS idx_regime
            ON error_replay_buffer(regime)
        ''')

        conn.commit()
        conn.close()

    def add_error(
        self,
        features: np.ndarray,
        label: int,
        prediction: int,
        confidence: float,
        symbol: str = "",
        date: str = "",
        regime: str = ""
    ):
        """
        Add a prediction error to the buffer

        Only stores if:
        1. Prediction is wrong (label != prediction)
        2. Confidence is high (error score is high)
        3. Buffer has space OR error score is worse than current worst

        Args:
            features: Feature vector (1D array)
            label: True label (0=buy, 1=hold, 2=sell)
            prediction: Model prediction (0=buy, 1=hold, 2=sell)
            confidence: Prediction confidence [0-1]
            symbol: Stock symbol
            date: Prediction date
            regime: Market regime
        """
        # Only store errors (wrong predictions)
        if label == prediction:
            return

        # Calculate error score (high confidence wrong = worst errors)
        error_score = float(confidence)

        # Convert features to JSON
        features_json = json.dumps(features.tolist() if hasattr(features, 'tolist') else list(features))

        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()

        # Check current buffer size
        cursor.execute('SELECT COUNT(*) FROM error_replay_buffer')
        current_size = cursor.fetchone()[0]

        # If buffer full, check if this error is worse than the best error in buffer
        if current_size >= self.max_size:
            cursor.execute('SELECT MIN(error_score) FROM error_replay_buffer')
            min_error = cursor.fetchone()[0]

            # If new error is not worse than current worst, skip it
            if error_score <= min_error:
                conn.close()
                return

            # Remove the lowest error score to make room
            cursor.execute('''
                DELETE FROM error_replay_buffer
                WHERE id = (
                    SELECT id FROM error_replay_buffer
                    ORDER BY error_score ASC
                    LIMIT 1
                )
            ''')

        # Insert new error
        cursor.execute('''
            INSERT INTO error_replay_buffer
            (symbol, date, regime, features_json, true_label, predicted_label,
             confidence, error_score, added_at, replayed_count)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, 0)
        ''', (
            symbol,
            date,
            regime,
            features_json,
            int(label),
            int(prediction),
            float(confidence),
            float(error_score),
            datetime.now().isoformat()
        ))

        conn.commit()
        conn.close()

    def add_batch_errors(
        self,
        features: np.ndarray,
        labels: np.ndarray,
        predictions: np.ndarray,
        confidences: np.ndarray,
        symbols: List[str] = None,
        dates: List[str] = None,
        regimes: List[str] = None
    ):
        """
        Add multiple errors in batch (more efficient)

        Args:
            features: Feature matrix (n_samples, n_features)
            labels: True labels (n_samples,)
            predictions: Model predictions (n_samples,)
            confidences: Prediction confidences (n_samples,)
            symbols: Symbol for each sample (optional)
            dates: Date for each sample (optional)
            regimes: Regime for each sample (optional)
        """
        n_samples = len(labels)

        # Default values if not provided
        if symbols is None:
            symbols = [""] * n_samples
        if dates is None:
            dates = [""] * n_samples
        if regimes is None:
            regimes = [""] * n_samples

        # Find wrong predictions
        errors = []
        for i in range(n_samples):
            if labels[i] != predictions[i]:
                errors.append({
                    'features': features[i],
                    'label': labels[i],
                    'prediction': predictions[i],
                    'confidence': confidences[i],
                    'symbol': symbols[i],
                    'date': dates[i],
                    'regime': regimes[i]
                })

        # Sort by confidence (highest first)
        errors.sort(key=lambda x: x['confidence'], reverse=True)

        # Add top errors to buffer
        for error in errors:
            self.add_error(
                error['features'],
                error['label'],
                error['prediction'],
                error['confidence'],
                error['symbol'],
                error['date'],
                error['regime']
            )

    def get_replay_samples(
        self,
        n_samples: int = None,
        regime: str = None
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Retrieve error samples for replay training

        Args:
            n_samples: Number of samples to retrieve (None = all)
            regime: Filter by regime (None = all regimes)

        Returns:
            (features, labels) tuple of numpy arrays
        """
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()

        # Build query
        if regime:
            if n_samples:
                cursor.execute('''
                    SELECT features_json, true_label, id
                    FROM error_replay_buffer
                    WHERE regime = ?
                    ORDER BY error_score DESC
                    LIMIT ?
                ''', (regime, n_samples))
            else:
                cursor.execute('''
                    SELECT features_json, true_label, id
                    FROM error_replay_buffer
                    WHERE regime = ?
                    ORDER BY error_score DESC
                ''', (regime,))
        else:
            if n_samples:
                cursor.execute('''
                    SELECT features_json, true_label, id
                    FROM error_replay_buffer
                    ORDER BY error_score DESC
                    LIMIT ?
                ''', (n_samples,))
            else:
                cursor.execute('''
                    SELECT features_json, true_label, id
                    FROM error_replay_buffer
                    ORDER BY error_score DESC
                ''')

        rows = cursor.fetchall()

        if not rows:
            conn.close()
            return np.array([]), np.array([])

        # Parse features and labels
        feature_list = []
        label_list = []
        ids = []

        for row in rows:
            features = np.array(json.loads(row[0]))
            label = row[1]
            sample_id = row[2]

            feature_list.append(features)
            label_list.append(label)
            ids.append(sample_id)

        # Increment replay count
        for sample_id in ids:
            cursor.execute('''
                UPDATE error_replay_buffer
                SET replayed_count = replayed_count + 1
                WHERE id = ?
            ''', (sample_id,))

        conn.commit()
        conn.close()

        # Convert to numpy arrays
        X = np.array(feature_list)
        y = np.array(label_list)

        return X, y

    def get_error_stats(self) -> Dict[str, Any]:
        """
        Get statistics about stored errors

        Returns:
            Dict with error statistics:
            {
                'total_errors': int,
                'by_regime': Dict[str, int],
                'by_label': Dict[str, int],
                'avg_confidence': float,
                'max_confidence': float,
                'most_replayed': int
            }
        """
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()

        # Total errors
        cursor.execute('SELECT COUNT(*) FROM error_replay_buffer')
        total_errors = cursor.fetchone()[0]

        # By regime
        cursor.execute('''
            SELECT regime, COUNT(*) as count
            FROM error_replay_buffer
            GROUP BY regime
        ''')
        by_regime = {row[0]: row[1] for row in cursor.fetchall()}

        # By label (what they should have been)
        cursor.execute('''
            SELECT true_label, COUNT(*) as count
            FROM error_replay_buffer
            GROUP BY true_label
        ''')
        label_map = {0: 'buy', 1: 'hold', 2: 'sell'}
        by_label = {label_map[row[0]]: row[1] for row in cursor.fetchall()}

        # Confidence stats
        cursor.execute('''
            SELECT AVG(confidence), MAX(confidence), MAX(replayed_count)
            FROM error_replay_buffer
        ''')
        row = cursor.fetchone()
        avg_confidence = row[0] if row[0] else 0.0
        max_confidence = row[1] if row[1] else 0.0
        most_replayed = row[2] if row[2] else 0

        conn.close()

        return {
            'total_errors': total_errors,
            'by_regime': by_regime,
            'by_label': by_label,
            'avg_confidence': float(avg_confidence),
            'max_confidence': float(max_confidence),
            'most_replayed': most_replayed
        }

    def clear_buffer(self):
        """Clear all stored errors (after successful retraining)"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()

        cursor.execute('DELETE FROM error_replay_buffer')

        conn.commit()
        conn.close()

        print("[OK] Error replay buffer cleared")

    def prune_low_priority(self, keep_top_n: int = 500):
        """
        Remove lower priority errors to make room

        Args:
            keep_top_n: Number of highest-error samples to keep
        """
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()

        cursor.execute('''
            DELETE FROM error_replay_buffer
            WHERE id NOT IN (
                SELECT id FROM error_replay_buffer
                ORDER BY error_score DESC
                LIMIT ?
            )
        ''', (keep_top_n,))

        deleted = cursor.rowcount

        conn.commit()
        conn.close()

        print(f"[OK] Pruned {deleted} low-priority errors, kept top {keep_top_n}")


if __name__ == '__main__':
    # Test error replay buffer
    print("Testing Error Replay Buffer...\n")

    # Create buffer
    buffer = ErrorReplayBuffer(max_size=100)

    print("[TEST 1] Adding individual errors")
    print("=" * 50)

    # Simulate 10 high-confidence wrong predictions
    np.random.seed(42)
    for i in range(10):
        features = np.random.randn(50)
        true_label = 0  # Should be buy
        predicted_label = 2  # Predicted sell (WRONG!)
        confidence = 0.8 + np.random.rand() * 0.2  # High confidence (0.8-1.0)

        buffer.add_error(
            features=features,
            label=true_label,
            prediction=predicted_label,
            confidence=confidence,
            symbol=f"SYMBOL{i}",
            date=f"2024-01-{i+1:02d}",
            regime="crash"
        )

    stats = buffer.get_error_stats()
    print(f"Total errors stored: {stats['total_errors']}")
    print(f"Average confidence: {stats['avg_confidence']:.3f}")
    print(f"By regime: {stats['by_regime']}")
    print()

    print("[TEST 2] Adding batch errors")
    print("=" * 50)

    # Simulate validation set
    n_samples = 50
    features = np.random.randn(n_samples, 50)
    labels = np.random.choice([0, 1, 2], size=n_samples)
    predictions = np.random.choice([0, 1, 2], size=n_samples)
    confidences = np.random.rand(n_samples) * 0.5 + 0.5  # 0.5-1.0

    regimes = ['normal'] * 30 + ['crash'] * 10 + ['recovery'] * 10

    buffer.add_batch_errors(
        features, labels, predictions, confidences,
        regimes=regimes
    )

    stats = buffer.get_error_stats()
    print(f"Total errors after batch: {stats['total_errors']}")
    print(f"By label: {stats['by_label']}")
    print()

    print("[TEST 3] Retrieving replay samples")
    print("=" * 50)

    # Get top 20 errors for replay
    X_replay, y_replay = buffer.get_replay_samples(n_samples=20)
    print(f"Retrieved {len(X_replay)} samples for replay")
    print(f"Feature shape: {X_replay.shape}")
    print(f"Label shape: {y_replay.shape}")
    print()

    # Get crash-specific errors
    X_crash, y_crash = buffer.get_replay_samples(regime="crash")
    print(f"Retrieved {len(X_crash)} crash samples")
    print()

    print("[TEST 4] Pruning buffer")
    print("=" * 50)

    buffer.prune_low_priority(keep_top_n=50)

    stats = buffer.get_error_stats()
    print(f"Total errors after pruning: {stats['total_errors']}")
    print(f"Most replayed count: {stats['most_replayed']}")
    print()

    print("=" * 50)
    print("[OK] Error Replay Buffer Tests Complete")
    print("=" * 50)
