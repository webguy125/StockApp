"""
Model Promotion Gate Module

Multi-criteria validation before promoting a model to production.

Criteria (ALL must pass):
1. Overall Accuracy > 65%
2. Crash Regime Accuracy > 65% (most important)
3. No sector accuracy < 55% (prevent weak sectors)
4. Drift score < 0.20 (model stable)
5. Error replay improvement > 0% (learning from mistakes)

Only if ALL criteria pass → promote model
"""

import sqlite3
import json
import os
import shutil
from typing import Dict, List, Any, Optional
from datetime import datetime


class ModelPromotionGate:
    """
    Validates ML models before production deployment

    Prevents bad models from reaching live trading by enforcing
    strict quality criteria across multiple dimensions.
    """

    def __init__(self, db_path: str = "backend/data/advanced_ml_system.db"):
        """
        Initialize model promotion gate

        Args:
            db_path: Path to database
        """
        self.db_path = db_path

        # Validation thresholds
        self.min_overall_accuracy = 0.65
        self.min_crash_accuracy = 0.65
        self.min_sector_accuracy = 0.55
        self.max_drift_score = 0.20
        self.min_replay_improvement = 0.0

        self._init_table()

    def _init_table(self):
        """Create model_promotion_history table if not exists"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()

        cursor.execute('''
            CREATE TABLE IF NOT EXISTS model_promotion_history (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                model_path TEXT NOT NULL,
                validation_timestamp TEXT NOT NULL,
                approved INTEGER NOT NULL,
                overall_accuracy REAL,
                crash_accuracy REAL,
                min_sector_accuracy REAL,
                drift_score REAL,
                replay_improvement REAL,
                checks_passed TEXT,
                checks_failed TEXT,
                promoted_to_production INTEGER DEFAULT 0,
                notes TEXT
            )
        ''')

        conn.commit()
        conn.close()

    def validate_model(
        self,
        model_path: str,
        validation_results: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        Run all validation checks

        Args:
            model_path: Path to trained model file
            validation_results: Dict from training with:
                - overall_accuracy: float
                - regime_accuracy: Dict[str, float]
                - sector_accuracy: Dict[str, float]
                - drift_score: float (optional)
                - replay_improvement: float (optional)

        Returns:
            Dict with validation results:
            {
                'approved': bool,
                'checks_passed': List[str],
                'checks_failed': List[str],
                'details': Dict[str, Any],
                'recommendation': str  # "PROMOTE" or "REJECT"
            }
        """
        checks_passed = []
        checks_failed = []
        details = {}

        # Check 1: Overall Accuracy
        overall_acc = validation_results.get('overall_accuracy', 0.0)
        details['overall_accuracy'] = overall_acc

        if overall_acc >= self.min_overall_accuracy:
            checks_passed.append(f"Overall Accuracy ({overall_acc:.1%} >= {self.min_overall_accuracy:.1%})")
        else:
            checks_failed.append(f"Overall Accuracy ({overall_acc:.1%} < {self.min_overall_accuracy:.1%})")

        # Check 2: Crash Regime Accuracy (MOST IMPORTANT)
        regime_acc = validation_results.get('regime_accuracy', {})
        crash_acc = regime_acc.get('crash', regime_acc.get('market_crash', 0.0))
        details['crash_accuracy'] = crash_acc

        if crash_acc >= self.min_crash_accuracy:
            checks_passed.append(f"Crash Accuracy ({crash_acc:.1%} >= {self.min_crash_accuracy:.1%})")
        else:
            checks_failed.append(f"Crash Accuracy ({crash_acc:.1%} < {self.min_crash_accuracy:.1%})")

        # Check 3: Sector Accuracy (no weak sectors)
        sector_acc = validation_results.get('sector_accuracy', {})
        if sector_acc:
            min_sector = min(sector_acc.values())
            min_sector_name = [name for name, acc in sector_acc.items() if acc == min_sector][0]
            details['min_sector_accuracy'] = min_sector
            details['weakest_sector'] = min_sector_name

            if min_sector >= self.min_sector_accuracy:
                checks_passed.append(f"Min Sector Accuracy ({min_sector_name}: {min_sector:.1%} >= {self.min_sector_accuracy:.1%})")
            else:
                checks_failed.append(f"Min Sector Accuracy ({min_sector_name}: {min_sector:.1%} < {self.min_sector_accuracy:.1%})")
        else:
            # No sector data - skip this check
            details['min_sector_accuracy'] = None
            checks_passed.append("Sector Accuracy (no data, skipped)")

        # Check 4: Drift Score
        drift_score = validation_results.get('drift_score', 0.0)
        details['drift_score'] = drift_score

        if drift_score <= self.max_drift_score:
            checks_passed.append(f"Drift Score ({drift_score:.2f} <= {self.max_drift_score:.2f})")
        else:
            checks_failed.append(f"Drift Score ({drift_score:.2f} > {self.max_drift_score:.2f})")

        # Check 5: Error Replay Improvement
        replay_improvement = validation_results.get('replay_improvement', 0.0)
        details['replay_improvement'] = replay_improvement

        if replay_improvement >= self.min_replay_improvement:
            checks_passed.append(f"Replay Improvement ({replay_improvement:.1%} >= {self.min_replay_improvement:.1%})")
        else:
            checks_failed.append(f"Replay Improvement ({replay_improvement:.1%} < {self.min_replay_improvement:.1%})")

        # Final decision
        approved = len(checks_failed) == 0
        recommendation = "PROMOTE" if approved else "REJECT"

        result = {
            'approved': approved,
            'checks_passed': checks_passed,
            'checks_failed': checks_failed,
            'details': details,
            'recommendation': recommendation
        }

        # Log validation to database
        self._log_validation(model_path, result, validation_results)

        return result

    def _log_validation(
        self,
        model_path: str,
        result: Dict[str, Any],
        validation_results: Dict[str, Any]
    ):
        """
        Log validation attempt to database

        Args:
            model_path: Path to model
            result: Validation result dict
            validation_results: Original validation data
        """
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()

        cursor.execute('''
            INSERT INTO model_promotion_history
            (model_path, validation_timestamp, approved, overall_accuracy,
             crash_accuracy, min_sector_accuracy, drift_score, replay_improvement,
             checks_passed, checks_failed, promoted_to_production, notes)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, 0, ?)
        ''', (
            model_path,
            datetime.now().isoformat(),
            1 if result['approved'] else 0,
            result['details'].get('overall_accuracy'),
            result['details'].get('crash_accuracy'),
            result['details'].get('min_sector_accuracy'),
            result['details'].get('drift_score'),
            result['details'].get('replay_improvement'),
            json.dumps(result['checks_passed']),
            json.dumps(result['checks_failed']),
            result['recommendation']
        ))

        conn.commit()
        conn.close()

    def promote_model(
        self,
        model_path: str,
        production_path: str = "backend/data/ml_models/meta_learner_production.pkl",
        backup_old: bool = True
    ) -> bool:
        """
        Copy validated model to production path

        Args:
            model_path: Path to validated model
            production_path: Production model path
            backup_old: Whether to backup existing production model

        Returns:
            True if successful, False otherwise
        """
        try:
            # Ensure production directory exists
            prod_dir = os.path.dirname(production_path)
            if not os.path.exists(prod_dir):
                os.makedirs(prod_dir, exist_ok=True)

            # Backup existing production model
            if backup_old and os.path.exists(production_path):
                backup_path = production_path.replace('.pkl', f'_backup_{datetime.now().strftime("%Y%m%d_%H%M%S")}.pkl')
                shutil.copy2(production_path, backup_path)
                print(f"[OK] Backed up old model to: {backup_path}")

            # Copy new model to production
            shutil.copy2(model_path, production_path)
            print(f"[OK] Promoted model to production: {production_path}")

            # Update database
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()

            cursor.execute('''
                UPDATE model_promotion_history
                SET promoted_to_production = 1
                WHERE model_path = ?
                ORDER BY id DESC
                LIMIT 1
            ''', (model_path,))

            conn.commit()
            conn.close()

            return True

        except Exception as e:
            print(f"[ERROR] Failed to promote model: {e}")
            return False

    def get_promotion_history(self, limit: int = 10) -> List[Dict[str, Any]]:
        """
        Get recent promotion history

        Args:
            limit: Number of records to return

        Returns:
            List of promotion attempt dicts
        """
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()

        cursor.execute('''
            SELECT model_path, validation_timestamp, approved,
                   overall_accuracy, crash_accuracy, min_sector_accuracy,
                   drift_score, replay_improvement, checks_passed,
                   checks_failed, promoted_to_production, notes
            FROM model_promotion_history
            ORDER BY id DESC
            LIMIT ?
        ''', (limit,))

        rows = cursor.fetchall()
        conn.close()

        history = []
        for row in rows:
            history.append({
                'model_path': row[0],
                'timestamp': row[1],
                'approved': bool(row[2]),
                'overall_accuracy': row[3],
                'crash_accuracy': row[4],
                'min_sector_accuracy': row[5],
                'drift_score': row[6],
                'replay_improvement': row[7],
                'checks_passed': json.loads(row[8]) if row[8] else [],
                'checks_failed': json.loads(row[9]) if row[9] else [],
                'promoted': bool(row[10]),
                'notes': row[11]
            })

        return history

    def get_promotion_stats(self) -> Dict[str, Any]:
        """
        Get summary statistics of promotion attempts

        Returns:
            Dict with promotion stats
        """
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()

        # Total attempts
        cursor.execute('SELECT COUNT(*) FROM model_promotion_history')
        total_attempts = cursor.fetchone()[0]

        # Approved vs rejected
        cursor.execute('SELECT COUNT(*) FROM model_promotion_history WHERE approved = 1')
        approved = cursor.fetchone()[0]

        # Promoted to production
        cursor.execute('SELECT COUNT(*) FROM model_promotion_history WHERE promoted_to_production = 1')
        promoted = cursor.fetchone()[0]

        # Average metrics
        cursor.execute('''
            SELECT AVG(overall_accuracy), AVG(crash_accuracy), AVG(min_sector_accuracy)
            FROM model_promotion_history
        ''')
        row = cursor.fetchone()
        avg_overall = row[0] if row[0] else 0.0
        avg_crash = row[1] if row[1] else 0.0
        avg_sector = row[2] if row[2] else 0.0

        conn.close()

        return {
            'total_attempts': total_attempts,
            'approved': approved,
            'rejected': total_attempts - approved,
            'promoted': promoted,
            'approval_rate': approved / total_attempts if total_attempts > 0 else 0.0,
            'avg_overall_accuracy': avg_overall,
            'avg_crash_accuracy': avg_crash,
            'avg_min_sector_accuracy': avg_sector
        }


if __name__ == '__main__':
    # Test model promotion gate
    print("Testing Model Promotion Gate...\n")

    # Create gate
    gate = ModelPromotionGate()

    print("[TEST 1] Good Model - Should APPROVE")
    print("=" * 60)
    good_results = {
        'overall_accuracy': 0.72,
        'regime_accuracy': {
            'crash': 0.68,
            'normal': 0.75,
            'recovery': 0.70
        },
        'sector_accuracy': {
            'Technology': 0.75,
            'Energy': 0.60,
            'Healthcare': 0.70
        },
        'drift_score': 0.12,
        'replay_improvement': 0.05
    }

    result = gate.validate_model("test_model_good.pkl", good_results)
    print(f"Approved: {result['approved']}")
    print(f"Recommendation: {result['recommendation']}")
    print(f"Checks Passed: {len(result['checks_passed'])}")
    print(f"Checks Failed: {len(result['checks_failed'])}")
    for check in result['checks_passed']:
        print(f"  [PASS] {check}")
    for check in result['checks_failed']:
        print(f"  [FAIL] {check}")
    print()

    print("[TEST 2] Bad Model - Poor Crash Accuracy")
    print("=" * 60)
    bad_results = {
        'overall_accuracy': 0.70,
        'regime_accuracy': {
            'crash': 0.55,  # TOO LOW!
            'normal': 0.75,
            'recovery': 0.72
        },
        'sector_accuracy': {
            'Technology': 0.75,
            'Energy': 0.65,
            'Healthcare': 0.70
        },
        'drift_score': 0.10,
        'replay_improvement': 0.03
    }

    result = gate.validate_model("test_model_bad.pkl", bad_results)
    print(f"Approved: {result['approved']}")
    print(f"Recommendation: {result['recommendation']}")
    print(f"Checks Passed: {len(result['checks_passed'])}")
    print(f"Checks Failed: {len(result['checks_failed'])}")
    for check in result['checks_passed']:
        print(f"  [PASS] {check}")
    for check in result['checks_failed']:
        print(f"  [FAIL] {check}")
    print()

    print("[TEST 3] Promotion History")
    print("=" * 60)
    history = gate.get_promotion_history(limit=5)
    print(f"Total validation attempts: {len(history)}")
    for i, attempt in enumerate(history):
        status = "APPROVED" if attempt['approved'] else "REJECTED"
        print(f"  {i+1}. {status:10s} - Overall: {attempt['overall_accuracy']:.1%}, Crash: {attempt['crash_accuracy']:.1%}")
    print()

    print("[TEST 4] Promotion Statistics")
    print("=" * 60)
    stats = gate.get_promotion_stats()
    print(f"Total Attempts: {stats['total_attempts']}")
    print(f"Approved: {stats['approved']}")
    print(f"Rejected: {stats['rejected']}")
    print(f"Approval Rate: {stats['approval_rate']:.1%}")
    print(f"Avg Crash Accuracy: {stats['avg_crash_accuracy']:.1%}")
    print()

    print("=" * 60)
    print("[OK] Model Promotion Gate Tests Complete")
    print("=" * 60)
