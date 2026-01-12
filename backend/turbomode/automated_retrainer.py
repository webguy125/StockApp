"""
TurboMode Automated Retrainer
Monthly model retraining with accumulated data
Validates new models before deployment

This runs monthly (1st of month at 4 AM) to retrain models with new data
"""

import os
import sys
import sqlite3
import json
import shutil
from datetime import datetime
from typing import Dict, Any
import logging
import subprocess

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger('automated_retrainer')

# Add parent directory to path
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
if parent_dir not in sys.path:
    sys.path.insert(0, parent_dir)


class AutomatedRetrainer:
    """
    Automated monthly model retraining
    Validates new models and deploys only if they improve performance
    """

    def __init__(self, db_path: str = None, models_dir: str = None):
        """
        Initialize automated retrainer

        Args:
            db_path: Path to turbomode.db
            models_dir: Path to turbomode_models directory
        """
        if db_path is None:
            db_path = os.path.join(parent_dir, 'data', 'turbomode.db')

        if models_dir is None:
            models_dir = os.path.join(parent_dir, 'data', 'turbomode_models')

        self.db_path = db_path
        self.models_dir = models_dir
        self.min_new_samples = 100  # Minimum new samples required for retraining

    def get_training_sample_stats(self) -> Dict[str, Any]:
        """
        Get statistics about training samples

        Returns:
            Dictionary with sample counts and metadata
        """
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()

        # Count total samples
        cursor.execute("SELECT COUNT(*) FROM trades")
        total_samples = cursor.fetchone()[0]

        # Count samples by type
        cursor.execute("""
            SELECT trade_type, COUNT(*)
            FROM trades
            GROUP BY trade_type
        """)
        samples_by_type = dict(cursor.fetchall())

        # Count samples by outcome
        cursor.execute("""
            SELECT outcome, COUNT(*)
            FROM trades
            GROUP BY outcome
        """)
        samples_by_outcome = dict(cursor.fetchall())

        # Get date range
        cursor.execute("""
            SELECT MIN(entry_date), MAX(entry_date)
            FROM trades
        """)
        date_range = cursor.fetchone()

        # Count samples added since last retrain
        cursor.execute("""
            SELECT COUNT(*)
            FROM trades
            WHERE trade_type = 'real_prediction'
        """)
        real_prediction_samples = cursor.fetchone()[0]

        conn.close()

        return {
            'total_samples': total_samples,
            'samples_by_type': samples_by_type,
            'samples_by_outcome': samples_by_outcome,
            'date_range': date_range,
            'real_prediction_samples': real_prediction_samples
        }

    def check_if_retraining_needed(self) -> Dict[str, Any]:
        """
        Check if retraining is needed based on new samples

        Returns:
            Dictionary with decision and reasoning
        """
        stats = self.get_training_sample_stats()

        # Check if we have any samples at all
        if stats['total_samples'] == 0:
            return {
                'should_retrain': False,
                'reason': 'No training samples available',
                'stats': stats
            }

        # Check if we have minimum samples
        if stats['total_samples'] < 1000:
            return {
                'should_retrain': False,
                'reason': f'Insufficient total samples ({stats["total_samples"]} < 1000)',
                'stats': stats
            }

        # Check if we have enough new real prediction samples
        real_samples = stats['real_prediction_samples']
        if real_samples < self.min_new_samples:
            return {
                'should_retrain': False,
                'reason': f'Insufficient new samples ({real_samples} < {self.min_new_samples})',
                'stats': stats
            }

        return {
            'should_retrain': True,
            'reason': f'Ready to retrain with {real_samples} new samples',
            'stats': stats
        }

    def backup_current_models(self) -> str:
        """
        Backup current models before retraining

        Returns:
            Path to backup directory
        """
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        backup_dir = f"{self.models_dir}_backup_{timestamp}"

        if os.path.exists(self.models_dir):
            shutil.copytree(self.models_dir, backup_dir)
            logger.info(f"[OK] Models backed up to: {backup_dir}")
        else:
            logger.warning(f"[SKIP] No existing models to backup")
            backup_dir = None

        return backup_dir

    def run_training_script(self) -> Dict[str, Any]:
        """
        Run the training script and capture results

        Returns:
            Training results dictionary
        """
        training_script = os.path.join(current_dir, 'train_turbomode_models.py')

        logger.info(f"[INFO] Running training script: {training_script}")

        try:
            # Run training script
            result = subprocess.run(
                [sys.executable, training_script],
                capture_output=True,
                text=True,
                timeout=7200  # 2 hour timeout
            )

            if result.returncode != 0:
                return {
                    'success': False,
                    'error': result.stderr,
                    'stdout': result.stdout
                }

            return {
                'success': True,
                'stdout': result.stdout,
                'stderr': result.stderr
            }

        except subprocess.TimeoutExpired:
            return {
                'success': False,
                'error': 'Training script timed out (exceeded 2 hours)'
            }
        except Exception as e:
            return {
                'success': False,
                'error': str(e)
            }

    def validate_new_models(self) -> Dict[str, Any]:
        """
        Validate new models by checking metadata and performance

        Returns:
            Validation results
        """
        # Check if models directory exists
        if not os.path.exists(self.models_dir):
            return {
                'valid': False,
                'reason': 'Models directory not found'
            }

        # Check for meta-learner (required)
        meta_learner_path = os.path.join(self.models_dir, 'meta_learner', 'metadata.json')
        if not os.path.exists(meta_learner_path):
            return {
                'valid': False,
                'reason': 'Meta-learner metadata not found'
            }

        # Read meta-learner metadata
        try:
            with open(meta_learner_path, 'r') as f:
                meta_data = json.load(f)

            accuracy = meta_data.get('test_accuracy', 0)
            precision = meta_data.get('test_precision', 0)
            recall = meta_data.get('test_recall', 0)

            # Minimum performance thresholds
            min_accuracy = 0.60  # 60% minimum
            min_precision = 0.55
            min_recall = 0.55

            if accuracy < min_accuracy:
                return {
                    'valid': False,
                    'reason': f'Accuracy too low ({accuracy:.1%} < {min_accuracy:.1%})',
                    'metrics': meta_data
                }

            if precision < min_precision:
                return {
                    'valid': False,
                    'reason': f'Precision too low ({precision:.1%} < {min_precision:.1%})',
                    'metrics': meta_data
                }

            if recall < min_recall:
                return {
                    'valid': False,
                    'reason': f'Recall too low ({recall:.1%} < {min_recall:.1%})',
                    'metrics': meta_data
                }

            return {
                'valid': True,
                'reason': 'Models pass validation',
                'metrics': meta_data
            }

        except Exception as e:
            return {
                'valid': False,
                'reason': f'Error reading metadata: {str(e)}'
            }

    def compare_with_backup(self, backup_dir: str) -> Dict[str, Any]:
        """
        Compare new models with backup to see if they improved

        Args:
            backup_dir: Path to backup directory

        Returns:
            Comparison results
        """
        if backup_dir is None or not os.path.exists(backup_dir):
            return {
                'comparison': 'no_backup',
                'reason': 'No backup available for comparison'
            }

        # Read old metadata
        old_meta_path = os.path.join(backup_dir, 'meta_learner', 'metadata.json')
        new_meta_path = os.path.join(self.models_dir, 'meta_learner', 'metadata.json')

        try:
            with open(old_meta_path, 'r') as f:
                old_meta = json.load(f)

            with open(new_meta_path, 'r') as f:
                new_meta = json.load(f)

            old_acc = old_meta.get('test_accuracy', 0)
            new_acc = new_meta.get('test_accuracy', 0)

            improvement = new_acc - old_acc

            return {
                'comparison': 'completed',
                'old_accuracy': old_acc,
                'new_accuracy': new_acc,
                'improvement': improvement,
                'improved': improvement > 0,
                'old_metadata': old_meta,
                'new_metadata': new_meta
            }

        except Exception as e:
            return {
                'comparison': 'error',
                'reason': f'Error comparing models: {str(e)}'
            }

    def restore_backup(self, backup_dir: str):
        """
        Restore models from backup if new models are worse

        Args:
            backup_dir: Path to backup directory
        """
        if backup_dir is None or not os.path.exists(backup_dir):
            logger.error("[ERROR] Cannot restore: backup not found")
            return False

        try:
            # Remove new models
            if os.path.exists(self.models_dir):
                shutil.rmtree(self.models_dir)

            # Restore backup
            shutil.copytree(backup_dir, self.models_dir)

            logger.info(f"[OK] Models restored from backup")
            return True

        except Exception as e:
            logger.error(f"[ERROR] Failed to restore backup: {e}")
            return False

    def automated_retrain(self) -> Dict[str, Any]:
        """
        Main function: Automated model retraining with validation

        Returns:
            Summary of retraining process
        """
        logger.info("=" * 80)
        logger.info("TURBOMODE AUTOMATED RETRAINER")
        logger.info(f"Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        logger.info("=" * 80)

        # Step 1: Check if retraining is needed
        check = self.check_if_retraining_needed()

        logger.info(f"\n[INFO] Retraining Check:")
        logger.info(f"  Total Samples: {check['stats']['total_samples']}")
        logger.info(f"  Real Prediction Samples: {check['stats']['real_prediction_samples']}")
        logger.info(f"  Decision: {check['reason']}")

        if not check['should_retrain']:
            logger.info(f"\n[SKIP] Retraining not needed")
            return {
                'retrained': False,
                'reason': check['reason'],
                'stats': check['stats']
            }

        # Step 2: Backup current models
        logger.info(f"\n[INFO] Backing up current models...")
        backup_dir = self.backup_current_models()

        # Step 3: Run training script
        logger.info(f"\n[INFO] Starting model training...")
        training_result = self.run_training_script()

        if not training_result['success']:
            logger.error(f"[ERROR] Training failed: {training_result['error']}")
            logger.info(f"\n[INFO] Restoring backup...")
            self.restore_backup(backup_dir)
            return {
                'retrained': False,
                'reason': 'Training script failed',
                'error': training_result['error'],
                'backup_restored': True
            }

        logger.info(f"[OK] Training completed successfully")

        # Step 4: Validate new models
        logger.info(f"\n[INFO] Validating new models...")
        validation = self.validate_new_models()

        if not validation['valid']:
            logger.error(f"[ERROR] Validation failed: {validation['reason']}")
            logger.info(f"\n[INFO] Restoring backup...")
            self.restore_backup(backup_dir)
            return {
                'retrained': False,
                'reason': 'New models failed validation',
                'validation': validation,
                'backup_restored': True
            }

        logger.info(f"[OK] Models passed validation")
        logger.info(f"  Accuracy: {validation['metrics']['test_accuracy']:.1%}")
        logger.info(f"  Precision: {validation['metrics']['test_precision']:.1%}")
        logger.info(f"  Recall: {validation['metrics']['test_recall']:.1%}")

        # Step 5: Compare with old models
        logger.info(f"\n[INFO] Comparing with previous models...")
        comparison = self.compare_with_backup(backup_dir)

        if comparison['comparison'] == 'completed':
            logger.info(f"  Old Accuracy: {comparison['old_accuracy']:.1%}")
            logger.info(f"  New Accuracy: {comparison['new_accuracy']:.1%}")
            logger.info(f"  Improvement: {comparison['improvement']:+.1%}")

            if not comparison['improved']:
                logger.warning(f"[WARNING] New models did not improve accuracy")
                logger.info(f"\n[INFO] Restoring backup...")
                self.restore_backup(backup_dir)
                return {
                    'retrained': False,
                    'reason': 'New models did not improve performance',
                    'comparison': comparison,
                    'backup_restored': True
                }

        # Step 6: Success - new models are deployed
        logger.info("\n" + "=" * 80)
        logger.info("RETRAINING SUCCESSFUL")
        logger.info("=" * 80)
        logger.info(f"[OK] New models deployed")
        logger.info(f"[OK] Backup available at: {backup_dir}")
        logger.info("=" * 80)

        return {
            'retrained': True,
            'reason': 'New models successfully deployed',
            'stats': check['stats'],
            'validation': validation,
            'comparison': comparison,
            'backup_dir': backup_dir
        }


def automated_model_retraining():
    """
    Main entry point for scheduled job
    Called monthly by Flask scheduler (1st of month at 4 AM)
    """
    retrainer = AutomatedRetrainer()
    return retrainer.automated_retrain()


if __name__ == '__main__':
    # Test the automated retrainer
    print("Testing Automated Retrainer...")
    print("=" * 80)

    retrainer = AutomatedRetrainer()
    results = retrainer.automated_retrain()

    print("\n[OK] Retraining process complete!")
    print(f"Retrained: {results['retrained']}")
    print(f"Reason: {results['reason']}")
