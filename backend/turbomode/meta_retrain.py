"""
Meta-Learner Retraining Module
Regenerates meta-predictions and retrains meta-learner with override-aware features
Runs every 6 weeks on Sunday at 23:45
"""

import os
import sys
import logging
from pathlib import Path
from datetime import datetime

# Add parent directory to path
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
if parent_dir not in sys.path:
    sys.path.insert(0, parent_dir)

# Setup logging
logger = logging.getLogger('meta_retrain')
logger.setLevel(logging.INFO)


def maybe_retrain_meta():
    """
    Regenerate meta-predictions and retrain meta-learner with override-aware features.

    This is a long-running task (2-3 hours) that:
    1. Loads all 8 base models
    2. Generates predictions for all 169,400 training samples
    3. Adds override-aware features
    4. Retrains the final meta-learner with 55 features

    Returns:
        bool: True if successful, False otherwise
    """
    logger.info("=" * 80)
    logger.info("META-LEARNER RETRAINING - STARTING")
    logger.info(f"Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    logger.info("=" * 80)

    try:
        # Import here to avoid circular imports
        from backend.turbomode.generate_meta_predictions import generate_meta_predictions
        from backend.turbomode.retrain_meta_with_override_features import retrain_meta_learner

        # Paths
        db_path = Path(__file__).parent.parent / 'data' / 'turbomode.db'
        models_dir = Path(__file__).parent.parent / 'data' / 'turbomode_models'

        logger.info("\n[STEP 1/2] Generating meta-predictions table...")
        logger.info("           (This will take ~2 minutes for 169,400 samples)")

        # Generate meta-predictions (base model outputs on training data)
        success = generate_meta_predictions(
            str(db_path),
            str(models_dir),
            batch_size=5000  # Optimized batch size
        )

        if not success:
            logger.error("❌ Meta-predictions generation failed")
            return False

        logger.info("✅ Meta-predictions table generated successfully")

        logger.info("\n[STEP 2/2] Retraining meta-learner with override-aware features...")
        logger.info("           (This will take ~1 minute)")

        # Retrain meta-learner with 55 features (24 base + 24 override + 7 aggregate)
        result = retrain_meta_learner(
            training_db_path=str(db_path),
            output_path=None,  # Uses default: meta_learner_v2
            use_class_weights=True,
            test_size=0.2,
            save_model=True
        )

        if result is None:
            logger.error("❌ Meta-learner retraining failed")
            return False

        logger.info("✅ Meta-learner retrained successfully")
        logger.info(f"   Validation accuracy: {result['val_accuracy']:.2%}")
        logger.info(f"   Model saved to: backend/data/turbomode_models/meta_learner_v2")

        logger.info("\n" + "=" * 80)
        logger.info("META-LEARNER RETRAINING - COMPLETED")
        logger.info("=" * 80)

        return True

    except Exception as e:
        logger.error(f"❌ Meta-learner retraining failed: {e}")
        import traceback
        traceback.print_exc()
        return False


if __name__ == '__main__':
    # Test the retraining
    print("Testing Meta-Learner Retraining...")
    print("WARNING: This is a long-running task (~3 minutes)")
    print("Press Ctrl+C to cancel, or wait 5 seconds to continue...")

    import time
    try:
        time.sleep(5)
    except KeyboardInterrupt:
        print("\n\nCancelled by user.")
        sys.exit(0)

    # Run retraining
    success = maybe_retrain_meta()

    if success:
        print("\n✅ Meta-learner retraining completed successfully!")
    else:
        print("\n❌ Meta-learner retraining failed. Check logs above.")
