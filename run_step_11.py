"""
Step 11: Full 80-Symbol Backtest
Final validation of 8-model ensemble on complete dataset
"""

import sys
import os

# Add backend to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'backend'))

from advanced_ml.config import get_all_core_symbols
from advanced_ml.training.training_pipeline import TrainingPipeline

def main():
    print("=" * 70)
    print("STEP 11: FULL 80-SYMBOL BACKTEST (8-MODEL ENSEMBLE)")
    print("=" * 70)
    print("\nTraining on ALL 80 core symbols (2 years data)")
    print("This is the final validation before production deployment")
    print()
    print("Expected:")
    print("  - ~32,000 training samples")
    print("  - Runtime: 2-3 hours")
    print("  - Meta-learner accuracy: 90%+")
    print()

    # Get ALL core symbols
    all_core = get_all_core_symbols()

    print(f"Core Symbols: {len(all_core)} symbols")
    print(f"Symbols: {', '.join(all_core[:10])}... (showing first 10)")
    print()

    # Initialize pipeline
    pipeline = TrainingPipeline()

    # Run full pipeline on ALL 80 symbols
    try:
        results = pipeline.run_full_pipeline(
            symbols=all_core,
            years=2,  # 2 years for good bull/bear coverage
            test_size=0.2,
            use_existing_data=False  # Fresh backtest
        )

        # Check results
        print("\n" + "=" * 70)
        print("STEP 11 RESULTS")
        print("=" * 70)

        if results and 'rf_eval' in results:
            rf_acc = results['rf_eval']['accuracy']
            xgb_acc = results['xgb_eval']['accuracy']
            lgbm_acc = results['lgbm_eval']['accuracy']
            et_acc = results['et_eval']['accuracy']
            gb_acc = results['gb_eval']['accuracy']
            nn_acc = results['nn_eval']['accuracy']
            lr_acc = results['lr_eval']['accuracy']
            svm_acc = results['svm_eval']['accuracy']
            meta_acc = results['meta_eval']['accuracy']
            best_model = results.get('best_model', 'unknown')
            best_acc = results.get('best_accuracy', 0.0)

            print(f"\n[OK] Test SUCCESSFUL!")
            print(f"\nBase Model Accuracies:")
            print(f"  Random Forest:         {rf_acc:.4f}")
            print(f"  XGBoost:               {xgb_acc:.4f}")
            print(f"  LightGBM:              {lgbm_acc:.4f}")
            print(f"  Extra Trees:           {et_acc:.4f}")
            print(f"  Gradient Boosting:     {gb_acc:.4f}")
            print(f"  Neural Network:        {nn_acc:.4f}")
            print(f"  Logistic Regression:   {lr_acc:.4f}")
            print(f"  SVM:                   {svm_acc:.4f}")
            print(f"\nEnsemble:")
            print(f"  Meta-Learner:          {meta_acc:.4f}")
            print(f"\nBest Model: {best_model} ({best_acc:.4f})")

            # Analysis
            print("\n" + "=" * 70)
            print("VALIDATION ANALYSIS")
            print("=" * 70)

            # Compare to Step 10 baseline
            step10_meta = 0.9000
            delta = meta_acc - step10_meta

            print(f"\nStep 10 (10 symbols, 1 year):  {step10_meta:.4f}")
            print(f"Step 11 (80 symbols, 2 years): {meta_acc:.4f}")
            print(f"Delta: {delta:+.4f} ({delta*100:+.2f}%)")

            if meta_acc >= 0.90:
                print(f"\n[OUTSTANDING] Meta-learner maintained 90%+ accuracy!")
                print(f"System is VALIDATED for production deployment.")
                status = "READY FOR PRODUCTION"
            elif meta_acc >= 0.88:
                print(f"\n[EXCELLENT] Meta-learner achieved 88%+ accuracy!")
                print(f"Slight drop is normal with more diverse data.")
                print(f"System is VALIDATED for production deployment.")
                status = "READY FOR PRODUCTION"
            elif meta_acc >= 0.85:
                print(f"\n[GOOD] Meta-learner achieved 85%+ accuracy.")
                print(f"System is functional but could benefit from improvements.")
                status = "READY - Consider Phase 1 improvements"
            else:
                print(f"\n[NEEDS REVIEW] Meta-learner below 85% accuracy.")
                print(f"May need to investigate overfitting or feature engineering.")
                status = "NEEDS IMPROVEMENT"

            # Check overfitting
            if 'rf_metrics' in results:
                rf_train = results['rf_metrics'].get('train_accuracy', 0)
                overfit_gap = rf_train - rf_acc
                print(f"\nOverfitting Check (Random Forest):")
                print(f"  Train: {rf_train:.4f}")
                print(f"  Test:  {rf_acc:.4f}")
                print(f"  Gap:   {overfit_gap:.4f}")

                if overfit_gap < 0.05:
                    print(f"  [OK] No significant overfitting")
                elif overfit_gap < 0.10:
                    print(f"  [ACCEPTABLE] Minor overfitting")
                else:
                    print(f"  [WARNING] Significant overfitting detected")

            # Model weight distribution
            print(f"\nMeta-Learner Weight Distribution:")
            if 'improved_meta_metrics' in results:
                weights = results['improved_meta_metrics'].get('model_weights', {})
                for name, weight in sorted(weights.items(), key=lambda x: -x[1]):
                    acc = results['improved_meta_metrics']['model_accuracies'].get(name, 0)
                    print(f"  {name:25s} {weight*100:5.2f}%  (accuracy: {acc:.4f})")

            print("\n" + "=" * 70)
            print(f"FINAL STATUS: {status}")
            print("=" * 70)

            # Next steps
            if meta_acc >= 0.88:
                print("\nNext Steps:")
                print("  1. Review SESSION_SUMMARY_2025-12-21.md")
                print("  2. Consider Phase 1 improvements:")
                print("     - Increase to 3-5 years historical data")
                print("     - Add macro indicators (VIX, yields, DXY)")
                print("  3. Deploy to production when ready")
                return True
            else:
                print("\nNext Steps:")
                print("  1. Investigate why accuracy dropped")
                print("  2. Check feature importance")
                print("  3. Review training samples for quality")
                print("  4. Consider data preprocessing improvements")
                return False

        else:
            print("\n[FAIL] Test FAILED - Results incomplete")
            return False

    except Exception as e:
        print(f"\n[FAIL] Test FAILED with error: {e}")
        import traceback
        traceback.print_exc()
        return False


if __name__ == '__main__':
    success = main()

    if success:
        print("\n[OK] Step 11 COMPLETE - System validated!")
        sys.exit(0)
    else:
        print("\n[FAIL] Step 11 needs review")
        sys.exit(1)
