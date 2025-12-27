"""
Step 11 Phase 1: Full 80-Symbol Backtest with Improvements
Enhanced validation with regime/macro features and better regularization

Phase 1 Improvements:
- 204 total features (179 base + 25 regime/macro)
- Better model regularization (reduced overfitting)
- 5 years of historical data (vs 2 years)
- Market regime awareness
- Macro economic indicators
"""

import sys
import os

# Add backend to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'backend'))

from advanced_ml.config import get_all_core_symbols
from advanced_ml.training.training_pipeline import TrainingPipeline

def main():
    print("=" * 70)
    print("STEP 11 PHASE 1: FULL 80-SYMBOL BACKTEST (8-MODEL ENSEMBLE)")
    print("=" * 70)
    print("\nTraining on ALL 80 core symbols (5 years data)")
    print("Phase 1 Improvements Applied:")
    print("  - 204 features (179 base + 25 regime/macro)")
    print("  - Better regularization (reduced overfitting)")
    print("  - 5 years historical data (more market cycles)")
    print()
    print("Expected:")
    print("  - ~85,000 training samples (vs 33,930 baseline)")
    print("  - Runtime: 8-12 hours (more data = longer training)")
    print("  - Meta-learner accuracy: 90-92%+")
    print("  - Overfitting gap: <5% (vs 10.12% baseline)")
    print()

    # Get ALL core symbols
    all_core = get_all_core_symbols()

    print(f"Core Symbols: {len(all_core)} symbols")
    print(f"Symbols: {', '.join(all_core[:10])}... (showing first 10)")
    print()

    # Initialize pipeline
    pipeline = TrainingPipeline()

    # Run full pipeline on ALL 80 symbols with 5 years of data
    try:
        results = pipeline.run_full_pipeline(
            symbols=all_core,
            years=5,  # PHASE 1: 5 years for more data and market cycles
            test_size=0.2,
            use_existing_data=False  # Fresh backtest
        )

        # Check results
        print("\n" + "=" * 70)
        print("STEP 11 PHASE 1 RESULTS")
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
            print("PHASE 1 IMPROVEMENT ANALYSIS")
            print("=" * 70)

            # Compare to baseline (Step 11 without improvements)
            baseline_meta = 0.8518  # From baseline run
            baseline_overfit = 0.1012  # 10.12% overfitting gap
            delta = meta_acc - baseline_meta

            print(f"\nBaseline (2 years, 179 features):   {baseline_meta:.4f}")
            print(f"Phase 1  (5 years, 204 features):   {meta_acc:.4f}")
            print(f"Improvement: {delta:+.4f} ({delta*100:+.2f}%)")

            if meta_acc >= 0.92:
                print(f"\n[OUTSTANDING] Meta-learner achieved 92%+ accuracy!")
                print(f"Phase 1 improvements EXCEEDED expectations!")
                status = "EXCEPTIONAL - READY FOR PRODUCTION"
            elif meta_acc >= 0.90:
                print(f"\n[EXCELLENT] Meta-learner achieved 90%+ accuracy!")
                print(f"Phase 1 improvements SUCCESSFUL!")
                print(f"System is VALIDATED for production deployment.")
                status = "READY FOR PRODUCTION"
            elif meta_acc >= 0.88:
                print(f"\n[GOOD] Meta-learner achieved 88%+ accuracy.")
                print(f"Phase 1 improvements showed positive impact.")
                print(f"System is functional and ready for production.")
                status = "READY FOR PRODUCTION"
            elif meta_acc >= 0.86:
                print(f"\n[ACCEPTABLE] Meta-learner achieved 86%+ accuracy.")
                print(f"Phase 1 improvements showed some benefit.")
                print(f"Consider Phase 2 for further optimization.")
                status = "READY - Consider Phase 2"
            else:
                print(f"\n[NEEDS REVIEW] Meta-learner below 86% accuracy.")
                print(f"Phase 1 improvements may not have been fully applied.")
                print(f"Verify regime/macro features are integrated.")
                status = "NEEDS INVESTIGATION"

            # Check overfitting improvement
            if 'rf_metrics' in results:
                rf_train = results['rf_metrics'].get('train_accuracy', 0)
                overfit_gap = rf_train - rf_acc
                print(f"\nOverfitting Analysis:")
                print(f"  Baseline Gap:  {baseline_overfit:.4f} (10.12%)")
                print(f"  Phase 1 Gap:   {overfit_gap:.4f}")
                print(f"  Improvement:   {baseline_overfit - overfit_gap:+.4f}")

                if overfit_gap < 0.05:
                    print(f"  [EXCELLENT] Overfitting eliminated (<5%)")
                elif overfit_gap < 0.08:
                    print(f"  [GOOD] Overfitting significantly reduced")
                elif overfit_gap < baseline_overfit:
                    print(f"  [OK] Some improvement over baseline")
                else:
                    print(f"  [WARNING] No improvement in overfitting")

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
            if meta_acc >= 0.90:
                print("\nNext Steps:")
                print("  1. [OK] Phase 1 improvements SUCCESSFUL!")
                print("  2. System ready for production deployment")
                print("  3. Optional: Consider Phase 2 improvements:")
                print("     - Regime-specific models")
                print("     - Advanced feature selection")
                print("     - Hyperparameter optimization")
                print("  4. Implement 1 AM automation")
                return True
            elif meta_acc >= 0.86:
                print("\nNext Steps:")
                print("  1. Phase 1 showed positive results")
                print("  2. Can deploy to production")
                print("  3. Monitor performance and consider Phase 2")
                print("  4. Implement 1 AM automation")
                return True
            else:
                print("\nNext Steps:")
                print("  1. Verify all Phase 1 changes were applied:")
                print("     - Check regime_macro_features.py exists")
                print("     - Verify historical_backtest.py has import")
                print("     - Confirm model hyperparameters updated")
                print("  2. Review training logs for errors")
                print("  3. Check feature count (should be ~204)")
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
        print("\n[OK] Step 11 Phase 1 COMPLETE - Improvements validated!")
        sys.exit(0)
    else:
        print("\n[FAIL] Step 11 Phase 1 needs review")
        sys.exit(1)
