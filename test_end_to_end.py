"""
End-to-End Test of Advanced ML System
Tests complete workflow with sample data from core symbols
"""

import sys
import os

# Add backend to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'backend'))

from advanced_ml.config import get_all_core_symbols
from advanced_ml.training.training_pipeline import TrainingPipeline

def main():
    print("=" * 70)
    print("STEP 10: END-TO-END SYSTEM TEST (8-MODEL ENSEMBLE)")
    print("=" * 70)
    print("\nTesting with 10 core symbols (1 year data)")
    print("This will verify:")
    print("  1. Historical backtest engine")
    print("  2. Feature engineering (179 features)")
    print("  3. Random Forest training")
    print("  4. XGBoost training")
    print("  5. LightGBM training")
    print("  6. Extra Trees training")
    print("  7. Gradient Boosting training")
    print("  8. Neural Network training")
    print("  9. Logistic Regression training")
    print(" 10. SVM training")
    print(" 11. Meta-learner ensemble (stacking)")
    print(" 12. Model evaluation")
    print()

    # Get subset of core symbols (10 symbols, diverse sectors)
    all_core = get_all_core_symbols()

    # Select 10 diverse symbols
    test_symbols = [
        'AAPL',   # Tech, Large
        'JPM',    # Financial, Large
        'JNJ',    # Healthcare, Large
        'XOM',    # Energy, Large
        'NEE',    # Utility, Large
        'PLTR',   # Tech, Mid
        'SCHW',   # Financial, Mid
        'DXCM',   # Healthcare, Mid
        'FANG',   # Energy, Mid
        'AES',    # Utility, Mid
    ]

    print(f"Test symbols: {', '.join(test_symbols)}")
    print()

    # Initialize pipeline
    pipeline = TrainingPipeline()

    # Run full pipeline
    try:
        results = pipeline.run_full_pipeline(
            symbols=test_symbols,
            years=1,  # 1 year for speed
            test_size=0.2,
            use_existing_data=False  # Fresh backtest
        )

        # Check results
        print("\n" + "=" * 70)
        print("STEP 10 RESULTS")
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

            # Check if best model has reasonable accuracy
            if best_acc >= 0.75:
                print(f"\n[OK] Accuracy check PASSED (>= 75%)")
                print("\n" + "=" * 70)
                print("PROCEEDING TO STEP 11: FULL BACKTEST ON 80 CORE SYMBOLS")
                print("=" * 70)
                return True
            else:
                print(f"\n[FAIL] Accuracy check FAILED (< 75%)")
                print("  System needs review before full backtest")
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
        print("\n[OK] Step 10 COMPLETE - Ready for Step 11")
        sys.exit(0)
    else:
        print("\n[FAIL] Step 10 FAILED - Fix issues before Step 11")
        sys.exit(1)
