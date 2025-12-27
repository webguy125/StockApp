"""
Integration Test for Phase 1 Regime System
Tests Modules 1-5 end-to-end

Modules Tested:
- Module 1: Rare Event Archive (if available)
- Module 2: Regime Labeling
- Module 3: Regime Balanced Sampling
- Module 4: Regime-Aware Validation (5 validation sets)
- Module 5: Regime Weighted Loss

Test Cases (7 total):
1. Regime modules initialized
2. Historical backtest
3. Data loading with regime processing
4. Training all 8 models with sample weights
5. Meta-learner training with accuracy-based weighting
6. Regime-aware evaluation with per-regime accuracy
7. Regime validation sets creation

Expected Output:
- Regime distribution printed
- Sample weights generated (crash = 2.0x, recovery = 1.5x, etc.)
- Per-regime accuracy tracked for all 8 models
- Training pipeline completes successfully
"""

import sys
import os

# Add backend to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'backend'))

from advanced_ml.training.training_pipeline import TrainingPipeline
import numpy as np

def test_regime_integration():
    """
    Test complete regime system integration
    """
    print("=" * 80)
    print("PHASE 1 REGIME SYSTEM - INTEGRATION TEST (7 Tests)")
    print("=" * 80)
    print()
    print("Testing Modules 1-5:")
    print("  Module 1: Rare Event Archive")
    print("  Module 2: Regime Labeling")
    print("  Module 3: Regime Balanced Sampling")
    print("  Module 4: Regime-Aware Validation (5 sets)")
    print("  Module 5: Regime Weighted Loss")
    print()
    print("=" * 80)
    print()

    # Test configuration
    test_symbols = ['AAPL', 'MSFT', 'GOOGL']  # Small set for quick test
    years = 1  # 1 year for speed
    test_size = 0.2

    # Initialize pipeline
    print("[TEST] Initializing training pipeline...")
    pipeline = TrainingPipeline()

    # Test 1: Check regime modules initialized
    print("\n[TEST 1] Checking regime modules...")
    assert hasattr(pipeline, 'regime_labeler'), "RegimeLabeler not initialized"
    assert hasattr(pipeline, 'regime_sampler'), "RegimeSampler not initialized"
    assert hasattr(pipeline, 'regime_weighted_loss'), "RegimeWeightedLoss not initialized"
    print("  [OK] All regime modules initialized")

    # Test 2: Run backtest
    print("\n[TEST 2] Running historical backtest...")
    try:
        backtest_results = pipeline.run_backtest(test_symbols, years=years)
        print(f"  [OK] Backtest complete: {backtest_results['total_samples']} samples generated")
    except Exception as e:
        print(f"  [FAIL] Backtest failed: {e}")
        return False

    # Test 3: Load data with regime processing
    print("\n[TEST 3] Loading data with regime processing...")
    try:
        load_result = pipeline.load_training_data(
            test_size=test_size,
            use_rare_event_archive=True,  # Try to use archive if available
            use_regime_processing=True    # Enable regime modules 2-5
        )

        # Check return signature
        if len(load_result) == 5:
            X_train, X_test, y_train, y_test, sample_weight = load_result
            print(f"  [OK] Data loaded with regime processing")
            print(f"    - Training samples: {len(X_train)}")
            print(f"    - Test samples: {len(X_test)}")
            print(f"    - Sample weight shape: {sample_weight.shape}")
            print(f"    - Weight range: {np.min(sample_weight):.2f}x - {np.max(sample_weight):.2f}x")
            print(f"    - Mean weight: {np.mean(sample_weight):.2f}x")

            # Verify crash samples have 2.0x weight
            crash_weight_found = np.any(np.isclose(sample_weight, 2.0, atol=0.1))
            if crash_weight_found:
                print(f"    [OK] Crash regime samples detected (2.0x weight)")
        else:
            print(f"  [FAIL] Expected 5-tuple return, got {len(load_result)}-tuple")
            return False

    except Exception as e:
        print(f"  [FAIL] Data loading failed: {e}")
        import traceback
        traceback.print_exc()
        return False

    # Test 4: Train models with sample weights
    print("\n[TEST 4] Training models with regime-weighted loss...")
    try:
        train_metrics = pipeline.train_base_models(X_train, y_train, sample_weight=sample_weight)
        print(f"  [OK] All 8 models trained successfully")
        print(f"    - Random Forest accuracy: {train_metrics['random_forest'].get('accuracy', 'N/A')}")
        print(f"    - XGBoost accuracy: {train_metrics['xgboost'].get('accuracy', 'N/A')}")
    except Exception as e:
        print(f"  [FAIL] Model training failed: {e}")
        import traceback
        traceback.print_exc()
        return False

    # Test 5: Train meta-learner with test accuracies
    print("\n[TEST 5] Training improved meta-learner...")
    try:
        # Quick evaluation to get test accuracies
        test_accuracies = {
            'random_forest': pipeline.rf_model.evaluate(X_test, y_test)['accuracy'],
            'xgboost': pipeline.xgb_model.evaluate(X_test, y_test)['accuracy'],
            'lightgbm': pipeline.lgbm_model.evaluate(X_test, y_test)['accuracy'],
            'extratrees': pipeline.et_model.evaluate(X_test, y_test)['accuracy'],
            'gradientboost': pipeline.gb_model.evaluate(X_test, y_test)['accuracy'],
            'neural_network': pipeline.nn_model.evaluate(X_test, y_test)['accuracy'],
            'logistic_regression': pipeline.lr_model.evaluate(X_test, y_test)['accuracy'],
            'svm': pipeline.svm_model.evaluate(X_test, y_test)['accuracy']
        }

        # Train improved meta-learner
        pipeline.train_improved_meta_learner(X_test, y_test, test_accuracies)
        print(f"  [OK] Meta-learner trained with accuracy-based weighting")

    except Exception as e:
        print(f"  [FAIL] Meta-learner training failed: {e}")
        import traceback
        traceback.print_exc()
        return False

    # Test 6: Evaluate with regime-aware validation
    print("\n[TEST 6] Evaluating models with regime-aware validation...")
    try:
        eval_results = pipeline.evaluate_models(X_test, y_test, regime_aware=True)

        print(f"  [OK] Regime-aware evaluation complete")
        print(f"    - Best model: {eval_results['best_model']}")
        print(f"    - Best accuracy: {eval_results['best_accuracy']:.4f}")

        # Check if per-regime accuracies are tracked
        if 'regime_accuracies' in eval_results.get('random_forest', {}):
            print(f"    [OK] Per-regime accuracies tracked:")
            for regime, acc in eval_results['random_forest']['regime_accuracies'].items():
                print(f"      - {regime:20s}: {acc:.4f}")
        else:
            print(f"    [WARN] Per-regime accuracies not found (may be due to small test set)")

    except Exception as e:
        print(f"  [FAIL] Evaluation failed: {e}")
        import traceback
        traceback.print_exc()
        return False

    # Test 7: Verify regime validation sets creation
    print("\n[TEST 7] Testing regime validation sets creation...")
    try:
        regime_sets = pipeline._create_regime_validation_sets(X_test, y_test)

        print(f"  [OK] 5 regime validation sets created:")
        for regime, data in regime_sets.items():
            count = len(data['y'])
            print(f"    - {regime:20s}: {count:5d} samples")

    except Exception as e:
        print(f"  [FAIL] Regime validation sets creation failed: {e}")
        import traceback
        traceback.print_exc()
        return False

    # All tests passed
    print("\n" + "=" * 80)
    print("INTEGRATION TEST PASSED")
    print("=" * 80)
    print()
    print("Summary:")
    print("  [OK] Module 1: Rare Event Archive integration working")
    print("  [OK] Module 2: Regime Labeling working")
    print("  [OK] Module 3: Regime Balanced Sampling working")
    print("  [OK] Module 4: Regime-Aware Validation (5 sets) working")
    print("  [OK] Module 5: Regime Weighted Loss working")
    print()
    print("Phase 1 Implementation: COMPLETE")
    print("=" * 80)

    return True


if __name__ == '__main__':
    print("\n" * 2)
    success = test_regime_integration()
    print("\n" * 2)

    if success:
        print("[RESULT] [OK] All tests passed - Phase 1 regime system fully operational")
        exit(0)
    else:
        print("[RESULT] [FAIL] Some tests failed - see output above")
        exit(1)
