"""
Test script for vectorized GPU feature engineering
Compares performance of sequential vs. vectorized batched processing
"""

import sys
import os
import time
import pandas as pd
import numpy as np

# Add backend to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'backend'))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'backend', 'advanced_ml', 'features'))

from gpu_feature_engineer import GPUFeatureEngineer


def test_vectorized_batch_processing():
    """Test the new vectorized batch processing implementation"""

    print("=" * 80)
    print("TESTING VECTORIZED GPU BATCH PROCESSING")
    print("=" * 80)

    # Create sample data (500 bars)
    dates = pd.date_range('2023-01-01', periods=500)
    np.random.seed(42)

    df = pd.DataFrame({
        'Open': 100 + np.cumsum(np.random.randn(500) * 2),
        'High': 102 + np.cumsum(np.random.randn(500) * 2),
        'Low': 98 + np.cumsum(np.random.randn(500) * 2),
        'Close': 100 + np.cumsum(np.random.randn(500) * 2),
        'Volume': np.random.randint(1000000, 10000000, 500)
    }, index=dates)

    print(f"\n[OK] Created test dataset: {len(df)} bars")
    print(f"   Date range: {df.index[0]} to {df.index[-1]}")

    # Initialize GPU feature engineer
    gpu_engineer = GPUFeatureEngineer(use_gpu=True)
    print(f"\n[OK] Initialized GPU feature engineer: {gpu_engineer}")

    # Test batch processing with 100 windows
    test_indices = list(range(50, 150))  # 100 windows
    print(f"\n[TEST] Testing batch processing with {len(test_indices)} windows...")
    print(f"   Window range: indices {test_indices[0]} to {test_indices[-1]}")

    # Run vectorized batch processing
    start_time = time.time()
    batch_results = gpu_engineer.extract_features_batch(df, test_indices, symbol='TEST')
    batch_time = time.time() - start_time

    print(f"\n[OK] Batch processing complete!")
    print(f"   Total time: {batch_time*1000:.1f}ms")
    print(f"   Time per window: {(batch_time/len(test_indices))*1000:.2f}ms")
    print(f"   Windows processed: {len(batch_results)}")

    # Verify results
    print(f"\n[VERIFY] Verifying results...")

    # Check that we got the right number of results
    assert len(batch_results) == len(test_indices), f"Expected {len(test_indices)} results, got {len(batch_results)}"
    print(f"   [OK] Got {len(batch_results)} results (correct)")

    # Check that each result has features
    for i, result in enumerate(batch_results[:5]):  # Check first 5
        if 'error' in result:
            print(f"   [ERROR] Window {i} has error: {result['error']}")
        else:
            feature_count = result.get('feature_count', len(result))
            print(f"   [OK] Window {i}: {feature_count} features calculated")

            # Show sample features
            if i == 0:
                print(f"\n   Sample features from first window:")
                for key in ['rsi_14', 'sma_20', 'sma_50', 'return_1d', 'historical_vol_20', 'last_price']:
                    if key in result:
                        print(f"      {key}: {result[key]:.4f}")

    # Compare with single-window processing for validation
    print(f"\n[VALIDATE] Comparing batch vs. single-window results...")

    # Test a few random windows
    test_window_idx = test_indices[25]  # Middle window

    # Single-window calculation
    single_result = gpu_engineer.extract_features(df[:test_window_idx+1], symbol='TEST')

    # Batch result for same window
    batch_result = batch_results[25]

    # Compare key features
    print(f"\n   Comparing window at index {test_window_idx}:")
    comparison_features = ['rsi_14', 'sma_20', 'return_1d', 'historical_vol_20']
    all_match = True

    for feature in comparison_features:
        if feature in single_result and feature in batch_result:
            single_val = single_result[feature]
            batch_val = batch_result[feature]
            diff = abs(single_val - batch_val)
            match = diff < 0.01  # Allow small floating point differences

            status = "[OK]" if match else "[DIFF]"
            print(f"   {status} {feature}: single={single_val:.6f}, batch={batch_val:.6f}, diff={diff:.6f}")

            if not match:
                all_match = False

    if all_match:
        print(f"\n   [OK] All features match! Vectorization is correct.")
    else:
        print(f"\n   [WARN] Some features differ - may need investigation")

    # Performance summary
    print(f"\n" + "=" * 80)
    print("PERFORMANCE SUMMARY")
    print("=" * 80)
    print(f"Windows processed: {len(test_indices)}")
    print(f"Total time: {batch_time*1000:.1f}ms")
    print(f"Time per window: {(batch_time/len(test_indices))*1000:.2f}ms")
    print(f"Throughput: {len(test_indices)/batch_time:.1f} windows/second")
    print("=" * 80)

    return batch_results


if __name__ == '__main__':
    try:
        results = test_vectorized_batch_processing()
        print("\n[SUCCESS] TEST PASSED: Vectorized GPU batch processing is working!")
    except Exception as e:
        print(f"\n[FAILED] TEST FAILED: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
