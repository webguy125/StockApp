"""
Integration tests for snapshot system (reader/writer)
"""
import os
import sys

# Add backend to path
TEST_DIR = os.path.dirname(__file__)
OPTIONS_DIR = os.path.abspath(os.path.join(TEST_DIR, '..'))
BACKEND_DIR = os.path.abspath(os.path.join(OPTIONS_DIR, '..', '..'))
sys.path.insert(0, BACKEND_DIR)

# Import modules
sys.path.insert(0, os.path.join(OPTIONS_DIR, 'Data'))
from snapshot_writer import write_snapshot
from snapshot_loader import load_snapshot, load_all_snapshots


def test_snapshot_writer_import():
    """Test that snapshot writer can be imported"""
    try:
        assert write_snapshot is not None
        print("[PASS] Snapshot writer imported successfully")
        return True
    except Exception as e:
        print(f"[FAIL] Snapshot writer import failed: {e}")
        return False


def test_snapshot_loader_import():
    """Test that snapshot loader can be imported"""
    try:
        assert load_snapshot is not None
        assert load_all_snapshots is not None
        print("[PASS] Snapshot loader imported successfully")
        return True
    except Exception as e:
        print(f"[FAIL] Snapshot loader import failed: {e}")
        return False


def test_write_snapshot():
    """Test writing a snapshot"""
    try:
        sample_data = {
            'symbol': 'TEST_SNAPSHOT',
            'base_signal': 'BUY',
            'model_confidence': 0.85,
            'signal_quality_score': 75
        }

        result = write_snapshot('TEST_SNAPSHOT', sample_data)
        assert result is True, "write_snapshot should return True"

        print("[PASS] Snapshot written successfully")
        return True

    except Exception as e:
        print(f"[FAIL] Write snapshot failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_load_snapshot():
    """Test loading a snapshot"""
    try:
        # First write a snapshot
        sample_data = {
            'symbol': 'TEST_LOAD',
            'base_signal': 'SELL',
            'model_confidence': 0.90
        }
        write_snapshot('TEST_LOAD', sample_data)

        # Then load it
        loaded = load_snapshot('TEST_LOAD')

        assert loaded is not None, "Should load snapshot"
        assert loaded['symbol'] == 'TEST_LOAD', "Symbol should match"
        assert loaded['base_signal'] == 'SELL', "Signal type should match"

        print("[PASS] Snapshot loaded successfully")
        print(f"  Loaded: {loaded.get('symbol')}")
        return True

    except Exception as e:
        print(f"[FAIL] Load snapshot failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_load_all_snapshots():
    """Test loading all snapshots"""
    try:
        all_snapshots = load_all_snapshots()

        assert isinstance(all_snapshots, list), "Should return list"

        print(f"[PASS] All snapshots loaded successfully")
        print(f"  Count: {len(all_snapshots)}")
        return True

    except Exception as e:
        print(f"[FAIL] Load all snapshots failed: {e}")
        return False


def test_snapshot_roundtrip():
    """Test complete write-read roundtrip"""
    try:
        # Write comprehensive snapshot
        test_symbol = 'ROUNDTRIP_TEST'
        original_data = {
            'symbol': test_symbol,
            'base_signal': 'BUY',
            'model_confidence': 0.88,
            'model_probability': 0.92,
            'signal_quality_score': 85,
            'drift': 0.05,
            'pressure_index': 0.6,
            'volatility_regime': 'Medium',
            'regime_state': 'Bullish'
        }

        # Write
        write_result = write_snapshot(test_symbol, original_data)
        assert write_result is True

        # Read
        loaded_data = load_snapshot(test_symbol)
        assert loaded_data is not None

        # Verify key fields
        assert loaded_data['symbol'] == original_data['symbol']
        assert loaded_data['base_signal'] == original_data['base_signal']
        assert loaded_data['model_confidence'] == original_data['model_confidence']

        print("[PASS] Snapshot roundtrip successful")
        print(f"  Symbol: {loaded_data['symbol']}")
        print(f"  Signal: {loaded_data['base_signal']}")
        print(f"  Confidence: {loaded_data['model_confidence']}")
        return True

    except Exception as e:
        print(f"[FAIL] Snapshot roundtrip failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def run_all_tests():
    """Run all snapshot system tests"""
    print("\n=== Running Snapshot System Integration Tests ===\n")

    results = []
    results.append(test_snapshot_writer_import())
    results.append(test_snapshot_loader_import())
    results.append(test_write_snapshot())
    results.append(test_load_snapshot())
    results.append(test_load_all_snapshots())
    results.append(test_snapshot_roundtrip())

    passed = sum(results)
    total = len(results)

    print(f"\n{'='*50}")
    print(f"Results: {passed}/{total} tests passed")
    print(f"{'='*50}\n")

    return all(results)


if __name__ == '__main__':
    success = run_all_tests()
    sys.exit(0 if success else 1)
