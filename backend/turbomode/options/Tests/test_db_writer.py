"""
Unit tests for DB writer functionality
"""
import os
import sys
import sqlite3

# Add backend to path
TEST_DIR = os.path.dirname(__file__)
OPTIONS_DIR = os.path.abspath(os.path.join(TEST_DIR, '..'))
BACKEND_DIR = os.path.abspath(os.path.join(OPTIONS_DIR, '..', '..'))
sys.path.insert(0, BACKEND_DIR)

# Import directly from relative path
sys.path.insert(0, os.path.join(OPTIONS_DIR, 'Data'))
from db_writer import write_enriched_to_db

DB_PATH = os.path.join(os.path.dirname(__file__), '..', 'Data', 'options_intel.db')


def test_db_insert():
    """Test basic database insertion"""
    sample = {
        'symbol': 'AAPL',
        'base_signal': 'BUY',
        'model_confidence': 0.85,
        'model_probability': 0.92,
        'signal_quality_score': 88
    }

    result = write_enriched_to_db(sample)
    assert result is True, "DB write should succeed"

    # Verify insertion
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    cursor.execute('SELECT * FROM enriched_options_signals WHERE symbol = ?', ('AAPL',))
    row = cursor.fetchone()
    conn.close()

    assert row is not None, "Record should exist in database"
    print("[PASS] test_db_insert passed")


def test_db_update():
    """Test database update (INSERT OR REPLACE)"""
    initial_data = {
        'symbol': 'TSLA',
        'base_signal': 'SELL',
        'model_confidence': 0.75,
        'signal_quality_score': 70
    }

    # Insert initial
    result1 = write_enriched_to_db(initial_data)
    assert result1 is True

    # Update with new data
    updated_data = {
        'symbol': 'TSLA',
        'base_signal': 'SELL',
        'model_confidence': 0.95,  # Changed
        'signal_quality_score': 92  # Changed
    }

    result2 = write_enriched_to_db(updated_data)
    assert result2 is True

    # Verify update
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    cursor = conn.cursor()
    cursor.execute('SELECT * FROM enriched_options_signals WHERE symbol = ?', ('TSLA',))
    row = dict(cursor.fetchone())
    conn.close()

    assert row['model_confidence'] == 0.95, "Confidence should be updated"
    assert row['signal_quality_score'] == 92, "Quality score should be updated"
    print("[PASS] test_db_update passed")


def test_db_unique_constraint():
    """Test that UNIQUE constraint on symbol works"""
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()

    # Count MSFT records before
    cursor.execute('SELECT COUNT(*) FROM enriched_options_signals WHERE symbol = ?', ('MSFT',))
    count_before = cursor.fetchone()[0]

    # Insert MSFT twice
    data1 = {'symbol': 'MSFT', 'base_signal': 'BUY', 'model_confidence': 0.80}
    data2 = {'symbol': 'MSFT', 'base_signal': 'BUY', 'model_confidence': 0.85}

    write_enriched_to_db(data1)
    write_enriched_to_db(data2)

    # Count MSFT records after
    cursor.execute('SELECT COUNT(*) FROM enriched_options_signals WHERE symbol = ?', ('MSFT',))
    count_after = cursor.fetchone()[0]

    conn.close()

    # Should still only have 1 record (replaced)
    assert count_after == count_before + 1, "Should have exactly one MSFT record"
    print("[PASS] test_db_unique_constraint passed")


def run_all_tests():
    """Run all unit tests"""
    print("\n=== Running DB Writer Unit Tests ===\n")

    try:
        test_db_insert()
        test_db_update()
        test_db_unique_constraint()

        print("\n[SUCCESS] All tests passed!\n")
        return True

    except AssertionError as e:
        print(f"\n[FAIL] Test failed: {e}\n")
        return False

    except Exception as e:
        print(f"\n[ERROR] Test error: {e}\n")
        return False


if __name__ == '__main__':
    success = run_all_tests()
    sys.exit(0 if success else 1)
