"""
End-to-end integration test for options scanner
"""
import os
import sys

# Add backend to path
TEST_DIR = os.path.dirname(__file__)
OPTIONS_DIR = os.path.abspath(os.path.join(TEST_DIR, '..'))
SCANNER_DIR = os.path.join(OPTIONS_DIR, 'Scanner')
BACKEND_DIR = os.path.abspath(os.path.join(OPTIONS_DIR, '..', '..'))
sys.path.insert(0, BACKEND_DIR)
sys.path.insert(0, SCANNER_DIR)


def test_scanner_module_import():
    """Test that scanner module can be imported"""
    try:
        # Mock dependencies
        from unittest.mock import Mock
        sys.modules['backend'] = Mock()
        sys.modules['backend.turbomode'] = Mock()
        sys.modules['backend.turbomode.Options'] = Mock()
        sys.modules['backend.turbomode.Options.unified_options_client'] = Mock()
        sys.modules['backend.turbomode.Options.Engines'] = Mock()
        sys.modules['backend.turbomode.Options.Engines.expiration_selector'] = Mock()
        sys.modules['backend.turbomode.Options.Engines.wing_selector'] = Mock()
        sys.modules['backend.turbomode.Options.Engines.condor_pricing'] = Mock()
        sys.modules['backend.turbomode.Options.Engines.analytics_engine'] = Mock()
        sys.modules['backend.turbomode.Options.Engines.regime_engine'] = Mock()
        sys.modules['backend.turbomode.Options.Engines.transition_engine'] = Mock()
        sys.modules['backend.turbomode.Options.Engines.signal_intelligence'] = Mock()
        sys.modules['backend.turbomode.Options.Data'] = Mock()
        sys.modules['backend.turbomode.Options.Data.snapshot_loader'] = Mock()
        sys.modules['backend.turbomode.Options.Data.snapshot_writer'] = Mock()
        sys.modules['backend.turbomode.Options.Data.db_writer'] = Mock()
        sys.modules['backend.turbomode.Options.Scanner'] = Mock()
        sys.modules['backend.turbomode.Options.Scanner.scanner_logger'] = Mock()

        import options_scanner
        assert hasattr(options_scanner, 'run_options_scanner')

        print("[PASS] Scanner module imported successfully")
        return True

    except Exception as e:
        print(f"[FAIL] Scanner import failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_scanner_function_exists():
    """Test that main scanner function exists"""
    try:
        from unittest.mock import Mock

        # Mock all dependencies first
        sys.modules['backend'] = Mock()
        sys.modules['backend.turbomode'] = Mock()
        sys.modules['backend.turbomode.Options'] = Mock()
        sys.modules['backend.turbomode.Options.unified_options_client'] = Mock()
        sys.modules['backend.turbomode.Options.Engines'] = Mock()
        sys.modules['backend.turbomode.Options.Engines.expiration_selector'] = Mock()
        sys.modules['backend.turbomode.Options.Engines.wing_selector'] = Mock()
        sys.modules['backend.turbomode.Options.Engines.condor_pricing'] = Mock()
        sys.modules['backend.turbomode.Options.Engines.analytics_engine'] = Mock()
        sys.modules['backend.turbomode.Options.Engines.regime_engine'] = Mock()
        sys.modules['backend.turbomode.Options.Engines.transition_engine'] = Mock()
        sys.modules['backend.turbomode.Options.Engines.signal_intelligence'] = Mock()
        sys.modules['backend.turbomode.Options.Data'] = Mock()
        sys.modules['backend.turbomode.Options.Data.snapshot_loader'] = Mock()
        sys.modules['backend.turbomode.Options.Data.snapshot_writer'] = Mock()
        sys.modules['backend.turbomode.Options.Data.db_writer'] = Mock()
        sys.modules['backend.turbomode.Options.Scanner'] = Mock()
        sys.modules['backend.turbomode.Options.Scanner.scanner_logger'] = Mock()

        import options_scanner

        assert callable(options_scanner.run_options_scanner)

        print("[PASS] Scanner function exists and is callable")
        return True

    except Exception as e:
        print(f"[FAIL] Scanner function check failed: {e}")
        return False


def test_scanner_paths_exist():
    """Test that scanner file exists at expected location"""
    scanner_path = os.path.join(SCANNER_DIR, 'options_scanner.py')

    if os.path.exists(scanner_path):
        print(f"[PASS] Scanner file exists at {scanner_path}")
        return True
    else:
        print(f"[FAIL] Scanner file not found at {scanner_path}")
        return False


def test_scanner_logger_exists():
    """Test that scanner logger exists"""
    logger_path = os.path.join(SCANNER_DIR, 'scanner_logger.py')

    if os.path.exists(logger_path):
        print(f"[PASS] Scanner logger exists at {logger_path}")
        return True
    else:
        print(f"[FAIL] Scanner logger not found at {logger_path}")
        return False


def test_database_paths():
    """Test that database paths are correct"""
    db_path = os.path.join(OPTIONS_DIR, 'Data', 'options_intel.db')
    schema_path = os.path.join(OPTIONS_DIR, 'Data', 'schema_options_intel.sql')

    db_exists = os.path.exists(db_path)
    schema_exists = os.path.exists(schema_path)

    if db_exists and schema_exists:
        print(f"[PASS] Database and schema files exist")
        return True
    else:
        print(f"[FAIL] Database exists: {db_exists}, Schema exists: {schema_exists}")
        return False


def run_all_tests():
    """Run all end-to-end scanner tests"""
    print("\n=== Running Scanner End-to-End Integration Tests ===\n")

    results = []
    results.append(test_scanner_paths_exist())
    results.append(test_scanner_logger_exists())
    results.append(test_database_paths())
    results.append(test_scanner_module_import())
    results.append(test_scanner_function_exists())

    passed = sum(results)
    total = len(results)

    print(f"\n{'='*50}")
    print(f"Results: {passed}/{total} tests passed")
    print(f"{'='*50}\n")

    return all(results)


if __name__ == '__main__':
    success = run_all_tests()
    sys.exit(0 if success else 1)
