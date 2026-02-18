"""
Integration tests for Options Scanner Scheduler
"""
import os
import sys
from datetime import datetime, time as dt_time
from unittest.mock import Mock, patch

# Add paths
TEST_DIR = os.path.dirname(__file__)
OPTIONS_DIR = os.path.abspath(os.path.join(TEST_DIR, '..'))
SCANNER_DIR = os.path.join(OPTIONS_DIR, 'Scanner')
BACKEND_DIR = os.path.abspath(os.path.join(OPTIONS_DIR, '..', '..'))
sys.path.insert(0, BACKEND_DIR)
sys.path.insert(0, SCANNER_DIR)


def test_scheduler_can_be_imported():
    """Test that scheduler module can be imported with mocked dependencies"""
    print("[TEST] Testing scheduler import with mocked dependencies...")

    # Mock the scanner dependencies
    sys.modules['backend'] = Mock()
    sys.modules['backend.turbomode'] = Mock()
    sys.modules['backend.turbomode.Options'] = Mock()
    sys.modules['backend.turbomode.Options.Scanner'] = Mock()
    sys.modules['backend.turbomode.Options.Scanner.options_scanner'] = Mock()
    sys.modules['backend.turbomode.Options.Scanner.scanner_logger'] = Mock()

    try:
        import options_scanner_scheduler as scheduler_module
        print("[PASS] Scheduler module imported successfully")
        return scheduler_module
    except Exception as e:
        raise AssertionError(f"Failed to import scheduler: {e}")


def test_scheduler_functions_exist(scheduler_module):
    """Test that required functions exist"""
    required_functions = [
        'start_scheduler',
        'safe_run_scanner',
        'should_run_scanner',
        'get_scheduler_stats',
        'is_market_hours',
        'is_weekday',
        'log_scheduler'
    ]

    for func_name in required_functions:
        assert hasattr(scheduler_module, func_name), f"Missing function: {func_name}"
        assert callable(getattr(scheduler_module, func_name)), f"{func_name} is not callable"

    print(f"[PASS] All {len(required_functions)} required functions exist")


def test_scheduler_configuration(scheduler_module):
    """Test scheduler configuration values"""
    assert hasattr(scheduler_module, 'SCAN_INTERVAL_MINUTES'), "Missing SCAN_INTERVAL_MINUTES"
    assert hasattr(scheduler_module, 'MARKET_OPEN'), "Missing MARKET_OPEN"
    assert hasattr(scheduler_module, 'MARKET_CLOSE'), "Missing MARKET_CLOSE"

    # Verify types
    assert isinstance(scheduler_module.SCAN_INTERVAL_MINUTES, int), "SCAN_INTERVAL_MINUTES should be int"
    assert isinstance(scheduler_module.MARKET_OPEN, dt_time), "MARKET_OPEN should be time"
    assert isinstance(scheduler_module.MARKET_CLOSE, dt_time), "MARKET_CLOSE should be time"

    # Verify values are reasonable
    assert 1 <= scheduler_module.SCAN_INTERVAL_MINUTES <= 60, "Scan interval should be 1-60 minutes"
    assert scheduler_module.MARKET_OPEN < scheduler_module.MARKET_CLOSE, "Market open should be before close"

    print(f"[PASS] Configuration validated")
    print(f"  Scan interval: {scheduler_module.SCAN_INTERVAL_MINUTES} minutes")
    print(f"  Market hours: {scheduler_module.MARKET_OPEN} - {scheduler_module.MARKET_CLOSE}")


def test_market_hours_logic(scheduler_module):
    """Test market hours detection logic"""
    result = scheduler_module.is_market_hours()
    assert isinstance(result, bool), "is_market_hours should return boolean"

    now = datetime.now().time()
    expected = scheduler_module.MARKET_OPEN <= now <= scheduler_module.MARKET_CLOSE

    assert result == expected, f"Market hours logic incorrect: got {result}, expected {expected}"
    print(f"[PASS] Market hours logic validated (current: {result})")


def test_weekday_logic(scheduler_module):
    """Test weekday detection logic"""
    result = scheduler_module.is_weekday()
    assert isinstance(result, bool), "is_weekday should return boolean"

    weekday = datetime.now().weekday()
    expected = weekday < 5  # Mon-Fri

    assert result == expected, f"Weekday logic incorrect: got {result}, expected {expected}"
    print(f"[PASS] Weekday logic validated (weekday={weekday}, is_weekday={result})")


def test_should_run_scanner_logic(scheduler_module):
    """Test combined market hours + weekday logic"""
    result = scheduler_module.should_run_scanner()
    assert isinstance(result, bool), "should_run_scanner should return boolean"

    is_weekday = scheduler_module.is_weekday()
    is_market_hours = scheduler_module.is_market_hours()
    expected = is_weekday and is_market_hours

    assert result == expected, f"should_run_scanner logic incorrect: got {result}, expected {expected}"
    print(f"[PASS] should_run_scanner logic validated (result={result})")


def test_scheduler_stats_structure(scheduler_module):
    """Test scheduler statistics structure"""
    stats = scheduler_module.get_scheduler_stats()

    # Verify structure
    assert isinstance(stats, dict), "get_scheduler_stats should return dict"

    required_keys = ['total_scans', 'total_errors', 'error_rate', 'avg_runtime', 'total_runtime']
    for key in required_keys:
        assert key in stats, f"Missing key in stats: {key}"

    # Verify types
    assert isinstance(stats['total_scans'], int), "total_scans should be int"
    assert isinstance(stats['total_errors'], int), "total_errors should be int"
    assert isinstance(stats['error_rate'], float), "error_rate should be float"
    assert isinstance(stats['avg_runtime'], float), "avg_runtime should be float"
    assert isinstance(stats['total_runtime'], float), "total_runtime should be float"

    print(f"[PASS] Scheduler stats structure validated: {stats}")


def test_metrics_tracking(scheduler_module):
    """Test that metrics tracking variables exist"""
    assert hasattr(scheduler_module, 'scan_count'), "Missing scan_count variable"
    assert hasattr(scheduler_module, 'error_count'), "Missing error_count variable"
    assert hasattr(scheduler_module, 'total_runtime'), "Missing total_runtime variable"

    assert isinstance(scheduler_module.scan_count, int), "scan_count should be int"
    assert isinstance(scheduler_module.error_count, int), "error_count should be int"
    assert isinstance(scheduler_module.total_runtime, float), "total_runtime should be float"

    print("[PASS] Metrics tracking variables validated")


def run_all_tests():
    """Run all scheduler integration tests"""
    print("\n=== Running Scheduler Integration Tests ===\n")

    try:
        # Import with mocked dependencies
        scheduler_module = test_scheduler_can_be_imported()

        # Run tests
        test_scheduler_functions_exist(scheduler_module)
        test_scheduler_configuration(scheduler_module)
        test_market_hours_logic(scheduler_module)
        test_weekday_logic(scheduler_module)
        test_should_run_scanner_logic(scheduler_module)
        test_scheduler_stats_structure(scheduler_module)
        test_metrics_tracking(scheduler_module)

        print("\n[SUCCESS] All scheduler tests passed!\n")
        return True

    except AssertionError as e:
        print(f"\n[FAIL] Test failed: {e}\n")
        return False

    except Exception as e:
        print(f"\n[ERROR] Test error: {e}\n")
        import traceback
        traceback.print_exc()
        return False


if __name__ == '__main__':
    success = run_all_tests()
    sys.exit(0 if success else 1)
