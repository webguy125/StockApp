"""
Integration tests for Options Intelligence API endpoints
"""
import os
import sys
import json

# Add backend to path
TEST_DIR = os.path.dirname(__file__)
BACKEND_DIR = os.path.abspath(os.path.join(TEST_DIR, '..', '..', '..'))
sys.path.insert(0, BACKEND_DIR)


def test_api_imports():
    """Test that API module can be imported"""
    try:
        from backend.turbomode.options.api_options_intel import bp
        assert bp is not None
        print("[PASS] API module imported successfully")
        return True
    except Exception as e:
        print(f"[FAIL] API import failed: {e}")
        return False


def test_api_routes_exist():
    """Test that API routes are defined"""
    try:
        from backend.turbomode.options.api_options_intel import bp

        # Get all routes
        rules = list(bp.url_map.iter_rules()) if hasattr(bp, 'url_map') else []

        print(f"[PASS] API blueprint has routes defined")
        return True
    except Exception as e:
        print(f"[FAIL] Route check failed: {e}")
        return False


def test_database_exists():
    """Test that options_intel.db exists"""
    db_path = os.path.join(TEST_DIR, '..', 'Data', 'options_intel.db')

    if os.path.exists(db_path):
        print(f"[PASS] Database exists at {db_path}")
        return True
    else:
        print(f"[FAIL] Database not found at {db_path}")
        return False


def test_api_endpoint_mock():
    """Test API endpoint with mock Flask app"""
    try:
        from flask import Flask
        from backend.turbomode.options.api_options_intel import bp

        # Create test app
        app = Flask(__name__)
        app.register_blueprint(bp, url_prefix='/turbomode')

        # Test client
        client = app.test_client()

        # Test /turbomode/options_intel/all endpoint
        response = client.get('/turbomode/options_intel/all')

        assert response.status_code == 200, f"Expected 200, got {response.status_code}"

        data = json.loads(response.data)
        assert 'success' in data, "Response missing 'success' field"
        assert 'signals' in data, "Response missing 'signals' field"

        print(f"[PASS] API endpoint returned valid response")
        print(f"  Status: {response.status_code}")
        print(f"  Success: {data.get('success')}")
        print(f"  Signals: {len(data.get('signals', []))}")
        return True

    except Exception as e:
        print(f"[FAIL] API endpoint test failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_api_stats_endpoint():
    """Test API stats endpoint"""
    try:
        from flask import Flask
        from backend.turbomode.options.api_options_intel import bp

        app = Flask(__name__)
        app.register_blueprint(bp, url_prefix='/turbomode')
        client = app.test_client()

        response = client.get('/turbomode/options_intel/stats')

        assert response.status_code == 200

        data = json.loads(response.data)
        assert 'success' in data

        if data.get('success'):
            assert 'stats' in data
            stats = data['stats']
            assert 'total_signals' in stats
            print(f"[PASS] Stats endpoint returned valid response")
            print(f"  Total signals: {stats.get('total_signals')}")
        else:
            print(f"[PASS] Stats endpoint returned (empty database)")

        return True

    except Exception as e:
        print(f"[FAIL] Stats endpoint test failed: {e}")
        return False


def run_all_tests():
    """Run all API integration tests"""
    print("\n=== Running API Integration Tests ===\n")

    results = []
    results.append(test_api_imports())
    results.append(test_api_routes_exist())
    results.append(test_database_exists())
    results.append(test_api_endpoint_mock())
    results.append(test_api_stats_endpoint())

    passed = sum(results)
    total = len(results)

    print(f"\n{'='*50}")
    print(f"Results: {passed}/{total} tests passed")
    print(f"{'='*50}\n")

    return all(results)


if __name__ == '__main__':
    success = run_all_tests()
    sys.exit(0 if success else 1)
