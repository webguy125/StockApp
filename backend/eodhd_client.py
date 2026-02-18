"""
EOD Historical Data (EODHD) API Client
Provides historical options data for training and backtesting
"""
import requests
import os
import json
from datetime import datetime, timedelta
from typing import Dict, List, Optional
import logging

logger = logging.getLogger(__name__)

# EODHD API configuration
def load_api_key():
    """Load API key from config file or environment variable"""
    # Try config file first
    config_path = os.path.join(os.path.dirname(__file__), '..', 'config', 'api_keys.json')
    if os.path.exists(config_path):
        try:
            with open(config_path, 'r') as f:
                config = json.load(f)
                return config.get('EODHD_API_KEY', '')
        except:
            pass
    # Fallback to environment variable
    return os.environ.get('EODHD_API_KEY', '')

EODHD_API_KEY = load_api_key()
EODHD_API_URL = "https://eodhd.com/api"


class EODHDClient:
    """
    EOD Historical Data API Client for historical options data
    """

    def __init__(self, api_key: str = None):
        """
        Initialize EODHD client

        Args:
            api_key: EODHD API key (defaults to environment variable)
        """
        self.api_key = api_key or EODHD_API_KEY
        if not self.api_key:
            logger.warning("[EODHD] No API key provided - set EODHD_API_KEY environment variable")
        self.base_url = EODHD_API_URL

    def get_options_data(self, symbol: str, date: str = None) -> Optional[Dict]:
        """
        Get historical options data for a symbol on a specific date

        Args:
            symbol: Stock ticker (e.g., 'AAPL', 'MSFT')
            date: Date in YYYY-MM-DD format (defaults to most recent trading day)

        Returns:
            Dictionary with options data:
            {
                'symbol': 'AAPL',
                'date': '2026-02-06',
                'expirations': ['2026-02-14', '2026-02-21', ...],
                'data': [
                    {
                        'contract_name': 'AAPL260214C00150000',
                        'expiration': '2026-02-14',
                        'strike': 150.0,
                        'type': 'call',
                        'last_price': 2.50,
                        'bid': 2.48,
                        'ask': 2.52,
                        'volume': 1234,
                        'open_interest': 5678,
                        'implied_volatility': 0.25,
                        'delta': 0.52,
                        'gamma': 0.03,
                        'theta': -0.05,
                        'vega': 0.15
                    },
                    ...
                ]
            }

            Returns None if request fails
        """
        if not self.api_key:
            logger.error("[EODHD] Cannot fetch options data - no API key")
            return None

        try:
            # Default to previous trading day if no date specified
            if not date:
                today = datetime.now()
                # Go back one day (basic approximation - doesn't handle weekends)
                date = (today - timedelta(days=1)).strftime('%Y-%m-%d')

            # Use the advanced EOD options endpoint
            params = {
                'api_token': self.api_key,
                'filter[underlying_symbol]': symbol,
                'limit': 10000  # Get up to 10,000 records per request
            }

            # Filter by date range (trade date)
            if date:
                params['filter[tradetime_from]'] = date
                params['filter[tradetime_to]'] = date

            response = requests.get(
                f"{self.base_url}/mp/unicornbay/options/eod",
                params=params,
                timeout=30
            )

            if response.status_code != 200:
                logger.warning(f"[EODHD] Options request failed for {symbol}: {response.status_code}")
                return None

            result = response.json()

            if not result or 'data' not in result:
                logger.warning(f"[EODHD] No options data for {symbol} on {date}")
                return None

            data = result['data']

            # Convert EODHD format to our expected format
            # EODHD returns data nested in 'attributes' field
            expirations = []
            converted_data = []

            for option in data:
                # Extract attributes (EODHD specific structure)
                attrs = option.get('attributes', {})

                exp = attrs.get('exp_date')
                if exp and exp not in expirations:
                    expirations.append(exp)

                converted_data.append({
                    'contractName': attrs.get('contract'),
                    'expirationDate': attrs.get('exp_date'),
                    'strike': float(attrs.get('strike', 0)),
                    'type': attrs.get('type', '').lower(),
                    'lastTradePrice': float(attrs.get('last', 0)) if attrs.get('last') else None,
                    'bid': float(attrs.get('bid', 0)) if attrs.get('bid') else None,
                    'ask': float(attrs.get('ask', 0)) if attrs.get('ask') else None,
                    'volume': int(attrs.get('volume', 0)) if attrs.get('volume') else None,
                    'openInterest': int(attrs.get('open_interest', 0)) if attrs.get('open_interest') else None,
                    'impliedVolatility': float(attrs.get('volatility', 0)) if attrs.get('volatility') else None,
                    'delta': float(attrs.get('delta', 0)) if attrs.get('delta') else None,
                    'gamma': float(attrs.get('gamma', 0)) if attrs.get('gamma') else None,
                    'theta': float(attrs.get('theta', 0)) if attrs.get('theta') else None,
                    'vega': float(attrs.get('vega', 0)) if attrs.get('vega') else None,
                    'underlyingPrice': float(attrs.get('strike', 0)) * (1 / attrs.get('moneyness', 1)) if attrs.get('moneyness') else None
                })

            return {
                'symbol': symbol,
                'date': date,
                'expirations': sorted(expirations) if expirations else [],
                'data': converted_data
            }

        except requests.exceptions.Timeout:
            logger.warning(f"[EODHD] Timeout fetching options for {symbol}")
            return None
        except requests.exceptions.RequestException as e:
            logger.warning(f"[EODHD] Request error for {symbol}: {e}")
            return None
        except Exception as e:
            logger.error(f"[EODHD] Unexpected error fetching options for {symbol}: {e}")
            return None

    def get_bulk_options_data(self, symbol: str, start_date: str, end_date: str) -> List[Dict]:
        """
        Get historical options data for a symbol over a date range

        Args:
            symbol: Stock ticker
            start_date: Start date in YYYY-MM-DD format
            end_date: End date in YYYY-MM-DD format

        Returns:
            List of options data dictionaries (one per date)
        """
        results = []
        current_date = datetime.strptime(start_date, '%Y-%m-%d')
        end = datetime.strptime(end_date, '%Y-%m-%d')

        while current_date <= end:
            date_str = current_date.strftime('%Y-%m-%d')

            # Skip weekends (basic check)
            if current_date.weekday() < 5:  # Monday=0, Friday=4
                logger.info(f"[EODHD] Fetching options for {symbol} on {date_str}")
                data = self.get_options_data(symbol, date_str)
                if data:
                    results.append(data)

            current_date += timedelta(days=1)

        return results


# Singleton instance
_eodhd_client = None


def get_eodhd_client() -> EODHDClient:
    """
    Get singleton EODHDClient instance

    Returns:
        EODHDClient instance
    """
    global _eodhd_client
    if _eodhd_client is None:
        _eodhd_client = EODHDClient()
    return _eodhd_client


def get_historical_options(symbol: str, date: str = None) -> Optional[Dict]:
    """
    Get historical options data for a symbol

    Args:
        symbol: Stock ticker
        date: Date in YYYY-MM-DD format (optional)

    Returns:
        Options data dictionary, or None if unavailable
    """
    client = get_eodhd_client()
    return client.get_options_data(symbol, date)


if __name__ == '__main__':
    # Test EODHD client
    print("=" * 80)
    print("EODHD CLIENT TEST")
    print("=" * 80)

    client = EODHDClient()

    if not client.api_key:
        print("\n[ERROR] No EODHD_API_KEY environment variable set")
        print("Set it with: set EODHD_API_KEY=your_key_here (Windows)")
        print("Or: export EODHD_API_KEY=your_key_here (Linux/Mac)")
    else:
        # Test with AAPL
        symbol = 'AAPL'
        print(f"\nFetching options data for {symbol}...")

        data = client.get_options_data(symbol)

        if data:
            print(f"  Symbol: {data['symbol']}")
            print(f"  Date: {data['date']}")
            print(f"  Expirations: {len(data['expirations'])}")
            print(f"  Total options: {len(data['data'])}")

            if data['expirations']:
                print(f"  First expiration: {data['expirations'][0]}")
        else:
            print(f"  Failed to fetch options data")

    print("\n" + "=" * 80)
    print("TEST COMPLETE")
    print("=" * 80)
