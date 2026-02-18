"""
Tradier API Client
Provides real-time equity quotes via REST API
"""
import requests
import os
from datetime import datetime
from typing import Dict, Optional
import logging

logger = logging.getLogger(__name__)

# Tradier API configuration
TRADIER_API_KEY = "pplYfsA91vM8AAFoSmLB4naoaDa5"
TRADIER_API_URL = "https://api.tradier.com/v1"


class TradierClient:
    """
    Tradier API Client for real-time equity quotes
    """

    def __init__(self, api_key: str = None):
        """
        Initialize Tradier client

        Args:
            api_key: Tradier API key (defaults to hardcoded key)
        """
        self.api_key = api_key or TRADIER_API_KEY
        self.base_url = TRADIER_API_URL
        self.headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Accept": "application/json"
        }

    def get_history(self, symbol: str, interval: str = '1min', start: str = None, end: str = None) -> Optional[list]:
        """
        Get historical OHLC bars for a symbol

        Args:
            symbol: Stock ticker (e.g., 'AAPL', 'MSFT')
            interval: Time interval - 'daily', 'weekly', 'monthly'
                     Note: Tradier doesn't provide intraday historical bars via REST
            start: Start date YYYY-MM-DD (optional)
            end: End date YYYY-MM-DD (optional)

        Returns:
            List of OHLC candles:
            [
                {
                    'Date': '2026-02-05',
                    'Open': 150.25,
                    'High': 150.75,
                    'Low': 150.10,
                    'Close': 150.50,
                    'Volume': 12500000
                },
                ...
            ]

            Returns None if request fails
        """
        # Map interval to Tradier format
        interval_map = {
            '1d': 'daily',
            '1w': 'weekly',
            '1mo': 'monthly',
            'daily': 'daily',
            'weekly': 'weekly',
            'monthly': 'monthly'
        }

        tradier_interval = interval_map.get(interval, 'daily')

        try:
            params = {
                "symbol": symbol,
                "interval": tradier_interval
            }

            if start:
                params['start'] = start
            if end:
                params['end'] = end

            response = requests.get(
                f"{self.base_url}/markets/history",
                headers=self.headers,
                params=params,
                timeout=10
            )

            if response.status_code != 200:
                logger.warning(f"[TRADIER] History request failed: {response.status_code}")
                return None

            data = response.json()

            # Extract history from response
            if "history" not in data:
                logger.warning(f"[TRADIER] No history data for {symbol}")
                return None

            history = data["history"]

            # Check if history is None or has no data
            if history is None or "day" not in history:
                logger.warning(f"[TRADIER] Empty history for {symbol}")
                return None

            days = history["day"]

            # Handle single day response (not a list)
            if not isinstance(days, list):
                days = [days]

            # Convert to standard OHLCV format
            candles = []
            for day in days:
                candles.append({
                    'Date': day.get('date', ''),
                    'Open': float(day.get('open', 0)),
                    'High': float(day.get('high', 0)),
                    'Low': float(day.get('low', 0)),
                    'Close': float(day.get('close', 0)),
                    'Volume': int(day.get('volume', 0))
                })

            return candles

        except requests.exceptions.Timeout:
            logger.warning(f"[TRADIER] Timeout fetching history for {symbol}")
            return None
        except requests.exceptions.RequestException as e:
            logger.warning(f"[TRADIER] Request error for {symbol}: {e}")
            return None
        except Exception as e:
            logger.error(f"[TRADIER] Unexpected error fetching history for {symbol}: {e}")
            return None

    def get_option_chain(self, symbol: str, expiration: str = None) -> Optional[Dict]:
        """
        Get options chain for a symbol

        Args:
            symbol: Stock ticker (e.g., 'AAPL', 'MSFT')
            expiration: Optional expiration date in YYYY-MM-DD format

        Returns:
            Dictionary with options chain data:
            {
                'symbol': 'AAPL',
                'expirations': ['2026-02-14', '2026-02-21', ...],
                'chains': {
                    '2026-02-14': {
                        'calls': {
                            150.0: {
                                'strike': 150.0,
                                'bid': 2.50,
                                'ask': 2.55,
                                'last': 2.52,
                                'volume': 1234,
                                'open_interest': 5678,
                                'implied_volatility': 0.25,
                                'delta': 0.52,
                                'gamma': 0.03,
                                'theta': -0.05,
                                'vega': 0.15
                            },
                            ...
                        },
                        'puts': {
                            150.0: {...},
                            ...
                        }
                    },
                    ...
                }
            }

            Returns None if request fails
        """
        try:
            # First, get available expirations
            exp_response = requests.get(
                f"{self.base_url}/markets/options/expirations",
                headers=self.headers,
                params={"symbol": symbol, "includeAllRoots": "true"},
                timeout=10
            )

            if exp_response.status_code != 200:
                logger.warning(f"[TRADIER] Expirations request failed: {exp_response.status_code}")
                return None

            exp_data = exp_response.json()

            if "expirations" not in exp_data or "date" not in exp_data["expirations"]:
                logger.warning(f"[TRADIER] No expirations data for {symbol}")
                return None

            expirations = exp_data["expirations"]["date"]
            if not isinstance(expirations, list):
                expirations = [expirations]

            # If specific expiration requested, use only that
            if expiration:
                if expiration in expirations:
                    expirations = [expiration]
                else:
                    logger.warning(f"[TRADIER] Requested expiration {expiration} not available for {symbol}")
                    return None

            # Get chains for all expirations (or just the requested one)
            chains = {}
            for exp in expirations[:10]:  # Limit to first 10 expirations for performance
                chain_response = requests.get(
                    f"{self.base_url}/markets/options/chains",
                    headers=self.headers,
                    params={"symbol": symbol, "expiration": exp, "greeks": "true"},
                    timeout=10
                )

                if chain_response.status_code != 200:
                    logger.warning(f"[TRADIER] Chain request failed for {symbol} exp {exp}: {chain_response.status_code}")
                    continue

                chain_data = chain_response.json()

                if "options" not in chain_data or "option" not in chain_data["options"]:
                    logger.warning(f"[TRADIER] No options data for {symbol} exp {exp}")
                    continue

                options = chain_data["options"]["option"]
                if not isinstance(options, list):
                    options = [options]

                # Organize by strike and type
                calls = {}
                puts = {}

                for opt in options:
                    strike = float(opt.get('strike', 0))
                    option_type = opt.get('option_type', '')

                    option_data = {
                        'strike': strike,
                        'bid': float(opt.get('bid', 0)) if opt.get('bid') else 0,
                        'ask': float(opt.get('ask', 0)) if opt.get('ask') else 0,
                        'last': float(opt.get('last', 0)) if opt.get('last') else 0,
                        'volume': int(opt.get('volume', 0)) if opt.get('volume') else 0,
                        'open_interest': int(opt.get('open_interest', 0)) if opt.get('open_interest') else 0,
                        'implied_volatility': float(opt.get('greeks', {}).get('mid_iv', 0)) if opt.get('greeks') else 0,
                        'delta': float(opt.get('greeks', {}).get('delta', 0)) if opt.get('greeks') else 0,
                        'gamma': float(opt.get('greeks', {}).get('gamma', 0)) if opt.get('greeks') else 0,
                        'theta': float(opt.get('greeks', {}).get('theta', 0)) if opt.get('greeks') else 0,
                        'vega': float(opt.get('greeks', {}).get('vega', 0)) if opt.get('greeks') else 0
                    }

                    if option_type == 'call':
                        calls[strike] = option_data
                    elif option_type == 'put':
                        puts[strike] = option_data

                chains[exp] = {
                    'calls': calls,
                    'puts': puts
                }

            return {
                'symbol': symbol,
                'expirations': expirations,
                'chains': chains
            }

        except requests.exceptions.Timeout:
            logger.warning(f"[TRADIER] Timeout fetching option chain for {symbol}")
            return None
        except requests.exceptions.RequestException as e:
            logger.warning(f"[TRADIER] Request error for {symbol}: {e}")
            return None
        except Exception as e:
            logger.error(f"[TRADIER] Unexpected error fetching option chain for {symbol}: {e}")
            return None

    def get_quote(self, symbol: str) -> Optional[Dict]:
        """
        Get real-time quote for a symbol

        Args:
            symbol: Stock ticker (e.g., 'AAPL', 'MSFT')

        Returns:
            Dictionary with quote data:
            {
                'symbol': 'AAPL',
                'last': 150.25,
                'bid': 150.24,
                'ask': 150.26,
                'volume': 52341234,
                'open': 149.50,
                'high': 150.75,
                'low': 149.25,
                'close': 150.25,  # Previous close
                'change': 0.75,
                'change_percentage': 0.50,
                'last_trade_time': '2026-02-05T13:45:23.000Z'
            }

            Returns None if request fails
        """
        try:
            response = requests.get(
                f"{self.base_url}/markets/quotes",
                headers=self.headers,
                params={"symbols": symbol},
                timeout=5
            )

            if response.status_code != 200:
                logger.warning(f"[TRADIER] Quote request failed: {response.status_code}")
                return None

            data = response.json()

            # Extract quote from response
            if "quotes" not in data or "quote" not in data["quotes"]:
                logger.warning(f"[TRADIER] No quote data for {symbol}")
                return None

            quote = data["quotes"]["quote"]

            # Handle case where quote is a list (shouldn't happen with single symbol)
            if isinstance(quote, list):
                quote = quote[0] if quote else {}

            # Return normalized quote data
            return {
                'symbol': quote.get('symbol', symbol),
                'last': float(quote.get('last', 0)) if quote.get('last') else None,
                'bid': float(quote.get('bid', 0)) if quote.get('bid') else None,
                'ask': float(quote.get('ask', 0)) if quote.get('ask') else None,
                'volume': int(quote.get('volume', 0)) if quote.get('volume') else 0,
                'open': float(quote.get('open', 0)) if quote.get('open') else None,
                'high': float(quote.get('high', 0)) if quote.get('high') else None,
                'low': float(quote.get('low', 0)) if quote.get('low') else None,
                'close': float(quote.get('close', 0)) if quote.get('close') else None,  # Previous close
                'change': float(quote.get('change', 0)) if quote.get('change') else None,
                'change_percentage': float(quote.get('change_percentage', 0)) if quote.get('change_percentage') else None,
                'last_trade_time': quote.get('trade_date', datetime.utcnow().isoformat() + 'Z')
            }

        except requests.exceptions.Timeout:
            logger.warning(f"[TRADIER] Timeout fetching quote for {symbol}")
            return None
        except requests.exceptions.RequestException as e:
            logger.warning(f"[TRADIER] Request error for {symbol}: {e}")
            return None
        except Exception as e:
            logger.error(f"[TRADIER] Unexpected error fetching quote for {symbol}: {e}")
            return None


# Singleton instance
_tradier_client = None


def get_tradier_client() -> TradierClient:
    """
    Get singleton TradierClient instance

    Returns:
        TradierClient instance
    """
    global _tradier_client
    if _tradier_client is None:
        _tradier_client = TradierClient()
    return _tradier_client


def get_last_price(symbol: str) -> Optional[float]:
    """
    Get last traded price for a symbol

    Args:
        symbol: Stock ticker

    Returns:
        Last price as float, or None if unavailable
    """
    client = get_tradier_client()
    quote = client.get_quote(symbol)
    if quote and quote.get('last'):
        return quote['last']
    return None


def get_option_chain(symbol: str, expiration: str = None) -> Optional[Dict]:
    """
    Get options chain for a symbol

    Args:
        symbol: Stock ticker
        expiration: Optional expiration date in YYYY-MM-DD format

    Returns:
        Options chain dictionary, or None if unavailable
    """
    client = get_tradier_client()
    return client.get_option_chain(symbol, expiration)


if __name__ == '__main__':
    # Test Tradier client
    print("=" * 80)
    print("TRADIER CLIENT TEST")
    print("=" * 80)

    client = TradierClient()

    # Test with a few symbols
    test_symbols = ['AAPL', 'MSFT', 'GOOGL']

    for symbol in test_symbols:
        print(f"\nFetching quote for {symbol}...")
        quote = client.get_quote(symbol)

        if quote:
            print(f"  Last: ${quote['last']:.2f}" if quote['last'] else "  Last: N/A")
            print(f"  Bid: ${quote['bid']:.2f}  Ask: ${quote['ask']:.2f}" if quote['bid'] and quote['ask'] else "  Bid/Ask: N/A")
            print(f"  Volume: {quote['volume']:,}")
            if quote['open'] and quote['high'] and quote['low'] and quote['close']:
                print(f"  OHLC: O=${quote['open']:.2f} H=${quote['high']:.2f} L=${quote['low']:.2f} C=${quote['close']:.2f}")
            else:
                print(f"  OHLC: Partial data available")
            if quote['change'] and quote['change_percentage']:
                print(f"  Change: ${quote['change']:+.2f} ({quote['change_percentage']:+.2f}%)")
        else:
            print(f"  Failed to fetch quote")

    print("\n" + "=" * 80)
    print("TEST COMPLETE")
    print("=" * 80)
