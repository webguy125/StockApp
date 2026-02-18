"""
Tradier Options REST Client
============================

Dedicated REST client for options data - completely separate from scheduler client.

DATA FIELDS PROVIDED:
=====================

1. UNDERLYING DATA:
   - last: Last traded price
   - bid: Current bid price
   - ask: Current ask price
   - mid: Mid price (bid + ask) / 2
   - open: Opening price
   - high: Day high
   - low: Day low
   - previous_close: Previous day's close
   - change: Price change from previous close
   - change_percent: Percent change from previous close
   - volume: Trading volume
   - timestamp: Quote timestamp

2. EXPIRATIONS:
   - expiration_dates: List of expiration dates (YYYY-MM-DD)
   - is_weekly: Boolean flag for weekly expirations
   - is_monthly: Boolean flag for monthly expirations (3rd Friday)
   - expiration_type: 'weekly' or 'monthly'

3. STRIKES:
   - strike_prices: List of available strike prices
   - strike_precision: Strike increment (e.g., $5.00, $10.00)

4. OPTION CONTRACT FIELDS:

   Metadata:
   - contract_symbol: OCC symbol (e.g., 'AAPL260221C00150000')
   - underlying_symbol: Stock ticker
   - expiration_date: Option expiration date
   - strike: Strike price
   - option_type: 'call' or 'put'
   - multiplier: Contract multiplier (usually 100)
   - contract_size: Number of shares per contract
   - root_symbol: Underlying ticker

   Market Data:
   - bid: Current bid price
   - ask: Current ask price
   - mid: Mid price (bid + ask) / 2
   - last: Last traded price
   - volume: Contract volume
   - open_interest: Open interest
   - previous_close: Previous day's close
   - change: Price change
   - change_percent: Percent change
   - quote_timestamp: Quote timestamp

   Greeks:
   - delta: Delta (rate of change relative to underlying)
   - gamma: Gamma (rate of change of delta)
   - theta: Theta (time decay)
   - vega: Vega (sensitivity to volatility)
   - rho: Rho (sensitivity to interest rates)

   Volatility:
   - implied_volatility: Mid IV
   - bid_iv: Bid-side implied volatility
   - ask_iv: Ask-side implied volatility

Architecture:
- REST-only (no WebSocket)
- Session auto-renewal (55 minutes)
- Thread-safe singleton
- 24/7 operation

Tradier API Endpoints:
- /v1/markets/quotes - Underlying quotes
- /v1/markets/options/expirations - Option expirations
- /v1/markets/options/strikes - Strike prices
- /v1/markets/options/chains - Full option chains with greeks
"""

import os
import requests
import time
from datetime import datetime, timedelta
from threading import Lock
from typing import Dict, List, Optional, Any
import logging

logger = logging.getLogger(__name__)

# Tradier API configuration
TRADIER_API_URL = "https://api.tradier.com"
TRADIER_SESSION_ENDPOINT = "/v1/markets/events/session"
TRADIER_QUOTES_ENDPOINT = "/v1/markets/quotes"
TRADIER_EXPIRATIONS_ENDPOINT = "/v1/markets/options/expirations"
TRADIER_STRIKES_ENDPOINT = "/v1/markets/options/strikes"
TRADIER_CHAINS_ENDPOINT = "/v1/markets/options/chains"

# Session renewal configuration
SESSION_RENEWAL_SECONDS = 55 * 60  # Renew at 55 minutes (expire at 60)


class TradierOptionsClient:
    """
    Dedicated REST client for Tradier options data.

    Completely separate from scheduler client - only for options.

    Features:
    - Full option chains with greeks
    - Underlying quotes
    - Expirations with weekly/monthly flags
    - Strike prices with precision
    - Session auto-renewal for 24/7 operation
    """

    def __init__(self, api_key: Optional[str] = None):
        """
        Initialize Tradier Options REST client.

        Args:
            api_key: Tradier API key (defaults to environment variable)
        """
        self.api_key = api_key or os.getenv("TRADIER_API_KEY")
        if not self.api_key:
            raise ValueError("TRADIER_API_KEY environment variable not set")

        # Session management
        self.session_id: Optional[str] = None
        self.session_created_at: Optional[datetime] = None
        self.session_lock = Lock()

        logger.info("[TRADIER OPTIONS] Client initialized")

    def _create_session(self) -> str:
        """
        Create a new Tradier session.

        Sessions expire after 60 minutes, auto-renew at 55 minutes.

        Returns:
            Session ID string

        Raises:
            Exception: If session creation fails
        """
        try:
            response = requests.post(
                TRADIER_API_URL + TRADIER_SESSION_ENDPOINT,
                headers={
                    "Authorization": f"Bearer {self.api_key}",
                    "Accept": "application/json"
                },
                timeout=10
            )
            response.raise_for_status()

            data = response.json()
            session_id = data["stream"]["sessionid"]

            logger.info(f"[TRADIER OPTIONS] Session created: {session_id[:8]}...")
            return session_id

        except Exception as e:
            logger.error(f"[TRADIER OPTIONS] Session creation failed: {e}")
            raise

    def _ensure_valid_session(self) -> str:
        """
        Ensure we have a valid session, creating/renewing as needed.

        Thread-safe session management with auto-renewal at 55 minutes.

        Returns:
            Valid session ID
        """
        with self.session_lock:
            # Check if we need to create/renew session
            needs_renewal = False

            if not self.session_id or not self.session_created_at:
                needs_renewal = True
            else:
                elapsed = (datetime.now() - self.session_created_at).total_seconds()
                if elapsed >= SESSION_RENEWAL_SECONDS:
                    needs_renewal = True
                    logger.info(f"[TRADIER OPTIONS] Session expired ({elapsed:.0f}s), renewing...")

            if needs_renewal:
                self.session_id = self._create_session()
                self.session_created_at = datetime.now()

            return self.session_id

    # ========================================================================
    # UNDERLYING DATA
    # ========================================================================

    def get_underlying_quote(self, symbol: str) -> Optional[Dict[str, Any]]:
        """
        Get underlying stock quote with full market data.

        Fields returned:
        - last, bid, ask, mid
        - open, high, low, previous_close
        - change, change_percent
        - volume, timestamp

        Args:
            symbol: Stock ticker (e.g., 'AAPL')

        Returns:
            Dictionary with quote data, or None if failed

        Example:
        {
            'symbol': 'AAPL',
            'last': 150.25,
            'bid': 150.24,
            'ask': 150.26,
            'mid': 150.25,
            'open': 149.50,
            'high': 150.75,
            'low': 149.25,
            'previous_close': 149.00,
            'change': 1.25,
            'change_percent': 0.84,
            'volume': 50000000,
            'timestamp': 1706803200
        }
        """
        try:
            session_id = self._ensure_valid_session()

            response = requests.get(
                TRADIER_API_URL + TRADIER_QUOTES_ENDPOINT,
                params={"symbols": symbol, "sessionid": session_id, "greeks": "false"},
                headers={
                    "Authorization": f"Bearer {self.api_key}",
                    "Accept": "application/json"
                },
                timeout=10
            )
            response.raise_for_status()

            data = response.json()

            # Parse response
            if "quotes" in data and "quote" in data["quotes"]:
                quote_data = data["quotes"]["quote"]

                # Handle single quote (dict) vs multiple quotes (list)
                if isinstance(quote_data, list):
                    quote_data = quote_data[0] if quote_data else None

                if not quote_data:
                    return None

                # Extract and normalize fields
                result = {
                    'symbol': quote_data.get('symbol'),
                    'last': float(quote_data.get('last', 0)),
                    'bid': float(quote_data.get('bid', 0)),
                    'ask': float(quote_data.get('ask', 0)),
                    'mid': (float(quote_data.get('bid', 0)) + float(quote_data.get('ask', 0))) / 2,
                    'open': float(quote_data.get('open', 0)),
                    'high': float(quote_data.get('high', 0)),
                    'low': float(quote_data.get('low', 0)),
                    'previous_close': float(quote_data.get('prevclose', 0)),
                    'change': float(quote_data.get('change', 0)),
                    'change_percent': float(quote_data.get('change_percentage', 0)),
                    'volume': int(quote_data.get('volume', 0)),
                    'timestamp': int(datetime.now().timestamp())
                }

                logger.info(f"[TRADIER OPTIONS] {symbol} quote: ${result['last']:.2f}")
                return result

            return None

        except Exception as e:
            logger.error(f"[TRADIER OPTIONS] Failed to fetch quote for {symbol}: {e}")
            return None

    # ========================================================================
    # EXPIRATIONS
    # ========================================================================

    def get_expirations(self, symbol: str, include_all_roots: bool = True) -> Optional[List[Dict[str, Any]]]:
        """
        Get option expirations with weekly/monthly flags.

        Args:
            symbol: Stock ticker
            include_all_roots: Include all option roots (default: True)

        Returns:
            List of expiration dictionaries, or None if failed

        Example:
        [
            {
                'date': '2026-02-20',
                'contract_size': 100,
                'expiration_type': 'weekly',
                'is_weekly': True,
                'is_monthly': False
            },
            {
                'date': '2026-02-21',
                'contract_size': 100,
                'expiration_type': 'monthly',
                'is_weekly': False,
                'is_monthly': True
            },
            ...
        ]
        """
        try:
            session_id = self._ensure_valid_session()

            response = requests.get(
                TRADIER_API_URL + TRADIER_EXPIRATIONS_ENDPOINT,
                params={
                    "symbol": symbol,
                    "includeAllRoots": str(include_all_roots).lower(),
                    "sessionid": session_id
                },
                headers={
                    "Authorization": f"Bearer {self.api_key}",
                    "Accept": "application/json"
                },
                timeout=10
            )
            response.raise_for_status()

            data = response.json()

            # Parse response - validate structure first
            if "expirations" not in data:
                logger.warning(f"[TRADIER OPTIONS] No expirations key in response for {symbol}")
                return None

            expirations_obj = data["expirations"]
            if expirations_obj is None:
                logger.warning(f"[TRADIER OPTIONS] Expirations is None for {symbol}")
                return None

            if "date" not in expirations_obj:
                logger.warning(f"[TRADIER OPTIONS] No date key in expirations for {symbol}")
                return None

            dates = expirations_obj["date"]
            if dates is None:
                logger.warning(f"[TRADIER OPTIONS] Dates list is None for {symbol}")
                return None

            # Ensure list
            if isinstance(dates, str):
                dates = [dates]

            # Normalize expiration data
            expirations = []
            for date_str in dates:
                # Parse date
                exp_date = datetime.strptime(date_str, '%Y-%m-%d')

                # Determine if weekly or monthly (3rd Friday = monthly)
                # Simple heuristic: if it's the 3rd Friday of the month, it's monthly
                day = exp_date.day
                weekday = exp_date.weekday()  # Monday = 0, Friday = 4

                # Check if 3rd Friday (days 15-21 and Friday)
                is_monthly = (15 <= day <= 21) and (weekday == 4)
                is_weekly = not is_monthly

                expirations.append({
                    'date': date_str,
                    'contract_size': 100,
                    'expiration_type': 'monthly' if is_monthly else 'weekly',
                    'is_weekly': is_weekly,
                    'is_monthly': is_monthly
                })

            logger.info(f"[TRADIER OPTIONS] {symbol}: {len(expirations)} expirations")
            return expirations

        except Exception as e:
            logger.error(f"[TRADIER OPTIONS] Failed to fetch expirations for {symbol}: {e}")
            return None

    # ========================================================================
    # STRIKES
    # ========================================================================

    def get_strikes(self, symbol: str, expiration: str) -> Optional[Dict[str, Any]]:
        """
        Get strike prices for expiration.

        Args:
            symbol: Stock ticker
            expiration: Expiration date (YYYY-MM-DD)

        Returns:
            Dictionary with strikes and precision, or None if failed

        Example:
        {
            'strikes': [140.0, 145.0, 150.0, 155.0, 160.0, ...],
            'strike_precision': 5.0  # Strike increment
        }
        """
        try:
            session_id = self._ensure_valid_session()

            response = requests.get(
                TRADIER_API_URL + TRADIER_STRIKES_ENDPOINT,
                params={
                    "symbol": symbol,
                    "expiration": expiration,
                    "sessionid": session_id
                },
                headers={
                    "Authorization": f"Bearer {self.api_key}",
                    "Accept": "application/json"
                },
                timeout=10
            )
            response.raise_for_status()

            data = response.json()

            # Parse response
            if "strikes" in data and "strike" in data["strikes"]:
                strikes = data["strikes"]["strike"]

                # Ensure list
                if isinstance(strikes, (int, float)):
                    strikes = [float(strikes)]
                elif isinstance(strikes, list):
                    strikes = [float(s) for s in strikes]
                else:
                    return None

                # Sort strikes
                strikes.sort()

                # Calculate strike precision (increment)
                if len(strikes) >= 2:
                    strike_precision = strikes[1] - strikes[0]
                else:
                    strike_precision = 5.0  # Default to $5 increments

                result = {
                    'strikes': strikes,
                    'strike_precision': strike_precision
                }

                logger.info(f"[TRADIER OPTIONS] {symbol} {expiration}: {len(strikes)} strikes (precision: ${strike_precision:.2f})")
                return result

            return None

        except Exception as e:
            logger.error(f"[TRADIER OPTIONS] Failed to fetch strikes for {symbol} {expiration}: {e}")
            return None

    # ========================================================================
    # OPTION CHAINS (Full Data with Greeks)
    # ========================================================================

    def get_option_chain(self, symbol: str, expiration: str, greeks: bool = True) -> Optional[Dict[str, Any]]:
        """
        Get full option chain with all market data, greeks, and volatility.

        Fields per contract:
        - Metadata: contract_symbol, underlying_symbol, expiration_date, strike, option_type
        - Market Data: bid, ask, mid, last, volume, open_interest, change, change_percent
        - Greeks: delta, gamma, theta, vega, rho
        - Volatility: implied_volatility

        Args:
            symbol: Stock ticker
            expiration: Expiration date (YYYY-MM-DD)
            greeks: Include greeks (default: True)

        Returns:
            Dictionary with calls and puts, or None if failed

        Example:
        {
            'calls': {
                150.0: {
                    'contract_symbol': 'AAPL260220C00150000',
                    'strike': 150.0,
                    'bid': 2.50,
                    'ask': 2.60,
                    'mid': 2.55,
                    'last': 2.55,
                    'volume': 1000,
                    'open_interest': 5000,
                    'change': 0.10,
                    'change_percent': 4.08,
                    'delta': 0.50,
                    'gamma': 0.05,
                    'theta': -0.10,
                    'vega': 0.20,
                    'rho': 0.05,
                    'implied_volatility': 0.25
                },
                ...
            },
            'puts': {
                150.0: {...},
                ...
            }
        }
        """
        try:
            session_id = self._ensure_valid_session()

            response = requests.get(
                TRADIER_API_URL + TRADIER_CHAINS_ENDPOINT,
                params={
                    "symbol": symbol,
                    "expiration": expiration,
                    "greeks": str(greeks).lower(),
                    "sessionid": session_id
                },
                headers={
                    "Authorization": f"Bearer {self.api_key}",
                    "Accept": "application/json"
                },
                timeout=10
            )
            response.raise_for_status()

            data = response.json()

            # Validate response is not None
            if data is None:
                logger.warning(f"[TRADIER OPTIONS] Response JSON is None for {symbol} {expiration}")
                return None

            # Parse response - validate structure first
            if "options" not in data:
                logger.warning(f"[TRADIER OPTIONS] No options key in response for {symbol} {expiration}")
                return None

            options_obj = data["options"]
            if options_obj is None:
                logger.warning(f"[TRADIER OPTIONS] Options object is None for {symbol} {expiration}")
                return None

            if "option" not in options_obj:
                logger.warning(f"[TRADIER OPTIONS] No option key in options for {symbol} {expiration}")
                return None

            options_list = options_obj["option"]
            if options_list is None:
                logger.warning(f"[TRADIER OPTIONS] Options list is None for {symbol} {expiration}")
                return None

            # Ensure list
            if isinstance(options_list, dict):
                options_list = [options_list]

            # Separate calls and puts
            calls = {}
            puts = {}

            for opt in options_list:
                try:
                    strike = float(opt.get('strike', 0))
                    if strike <= 0:
                        continue

                    option_type = opt.get('option_type', '').lower()

                    # Extract all fields
                    bid = float(opt.get('bid', 0))
                    ask = float(opt.get('ask', 0))
                    mid = (bid + ask) / 2 if bid > 0 and ask > 0 else 0

                    contract_data = {
                        # Metadata
                        'contract_symbol': opt.get('symbol', ''),
                        'underlying_symbol': opt.get('root_symbol', symbol),
                        'expiration_date': opt.get('expiration_date', expiration),
                        'strike': strike,
                        'option_type': option_type,
                        'multiplier': int(opt.get('contract_size', 100)),
                        'contract_size': int(opt.get('contract_size', 100)),
                        'root_symbol': opt.get('root_symbol', symbol),

                        # Market Data
                        'bid': round(bid, 2),
                        'ask': round(ask, 2),
                        'mid': round(mid, 2),
                        'last': float(opt.get('last', 0)),
                        'volume': int(opt.get('volume', 0)),
                        'open_interest': int(opt.get('open_interest', 0)),
                        'previous_close': float(opt.get('prevclose', 0)),
                        'change': float(opt.get('change', 0)),
                        'change_percent': float(opt.get('change_percentage', 0)),
                        'quote_timestamp': int(datetime.now().timestamp())
                    }

                    # Greeks (if available)
                    if greeks and 'greeks' in opt:
                        greeks_data = opt['greeks']
                        contract_data.update({
                            'delta': float(greeks_data.get('delta', 0)),
                            'gamma': float(greeks_data.get('gamma', 0)),
                            'theta': float(greeks_data.get('theta', 0)),
                            'vega': float(greeks_data.get('vega', 0)),
                            'rho': float(greeks_data.get('rho', 0)),
                            'implied_volatility': float(greeks_data.get('mid_iv', 0)),
                            'bid_iv': float(greeks_data.get('bid_iv', 0)),
                            'ask_iv': float(greeks_data.get('ask_iv', 0))
                        })

                    # Add to appropriate dict
                    if option_type == 'call':
                        calls[strike] = contract_data
                    elif option_type == 'put':
                        puts[strike] = contract_data

                except Exception as e:
                    # Suppress warnings for illiquid/OTM options with missing data
                    logger.debug(f"[TRADIER OPTIONS] Skipped option (missing data): {e}")
                    continue

            result = {
                'calls': calls,
                'puts': puts
            }

            logger.info(f"[TRADIER OPTIONS] {symbol} {expiration}: {len(calls)} calls, {len(puts)} puts")
            return result

        except Exception as e:
            logger.error(f"[TRADIER OPTIONS] Failed to fetch chain for {symbol} {expiration}: {e}")
            return None

    # ========================================================================
    # HEALTH CHECK
    # ========================================================================

    def test_connection(self) -> bool:
        """
        Test connection to Tradier API.

        Returns:
            True if connection successful, False otherwise
        """
        try:
            quote = self.get_underlying_quote("AAPL")
            if quote and quote.get('last', 0) > 0:
                logger.info("[TRADIER OPTIONS] Connection test successful")
                return True
            else:
                logger.error("[TRADIER OPTIONS] Connection test failed - no data returned")
                return False
        except Exception as e:
            logger.error(f"[TRADIER OPTIONS] Connection test failed: {e}")
            return False


# ============================================================================
# GLOBAL SINGLETON INSTANCE
# ============================================================================

_tradier_options_client: Optional[TradierOptionsClient] = None


def get_tradier_options_client() -> TradierOptionsClient:
    """
    Get singleton instance of TradierOptionsClient.

    Returns:
        Global TradierOptionsClient instance
    """
    global _tradier_options_client

    if _tradier_options_client is None:
        _tradier_options_client = TradierOptionsClient()

    return _tradier_options_client


# ============================================================================
# MAIN TEST
# ============================================================================

if __name__ == "__main__":
    # PRODUCTION MODE: Fetch options data for all HOLD signals in database
    from dotenv import load_dotenv
    from pathlib import Path
    import sqlite3
    import sys

    # Get project root (Options -> turbomode -> backend -> StockApp)
    project_root = Path(__file__).resolve().parents[3]
    env_path = project_root / "backend" / ".env"
    db_path = project_root / "backend" / "data" / "turbomode.db"

    print(f"Loading .env from: {env_path}")
    load_dotenv(env_path)

    # Verify API key loaded
    import os
    api_key = os.getenv("TRADIER_API_KEY")
    if not api_key:
        print("ERROR: TRADIER_API_KEY not found in .env file")
        sys.exit(1)

    # Setup logging
    logging.basicConfig(level=logging.INFO, format='%(message)s')

    print("=" * 80)
    print("TRADIER OPTIONS CLIENT - PRODUCTION MODE")
    print("Fetching options data for all HOLD signals")
    print("=" * 80)

    # Connect to database
    if not db_path.exists():
        print(f"ERROR: Database not found at {db_path}")
        sys.exit(1)

    conn = sqlite3.connect(str(db_path))
    conn.row_factory = sqlite3.Row
    cursor = conn.cursor()

    # Get all HOLD signals
    cursor.execute("""
        SELECT symbol, sector, confidence, entry_price, stop_upper, stop_lower
        FROM active_signals
        WHERE status = 'ACTIVE' AND signal_type = 'HOLD'
        ORDER BY sector, symbol
    """)

    hold_signals = cursor.fetchall()
    conn.close()

    if not hold_signals:
        print("\nNo HOLD signals found in database.")
        print("Run the overnight scanner first to generate HOLD signals.")
        sys.exit(0)

    print(f"\nFound {len(hold_signals)} HOLD signals")
    print("-" * 80)

    # Initialize client
    client = get_tradier_options_client()

    # Test connection
    if not client.test_connection():
        print("ERROR: Failed to connect to Tradier API")
        sys.exit(1)

    # Process each HOLD signal
    success_count = 0
    error_count = 0

    for idx, signal in enumerate(hold_signals, 1):
        symbol = signal['symbol']
        sector = signal['sector']

        print(f"\n[{idx}/{len(hold_signals)}] {symbol} ({sector})")
        print(f"  Entry: ${signal['entry_price']:.2f} | Stops: ${signal['stop_lower']:.2f} - ${signal['stop_upper']:.2f}")

        try:
            # Get underlying quote
            quote = client.get_underlying_quote(symbol)
            if not quote:
                print(f"  [FAIL] Failed to get quote")
                error_count += 1
                continue

            current_price = quote['last']
            print(f"  Current: ${current_price:.2f}")

            # Get expirations
            expirations = client.get_expirations(symbol)
            if not expirations or len(expirations) == 0:
                print(f"  [FAIL] No expirations found")
                error_count += 1
                continue

            # Use first expiration (nearest)
            expiration = expirations[0]['date']
            exp_type = expirations[0]['expiration_type']
            print(f"  Expiration: {expiration} ({exp_type})")

            # Get option chain with greeks
            chain = client.get_option_chain(symbol, expiration, greeks=True)
            if not chain:
                print(f"  [FAIL] Failed to get option chain")
                error_count += 1
                continue

            calls_count = len(chain['calls'])
            puts_count = len(chain['puts'])
            print(f"  [OK] Chain loaded: {calls_count} calls, {puts_count} puts")

            # Find ATM strike and show ALL data fields
            if chain['calls']:
                atm_strike = min(chain['calls'].keys(), key=lambda x: abs(x - current_price))
                atm_call = chain['calls'][atm_strike]

                print(f"\n  ATM Call (strike {atm_strike}) - COMPLETE DATA:")
                print(f"\n  === Metadata ===")
                print(f"    Contract Symbol: {atm_call.get('contract_symbol', 'N/A')}")
                print(f"    Underlying: {atm_call.get('underlying_symbol', 'N/A')}")
                print(f"    Expiration: {atm_call.get('expiration_date', 'N/A')}")
                print(f"    Strike: ${atm_call.get('strike', 0):.2f}")
                print(f"    Type: {atm_call.get('option_type', 'N/A')}")
                print(f"    Multiplier: {atm_call.get('multiplier', 0)}")
                print(f"    Contract Size: {atm_call.get('contract_size', 0)}")
                print(f"    Root Symbol: {atm_call.get('root_symbol', 'N/A')}")

                print(f"\n  === Market Data ===")
                print(f"    Bid: ${atm_call.get('bid', 0):.2f}")
                print(f"    Ask: ${atm_call.get('ask', 0):.2f}")
                print(f"    Mid: ${atm_call.get('mid', 0):.2f}")
                print(f"    Last: ${atm_call.get('last', 0):.2f}")
                print(f"    Volume: {atm_call.get('volume', 0):,}")
                print(f"    Open Interest: {atm_call.get('open_interest', 0):,}")
                print(f"    Previous Close: ${atm_call.get('previous_close', 0):.2f}")
                print(f"    Change: ${atm_call.get('change', 0):.2f}")
                print(f"    Change %: {atm_call.get('change_percent', 0):.2f}%")
                print(f"    Quote Timestamp: {atm_call.get('quote_timestamp', 0)}")

                print(f"\n  === Greeks ===")
                print(f"    Delta: {atm_call.get('delta', 0):.4f}")
                print(f"    Gamma: {atm_call.get('gamma', 0):.4f}")
                print(f"    Theta: {atm_call.get('theta', 0):.4f}")
                print(f"    Vega: {atm_call.get('vega', 0):.4f}")
                print(f"    Rho: {atm_call.get('rho', 0):.4f}")

                print(f"\n  === Volatility ===")
                print(f"    Implied Volatility (mid): {atm_call.get('implied_volatility', 0):.4f} ({atm_call.get('implied_volatility', 0)*100:.2f}%)")
                print(f"    Bid IV: {atm_call.get('bid_iv', 0):.4f}")
                print(f"    Ask IV: {atm_call.get('ask_iv', 0):.4f}")

            success_count += 1

        except Exception as e:
            print(f"  [ERROR] {e}")
            error_count += 1

    # Summary
    print("\n" + "=" * 80)
    print("PRODUCTION RUN COMPLETE")
    print("=" * 80)
    print(f"Total HOLD signals: {len(hold_signals)}")
    print(f"Successfully processed: {success_count}")
    print(f"Errors: {error_count}")
    print("=" * 80)
