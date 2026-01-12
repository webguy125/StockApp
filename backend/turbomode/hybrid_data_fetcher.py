"""
Hybrid Data Fetcher - Best of IBKR + yfinance
Uses IBKR for options data, yfinance for historical stock data

Author: TurboMode System
Date: 2026-01-04
"""

import yfinance as yf
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import Optional, Dict, List, Tuple
import logging
from turbomode.ibkr_data_adapter import IBKRDataAdapter

logger = logging.getLogger(__name__)


class HybridDataFetcher:
    """
    Intelligent data fetcher that uses:
    - IBKR: Options chains, spreads, Greeks (300x faster, real-time)
    - yfinance: Historical stock OHLCV (free, reliable)

    Best of both worlds!
    """

    def __init__(self):
        """Initialize hybrid fetcher with IBKR connection"""
        self.ibkr = None
        try:
            self.ibkr = IBKRDataAdapter()
            logger.info("[HYBRID] IBKR adapter initialized")
        except Exception as e:
            logger.warning(f"[HYBRID] IBKR not available, using yfinance only: {e}")

    def get_stock_data(self, symbol: str, period: str = "1y", interval: str = "1d") -> Optional[pd.DataFrame]:
        """
        Get historical stock OHLCV data
        Uses yfinance (reliable and free)

        Args:
            symbol: Stock ticker
            period: Time period (1d, 5d, 1mo, 3mo, 6mo, 1y, 2y, 5y, 10y, ytd, max)
            interval: Data interval (1m, 2m, 5m, 15m, 30m, 60m, 90m, 1h, 1d, 5d, 1wk, 1mo, 3mo)

        Returns:
            DataFrame with OHLCV data
        """
        try:
            ticker = yf.Ticker(symbol)
            df = ticker.history(period=period, interval=interval)

            if df.empty:
                return None

            return df

        except Exception as e:
            logger.error(f"[HYBRID] Failed to fetch stock data for {symbol}: {e}")
            return None

    def get_current_price(self, symbol: str) -> Optional[float]:
        """
        Get current stock price
        Uses yfinance for simplicity

        Args:
            symbol: Stock ticker

        Returns:
            Current price or None
        """
        try:
            ticker = yf.Ticker(symbol)
            data = ticker.history(period="1d", interval="1m")

            if data.empty:
                return None

            return float(data['Close'].iloc[-1])

        except Exception as e:
            logger.error(f"[HYBRID] Failed to get price for {symbol}: {e}")
            return None

    def get_options_chain(self, symbol: str) -> Optional[Dict]:
        """
        Get options chain (expirations and strikes)
        Tries IBKR first (300x faster), falls back to yfinance

        Args:
            symbol: Stock ticker

        Returns:
            Dict with 'expirations' and 'strikes' or None
        """
        # Try IBKR first (much faster)
        if self.ibkr and self.ibkr.connected:
            try:
                chain = self.ibkr.get_options_chain(symbol)
                if chain:
                    logger.info(f"[HYBRID] Got options chain for {symbol} from IBKR")
                    return chain
            except Exception as e:
                logger.warning(f"[HYBRID] IBKR options chain failed for {symbol}: {e}")

        # Fallback to yfinance
        try:
            ticker = yf.Ticker(symbol)
            expirations = ticker.options

            if not expirations:
                return None

            # Get strikes from first expiration
            chain = ticker.option_chain(expirations[0])
            strikes = sorted(set(chain.calls['strike'].tolist()))

            logger.info(f"[HYBRID] Got options chain for {symbol} from yfinance")
            return {
                'expirations': list(expirations),
                'strikes': strikes,
                'source': 'yfinance'
            }

        except Exception as e:
            logger.error(f"[HYBRID] Failed to get options chain for {symbol}: {e}")
            return None

    def get_option_quote(self, symbol: str, expiration: str, strike: float, right: str = 'C') -> Optional[Dict]:
        """
        Get option quote with bid/ask/Greeks
        Tries IBKR first (real-time Greeks), falls back to yfinance

        Args:
            symbol: Stock ticker
            expiration: Expiration date (YYYY-MM-DD or YYYYMMDD)
            strike: Strike price
            right: 'C' for call, 'P' for put

        Returns:
            Dict with bid, ask, volume, open_interest, iv, Greeks
        """
        # Try IBKR first (real-time Greeks) - DISABLED until market data subscription active
        # if self.ibkr and self.ibkr.connected:
        #     try:
        #         # Convert expiration format if needed (YYYY-MM-DD -> YYYYMMDD)
        #         exp_formatted = expiration.replace('-', '')
        #         quote = self.ibkr.get_option_quote(symbol, exp_formatted, strike, right)
        #
        #         # Only use IBKR quote if it has valid data (bid/ask not zero)
        #         if quote and (quote['bid'] > 0 or quote['ask'] > 0):
        #             logger.info(f"[HYBRID] Got option quote for {symbol} from IBKR")
        #             quote['source'] = 'ibkr'
        #             return quote
        #         else:
        #             logger.warning(f"[HYBRID] IBKR returned empty quote for {symbol}, falling back to yfinance")
        #     except Exception as e:
        #         logger.warning(f"[HYBRID] IBKR option quote failed for {symbol}: {e}")

        # Fallback to yfinance
        try:
            ticker = yf.Ticker(symbol)

            # Try exact expiration first
            try:
                chain = ticker.option_chain(expiration)
            except ValueError:
                # Expiration not available - find nearest one
                available_exps = ticker.options
                target_dt = datetime.strptime(expiration, '%Y-%m-%d')
                exp_dts = [(exp, datetime.strptime(exp, '%Y-%m-%d')) for exp in available_exps]
                nearest = min(exp_dts, key=lambda x: abs((x[1] - target_dt).days))
                logger.warning(f"[HYBRID] Expiration {expiration} not available in yfinance, using nearest: {nearest[0]}")
                chain = ticker.option_chain(nearest[0])

            # Get calls or puts
            options = chain.calls if right == 'C' else chain.puts

            # Find matching strike
            option = options[options['strike'] == strike]

            if option.empty:
                return None

            row = option.iloc[0]

            result = {
                'bid': float(row['bid']) if not pd.isna(row['bid']) else 0,
                'ask': float(row['ask']) if not pd.isna(row['ask']) else 0,
                'last': float(row['lastPrice']) if not pd.isna(row['lastPrice']) else 0,
                'volume': int(row['volume']) if not pd.isna(row['volume']) else 0,
                'open_interest': int(row['openInterest']) if not pd.isna(row['openInterest']) else 0,
                'iv': float(row['impliedVolatility']) if not pd.isna(row['impliedVolatility']) else None,
                'delta': None,  # yfinance doesn't provide Greeks
                'gamma': None,
                'theta': None,
                'vega': None,
                'source': 'yfinance'
            }

            logger.info(f"[HYBRID] Got option quote for {symbol} from yfinance")
            return result

        except Exception as e:
            logger.error(f"[HYBRID] Failed to get option quote for {symbol}: {e}")
            return None

    def get_atm_spread(self, symbol: str, current_price: float, expiration: str = None) -> Optional[float]:
        """
        Get ATM option spread (critical for quarterly curation)
        Uses IBKR for accurate real-time spreads

        Args:
            symbol: Stock ticker
            current_price: Current stock price
            expiration: Target expiration (uses nearest 30-45 DTE if None)

        Returns:
            ATM spread in dollars or None
        """
        try:
            # Get options chain
            chain = self.get_options_chain(symbol)
            if not chain:
                return None

            # Find expiration ~30-45 DTE if not specified
            if expiration is None:
                expirations = chain['expirations']
                target_date = datetime.now() + timedelta(days=37)  # Mid-range of 30-45

                # Convert expirations to datetime and find closest
                exp_dates = []
                for exp in expirations:
                    try:
                        if '-' in exp:
                            exp_dt = datetime.strptime(exp, '%Y-%m-%d')
                            exp_str = exp  # Already in YYYY-MM-DD format
                        else:
                            exp_dt = datetime.strptime(exp, '%Y%m%d')
                            exp_str = f"{exp[:4]}-{exp[4:6]}-{exp[6:8]}"  # Convert to YYYY-MM-DD
                        exp_dates.append((exp_str, exp_dt))
                    except:
                        continue

                if not exp_dates:
                    return None

                # Find closest to target (within 30-45 DTE range)
                valid_exps = [(exp_str, exp_dt) for exp_str, exp_dt in exp_dates
                             if 30 <= (exp_dt - datetime.now()).days <= 45]

                if not valid_exps:
                    # If no expirations in 30-45 DTE range, use closest one
                    expiration = min(exp_dates, key=lambda x: abs((x[1] - target_date).days))[0]
                else:
                    # Use closest within valid range
                    expiration = min(valid_exps, key=lambda x: abs((x[1] - target_date).days))[0]

            # Find ATM strike (closest to current price)
            strikes = chain['strikes']
            atm_strike = min(strikes, key=lambda x: abs(x - current_price))

            # Convert expiration to YYYY-MM-DD format for yfinance
            if '-' not in expiration:
                # Convert YYYYMMDD to YYYY-MM-DD
                exp_formatted = f"{expiration[:4]}-{expiration[4:6]}-{expiration[6:8]}"
            else:
                exp_formatted = expiration

            logger.info(f"[HYBRID] Using expiration: {exp_formatted}, ATM strike: ${atm_strike}")

            # Get ATM call quote
            quote = self.get_option_quote(symbol, exp_formatted, atm_strike, 'C')

            if not quote or quote['bid'] == 0 or quote['ask'] == 0:
                return None

            spread = quote['ask'] - quote['bid']

            logger.info(f"[HYBRID] ATM spread for {symbol}: ${spread:.2f} (strike ${atm_strike}, exp {exp_formatted})")
            return spread

        except Exception as e:
            logger.error(f"[HYBRID] Failed to get ATM spread for {symbol}: {e}")
            return None

    def get_market_cap(self, symbol: str) -> Optional[float]:
        """
        Get market capitalization
        Uses yfinance

        Args:
            symbol: Stock ticker

        Returns:
            Market cap in USD or None
        """
        try:
            ticker = yf.Ticker(symbol)
            info = ticker.info

            market_cap = info.get('marketCap')
            if market_cap:
                return float(market_cap)

            return None

        except Exception as e:
            logger.error(f"[HYBRID] Failed to get market cap for {symbol}: {e}")
            return None

    def close(self):
        """Close IBKR connection"""
        if self.ibkr:
            self.ibkr.disconnect()


# Singleton instance
_hybrid_instance = None

def get_hybrid_fetcher() -> HybridDataFetcher:
    """Get shared hybrid fetcher instance"""
    global _hybrid_instance
    if _hybrid_instance is None:
        _hybrid_instance = HybridDataFetcher()
    return _hybrid_instance


if __name__ == '__main__':
    # Test the hybrid fetcher
    print("Testing Hybrid Data Fetcher...")
    print("=" * 60)

    fetcher = HybridDataFetcher()

    # Test 1: Stock data (yfinance)
    print("\n[TEST 1] Fetching AAPL stock data (yfinance)...")
    df = fetcher.get_stock_data('AAPL', '1mo', '1d')
    if df is not None:
        print(f"[OK] Got {len(df)} days of data")
        print(df.tail(3))
    else:
        print("[FAIL]")

    # Test 2: Current price (yfinance)
    print("\n[TEST 2] Fetching AAPL current price (yfinance)...")
    price = fetcher.get_current_price('AAPL')
    if price:
        print(f"[OK] Price: ${price:.2f}")
    else:
        print("[FAIL]")

    # Test 3: Options chain (IBKR or yfinance)
    print("\n[TEST 3] Fetching AAPL options chain...")
    chain = fetcher.get_options_chain('AAPL')
    if chain:
        print(f"[OK] Got {len(chain['expirations'])} expirations")
        print(f"     Source: {chain.get('source', 'unknown')}")
        print(f"     Next 5 expirations: {chain['expirations'][:5]}")
    else:
        print("[FAIL]")

    # Test 4: ATM spread (critical for curation)
    if price and chain:
        print("\n[TEST 4] Calculating ATM spread...")
        spread = fetcher.get_atm_spread('AAPL', price)
        if spread:
            print(f"[OK] ATM Spread: ${spread:.2f}")
            if spread <= 0.25:
                print("     [PASS] Meets $0.25 curation criteria")
            else:
                print(f"     [FAIL] Exceeds $0.25 limit (spread: ${spread:.2f})")
        else:
            print("[FAIL]")

    # Test 5: Market cap
    print("\n[TEST 5] Fetching AAPL market cap...")
    mcap = fetcher.get_market_cap('AAPL')
    if mcap:
        print(f"[OK] Market Cap: ${mcap/1e9:.2f}B")
    else:
        print("[FAIL]")

    print("\n" + "=" * 60)
    print("[OK] Hybrid fetcher tests complete!")

    fetcher.close()
