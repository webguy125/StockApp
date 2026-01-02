"""
Option Data Fetcher
Retrieves options chain data from yfinance (or Schwab API in future)
"""

import yfinance as yf
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple


class OptionDataFetcher:
    """
    Fetches and processes options data for analysis
    """

    def __init__(self, data_source: str = 'yfinance'):
        """
        Initialize data fetcher

        Args:
            data_source: 'yfinance' or 'schwab' (schwab not implemented yet)
        """
        self.data_source = data_source

    def get_options_chain(self, symbol: str, days_to_expiration: int = 14) -> Dict:
        """
        Get options chain for a symbol

        Args:
            symbol: Stock ticker (e.g., 'AAPL')
            days_to_expiration: Target days to expiration (will find closest)

        Returns:
            Dictionary with calls, puts, and metadata
        """
        try:
            ticker = yf.Ticker(symbol)

            # Get all available expirations
            expirations = ticker.options

            if not expirations:
                return {'error': f'No options available for {symbol}'}

            # Find expiration closest to target DTE
            target_date = datetime.now() + timedelta(days=days_to_expiration)
            expiration = self._find_closest_expiration(expirations, target_date)

            # Get options chain
            chain = ticker.option_chain(expiration)

            # Get current stock price
            info = ticker.history(period='1d')
            current_price = float(info['Close'].iloc[-1]) if len(info) > 0 else None

            return {
                'symbol': symbol,
                'current_price': current_price,
                'expiration': expiration,
                'days_to_expiration': (pd.to_datetime(expiration) - datetime.now()).days,
                'calls': chain.calls,
                'puts': chain.puts,
                'fetched_at': datetime.now().isoformat()
            }

        except Exception as e:
            return {'error': f'Failed to fetch options for {symbol}: {str(e)}'}

    def _find_closest_expiration(self, expirations: List[str], target_date: datetime) -> str:
        """Find expiration date closest to target"""
        expirations_dt = [pd.to_datetime(exp) for exp in expirations]
        differences = [abs((exp - target_date).days) for exp in expirations_dt]
        closest_idx = differences.index(min(differences))
        return expirations[closest_idx]

    def get_atm_options(self, symbol: str, days_to_expiration: int = 14) -> Dict:
        """
        Get ATM (At-The-Money) options

        Returns:
            Dictionary with ATM call and put
        """
        chain_data = self.get_options_chain(symbol, days_to_expiration)

        if 'error' in chain_data:
            return chain_data

        current_price = chain_data['current_price']
        calls = chain_data['calls']
        puts = chain_data['puts']

        # Find ATM strike (closest to current price)
        calls['distance'] = abs(calls['strike'] - current_price)
        puts['distance'] = abs(puts['strike'] - current_price)

        atm_call = calls.loc[calls['distance'].idxmin()]
        atm_put = puts.loc[puts['distance'].idxmin()]

        return {
            'symbol': symbol,
            'current_price': current_price,
            'expiration': chain_data['expiration'],
            'days_to_expiration': chain_data['days_to_expiration'],
            'atm_call': atm_call.to_dict(),
            'atm_put': atm_put.to_dict()
        }

    def get_option_by_delta(self, symbol: str, target_delta: float,
                           option_type: str = 'call',
                           days_to_expiration: int = 14) -> Dict:
        """
        Get option closest to target delta

        Args:
            symbol: Stock ticker
            target_delta: Target delta (e.g., 0.40 for slightly OTM)
            option_type: 'call' or 'put'
            days_to_expiration: Target DTE

        Returns:
            Option data dictionary
        """
        chain_data = self.get_options_chain(symbol, days_to_expiration)

        if 'error' in chain_data:
            return chain_data

        options = chain_data['calls'] if option_type == 'call' else chain_data['puts']

        # Some options may not have IV/delta from yfinance
        # Filter out options without impliedVolatility
        options = options[options['impliedVolatility'].notna()]

        if len(options) == 0:
            return {'error': f'No {option_type}s with delta data available'}

        # Calculate approximate delta if not provided
        # For calls: Delta ≈ 0.5 when ATM, higher ITM, lower OTM
        # For puts: Delta ≈ -0.5 when ATM, more negative ITM, closer to 0 OTM
        current_price = chain_data['current_price']

        if 'delta' not in options.columns or options['delta'].isna().all():
            # Approximate delta based on moneyness
            if option_type == 'call':
                options['estimated_delta'] = options['strike'].apply(
                    lambda strike: self._estimate_call_delta(current_price, strike)
                )
            else:
                options['estimated_delta'] = options['strike'].apply(
                    lambda strike: self._estimate_put_delta(current_price, strike)
                )
            delta_column = 'estimated_delta'
        else:
            delta_column = 'delta'

        # Find option closest to target delta
        options['delta_distance'] = abs(options[delta_column] - target_delta)
        best_option = options.loc[options['delta_distance'].idxmin()]

        return {
            'symbol': symbol,
            'current_price': current_price,
            'expiration': chain_data['expiration'],
            'days_to_expiration': chain_data['days_to_expiration'],
            'option_type': option_type,
            'target_delta': target_delta,
            'option': best_option.to_dict()
        }

    def _estimate_call_delta(self, spot: float, strike: float) -> float:
        """Rough delta estimate for calls"""
        moneyness = spot / strike
        if moneyness >= 1.10:  # Deep ITM
            return 0.80
        elif moneyness >= 1.05:
            return 0.65
        elif moneyness >= 1.00:
            return 0.55
        elif moneyness >= 0.95:  # ATM
            return 0.50
        elif moneyness >= 0.90:
            return 0.35
        elif moneyness >= 0.85:
            return 0.20
        else:  # Deep OTM
            return 0.10

    def _estimate_put_delta(self, spot: float, strike: float) -> float:
        """Rough delta estimate for puts (negative values)"""
        moneyness = spot / strike
        if moneyness <= 0.90:  # Deep ITM
            return -0.80
        elif moneyness <= 0.95:
            return -0.65
        elif moneyness <= 1.00:  # ATM
            return -0.50
        elif moneyness <= 1.05:
            return -0.35
        elif moneyness <= 1.10:
            return -0.20
        else:  # Deep OTM
            return -0.10


if __name__ == '__main__':
    # Test the fetcher
    fetcher = OptionDataFetcher()

    # Test 1: Get options chain
    print("=" * 60)
    print("TEST 1: Get Options Chain for AAPL (14 DTE)")
    print("=" * 60)
    chain = fetcher.get_options_chain('AAPL', days_to_expiration=14)

    if 'error' not in chain:
        print(f"Symbol: {chain['symbol']}")
        print(f"Current Price: ${chain['current_price']:.2f}")
        print(f"Expiration: {chain['expiration']}")
        print(f"Days to Expiration: {chain['days_to_expiration']}")
        print(f"Calls: {len(chain['calls'])} strikes")
        print(f"Puts: {len(chain['puts'])} strikes")
    else:
        print(f"Error: {chain['error']}")

    # Test 2: Get ATM options
    print("\n" + "=" * 60)
    print("TEST 2: Get ATM Options")
    print("=" * 60)
    atm = fetcher.get_atm_options('AAPL', days_to_expiration=14)

    if 'error' not in atm:
        print(f"\nATM Call:")
        print(f"  Strike: ${atm['atm_call']['strike']:.2f}")
        print(f"  Bid: ${atm['atm_call']['bid']:.2f}")
        print(f"  Ask: ${atm['atm_call']['ask']:.2f}")
        print(f"  IV: {atm['atm_call']['impliedVolatility']:.2%}")

        print(f"\nATM Put:")
        print(f"  Strike: ${atm['atm_put']['strike']:.2f}")
        print(f"  Bid: ${atm['atm_put']['bid']:.2f}")
        print(f"  Ask: ${atm['atm_put']['ask']:.2f}")
        print(f"  IV: {atm['atm_put']['impliedVolatility']:.2%}")
    else:
        print(f"Error: {atm['error']}")

    # Test 3: Get option by delta
    print("\n" + "=" * 60)
    print("TEST 3: Get Call with Delta ~0.40")
    print("=" * 60)
    delta_option = fetcher.get_option_by_delta('AAPL', target_delta=0.40,
                                               option_type='call',
                                               days_to_expiration=14)

    if 'error' not in delta_option:
        opt = delta_option['option']
        print(f"Strike: ${opt['strike']:.2f}")
        print(f"Bid: ${opt['bid']:.2f}")
        print(f"Ask: ${opt['ask']:.2f}")
        print(f"IV: {opt['impliedVolatility']:.2%}")
        if 'delta' in opt and not pd.isna(opt['delta']):
            print(f"Delta: {opt['delta']:.3f}")
        elif 'estimated_delta' in opt:
            print(f"Estimated Delta: {opt['estimated_delta']:.3f}")
    else:
        print(f"Error: {delta_option['error']}")
