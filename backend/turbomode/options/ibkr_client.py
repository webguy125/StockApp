"""
IBKR Option Chain Fetcher
Fetches option chain data from Interactive Brokers
Returns normalized chain with expirations, strikes, bid/ask
On failure returns None
"""

from ib_insync import IB, Stock, Option
from typing import Optional, Dict, List
import logging

logger = logging.getLogger(__name__)


class IBKROptionClient:
    """Fetch option chain from IBKR"""

    def __init__(self, host: str = '127.0.0.1', port: int = 4002):
        """
        Initialize IBKR client

        Args:
            host: IBKR Gateway host
            port: IBKR Gateway port (4002 = paper, 7496 = live)
        """
        self.host = host
        self.port = port
        self.ib = None

    def connect(self) -> bool:
        """
        Connect to IBKR Gateway

        Returns:
            True if connected, False otherwise
        """
        try:
            if self.ib and self.ib.isConnected():
                return True

            self.ib = IB()
            self.ib.connect(self.host, self.port, clientId=998, readonly=True)
            return self.ib.isConnected()
        except Exception as e:
            logger.warning(f"[IBKR] Connection failed: {e}")
            return False

    def disconnect(self):
        """Disconnect from IBKR Gateway"""
        if self.ib and self.ib.isConnected():
            try:
                self.ib.disconnect()
            except:
                pass

    def fetch_option_chain(self, symbol: str) -> Optional[Dict]:
        """
        Fetch option chain for a symbol

        Args:
            symbol: Stock ticker

        Returns:
            Dictionary with structure:
            {
                'expirations': ['20260220', '20260227', ...],
                'chains': {
                    '20260220': {
                        'calls': {
                            150.0: {'bid': 2.5, 'ask': 2.6, 'mid': 2.55},
                            155.0: {'bid': 1.2, 'ask': 1.3, 'mid': 1.25},
                            ...
                        },
                        'puts': {
                            150.0: {'bid': 1.8, 'ask': 1.9, 'mid': 1.85},
                            145.0: {'bid': 3.1, 'ask': 3.2, 'mid': 3.15},
                            ...
                        }
                    }
                }
            }

            Returns None if fetch fails
        """
        if not self.connect():
            logger.warning(f"[IBKR] Cannot fetch options for {symbol}: not connected")
            return None

        try:
            # Create stock contract
            stock = Stock(symbol, 'SMART', 'USD')

            # Qualify contract
            contracts = self.ib.qualifyContracts(stock)
            if not contracts:
                logger.warning(f"[IBKR] Failed to qualify {symbol}")
                return None

            stock = contracts[0]

            # Request option chain
            chains = self.ib.reqSecDefOptParams(stock.symbol, '', stock.secType, stock.conId)

            if not chains:
                logger.warning(f"[IBKR] No option chains found for {symbol}")
                return None

            # Get first chain (usually the primary exchange)
            chain = chains[0]
            expirations = sorted(chain.expirations)

            if not expirations:
                logger.warning(f"[IBKR] No expirations found for {symbol}")
                return None

            # Fetch quotes for all strikes and expirations
            result = {
                'expirations': expirations,
                'chains': {}
            }

            # For each expiration, fetch option quotes
            for expiration in expirations[:8]:  # Limit to first 8 expirations for speed
                strikes = sorted(chain.strikes)

                if not strikes:
                    continue

                # Create option contracts for this expiration
                call_contracts = []
                put_contracts = []

                for strike in strikes:
                    call_contracts.append(Option(symbol, expiration, strike, 'C', chain.exchange))
                    put_contracts.append(Option(symbol, expiration, strike, 'P', chain.exchange))

                # Qualify all contracts at once
                all_contracts = call_contracts + put_contracts
                qualified = self.ib.qualifyContracts(*all_contracts)

                if not qualified:
                    continue

                # Request market data for all contracts
                tickers = []
                for contract in qualified:
                    ticker = self.ib.reqMktData(contract, '', False, False)
                    tickers.append((contract, ticker))

                # Wait for data to populate
                self.ib.sleep(0.5)

                # Parse results
                calls = {}
                puts = {}

                for contract, ticker in tickers:
                    if ticker.bid and ticker.ask and ticker.bid > 0 and ticker.ask > 0:
                        strike = contract.strike
                        bid = ticker.bid
                        ask = ticker.ask
                        mid = (bid + ask) / 2

                        option_data = {'bid': bid, 'ask': ask, 'mid': mid}

                        if contract.right == 'C':
                            calls[strike] = option_data
                        else:
                            puts[strike] = option_data

                    # Cancel market data
                    self.ib.cancelMktData(contract)

                if calls or puts:
                    result['chains'][expiration] = {
                        'calls': calls,
                        'puts': puts
                    }

            if not result['chains']:
                logger.warning(f"[IBKR] No valid option data for {symbol}")
                return None

            logger.info(f"[IBKR] Fetched option chain for {symbol}: {len(result['chains'])} expirations")
            return result

        except Exception as e:
            logger.error(f"[IBKR] Error fetching option chain for {symbol}: {e}")
            return None
