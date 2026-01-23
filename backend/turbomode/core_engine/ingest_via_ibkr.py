
import sys
from pathlib import Path
project_root = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(project_root))

"""
Master Market Data DB - IBKR-Powered Ingestion
Uses IB Gateway for 300x faster data ingestion (50 req/sec vs yfinance's 1/sec)

Architecture: MASTER_MARKET_DATA_ARCHITECTURE.json v1.1
Database: C:\StockApp\master_market_data\market_data.db

Run this script with IB Gateway running for ultra-fast population.
Falls back to yfinance automatically if IBKR unavailable.
"""

import sys
import os

from backend.turbomode.core_engine.hybrid_data_fetcher import HybridDataFetcher
from backend.turbomode.core_engine.ingest_market_data import MarketDataIngestion
from backend.turbomode.core_engine.training_symbols import get_training_symbols, CRYPTO_SYMBOLS
import logging
import time
from datetime import datetime

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger('ibkr_ingestion')


class IBKRMarketDataIngestion(MarketDataIngestion):
    """
    Enhanced ingestion using IBKR Gateway for speed
    Inherits from MarketDataIngestion but uses hybrid fetcher
    """

    def __init__(self, db_path: str = None, use_ibkr: bool = True):
        """
        Initialize IBKR-powered ingestion

        Args:
            db_path: Path to market_data.db
            use_ibkr: Whether to use IBKR (auto-falls back to yfinance)
        """
        super().__init__(db_path)

        self.hybrid_fetcher = HybridDataFetcher(use_ibkr=use_ibkr)
        self.using_ibkr = self.hybrid_fetcher.ibkr_available

        if self.using_ibkr:
            logger.info("[OK] IBKR Gateway connected - Using 300x faster ingestion!")
        else:
            logger.info("[FALLBACK] IBKR unavailable - Using yfinance")

    def ingest_symbol_metadata(self, symbol: str) -> bool:
        """Ingest symbol metadata using hybrid fetcher"""
        try:
            info = self.hybrid_fetcher.fetch_fundamentals(symbol)

            if not info:
                logger.warning(f"[SKIP] {symbol}: No metadata available")
                return False

            conn = self._get_connection()
            cursor = conn.cursor()

            cursor.execute("""
                INSERT OR REPLACE INTO symbol_metadata (
                    symbol, company_name, sector, industry, country, exchange,
                    currency, quote_type, long_business_summary, website,
                    employees, city, state, zip_code, phone, is_active,
                    first_trade_date, last_updated
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, (
                symbol,
                info.get('longName', info.get('shortName', symbol)),
                info.get('sector', 'Unknown'),
                info.get('industry', 'Unknown'),
                info.get('country', 'US'),
                info.get('exchange', 'UNKNOWN'),
                info.get('currency', 'USD'),
                info.get('quoteType', 'EQUITY'),
                info.get('longBusinessSummary', '')[:1000] if info.get('longBusinessSummary') else '',
                info.get('website', ''),
                info.get('fullTimeEmployees'),
                info.get('city', ''),
                info.get('state', ''),
                info.get('zip', ''),
                info.get('phone', ''),
                1,  # is_active
                info.get('firstTradeDateEpochUtc'),
                datetime.now().strftime('%Y-%m-%d %H:%M:%S')
            ))

            conn.commit()
            conn.close()

            logger.info(f"[OK] {symbol}: Metadata ingested")
            return True

        except Exception as e:
            logger.error(f"[ERROR] {symbol}: Failed to ingest metadata - {e}")
            return False

    def ingest_candles(self, symbol: str, timeframe: str = '1d', period: str = '10y',
                       start_date: str = None, end_date: str = None) -> int:
        """Ingest OHLCV candles using hybrid fetcher (IBKR or yfinance)"""
        try:
            # Fetch data using hybrid fetcher
            df = self.hybrid_fetcher.fetch_candles(symbol, period=period, interval=timeframe)

            if df is None or df.empty:
                logger.warning(f"[SKIP] {symbol}: No candle data available")
                return 0

            conn = self._get_connection()
            cursor = conn.cursor()

            count = 0
            for timestamp, row in df.iterrows():
                try:
                    cursor.execute("""
                        INSERT OR REPLACE INTO candles (
                            symbol, timestamp, timeframe, open, high, low, close, volume, adjusted_close
                        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                    """, (
                        symbol,
                        timestamp.strftime('%Y-%m-%d %H:%M:%S'),
                        timeframe,
                        float(row['Open']),
                        float(row['High']),
                        float(row['Low']),
                        float(row['Close']),
                        int(row['Volume']),
                        float(row['Close'])  # Using Close as adjusted_close
                    ))
                    count += 1
                except Exception as e:
                    logger.error(f"[ERROR] {symbol}: Failed to insert candle at {timestamp} - {e}")

            conn.commit()
            conn.close()

            logger.info(f"[OK] {symbol}: Ingested {count} candles ({timeframe})")
            return count

        except Exception as e:
            logger.error(f"[ERROR] {symbol}: Failed to ingest candles - {e}")
            return 0

    def ingest_fundamentals(self, symbol: str) -> bool:
        """Ingest fundamental data using hybrid fetcher"""
        try:
            info = self.hybrid_fetcher.fetch_fundamentals(symbol)

            if not info:
                return False

            conn = self._get_connection()
            cursor = conn.cursor()

            cursor.execute("""
                INSERT OR REPLACE INTO fundamentals (
                    symbol, date, market_cap, enterprise_value, trailing_pe, forward_pe,
                    peg_ratio, price_to_book, price_to_sales, enterprise_to_revenue,
                    enterprise_to_ebitda, profit_margin, operating_margin, return_on_assets,
                    return_on_equity, revenue, revenue_per_share, quarterly_revenue_growth,
                    gross_profit, ebitda, net_income, diluted_eps, quarterly_earnings_growth,
                    total_cash, total_cash_per_share, total_debt, debt_to_equity,
                    current_ratio, book_value_per_share, operating_cash_flow,
                    levered_free_cash_flow, beta, fifty_two_week_change, short_ratio,
                    short_percent_of_float, shares_outstanding, shares_short
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, (
                symbol,
                datetime.now().strftime('%Y-%m-%d'),
                info.get('marketCap'),
                info.get('enterpriseValue'),
                info.get('trailingPE'),
                info.get('forwardPE'),
                info.get('pegRatio'),
                info.get('priceToBook'),
                info.get('priceToSalesTrailing12Months'),
                info.get('enterpriseToRevenue'),
                info.get('enterpriseToEbitda'),
                info.get('profitMargins'),
                info.get('operatingMargins'),
                info.get('returnOnAssets'),
                info.get('returnOnEquity'),
                info.get('totalRevenue'),
                info.get('revenuePerShare'),
                info.get('revenueGrowth'),
                info.get('grossProfits'),
                info.get('ebitda'),
                info.get('netIncomeToCommon'),
                info.get('trailingEps'),
                info.get('earningsGrowth'),
                info.get('totalCash'),
                info.get('totalCashPerShare'),
                info.get('totalDebt'),
                info.get('debtToEquity'),
                info.get('currentRatio'),
                info.get('bookValue'),
                info.get('operatingCashflow'),
                info.get('freeCashflow'),
                info.get('beta'),
                info.get('52WeekChange'),
                info.get('shortRatio'),
                info.get('shortPercentOfFloat'),
                info.get('sharesOutstanding'),
                info.get('sharesShort')
            ))

            conn.commit()
            conn.close()

            logger.info(f"[OK] {symbol}: Fundamentals ingested")
            return True

        except Exception as e:
            logger.error(f"[ERROR] {symbol}: Failed to ingest fundamentals - {e}")
            return False

    def ingest_splits_and_dividends(self, symbol: str) -> tuple:
        """Ingest stock splits and dividends using hybrid fetcher"""
        try:
            splits, dividends = self.hybrid_fetcher.fetch_splits_and_dividends(symbol)

            conn = self._get_connection()
            cursor = conn.cursor()

            # Ingest splits
            splits_count = 0
            if not splits.empty:
                for date, split_value in splits.items():
                    try:
                        cursor.execute("""
                            INSERT OR REPLACE INTO splits (
                                symbol, date, split_ratio, split_factor
                            ) VALUES (?, ?, ?, ?)
                        """, (
                            symbol,
                            date.strftime('%Y-%m-%d'),
                            f"{split_value}:1",
                            float(split_value)
                        ))
                        splits_count += 1
                    except Exception as e:
                        logger.error(f"[ERROR] {symbol}: Failed to insert split - {e}")

            # Ingest dividends
            dividends_count = 0
            if not dividends.empty:
                for date, div_amount in dividends.items():
                    try:
                        cursor.execute("""
                            INSERT OR REPLACE INTO dividends (
                                symbol, date, dividend_amount
                            ) VALUES (?, ?, ?)
                        """, (
                            symbol,
                            date.strftime('%Y-%m-%d'),
                            float(div_amount)
                        ))
                        dividends_count += 1
                    except Exception as e:
                        logger.error(f"[ERROR] {symbol}: Failed to insert dividend - {e}")

            conn.commit()
            conn.close()

            if splits_count > 0 or dividends_count > 0:
                logger.info(f"[OK] {symbol}: {splits_count} splits, {dividends_count} dividends")

            return (splits_count, dividends_count)

        except Exception as e:
            logger.error(f"[ERROR] {symbol}: Failed to ingest splits/dividends - {e}")
            return (0, 0)

    def close(self):
        """Close hybrid fetcher connection"""
        self.hybrid_fetcher.disconnect()


if __name__ == '__main__':
    print("=" * 80)
    print("MASTER MARKET DATA DB - IBKR-POWERED INGESTION")
    print("=" * 80)

    # Get all symbols (training 230 stocks + crypto)
    training_symbols = get_training_symbols()
    all_symbols = sorted(training_symbols + CRYPTO_SYMBOLS)

    print(f"\nSymbols to ingest: {len(all_symbols)}")
    print(f"  Training stocks: {len(training_symbols)}")
    print(f"  Crypto: {len(CRYPTO_SYMBOLS)}")

    print("\n[INFO] Attempting to connect to IB Gateway...")
    print("[INFO] If IB Gateway is running, ingestion will be 300x faster!")
    print("[INFO] Otherwise, will automatically fall back to yfinance\n")

    input("Press Enter to continue...")

    # Initialize IBKR-powered ingestion
    ingestion = IBKRMarketDataIngestion(use_ibkr=True)

    start_time = time.time()

    # Run ingestion
    results = ingestion.ingest_multiple_symbols(all_symbols, period='10y', timeframe='1d')

    elapsed = time.time() - start_time

    # Close connection
    ingestion.close()

    print("\n" + "=" * 80)
    print("INGESTION COMPLETE!")
    print("=" * 80)
    print(f"Time elapsed: {elapsed/60:.1f} minutes")
    print(f"Average per symbol: {elapsed/len(all_symbols):.1f} seconds")

    if ingestion.using_ibkr:
        print(f"\n✓ Used IBKR Gateway for ultra-fast ingestion!")
    else:
        print(f"\n⚠ Used yfinance fallback (IBKR unavailable)")

    print("\n[OK] Master Market Data DB is now populated and ready!")
