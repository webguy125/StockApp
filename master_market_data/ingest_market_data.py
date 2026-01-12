"""
Master Market Data DB - Data Ingestion Script
Populate the Master Market Data DB from yfinance

Architecture: MASTER_MARKET_DATA_ARCHITECTURE.json v1.1
Database: C:\StockApp\master_market_data\market_data.db

This script is for ADMIN use only (write access required).
Run this to populate the Master DB with historical data.
"""

import sqlite3
import os
import yfinance as yf
from datetime import datetime, timedelta
from typing import List, Optional
import logging
import time

# Import canonical symbol normalizer
from symbol_normalizer import validate_and_normalize, auto_correct

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger('data_ingestion')


class MarketDataIngestion:
    """
    Ingest market data from yfinance into Master Market Data DB
    Admin-only write access required
    """

    def __init__(self, db_path: str = None):
        """
        Initialize data ingestion

        Args:
            db_path: Path to market_data.db
        """
        if db_path is None:
            db_path = os.path.join(os.path.dirname(__file__), 'market_data.db')

        self.db_path = db_path

        if not os.path.exists(db_path):
            raise FileNotFoundError(f"Master Market Data DB not found: {db_path}")

        logger.info(f"[OK] MarketDataIngestion initialized")
        logger.info(f"Database: {db_path}")

    def _get_connection(self) -> sqlite3.Connection:
        """Get database connection with write access"""
        return sqlite3.connect(self.db_path)

    def ingest_symbol_metadata(self, symbol: str) -> bool:
        """
        Ingest symbol metadata from yfinance

        Args:
            symbol: Stock ticker (will be normalized to canonical format)

        Returns:
            True if successful, False otherwise
        """
        # Normalize symbol to canonical format (auto-correct if possible)
        try:
            canonical_symbol = validate_and_normalize(symbol, provider="yahoo", strict=False)
            if canonical_symbol != symbol:
                canonical, was_corrected, msg = auto_correct(symbol)
                logger.warning(f"[SYMBOL NORMALIZED] {msg}")
                symbol = canonical_symbol
        except ValueError as e:
            logger.error(f"[ERROR] Cannot normalize symbol '{symbol}': {e}")
            return False

        try:
            ticker = yf.Ticker(symbol)
            info = ticker.info

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

    def ingest_candles(
        self,
        symbol: str,
        timeframe: str = '1d',
        period: str = '10y',
        start_date: Optional[str] = None,
        end_date: Optional[str] = None
    ) -> int:
        """
        Ingest OHLCV candles from yfinance

        Args:
            symbol: Stock ticker (will be normalized to canonical format)
            timeframe: Timeframe (1m, 5m, 15m, 30m, 1h, 1d, 1wk, 1mo)
            period: Period to fetch (1d, 5d, 1mo, 3mo, 6mo, 1y, 2y, 5y, 10y, ytd, max)
            start_date: Start date (YYYY-MM-DD)
            end_date: End date (YYYY-MM-DD)

        Returns:
            Number of candles ingested
        """
        # Normalize symbol to canonical format
        try:
            canonical_symbol = validate_and_normalize(symbol, provider="yahoo", strict=False)
            if canonical_symbol != symbol:
                canonical, was_corrected, msg = auto_correct(symbol)
                logger.warning(f"[SYMBOL NORMALIZED] {msg}")
                symbol = canonical_symbol
        except ValueError as e:
            logger.error(f"[ERROR] Cannot normalize symbol '{symbol}': {e}")
            return 0

        try:
            ticker = yf.Ticker(symbol)

            if start_date and end_date:
                hist = ticker.history(start=start_date, end=end_date, interval=timeframe)
            else:
                hist = ticker.history(period=period, interval=timeframe)

            if hist.empty:
                logger.warning(f"[SKIP] {symbol}: No data available")
                return 0

            conn = self._get_connection()
            cursor = conn.cursor()

            count = 0
            for timestamp, row in hist.iterrows():
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
                        float(row['Close'])  # Using Close as adjusted_close for now
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
        """
        Ingest fundamental data from yfinance

        Args:
            symbol: Stock ticker

        Returns:
            True if successful, False otherwise
        """
        try:
            ticker = yf.Ticker(symbol)
            info = ticker.info

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
        """
        Ingest stock splits and dividends

        Args:
            symbol: Stock ticker

        Returns:
            Tuple of (splits_count, dividends_count)
        """
        try:
            ticker = yf.Ticker(symbol)

            conn = self._get_connection()
            cursor = conn.cursor()

            # Ingest splits
            splits = ticker.splits
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
            dividends = ticker.dividends
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

    def ingest_symbol_complete(self, symbol: str, period: str = '10y', timeframe: str = '1d') -> dict:
        """
        Ingest all data for a symbol (metadata, candles, fundamentals, splits, dividends)

        Args:
            symbol: Stock ticker
            period: Period to fetch candles
            timeframe: Candle timeframe

        Returns:
            Dictionary with ingestion results
        """
        logger.info(f"[START] Ingesting {symbol}...")

        results = {
            'symbol': symbol,
            'metadata': False,
            'candles': 0,
            'fundamentals': False,
            'splits': 0,
            'dividends': 0
        }

        # Metadata
        results['metadata'] = self.ingest_symbol_metadata(symbol)
        time.sleep(0.5)  # Rate limiting

        # Candles
        results['candles'] = self.ingest_candles(symbol, timeframe=timeframe, period=period)
        time.sleep(0.5)

        # Fundamentals
        results['fundamentals'] = self.ingest_fundamentals(symbol)
        time.sleep(0.5)

        # Splits and Dividends
        splits, dividends = self.ingest_splits_and_dividends(symbol)
        results['splits'] = splits
        results['dividends'] = dividends

        logger.info(f"[COMPLETE] {symbol}: {results['candles']} candles, {results['splits']} splits, {results['dividends']} dividends")

        return results

    def ingest_multiple_symbols(self, symbols: List[str], period: str = '10y', timeframe: str = '1d') -> dict:
        """
        Ingest data for multiple symbols

        Args:
            symbols: List of stock tickers
            period: Period to fetch
            timeframe: Candle timeframe

        Returns:
            Dictionary with overall results
        """
        logger.info("=" * 80)
        logger.info(f"MASTER MARKET DATA INGESTION")
        logger.info(f"Symbols: {len(symbols)}")
        logger.info(f"Period: {period}, Timeframe: {timeframe}")
        logger.info("=" * 80)

        summary = {
            'total_symbols': len(symbols),
            'successful': 0,
            'failed': 0,
            'total_candles': 0,
            'total_splits': 0,
            'total_dividends': 0
        }

        for i, symbol in enumerate(symbols, 1):
            logger.info(f"\n[{i}/{len(symbols)}] Processing {symbol}...")

            try:
                result = self.ingest_symbol_complete(symbol, period=period, timeframe=timeframe)

                if result['candles'] > 0:
                    summary['successful'] += 1
                    summary['total_candles'] += result['candles']
                    summary['total_splits'] += result['splits']
                    summary['total_dividends'] += result['dividends']
                else:
                    summary['failed'] += 1

                # Rate limiting
                time.sleep(1)

            except Exception as e:
                logger.error(f"[ERROR] {symbol}: Complete failure - {e}")
                summary['failed'] += 1

        logger.info("\n" + "=" * 80)
        logger.info("INGESTION SUMMARY")
        logger.info("=" * 80)
        logger.info(f"Total Symbols:      {summary['total_symbols']}")
        logger.info(f"Successful:         {summary['successful']}")
        logger.info(f"Failed:             {summary['failed']}")
        logger.info(f"Total Candles:      {summary['total_candles']:,}")
        logger.info(f"Total Splits:       {summary['total_splits']}")
        logger.info(f"Total Dividends:    {summary['total_dividends']}")
        logger.info("=" * 80)

        return summary


if __name__ == '__main__':
    # Example: Ingest data for TurboMode's core symbols
    import sys
    sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

    from backend.advanced_ml.config.core_symbols import get_all_core_symbols, CRYPTO_SYMBOLS

    # Get all core symbols (77 stocks) + crypto (3 symbols) = 80 total
    all_symbols = get_all_core_symbols() + CRYPTO_SYMBOLS

    print("=" * 80)
    print("MASTER MARKET DATA DB - DATA INGESTION")
    print("=" * 80)
    print(f"Symbols to ingest: {len(all_symbols)}")
    print(f"  Core stocks: {len(get_all_core_symbols())}")
    print(f"  Crypto: {len(CRYPTO_SYMBOLS)}")
    print("\nThis will take approximately 2-3 hours...")
    print("Press Ctrl+C to cancel\n")

    input("Press Enter to continue...")

    ingestion = MarketDataIngestion()
    results = ingestion.ingest_multiple_symbols(all_symbols, period='10y', timeframe='1d')

    print("\n[OK] Data ingestion complete!")
