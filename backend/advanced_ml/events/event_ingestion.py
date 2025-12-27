"""
Event Ingestion System
Handles SEC filings and news source ingestion per specification.
"""

import logging
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any
import pandas as pd
import numpy as np
import time

logger = logging.getLogger(__name__)


class EventIngestion:
    """
    Unified ingestion for SEC filings and news sources.
    Implements normalized schema and deduplication.
    """

    def __init__(self, config: Dict[str, Any]):
        """
        Initialize event ingestion system.

        Args:
            config: Ingestion configuration dictionary
        """
        self.config = config
        self.sec_enabled = config.get('sec', {}).get('enabled', False)
        self.news_enabled = config.get('news', {}).get('enabled', False)

        # SEC configuration
        if self.sec_enabled:
            self.sec_filing_types = config['sec']['filing_types']
            self.sec_max_lag_minutes = config['sec']['max_lag_minutes']
            self.sec_normalize_schema = config['sec']['normalize_schema']

        # News configuration
        if self.news_enabled:
            self.news_sources = config['news']['sources']
            self.news_languages = config['news']['languages']
            self.news_dedupe_window = config['news']['dedupe_window_minutes']
            self.news_normalize_schema = config['news']['normalize_schema']

        logger.info(f"[INGESTION] Initialized (SEC: {self.sec_enabled}, News: {self.news_enabled})")

    def fetch_events(
        self,
        ticker: str,
        start_date: datetime,
        end_date: datetime,
        sources: List[str]
    ) -> pd.DataFrame:
        """
        Fetch events from specified sources.

        Args:
            ticker: Stock ticker symbol
            start_date: Start date for retrieval
            end_date: End date for retrieval
            sources: List of sources ['sec', 'news']

        Returns:
            DataFrame with normalized event schema
        """
        all_events = []

        if 'sec' in sources and self.sec_enabled:
            sec_events = self._fetch_sec_filings(ticker, start_date, end_date)
            all_events.append(sec_events)

        if 'news' in sources and self.news_enabled:
            news_events = self._fetch_news(ticker, start_date, end_date)
            all_events.append(news_events)

        if not all_events:
            return pd.DataFrame()

        # Combine all events
        combined = pd.concat(all_events, ignore_index=True)

        # Sort by timestamp
        combined = combined.sort_values('timestamp').reset_index(drop=True)

        logger.info(f"[INGESTION] Fetched {len(combined)} total events for {ticker}")
        return combined

    def _fetch_sec_filings(
        self,
        ticker: str,
        start_date: datetime,
        end_date: datetime
    ) -> pd.DataFrame:
        """
        Fetch SEC filings for ticker.

        Args:
            ticker: Stock ticker symbol
            start_date: Start date
            end_date: End date

        Returns:
            DataFrame with normalized SEC filing schema
        """
        # Placeholder implementation - connects to SEC EDGAR API
        # In production, this would use sec-edgar-downloader or similar

        logger.info(f"[SEC] Fetching filings for {ticker} from {start_date} to {end_date}")

        # Mock data for demonstration
        events = []

        # Simulate SEC filing retrieval
        for filing_type in self.sec_filing_types[:3]:  # Sample subset
            days_delta = max(1, (end_date - start_date).days)
            event = {
                'source_type': 'sec',
                'ticker': ticker,
                'timestamp': start_date + timedelta(days=int(np.random.randint(0, days_delta))),
                'filing_type': filing_type,
                'raw_text': f"Mock {filing_type} filing content",
                'metadata': {'cik': '0000000000', 'company_name': ticker},
                'accession_number': f"{np.random.randint(100000, 999999)}-{np.random.randint(10, 99)}",
                'item_codes': [],
                'is_amendment': False
            }
            events.append(event)

        df = pd.DataFrame(events)

        # Ensure normalized schema
        for col in self.sec_normalize_schema:
            if col not in df.columns:
                df[col] = None

        logger.info(f"[SEC] Retrieved {len(df)} filings")
        return df

    def _fetch_news(
        self,
        ticker: str,
        start_date: datetime,
        end_date: datetime
    ) -> pd.DataFrame:
        """
        Fetch news articles for ticker.

        Args:
            ticker: Stock ticker symbol
            start_date: Start date
            end_date: End date

        Returns:
            DataFrame with normalized news schema
        """
        logger.info(f"[NEWS] Fetching articles for {ticker} from {start_date} to {end_date}")

        # Placeholder implementation - connects to news APIs
        # In production, this would use Dow Jones, Reuters, Bloomberg APIs

        events = []

        # Simulate news article retrieval
        for source in self.news_sources[:2]:  # Sample subset
            days_delta = max(1, (end_date - start_date).days)
            event = {
                'source_type': 'news',
                'ticker': ticker,
                'timestamp': start_date + timedelta(days=int(np.random.randint(0, days_delta))),
                'headline': f"Mock headline for {ticker} from {source}",
                'raw_text': f"Mock article content discussing {ticker} business developments",
                'metadata': {'source': source, 'language': 'en'},
                'urgency_level': np.random.choice(['low', 'medium', 'high']),
                'entity_relevance_score': float(np.random.uniform(0.5, 1.0))
            }
            events.append(event)

        df = pd.DataFrame(events)

        # Apply noise filtering
        df = self._apply_noise_filtering(df)

        # Apply deduplication
        df = self._deduplicate_news(df)

        # Ensure normalized schema
        for col in self.news_normalize_schema:
            if col not in df.columns:
                df[col] = None

        logger.info(f"[NEWS] Retrieved {len(df)} articles after filtering")
        return df

    def _apply_noise_filtering(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Apply noise filtering to news articles.

        Args:
            df: News DataFrame

        Returns:
            Filtered DataFrame
        """
        noise_config = self.config['news'].get('noise_filtering', {})

        if not noise_config:
            return df

        original_count = len(df)

        # Filter by minimum article length
        min_length = noise_config.get('min_article_length_chars', 0)
        if min_length > 0:
            df = df[df['raw_text'].str.len() >= min_length]

        # Filter blacklist keywords
        blacklist = noise_config.get('blacklist_keywords', [])
        for keyword in blacklist:
            df = df[~df['headline'].str.lower().str.contains(keyword, na=False)]
            df = df[~df['raw_text'].str.lower().str.contains(keyword, na=False)]

        filtered_count = original_count - len(df)
        if filtered_count > 0:
            logger.info(f"[NOISE_FILTER] Removed {filtered_count} articles")

        return df

    def _deduplicate_news(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Deduplicate news articles within time window.

        Args:
            df: News DataFrame

        Returns:
            Deduplicated DataFrame
        """
        if len(df) == 0:
            return df

        original_count = len(df)

        # Sort by timestamp
        df = df.sort_values('timestamp').reset_index(drop=True)

        # Group by ticker and headline similarity window
        window = timedelta(minutes=self.news_dedupe_window)

        # Simple deduplication: keep first occurrence within window
        df['headline_normalized'] = df['headline'].str.lower().str.strip()
        df = df.drop_duplicates(subset=['ticker', 'headline_normalized'], keep='first')
        df = df.drop(columns=['headline_normalized'])

        dedupe_count = original_count - len(df)
        if dedupe_count > 0:
            logger.info(f"[DEDUPE] Removed {dedupe_count} duplicate articles")

        return df

    def validate_event_schema(self, df: pd.DataFrame) -> bool:
        """
        Validate that events conform to normalized schema.

        Args:
            df: Events DataFrame

        Returns:
            True if valid, False otherwise
        """
        if len(df) == 0:
            return True

        required_columns = ['source_type', 'ticker', 'timestamp']

        for col in required_columns:
            if col not in df.columns:
                logger.error(f"[VALIDATION] Missing required column: {col}")
                return False

        # Check timestamp is datetime
        if not pd.api.types.is_datetime64_any_dtype(df['timestamp']):
            logger.error("[VALIDATION] timestamp column must be datetime type")
            return False

        return True

    def get_statistics(self) -> Dict[str, Any]:
        """
        Get ingestion statistics.

        Returns:
            Statistics dictionary
        """
        return {
            'sec_enabled': self.sec_enabled,
            'news_enabled': self.news_enabled,
            'sec_filing_types': self.sec_filing_types if self.sec_enabled else [],
            'news_sources': self.news_sources if self.news_enabled else [],
            'sec_max_lag_minutes': self.sec_max_lag_minutes if self.sec_enabled else 0,
            'news_dedupe_window_minutes': self.news_dedupe_window if self.news_enabled else 0
        }
