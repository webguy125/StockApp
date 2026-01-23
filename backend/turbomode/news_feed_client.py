"""
TurboMode News Feed Client

Fetches, parses, and normalizes RSS headlines from approved sources.
Provides deterministic news headline extraction for risk assessment.
"""

import feedparser
from typing import List, Dict, Optional
from datetime import datetime, timezone
from dataclasses import dataclass, field
import logging
import time

logger = logging.getLogger(__name__)


# Approved RSS Feed Sources
RSS_SOURCES = {
    'primary_market_feeds': [
        'https://www.reuters.com/rssFeed/businessNews',
        'https://www.cnbc.com/id/100003114/device/rss/rss.html',
        'https://feeds.marketwatch.com/marketwatch/topstories/',
        'https://feeds.finance.yahoo.com/rss/2.0/headline?s=^GSPC&region=US&lang=en-US'
    ],
    'sec_feeds': [
        'https://www.sec.gov/cgi-bin/browse-edgar?action=getcurrent&type=8-K&owner=exclude&count=100&output=atom',
        'https://www.sec.gov/cgi-bin/browse-edgar?action=getcurrent&type=10-K&owner=exclude&count=100&output=atom',
        'https://www.sec.gov/cgi-bin/browse-edgar?action=getcurrent&type=10-Q&owner=exclude&count=100&output=atom'
    ],
    'macro_feeds': [
        'https://www.federalreserve.gov/feeds/press_all.xml'
    ]
}


@dataclass
class Headline:
    """Normalized headline object."""
    title: str
    summary: str
    timestamp_utc: datetime
    source: str
    link: str
    raw: Dict = field(default_factory=dict)

    def __repr__(self):
        return f"<Headline: {self.title[:50]}... ({self.source})>"


class NewsFeedClient:
    """
    RSS feed client for fetching and parsing news headlines.

    Fetches from approved RSS sources, normalizes timestamps,
    and returns structured headline objects.
    """

    def __init__(self, max_headlines_per_feed: int = 100, timeout_seconds: int = 10):
        """
        Initialize news feed client.

        Args:
            max_headlines_per_feed: Maximum headlines to fetch per feed
            timeout_seconds: HTTP timeout for RSS fetches
        """
        self.max_headlines_per_feed = max_headlines_per_feed
        self.timeout_seconds = timeout_seconds
        self.last_fetch_time = None
        self.fetch_count = 0

        logger.info("[NEWS FEED CLIENT] Initialized")
        logger.info(f"  Max headlines per feed: {max_headlines_per_feed}")
        logger.info(f"  Timeout: {timeout_seconds}s")

    def fetch_all_headlines(self) -> List[Headline]:
        """
        Fetch headlines from all approved RSS sources.

        Returns:
            List of normalized Headline objects
        """
        start_time = time.time()
        all_headlines = []

        # Flatten all RSS sources
        all_feeds = []
        for category, feeds in RSS_SOURCES.items():
            for feed_url in feeds:
                all_feeds.append((category, feed_url))

        logger.info(f"[RSS FETCH] Fetching from {len(all_feeds)} feeds...")

        success_count = 0
        error_count = 0

        for category, feed_url in all_feeds:
            try:
                headlines = self._fetch_single_feed(feed_url, category)
                all_headlines.extend(headlines)
                success_count += 1
                logger.debug(f"  ✓ {category}: {feed_url} ({len(headlines)} headlines)")

            except Exception as e:
                error_count += 1
                logger.warning(f"  ✗ {category}: {feed_url} - {e}")

        elapsed = time.time() - start_time
        self.last_fetch_time = datetime.now(timezone.utc)
        self.fetch_count += 1

        logger.info(f"[RSS FETCH] Complete: {success_count} success, {error_count} errors, "
                   f"{len(all_headlines)} total headlines in {elapsed:.2f}s")

        return all_headlines

    def _fetch_single_feed(self, feed_url: str, category: str) -> List[Headline]:
        """
        Fetch and parse a single RSS feed.

        Args:
            feed_url: RSS feed URL
            category: Feed category (primary_market_feeds, sec_feeds, macro_feeds)

        Returns:
            List of Headline objects
        """
        # Parse RSS feed
        feed = feedparser.parse(feed_url)

        if feed.bozo:  # feedparser detected malformed feed
            logger.warning(f"Malformed feed: {feed_url} - {feed.bozo_exception}")

        headlines = []

        # Extract entries (limit to max_headlines_per_feed)
        entries = feed.entries[:self.max_headlines_per_feed]

        for entry in entries:
            try:
                headline = self._parse_entry(entry, feed_url, category)
                if headline:
                    headlines.append(headline)
            except Exception as e:
                logger.debug(f"Failed to parse entry from {feed_url}: {e}")

        return headlines

    def _parse_entry(self, entry: Dict, feed_url: str, category: str) -> Optional[Headline]:
        """
        Parse a single RSS entry into a Headline object.

        Args:
            entry: feedparser entry dict
            feed_url: Source feed URL
            category: Feed category

        Returns:
            Headline object or None if parsing fails
        """
        # Extract title
        title = entry.get('title', '').strip()
        if not title:
            return None

        # Extract summary
        summary = entry.get('summary', entry.get('description', '')).strip()

        # Extract and normalize timestamp
        timestamp_utc = self._parse_timestamp(entry)

        # Extract link
        link = entry.get('link', feed_url)

        # Determine source name
        source = self._extract_source_name(feed_url, category)

        return Headline(
            title=title,
            summary=summary,
            timestamp_utc=timestamp_utc,
            source=source,
            link=link,
            raw=entry
        )

    def _parse_timestamp(self, entry: Dict) -> datetime:
        """
        Parse and normalize timestamp to UTC.

        Args:
            entry: feedparser entry dict

        Returns:
            datetime in UTC timezone
        """
        # Try published_parsed first
        if hasattr(entry, 'published_parsed') and entry.published_parsed:
            try:
                timestamp = datetime(*entry.published_parsed[:6], tzinfo=timezone.utc)
                return timestamp
            except:
                pass

        # Try updated_parsed
        if hasattr(entry, 'updated_parsed') and entry.updated_parsed:
            try:
                timestamp = datetime(*entry.updated_parsed[:6], tzinfo=timezone.utc)
                return timestamp
            except:
                pass

        # Fallback to current time
        return datetime.now(timezone.utc)

    def _extract_source_name(self, feed_url: str, category: str) -> str:
        """
        Extract human-readable source name from feed URL.

        Args:
            feed_url: RSS feed URL
            category: Feed category

        Returns:
            Source name string
        """
        # Extract domain from URL
        if 'reuters.com' in feed_url:
            return 'Reuters'
        elif 'cnbc.com' in feed_url:
            return 'CNBC'
        elif 'marketwatch.com' in feed_url:
            return 'MarketWatch'
        elif 'yahoo.com' in feed_url:
            return 'Yahoo Finance'
        elif 'sec.gov' in feed_url:
            if '8-K' in feed_url:
                return 'SEC 8-K'
            elif '10-K' in feed_url:
                return 'SEC 10-K'
            elif '10-Q' in feed_url:
                return 'SEC 10-Q'
            else:
                return 'SEC'
        elif 'federalreserve.gov' in feed_url:
            return 'Federal Reserve'
        elif 'gdeltproject.org' in feed_url:
            return 'GDELT'
        else:
            return category

    def get_recent_headlines(self, hours: int = 24) -> List[Headline]:
        """
        Fetch headlines from the last N hours.

        Args:
            hours: Number of hours to look back

        Returns:
            List of headlines within time window
        """
        all_headlines = self.fetch_all_headlines()

        cutoff_time = datetime.now(timezone.utc) - timedelta(hours=hours)

        recent = [h for h in all_headlines if h.timestamp_utc >= cutoff_time]

        logger.info(f"[RSS FILTER] {len(recent)}/{len(all_headlines)} headlines from last {hours} hours")

        return recent


# Convenience function for testing
def test_feed_client():
    """Test the news feed client."""
    client = NewsFeedClient()

    logger.info("Testing NewsFeedClient...")

    headlines = client.fetch_all_headlines()

    logger.info(f"\nFetched {len(headlines)} headlines")

    if headlines:
        logger.info("\nSample headlines:")
        for headline in headlines[:5]:
            logger.info(f"  [{headline.source}] {headline.title}")
            logger.info(f"    Time: {headline.timestamp_utc}")
            logger.info(f"    Link: {headline.link}")

    return headlines


if __name__ == '__main__':
    # Test the feed client
    logging.basicConfig(
        level=logging.INFO,
        format='[%(asctime)s] %(levelname)s - %(message)s'
    )

    from datetime import timedelta
    test_feed_client()
