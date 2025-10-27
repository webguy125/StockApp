"""
Yahoo Finance RSS News Service
Fetches real financial news from Yahoo Finance RSS feeds
"""

import feedparser
from datetime import datetime
import requests
from functools import lru_cache

class NewsService:
    """Service for fetching news from Yahoo Finance RSS feeds"""

    def __init__(self):
        self.base_url = "https://feeds.finance.yahoo.com/rss/2.0/headline"

    @lru_cache(maxsize=100)
    def get_market_news(self, max_items=20):
        """
        Get general market news
        Returns list of news articles
        """
        try:
            url = f"{self.base_url}?s=^GSPC&region=US&lang=en-US"
            feed = feedparser.parse(url)

            articles = []
            for entry in feed.entries[:max_items]:
                articles.append({
                    'title': entry.get('title', 'No title'),
                    'link': entry.get('link', ''),
                    'published': self._parse_date(entry.get('published', '')),
                    'summary': entry.get('summary', ''),
                    'source': 'Yahoo Finance',
                    'symbol': 'Market'
                })

            return articles
        except Exception as e:
            print(f"Error fetching market news: {e}")
            return []

    def get_symbol_news(self, symbol, max_items=15):
        """
        Get news for a specific stock symbol

        Args:
            symbol: Stock ticker symbol (e.g., 'AAPL', 'TSLA')
            max_items: Maximum number of articles to return

        Returns:
            List of news articles
        """
        try:
            # Yahoo Finance RSS feed for specific symbol
            url = f"{self.base_url}?s={symbol.upper()}&region=US&lang=en-US"
            feed = feedparser.parse(url)

            articles = []
            for entry in feed.entries[:max_items]:
                articles.append({
                    'title': entry.get('title', 'No title'),
                    'link': entry.get('link', ''),
                    'published': self._parse_date(entry.get('published', '')),
                    'summary': entry.get('summary', ''),
                    'source': 'Yahoo Finance',
                    'symbol': symbol.upper()
                })

            return articles
        except Exception as e:
            print(f"Error fetching news for {symbol}: {e}")
            return []

    def get_multiple_symbols_news(self, symbols, max_items_per_symbol=5):
        """
        Get news for multiple symbols combined

        Args:
            symbols: List of ticker symbols
            max_items_per_symbol: Max articles per symbol

        Returns:
            List of news articles sorted by date
        """
        all_articles = []

        for symbol in symbols:
            articles = self.get_symbol_news(symbol, max_items_per_symbol)
            all_articles.extend(articles)

        # Sort by published date (newest first)
        all_articles.sort(
            key=lambda x: x['published'],
            reverse=True
        )

        return all_articles

    def get_sector_news(self, sector='technology', max_items=15):
        """
        Get news for a specific sector

        Args:
            sector: Sector name (technology, finance, healthcare, etc.)
            max_items: Maximum number of articles

        Returns:
            List of news articles
        """
        # Map sectors to representative symbols or ETFs
        sector_symbols = {
            'technology': 'XLK',
            'finance': 'XLF',
            'healthcare': 'XLV',
            'energy': 'XLE',
            'consumer': 'XLY',
            'utilities': 'XLU',
            'materials': 'XLB',
            'industrials': 'XLI'
        }

        symbol = sector_symbols.get(sector.lower(), 'SPY')
        return self.get_symbol_news(symbol, max_items)

    def _parse_date(self, date_str):
        """
        Parse date string to datetime object

        Args:
            date_str: Date string from RSS feed

        Returns:
            datetime object or current time if parsing fails
        """
        try:
            # Try parsing common RSS date formats
            from dateutil import parser
            return parser.parse(date_str)
        except:
            # Return current time if parsing fails
            return datetime.now()

    def get_trending_news(self, max_items=20):
        """
        Get trending/popular financial news
        Uses general market feed
        """
        return self.get_market_news(max_items)

    def search_news(self, query, max_items=15):
        """
        Search for news containing specific keywords
        Note: Yahoo RSS doesn't support search, so this filters market news

        Args:
            query: Search query string
            max_items: Maximum results

        Returns:
            Filtered news articles
        """
        all_news = self.get_market_news(50)  # Get more articles to filter

        query_lower = query.lower()
        filtered = [
            article for article in all_news
            if query_lower in article['title'].lower() or
               query_lower in article['summary'].lower()
        ]

        return filtered[:max_items]


# Singleton instance
news_service = NewsService()
