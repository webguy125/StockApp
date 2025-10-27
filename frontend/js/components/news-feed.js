/**
 * News Feed Component
 * Displays market news and symbol-specific updates
 */

export class NewsFeed {
  constructor() {
    this.newsItems = [];
    this.currentFilter = 'all';
    this.currentSymbol = null;
    this.updateInterval = null;

    this.init();
  }

  init() {
    this.loadNews();
    this.bindEvents();
    this.startAutoRefresh();
  }

  bindEvents() {
    // Filter buttons could be added here
  }

  setSymbolFilter(symbol) {
    this.currentSymbol = symbol;
    this.currentFilter = symbol ? 'symbol' : 'all';
    this.loadNews(); // Reload news for new symbol
  }

  async loadNews() {
    try {
      let url;
      if (this.currentSymbol) {
        // Get news for specific symbol
        url = `/news/${this.currentSymbol}?limit=15`;
      } else {
        // Get general market news
        url = `/news/market?limit=20`;
      }

      const response = await fetch(url);
      const newsData = await response.json();

      // Convert Yahoo RSS format to our internal format
      this.newsItems = newsData.map((item, index) => ({
        id: `${item.symbol}-${index}`,
        source: item.source || 'Yahoo Finance',
        headline: item.title || 'No headline available',
        symbol: item.symbol !== 'Market' ? item.symbol : null,
        timestamp: item.published || Date.now(),
        preview: item.summary || 'No summary available',
        link: item.link || '#',
        type: item.symbol !== 'Market' ? 'symbol' : 'market'
      }));

      // If no news returned, generate mock headlines
      if (this.newsItems.length === 0 || !this.newsItems[0].headline) {
        this.newsItems = this.generateMockNews();
      }

      this.render();
    } catch (error) {
      console.error('Error loading news:', error);
      // Fallback to showing mock news
      this.newsItems = this.generateMockNews();
      this.render();
    }
  }

  generateMockNews() {
    const sources = ['Reuters', 'Bloomberg', 'CNBC', 'MarketWatch', 'WSJ', 'Financial Times'];
    const symbols = this.currentSymbol ? [this.currentSymbol] : ['AAPL', 'MSFT', 'GOOGL', 'TSLA', 'AMZN', 'NVDA', 'META', 'BTC-USD'];

    const headlines = [
      { text: 'Market opens higher as investors digest earnings reports', type: 'market' },
      { text: '{symbol} announces new product line, shares surge', type: 'symbol' },
      { text: 'Fed signals potential rate cut in upcoming meeting', type: 'market' },
      { text: '{symbol} beats earnings expectations, raises guidance', type: 'symbol' },
      { text: 'Tech sector leads market rally amid strong economic data', type: 'market' },
      { text: '{symbol} CEO discusses future growth strategy in interview', type: 'symbol' },
      { text: 'Oil prices climb as supply concerns mount', type: 'market' },
      { text: '{symbol} price targets raised by analysts', type: 'symbol' },
      { text: 'New trade agreement boosts investor confidence', type: 'market' },
      { text: '{symbol} announces strategic partnership with major player', type: 'symbol' },
      { text: 'Consumer confidence index reaches new high', type: 'market' },
      { text: '{symbol} reports record revenue in latest quarter', type: 'symbol' },
      { text: 'Global markets react to central bank policy changes', type: 'market' },
      { text: '{symbol} expands operations into new markets', type: 'symbol' },
      { text: 'Analysts predict continued growth in tech sector', type: 'market' },
      { text: '{symbol} trading volume exceeds expectations', type: 'symbol' },
      { text: '{symbol} institutional investors increase holdings', type: 'symbol' }
    ];

    const news = [];
    const now = Date.now();
    const limit = this.currentSymbol ? 15 : 20;

    for (let i = 0; i < limit; i++) {
      const headline = headlines[Math.floor(Math.random() * headlines.length)];
      const source = sources[Math.floor(Math.random() * sources.length)];
      const symbol = headline.type === 'symbol' ? symbols[Math.floor(Math.random() * symbols.length)] : null;
      const minutesAgo = Math.floor(Math.random() * 240); // 0-4 hours ago

      news.push({
        id: `mock-${i}`,
        source: source,
        headline: headline.text.replace('{symbol}', symbol || ''),
        symbol: symbol,
        timestamp: now - (minutesAgo * 60 * 1000),
        preview: this.generatePreview(),
        link: '#',
        type: headline.type
      });
    }

    // Sort by timestamp (newest first)
    news.sort((a, b) => b.timestamp - a.timestamp);

    return news;
  }

  generatePreview() {
    const previews = [
      'Market analysts are closely watching the developments as trading volumes increase significantly...',
      'The announcement comes amid growing investor interest in the sector and positive market sentiment...',
      'Experts suggest this could have significant implications for the broader market in coming weeks...',
      'Trading activity has picked up considerably following the news, with volumes exceeding expectations...',
      'Industry observers note this trend aligns with broader market dynamics and economic indicators...',
      'The development is expected to influence investor sentiment and trading patterns in the near term...',
      'Market participants are adjusting their positions in response to the latest information...',
      'This follows a series of similar announcements that have shaped market expectations recently...'
    ];

    return previews[Math.floor(Math.random() * previews.length)];
  }

  render() {
    const container = document.getElementById('news-content');
    if (!container) return;

    container.innerHTML = '';

    // Filter news based on current filter
    const filteredNews = this.filterNews();

    if (filteredNews.length === 0) {
      const emptyMessage = document.createElement('div');
      emptyMessage.style.cssText = 'padding: 16px; text-align: center; color: var(--tos-text-secondary);';
      emptyMessage.textContent = 'No news available';
      container.appendChild(emptyMessage);
      return;
    }

    filteredNews.forEach(newsItem => {
      const item = this.createNewsItem(newsItem);
      container.appendChild(item);
    });
  }

  filterNews() {
    if (this.currentFilter === 'all') {
      return this.newsItems;
    } else if (this.currentFilter === 'symbol' && this.currentSymbol) {
      return this.newsItems.filter(item => item.symbol === this.currentSymbol);
    } else if (this.currentFilter === 'market') {
      return this.newsItems.filter(item => item.type === 'market');
    }

    return this.newsItems;
  }

  createNewsItem(newsItem) {
    const item = document.createElement('div');
    item.className = 'tos-news-item';
    item.dataset.id = newsItem.id;

    const header = document.createElement('div');
    header.className = 'tos-news-header';

    const source = document.createElement('div');
    source.className = 'tos-news-source';
    source.textContent = newsItem.symbol ? `${newsItem.source} - ${newsItem.symbol}` : newsItem.source;

    const time = document.createElement('div');
    time.className = 'tos-news-time';
    time.textContent = this.formatTimestamp(newsItem.timestamp);

    header.appendChild(source);
    header.appendChild(time);

    const headline = document.createElement('div');
    headline.className = 'tos-news-headline';
    headline.textContent = newsItem.headline;
    headline.style.cursor = 'pointer';

    const preview = document.createElement('div');
    preview.className = 'tos-news-preview';
    preview.textContent = newsItem.preview;
    preview.style.display = 'none'; // Hidden by default

    item.appendChild(header);
    item.appendChild(headline);
    item.appendChild(preview);

    // Click headline to open article in new tab
    headline.addEventListener('click', (e) => {
      e.stopPropagation();
      if (newsItem.link) {
        window.open(newsItem.link, '_blank');
      }
    });

    // Click item to expand/collapse preview
    item.addEventListener('click', () => {
      const isExpanded = preview.style.display === 'block';
      preview.style.display = isExpanded ? 'none' : 'block';
    });

    return item;
  }

  formatTimestamp(timestamp) {
    const now = Date.now();
    const diff = now - timestamp;

    const minutes = Math.floor(diff / 60000);
    const hours = Math.floor(diff / 3600000);
    const days = Math.floor(diff / 86400000);

    if (minutes < 1) {
      return 'Just now';
    } else if (minutes < 60) {
      return `${minutes} min${minutes > 1 ? 's' : ''} ago`;
    } else if (hours < 24) {
      return `${hours} hour${hours > 1 ? 's' : ''} ago`;
    } else {
      return `${days} day${days > 1 ? 's' : ''} ago`;
    }
  }

  startAutoRefresh() {
    // Refresh news every 2 minutes
    this.updateInterval = setInterval(() => {
      this.refreshNews();
    }, 120000);
  }

  async refreshNews() {
    // Reload news from API
    await this.loadNews();
  }

  stopAutoRefresh() {
    if (this.updateInterval) {
      clearInterval(this.updateInterval);
      this.updateInterval = null;
    }
  }

  destroy() {
    this.stopAutoRefresh();
  }
}

export function initializeNewsFeed() {
  return new NewsFeed();
}
