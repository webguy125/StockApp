
import sys
from pathlib import Path
project_root = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(project_root))

"""
TurboMode News Engine

Provides real-time news risk assessment at three levels:
1. Global risk (market-wide events)
2. Sector risk (industry-specific events)
3. Symbol risk (company-specific events)

Risk levels: NONE, LOW, MEDIUM, HIGH, CRITICAL
"""

from typing import Dict, Optional, List, Tuple
from datetime import datetime, timedelta, timezone
from enum import Enum
from dataclasses import dataclass, field
import logging
import re

logger = logging.getLogger(__name__)


class RiskLevel(Enum):
    """News risk levels in ascending order of severity."""
    NONE = 0
    LOW = 1
    MEDIUM = 2
    HIGH = 3
    CRITICAL = 4

    def __lt__(self, other):
        if not isinstance(other, RiskLevel):
            return NotImplemented
        return self.value < other.value

    def __le__(self, other):
        if not isinstance(other, RiskLevel):
            return NotImplemented
        return self.value <= other.value

    def __gt__(self, other):
        if not isinstance(other, RiskLevel):
            return NotImplemented
        return self.value > other.value

    def __ge__(self, other):
        if not isinstance(other, RiskLevel):
            return NotImplemented
        return self.value >= other.value


@dataclass
class NewsState:
    """
    Centralized news risk state.

    Attributes:
        global_risk: Market-wide risk level
        global_sentiment: Market-wide directional bias ("bullish", "bearish", "neutral")
        sector_risk: Dictionary mapping sector -> RiskLevel
        sector_sentiment: Dictionary mapping sector -> sentiment bias ("bullish", "bearish", "neutral")
        symbol_risk: Dictionary mapping symbol -> RiskLevel
        symbol_sentiment: Dictionary mapping symbol -> sentiment bias ("bullish", "bearish", "neutral")
        last_updated: Timestamp of last risk assessment
    """
    global_risk: RiskLevel = RiskLevel.NONE
    global_sentiment: str = 'neutral'  # Global macro directional bias
    sector_risk: Dict[str, RiskLevel] = field(default_factory=dict)
    sector_sentiment: Dict[str, str] = field(default_factory=dict)  # Per-sector directional bias
    symbol_risk: Dict[str, RiskLevel] = field(default_factory=dict)
    symbol_sentiment: Dict[str, str] = field(default_factory=dict)  # Per-symbol directional bias
    last_updated: datetime = field(default_factory=datetime.now)

    def get_symbol_risk(self, symbol: str) -> RiskLevel:
        """Get risk level for a specific symbol (default: NONE)."""
        return self.symbol_risk.get(symbol, RiskLevel.NONE)

    def get_sector_risk(self, sector: str) -> RiskLevel:
        """Get risk level for a specific sector (default: NONE)."""
        return self.sector_risk.get(sector, RiskLevel.NONE)

    def set_global_risk(self, level: RiskLevel):
        """Set global market risk level."""
        if self.global_risk != level:
            logger.info(f"[NEWS] Global risk changed: {self.global_risk.name} -> {level.name}")
            self.global_risk = level
            self.last_updated = datetime.now()

    def set_sector_risk(self, sector: str, level: RiskLevel):
        """Set risk level for a specific sector."""
        old_level = self.sector_risk.get(sector, RiskLevel.NONE)
        if old_level != level:
            logger.info(f"[NEWS] Sector risk changed for {sector}: {old_level.name} -> {level.name}")
            self.sector_risk[sector] = level
            self.last_updated = datetime.now()

    def set_symbol_risk(self, symbol: str, level: RiskLevel):
        """Set risk level for a specific symbol."""
        old_level = self.symbol_risk.get(symbol, RiskLevel.NONE)
        if old_level != level:
            logger.info(f"[NEWS] Symbol risk changed for {symbol}: {old_level.name} -> {level.name}")
            self.symbol_risk[symbol] = level
            self.last_updated = datetime.now()

    def set_global_sentiment(self, sentiment: str):
        """Set global market sentiment ('bullish', 'bearish', 'neutral')."""
        if self.global_sentiment != sentiment:
            logger.info(f"[NEWS] Global sentiment changed: {self.global_sentiment} -> {sentiment}")
            self.global_sentiment = sentiment
            self.last_updated = datetime.now()

    def get_global_sentiment(self) -> str:
        """Get global market sentiment (default: 'neutral')."""
        return self.global_sentiment

    def set_sector_sentiment(self, sector: str, sentiment: str):
        """Set sentiment bias for a specific sector ('bullish', 'bearish', 'neutral')."""
        old_sentiment = self.sector_sentiment.get(sector, 'neutral')
        if old_sentiment != sentiment:
            logger.info(f"[NEWS] Sector sentiment changed for {sector}: {old_sentiment} -> {sentiment}")
            self.sector_sentiment[sector] = sentiment
            self.last_updated = datetime.now()

    def get_sector_sentiment(self, sector: str) -> str:
        """Get sentiment bias for a specific sector (default: 'neutral')."""
        return self.sector_sentiment.get(sector, 'neutral')

    def set_symbol_sentiment(self, symbol: str, sentiment: str):
        """Set sentiment bias for a specific symbol ('bullish', 'bearish', 'neutral')."""
        old_sentiment = self.symbol_sentiment.get(symbol, 'neutral')
        if old_sentiment != sentiment:
            logger.info(f"[NEWS] Symbol sentiment changed for {symbol}: {old_sentiment} -> {sentiment}")
            self.symbol_sentiment[symbol] = sentiment
            self.last_updated = datetime.now()

    def get_symbol_sentiment(self, symbol: str) -> str:
        """Get sentiment bias for a specific symbol (default: 'neutral')."""
        return self.symbol_sentiment.get(symbol, 'neutral')

    def get_max_risk_for_symbol(self, symbol: str, sector: str) -> Tuple[RiskLevel, str]:
        """
        Get the maximum risk level affecting a symbol.

        Returns:
            (max_risk_level, source) where source is 'global', 'sector', or 'symbol'
        """
        risks = [
            (self.global_risk, 'global'),
            (self.get_sector_risk(sector), 'sector'),
            (self.get_symbol_risk(symbol), 'symbol')
        ]
        max_risk = max(risks, key=lambda x: x[0])
        return max_risk


class NewsEngine:
    """
    News risk assessment engine.

    Provides real-time news risk state and risk-aware decision support.
    """

    # Directional bias strengths (stacked system: global + sector + symbol)
    GLOBAL_BIAS_STRENGTH = 0.05   # Weak macro bias
    SECTOR_BIAS_STRENGTH = 0.05   # Medium sector bias
    SYMBOL_BIAS_STRENGTH = 0.10   # Strong symbol bias

    def __init__(self, enable_sector_blocking: bool = True, enable_global_blocking: bool = True):
        """
        Initialize NewsEngine with RSS feed client.

        Args:
            enable_sector_blocking: If True, block entries when sector_risk >= HIGH
            enable_global_blocking: If True, block entries when global_risk >= HIGH
        """
        self.state = NewsState()
        self.enable_sector_blocking = enable_sector_blocking
        self.enable_global_blocking = enable_global_blocking

        # Initialize RSS feed client
        from backend.turbomode.news_feed_client import NewsFeedClient
        self.feed_client = NewsFeedClient(max_headlines_per_feed=100, timeout_seconds=10)

        # Store last headlines for debugging
        self.last_headlines = []

        logger.info("[NEWS ENGINE] Initialized with RSS feed client")
        logger.info(f"  Sector blocking: {enable_sector_blocking}")
        logger.info(f"  Global blocking: {enable_global_blocking}")

    def update_global_risk(self, level: RiskLevel):
        """Update global market risk level."""
        self.state.set_global_risk(level)

    def update_sector_risk(self, sector: str, level: RiskLevel):
        """Update risk level for a specific sector."""
        self.state.set_sector_risk(sector, level)

    def update_symbol_risk(self, symbol: str, level: RiskLevel):
        """Update risk level for a specific symbol."""
        self.state.set_symbol_risk(symbol, level)

    def should_block_entry(self, symbol: str, sector: str) -> Tuple[bool, Optional[str]]:
        """
        Determine if entry should be blocked based on news risk.

        PURE LONG/SHORT TRADING ENGINE:
        - NO automatic blocks based on risk levels or sentiment
        - Even "CRITICAL" news is tradable (short side for bad news, long side for good news)
        - Only TRUE market halts would block (halted/delisted states - NOT IMPLEMENTED YET)
        - System operates continuously under ALL conditions
        - All risk/sentiment converted to directional biasing instead

        Args:
            symbol: Stock symbol
            sector: Sector name

        Returns:
            (should_block, reason) - Always returns (False, None) for now
                                    Future: Check for actual trading halts/delistings
        """
        # TODO: Add explicit market halt detection (halted=True, delisted=True, etc.)
        # For now, NEVER block - everything is tradable via directional biasing

        return False, None

    def should_raise_entry_threshold(self) -> bool:
        """
        Determine if entry thresholds should be raised (0.60 -> 0.70).

        UPDATED BEHAVIOR:
        - GLOBAL_HIGH no longer raises threshold
        - Only GLOBAL_CRITICAL blocks entries (via should_block_entry)
        - System stays active during GLOBAL_HIGH with directional biasing

        Returns:
            False (threshold raising disabled per directional trading design)
        """
        return False  # GLOBAL_HIGH no longer raises threshold

    def should_tighten_stop(self, symbol: str, sector: str) -> Tuple[bool, float]:
        """
        Determine if stop should be tightened based on news risk.

        Args:
            symbol: Stock symbol
            sector: Sector name

        Returns:
            (should_tighten, multiplier) - True if stop should be tightened, with multiplier (e.g., 0.7)
        """
        max_risk, source = self.state.get_max_risk_for_symbol(symbol, sector)

        if max_risk >= RiskLevel.HIGH:
            return True, 0.7

        return False, 1.0

    def should_constrain_target(self, symbol: str, sector: str) -> bool:
        """
        Determine if target should be constrained (no extension during high risk).

        Args:
            symbol: Stock symbol
            sector: Sector name

        Returns:
            True if target should be constrained
        """
        max_risk, _ = self.state.get_max_risk_for_symbol(symbol, sector)
        return max_risk >= RiskLevel.HIGH

    def should_force_flatten(self, symbol: str, sector: str) -> Tuple[bool, str]:
        """
        Determine if position should be immediately flattened.

        Args:
            symbol: Stock symbol
            sector: Sector name

        Returns:
            (should_flatten, reason)
        """
        # Symbol-level CRITICAL risk
        if self.state.get_symbol_risk(symbol) == RiskLevel.CRITICAL:
            return True, f"Symbol risk CRITICAL"

        # Sector-level CRITICAL risk (if enabled)
        if self.enable_sector_blocking:
            if self.state.get_sector_risk(sector) == RiskLevel.CRITICAL:
                return True, f"Sector risk CRITICAL"

        # Global CRITICAL risk (if enabled)
        if self.enable_global_blocking:
            if self.state.global_risk == RiskLevel.CRITICAL:
                return True, f"Global risk CRITICAL"

        return False, None

    def get_risk_summary(self, symbol: str, sector: str) -> Dict:
        """
        Get complete risk summary for a symbol.

        Returns:
            Dictionary with all risk levels
        """
        max_risk, source = self.state.get_max_risk_for_symbol(symbol, sector)

        return {
            'symbol_risk': self.state.get_symbol_risk(symbol).name,
            'sector_risk': self.state.get_sector_risk(sector).name,
            'global_risk': self.state.global_risk.name,
            'max_risk': max_risk.name,
            'max_risk_source': source
        }

    def apply_directional_bias(self, symbol: str, sector: str, prob_buy: float, prob_sell: float) -> Tuple[float, float]:
        """
        Apply three-tier directional biasing to BUY/SELL probabilities.

        UNIFIED BIASING SYSTEM:
        - Global sentiment (weak ±0.05): Macro economic/market bias
        - Sector sentiment (medium ±0.05): Sector-specific bias
        - Symbol sentiment (strong ±0.10): Company-specific bias
        - All three biases STACK together (cumulative adjustment)
        - Priority: Symbol > Sector > Global (strongest to weakest)

        Args:
            symbol: Stock symbol
            sector: Sector name
            prob_buy: Original BUY probability from model
            prob_sell: Original SELL probability from model

        Returns:
            (adjusted_prob_buy, adjusted_prob_sell) after applying ALL sentiment biases
        """
        # Initialize cumulative bias (positive = bullish, negative = bearish)
        cumulative_bias = 0.0
        bias_components = []

        # TIER 1: Global sentiment (weak ±0.05)
        global_sentiment = self.state.get_global_sentiment()
        if global_sentiment == 'bearish':
            cumulative_bias -= self.GLOBAL_BIAS_STRENGTH
            bias_components.append(f"global_bearish(-{self.GLOBAL_BIAS_STRENGTH})")
        elif global_sentiment == 'bullish':
            cumulative_bias += self.GLOBAL_BIAS_STRENGTH
            bias_components.append(f"global_bullish(+{self.GLOBAL_BIAS_STRENGTH})")

        # TIER 2: Sector sentiment (medium ±0.05)
        sector_sentiment = self.state.get_sector_sentiment(sector)
        if sector_sentiment == 'bearish':
            cumulative_bias -= self.SECTOR_BIAS_STRENGTH
            bias_components.append(f"sector_bearish(-{self.SECTOR_BIAS_STRENGTH})")
        elif sector_sentiment == 'bullish':
            cumulative_bias += self.SECTOR_BIAS_STRENGTH
            bias_components.append(f"sector_bullish(+{self.SECTOR_BIAS_STRENGTH})")

        # TIER 3: Symbol sentiment (strong ±0.10)
        symbol_sentiment = self.state.get_symbol_sentiment(symbol)
        if symbol_sentiment == 'bearish':
            cumulative_bias -= self.SYMBOL_BIAS_STRENGTH
            bias_components.append(f"symbol_bearish(-{self.SYMBOL_BIAS_STRENGTH})")
        elif symbol_sentiment == 'bullish':
            cumulative_bias += self.SYMBOL_BIAS_STRENGTH
            bias_components.append(f"symbol_bullish(+{self.SYMBOL_BIAS_STRENGTH})")

        # Apply cumulative bias to probabilities
        if cumulative_bias != 0.0:
            adjusted_buy = max(0.0, min(1.0, prob_buy + cumulative_bias))
            adjusted_sell = max(0.0, min(1.0, prob_sell - cumulative_bias))

            bias_str = " + ".join(bias_components)
            logger.info(
                f"[UNIFIED BIAS] {symbol} ({sector}): {bias_str} = {cumulative_bias:+.3f} "
                f"(BUY {prob_buy:.3f}->{adjusted_buy:.3f}, SELL {prob_sell:.3f}->{adjusted_sell:.3f})"
            )
            return adjusted_buy, adjusted_sell

        # Default: neutral, no bias
        return prob_buy, prob_sell

    def reset(self):
        """Reset all risk levels to NONE."""
        logger.info("[NEWS ENGINE] Resetting all risk levels to NONE")
        self.state = NewsState()

    def update(self):
        """
        Fetch news headlines and update risk state.

        Fetches RSS headlines, classifies them, applies keyword rules,
        and updates symbol/sector/global risk levels.
        """
        logger.info("[NEWS UPDATE] Starting news risk update...")

        # Fetch headlines
        try:
            headlines = self.feed_client.fetch_all_headlines()
            self.last_headlines = headlines
        except Exception as e:
            logger.error(f"[NEWS UPDATE] Failed to fetch headlines: {e}")
            return

        if not headlines:
            logger.warning("[NEWS UPDATE] No headlines fetched")
            return

        # Classify and score headlines
        global_risk_scores = []
        sector_risks = {}
        symbol_risks = {}

        for headline in headlines:
            # Get risk score and classification
            risk_level, classification = self._score_headline(headline)

            # Ignore headlines with no keyword matches (NONE risk level)
            if risk_level == RiskLevel.NONE:
                continue

            if classification == 'global':
                global_risk_scores.append(risk_level)
            elif classification.startswith('sector:'):
                sector = classification.split(':')[1]
                if sector not in sector_risks:
                    sector_risks[sector] = []
                sector_risks[sector].append(risk_level)
            elif classification.startswith('symbol:'):
                symbol = classification.split(':')[1]
                if symbol not in symbol_risks:
                    symbol_risks[symbol] = []
                symbol_risks[symbol].append(risk_level)

        # Update global risk (take maximum)
        if global_risk_scores:
            max_global_risk = max(global_risk_scores, key=lambda x: x.value)
            self.state.set_global_risk(max_global_risk)

        # Update sector risks (take maximum per sector)
        for sector, scores in sector_risks.items():
            max_sector_risk = max(scores, key=lambda x: x.value)
            self.state.set_sector_risk(sector, max_sector_risk)

        # Update symbol risks (take maximum per symbol)
        for symbol, scores in symbol_risks.items():
            max_symbol_risk = max(scores, key=lambda x: x.value)
            self.state.set_symbol_risk(symbol, max_symbol_risk)

        # DIRECTIONAL BIASING: Detect sentiment for symbols
        symbol_sentiments = {}
        for headline in headlines:
            sentiment, symbol = self._detect_sentiment(headline)
            if sentiment and symbol:
                if symbol not in symbol_sentiments:
                    symbol_sentiments[symbol] = []
                symbol_sentiments[symbol].append(sentiment)

        # Apply sentiment (majority vote per symbol)
        for symbol, sentiments in symbol_sentiments.items():
            bearish_count = sentiments.count('bearish')
            bullish_count = sentiments.count('bullish')
            if bearish_count > bullish_count:
                self.state.set_symbol_sentiment(symbol, 'bearish')
            elif bullish_count > bearish_count:
                self.state.set_symbol_sentiment(symbol, 'bullish')
            else:
                self.state.set_symbol_sentiment(symbol, 'neutral')

        # UNIFIED BIASING: Derive global sentiment from headline trends
        global_sentiments = []
        for headline in headlines:
            global_sent = self._detect_global_sentiment(headline)
            if global_sent:
                global_sentiments.append(global_sent)

        # Apply global sentiment (majority vote)
        if global_sentiments:
            bearish_count = global_sentiments.count('bearish')
            bullish_count = global_sentiments.count('bullish')
            if bearish_count > bullish_count:
                self.state.set_global_sentiment('bearish')
            elif bullish_count > bearish_count:
                self.state.set_global_sentiment('bullish')
            else:
                self.state.set_global_sentiment('neutral')
        else:
            self.state.set_global_sentiment('neutral')

        # UNIFIED BIASING: Derive sector sentiment from sector-specific headlines
        sector_sentiments = {}
        for headline in headlines:
            sector_sent, sector = self._detect_sector_sentiment(headline)
            if sector_sent and sector:
                if sector not in sector_sentiments:
                    sector_sentiments[sector] = []
                sector_sentiments[sector].append(sector_sent)

        # Apply sector sentiment (majority vote per sector)
        for sector, sentiments in sector_sentiments.items():
            bearish_count = sentiments.count('bearish')
            bullish_count = sentiments.count('bullish')
            if bearish_count > bullish_count:
                self.state.set_sector_sentiment(sector, 'bearish')
            elif bullish_count > bearish_count:
                self.state.set_sector_sentiment(sector, 'bullish')
            else:
                self.state.set_sector_sentiment(sector, 'neutral')

        logger.info(f"[NEWS UPDATE] Complete - Processed {len(headlines)} headlines")
        logger.info(f"  Global risk: {self.state.global_risk.name}")
        logger.info(f"  Global sentiment: {self.state.global_sentiment}")
        logger.info(f"  Sector risks: {len(sector_risks)} sectors updated")
        logger.info(f"  Sector sentiments: {len(sector_sentiments)} sectors with directional bias")
        logger.info(f"  Symbol risks: {len(symbol_risks)} symbols updated")
        logger.info(f"  Symbol sentiments: {len(symbol_sentiments)} symbols with directional bias")

    def _score_headline(self, headline) -> Tuple[RiskLevel, str]:
        """
        Score a single headline using deterministic keyword matching.

        Uses exact substring matching (lowercase) per the Phase 2 keyword schema.
        No NLP, no sentiment analysis, no fuzzy matching.

        Args:
            headline: Headline object

        Returns:
            (RiskLevel, classification_string)
            classification_string can be:
                - 'global' (market-wide)
                - 'sector:<sector_name>'
                - 'symbol:<symbol>'
        """
        text = (headline.title + ' ' + headline.summary).lower()

        # Deterministic Keyword Schema - Symbol Level
        SYMBOL_CRITICAL = [
            'sec investigation', 'fraud', 'accounting irregularities', 'restatement',
            'criminal probe', 'fbi', 'whistleblower', 'data breach', 'security breach',
            'ceo resigns', 'cfo resigns', 'product recall', 'explosion', 'factory fire',
            'bankruptcy', 'chapter 11', 'chapter 7'
        ]

        SYMBOL_HIGH = [
            'earnings miss', 'revenue miss', 'profit warning', 'guidance cut',
            'downgrade', 'lawsuit', 'antitrust', 'regulatory action', 'sec filing',
            'layoffs', 'job cuts', 'merger terminated', 'acquisition fails'
        ]

        SYMBOL_MEDIUM = [
            'earnings beat', 'revenue beat', 'guidance raised', 'upgrade',
            'partnership', 'contract win', 'analyst note', 'price target',
            'new product', 'launch'
        ]

        SYMBOL_LOW = [
            'press release', 'minor update', 'market chatter', 'rumor',
            'industry commentary'
        ]

        # Deterministic Keyword Schema - Sector Level
        SECTOR_CRITICAL = [
            'industry collapse', 'sector-wide investigation', 'mass recall',
            'regulatory crackdown', 'supply chain shutdown', 'systemic failure'
        ]

        SECTOR_HIGH = [
            'demand collapse', 'supply shortage', 'regulation change', 'tariffs',
            'export ban', 'import ban', 'opec decision', 'fda rejection'
        ]

        SECTOR_MEDIUM = [
            'sector upgrade', 'sector downgrade', 'industry growth',
            'industry slowdown', 'pricing pressure'
        ]

        SECTOR_LOW = [
            'industry commentary', 'analyst sector note'
        ]

        # Deterministic Keyword Schema - Global Level
        GLOBAL_CRITICAL = [
            'military escalation', 'missile launch', 'missile strike', 'airstrike',
            'troop mobilization', 'border conflict', 'border clash', 'geopolitical conflict',
            'naval confrontation', 'drone strike', 'invasion', 'hostilities',
            'ceasefire breakdown', 'escalation in region', 'military tensions',
            'terror attack', 'pandemic', 'global shutdown',
            'financial crisis', 'bank collapse', 'systemic risk', 'market crash',
            'emergency rate hike'
        ]

        GLOBAL_HIGH = [
            'fed raises rates', 'fed cuts rates', 'inflation report', 'cpi', 'ppi',
            'jobs report', 'geopolitical tensions', 'trade war', 'oil shock',
            'oil sanctions', 'energy sanctions', 'OPEC sanctions', 'shipping sanctions',
            'maritime sanctions', 'export controls', 'semiconductor sanctions',
            'SWIFT sanctions', 'sovereign asset freeze', 'currency sanctions', 'trade retaliation'
        ]

        GLOBAL_MEDIUM = [
            'economic slowdown', 'recession fears', 'market volatility', 'macro uncertainty'
        ]

        GLOBAL_LOW = [
            'general market commentary', 'economic outlook'
        ]

        # Classification: Check in order of severity (CRITICAL > HIGH > MEDIUM > LOW)
        # Symbol-level keywords (check first - they override sector/global for specific symbols)
        for keyword in SYMBOL_CRITICAL:
            if keyword in text:
                logger.info(f"[NEWS CRITICAL/SYMBOL] '{headline.title}' (keyword: {keyword})")
                return (RiskLevel.CRITICAL, self._classify_headline(headline, text))

        for keyword in SYMBOL_HIGH:
            if keyword in text:
                logger.info(f"[NEWS HIGH/SYMBOL] '{headline.title}' (keyword: {keyword})")
                return (RiskLevel.HIGH, self._classify_headline(headline, text))

        # Sector-level keywords
        for keyword in SECTOR_CRITICAL:
            if keyword in text:
                logger.info(f"[NEWS CRITICAL/SECTOR] '{headline.title}' (keyword: {keyword})")
                classification = self._classify_headline(headline, text)
                # Force sector classification if not already
                if not classification.startswith('sector:'):
                    classification = 'global'  # Will be sector-classified by _classify_headline
                return (RiskLevel.CRITICAL, classification)

        for keyword in SECTOR_HIGH:
            if keyword in text:
                logger.info(f"[NEWS HIGH/SECTOR] '{headline.title}' (keyword: {keyword})")
                classification = self._classify_headline(headline, text)
                return (RiskLevel.HIGH, classification)

        # Global-level keywords
        for keyword in GLOBAL_CRITICAL:
            if keyword in text:
                logger.info(f"[NEWS CRITICAL/GLOBAL] '{headline.title}' (keyword: {keyword})")
                return (RiskLevel.CRITICAL, 'global')

        for keyword in GLOBAL_HIGH:
            if keyword in text:
                logger.info(f"[NEWS HIGH/GLOBAL] '{headline.title}' (keyword: {keyword})")
                return (RiskLevel.HIGH, 'global')

        # Medium-level keywords (symbol, sector, global)
        for keyword in SYMBOL_MEDIUM:
            if keyword in text:
                logger.debug(f"[NEWS MEDIUM/SYMBOL] '{headline.title}' (keyword: {keyword})")
                return (RiskLevel.MEDIUM, self._classify_headline(headline, text))

        for keyword in SECTOR_MEDIUM:
            if keyword in text:
                logger.debug(f"[NEWS MEDIUM/SECTOR] '{headline.title}' (keyword: {keyword})")
                classification = self._classify_headline(headline, text)
                return (RiskLevel.MEDIUM, classification)

        for keyword in GLOBAL_MEDIUM:
            if keyword in text:
                logger.debug(f"[NEWS MEDIUM/GLOBAL] '{headline.title}' (keyword: {keyword})")
                return (RiskLevel.MEDIUM, 'global')

        # Low-level keywords (symbol, sector, global)
        for keyword in SYMBOL_LOW:
            if keyword in text:
                logger.debug(f"[NEWS LOW/SYMBOL] '{headline.title}' (keyword: {keyword})")
                return (RiskLevel.LOW, self._classify_headline(headline, text))

        for keyword in SECTOR_LOW:
            if keyword in text:
                logger.debug(f"[NEWS LOW/SECTOR] '{headline.title}' (keyword: {keyword})")
                classification = self._classify_headline(headline, text)
                return (RiskLevel.LOW, classification)

        for keyword in GLOBAL_LOW:
            if keyword in text:
                logger.debug(f"[NEWS LOW/GLOBAL] '{headline.title}' (keyword: {keyword})")
                return (RiskLevel.LOW, 'global')

        # No keyword match - ignore headline (return NONE)
        return (RiskLevel.NONE, 'global')

    def _detect_sentiment(self, headline) -> Tuple[Optional[str], Optional[str]]:
        """
        Detect directional sentiment (bullish/bearish) for symbol-specific news.

        DIRECTIONAL TRADING DESIGN:
        - Negative news keywords → 'bearish' sentiment
        - Positive news keywords → 'bullish' sentiment
        - Uses simple keyword matching

        Args:
            headline: Headline object

        Returns:
            (sentiment, symbol) where sentiment is 'bullish'|'bearish'|None, symbol is ticker or None
        """
        text = (headline.title + ' ' + headline.summary).lower()

        # Bearish keywords (negative news)
        BEARISH_KEYWORDS = [
            'downgrade', 'layoffs', 'job cuts', 'guidance cut', 'guidance cuts',
            'earnings miss', 'revenue miss', 'profit warning', 'regulatory probe',
            'lawsuit', 'antitrust', 'fraud', 'sec investigation', 'product recall',
            'ceo resigns', 'cfo resigns', 'merger terminated', 'acquisition fails'
        ]

        # Bullish keywords (positive news)
        BULLISH_KEYWORDS = [
            'upgrade', 'raised guidance', 'guidance raise', 'contract win', 'major contract',
            'earnings beat', 'revenue beat', 'strong earnings', 'exceeded expectations',
            'positive outlook', 'product launch', 'breakthrough', 'partnership announced',
            'acquisition announced', 'merger approved'
        ]

        # Check for bearish sentiment
        for keyword in BEARISH_KEYWORDS:
            if keyword in text:
                # Try to extract symbol from headline
                symbol = self._extract_symbol_from_headline(headline, text)
                if symbol:
                    logger.info(f"[NEWS SENTIMENT] Bearish: {symbol} (keyword: {keyword})")
                    return ('bearish', symbol)

        # Check for bullish sentiment
        for keyword in BULLISH_KEYWORDS:
            if keyword in text:
                # Try to extract symbol from headline
                symbol = self._extract_symbol_from_headline(headline, text)
                if symbol:
                    logger.info(f"[NEWS SENTIMENT] Bullish: {symbol} (keyword: {keyword})")
                    return ('bullish', symbol)

        return (None, None)

    def _detect_global_sentiment(self, headline) -> Optional[str]:
        """
        Detect directional sentiment (bullish/bearish) for global/macro news.

        UNIFIED BIASING SYSTEM:
        - Global bearish keywords → 'bearish' sentiment (weak ±0.05 bias against longs)
        - Global bullish keywords → 'bullish' sentiment (weak ±0.05 bias for longs)
        - Uses simple keyword matching on macro economic/market news

        Args:
            headline: Headline object

        Returns:
            'bullish'|'bearish'|None
        """
        text = (headline.title + ' ' + headline.summary).lower()

        # Global bearish keywords (macro risk, market decline)
        GLOBAL_BEARISH_KEYWORDS = [
            'recession fears', 'economic slowdown', 'gdp decline', 'unemployment rising',
            'inflation surge', 'rate hike', 'fed hawkish', 'central bank tightening',
            'market selloff', 'broad decline', 'market crash', 'bear market',
            'credit crunch', 'liquidity crisis', 'debt ceiling', 'government shutdown',
            'geopolitical tension', 'trade war', 'supply chain disruption',
            'vix spike', 'volatility surge', 'risk-off'
        ]

        # Global bullish keywords (macro strength, market rally)
        GLOBAL_BULLISH_KEYWORDS = [
            'economic growth', 'gdp expansion', 'strong jobs report', 'unemployment falling',
            'inflation cooling', 'rate cut', 'fed dovish', 'central bank easing',
            'market rally', 'broad gains', 'bull market', 'all-time high',
            'credit expansion', 'liquidity boost', 'debt ceiling resolved',
            'geopolitical stability', 'trade deal', 'supply chain improving',
            'vix decline', 'volatility falling', 'risk-on'
        ]

        # Check for global bearish sentiment
        for keyword in GLOBAL_BEARISH_KEYWORDS:
            if keyword in text:
                logger.info(f"[GLOBAL SENTIMENT] Bearish (keyword: {keyword})")
                return 'bearish'

        # Check for global bullish sentiment
        for keyword in GLOBAL_BULLISH_KEYWORDS:
            if keyword in text:
                logger.info(f"[GLOBAL SENTIMENT] Bullish (keyword: {keyword})")
                return 'bullish'

        return None

    def _detect_sector_sentiment(self, headline) -> Tuple[Optional[str], Optional[str]]:
        """
        Detect directional sentiment (bullish/bearish) for sector-specific news.

        UNIFIED BIASING SYSTEM:
        - Sector bearish keywords → 'bearish' sentiment (medium ±0.05 bias)
        - Sector bullish keywords → 'bullish' sentiment (medium ±0.05 bias)
        - Uses simple keyword matching with sector identification

        Args:
            headline: Headline object

        Returns:
            (sentiment, sector) where sentiment is 'bullish'|'bearish'|None, sector is sector name or None
        """
        text = (headline.title + ' ' + headline.summary).lower()

        # Sector-specific bearish keywords (sector under pressure)
        SECTOR_BEARISH_KEYWORDS = {
            'technology': ['tech sector decline', 'chip shortage', 'semiconductor slowdown', 'software downturn'],
            'financials': ['banking crisis', 'credit tightening', 'loan defaults rising', 'financial sector under pressure'],
            'healthcare': ['drug pricing pressure', 'healthcare regulation', 'pharma investigation', 'biotech slowdown'],
            'energy': ['oil price collapse', 'energy demand falling', 'crude oversupply', 'gas glut'],
            'consumer_discretionary': ['consumer spending weak', 'retail sales decline', 'discretionary pullback'],
            'consumer_staples': ['staples under pressure', 'food price inflation'],
            'industrials': ['manufacturing decline', 'industrial slowdown', 'factory orders falling'],
            'materials': ['commodity price crash', 'materials sector weak', 'mining slowdown'],
            'real_estate': ['property market decline', 'real estate crisis', 'housing market crash'],
            'utilities': ['utility sector weak', 'power demand falling'],
            'communication_services': ['telecom under pressure', 'media sector decline']
        }

        # Sector-specific bullish keywords (sector strength)
        SECTOR_BULLISH_KEYWORDS = {
            'technology': ['tech sector rally', 'chip demand surge', 'semiconductor boom', 'software strength'],
            'financials': ['banking sector strong', 'credit growth', 'loan demand rising', 'financial sector rally'],
            'healthcare': ['drug approval surge', 'healthcare innovation', 'pharma breakthrough', 'biotech rally'],
            'energy': ['oil price rally', 'energy demand surge', 'crude shortage', 'gas boom'],
            'consumer_discretionary': ['consumer spending strong', 'retail sales beat', 'discretionary strength'],
            'consumer_staples': ['staples sector strong', 'food demand surge'],
            'industrials': ['manufacturing boom', 'industrial strength', 'factory orders surge'],
            'materials': ['commodity price rally', 'materials sector strong', 'mining boom'],
            'real_estate': ['property market rally', 'real estate boom', 'housing market strength'],
            'utilities': ['utility sector strong', 'power demand surge'],
            'communication_services': ['telecom strength', 'media sector rally']
        }

        # Check for sector bearish sentiment
        for sector, keywords in SECTOR_BEARISH_KEYWORDS.items():
            for keyword in keywords:
                if keyword in text:
                    logger.info(f"[SECTOR SENTIMENT] Bearish: {sector} (keyword: {keyword})")
                    return ('bearish', sector)

        # Check for sector bullish sentiment
        for sector, keywords in SECTOR_BULLISH_KEYWORDS.items():
            for keyword in keywords:
                if keyword in text:
                    logger.info(f"[SECTOR SENTIMENT] Bullish: {sector} (keyword: {keyword})")
                    return ('bullish', sector)

        return (None, None)

    def _extract_symbol_from_headline(self, headline, text: str) -> Optional[str]:
        """
        Extract stock symbol from headline text.

        Simple extraction: looks for ticker patterns in parentheses like (AAPL) or $AAPL.
        """
        import re

        # Pattern 1: Ticker in parentheses like (AAPL)
        match = re.search(r'\(([A-Z]{1,5})\)', headline.title + ' ' + headline.summary)
        if match:
            return match.group(1)

        # Pattern 2: Ticker with dollar sign like $AAPL
        match = re.search(r'\$([A-Z]{1,5})\b', headline.title + ' ' + headline.summary)
        if match:
            return match.group(1)

        return None

    def _classify_headline(self, headline, text: str) -> str:
        """
        Classify a headline as global, sector-specific, or symbol-specific.

        Args:
            headline: Headline object
            text: Lowercase text (title + summary)

        Returns:
            Classification string ('global', 'sector:<name>', 'symbol:<ticker>')
        """
        # Check for specific symbols (ticker patterns)
        # Simple heuristic: Look for uppercase 2-5 letter words that might be tickers
        import re
        ticker_pattern = r'\b[A-Z]{2,5}\b'
        tickers = re.findall(ticker_pattern, headline.title + ' ' + headline.summary)

        if tickers:
            # Return first ticker found
            return f'symbol:{tickers[0]}'

        # Check for sector keywords
        sector_keywords = {
            'technology': ['tech', 'software', 'semiconductor', 'chip', 'ai', 'cloud'],
            'financials': ['bank', 'financial', 'insurance', 'credit'],
            'healthcare': ['health', 'pharma', 'drug', 'medical', 'biotech'],
            'energy': ['oil', 'gas', 'energy', 'crude', 'petroleum'],
            'consumer_discretionary': ['retail', 'consumer', 'auto', 'automotive'],
            'consumer_staples': ['food', 'beverage', 'grocery', 'staples'],
            'industrials': ['industrial', 'manufacturing', 'aerospace', 'defense'],
            'materials': ['materials', 'metals', 'mining', 'steel'],
            'utilities': ['utility', 'utilities', 'electric', 'power', 'water'],
            'real_estate': ['real estate', 'reit', 'property', 'housing'],
            'communication_services': ['telecom', 'media', 'communications']
        }

        for sector, keywords in sector_keywords.items():
            for keyword in keywords:
                if keyword in text:
                    return f'sector:{sector}'

        # Default to global
        return 'global'


# Singleton instance for global access
_NEWS_ENGINE_INSTANCE = None


def get_news_engine() -> NewsEngine:
    """Get or create the global NewsEngine instance."""
    global _NEWS_ENGINE_INSTANCE
    if _NEWS_ENGINE_INSTANCE is None:
        _NEWS_ENGINE_INSTANCE = NewsEngine()
    return _NEWS_ENGINE_INSTANCE
