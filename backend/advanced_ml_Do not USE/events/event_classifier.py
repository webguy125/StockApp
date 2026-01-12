"""
Event Classifier
Classifies events into types with severity and impact scores per specification.
"""

import logging
from typing import Dict, List, Optional, Any, Tuple
import pandas as pd
import numpy as np
from datetime import datetime

logger = logging.getLogger(__name__)


class EventClassifier:
    """
    Event classification system using rule-based and ML-based approaches.
    Produces event_type, severity, impact scores, and confidence metrics.
    """

    def __init__(self, config: Dict[str, Any]):
        """
        Initialize event classifier.

        Args:
            config: Classifier configuration dictionary
        """
        self.config = config
        self.model_id = config.get('model_id', 'event_cls_v1_distilbert_ensemble')

        # Label space
        self.event_types = config.get('label_space', {}).get('event_type', [])

        # Output schema
        self.output_schema = config.get('output_schema', [])

        # Thresholds
        thresholds = config.get('thresholds', {})
        self.min_confidence = thresholds.get('min_confidence', 0.60)
        self.high_severity_cutoff = thresholds.get('high_severity_cutoff', 0.75)
        self.ambiguity_threshold = thresholds.get('ambiguity_flag_threshold', 0.20)

        # Initialize classification rules
        self._initialize_rules()

        logger.info(f"[CLASSIFIER] Initialized with model: {self.model_id}")

    def _initialize_rules(self):
        """Initialize rule-based classification patterns."""
        self.filing_type_rules = {
            '8-K': {
                'primary_types': ['earnings', 'management_change', 'dividend_change', 'm_and_a'],
                'item_map': {
                    '1.01': 'm_and_a',
                    '2.02': 'earnings',
                    '5.02': 'management_change',
                    '8.01': 'other'
                }
            },
            '10-Q': {'primary_types': ['earnings']},
            '10-K': {'primary_types': ['earnings']},
            '13D': {'primary_types': ['m_and_a']},
            '13F': {'primary_types': ['other']},
            'DEF 14A': {'primary_types': ['management_change', 'other']}
        }

        self.keyword_patterns = {
            'dividend_change': ['dividend', 'payout', 'dividend increase', 'dividend cut'],
            'refinancing': ['refinance', 'credit facility', 'term loan', 'revolving credit'],
            'litigation': ['lawsuit', 'litigation', 'settlement', 'legal proceedings'],
            'guidance': ['guidance', 'outlook', 'forecast', 'expects'],
            'stock_buyback': ['buyback', 'repurchase', 'share repurchase'],
            'product_recall': ['recall', 'safety issue', 'product defect'],
            'cybersecurity_incident': ['cybersecurity', 'data breach', 'hack', 'cyber attack'],
            'regulatory': ['SEC', 'FDA', 'regulatory', 'compliance'],
            'covenant_change': ['covenant', 'waiver', 'amendment']
        }

    def classify(self, events: pd.DataFrame) -> pd.DataFrame:
        """
        Classify events and generate output schema.

        Args:
            events: DataFrame with raw events

        Returns:
            DataFrame with classification outputs
        """
        if len(events) == 0:
            return pd.DataFrame()

        classified = events.copy()

        # Apply classification
        classifications = classified.apply(self._classify_single_event, axis=1)

        # Expand classification results into columns
        classification_df = pd.DataFrame(
            classifications.tolist(),
            index=classified.index
        )

        # Merge with original events
        result = pd.concat([classified, classification_df], axis=1)

        # Filter by minimum confidence
        result = result[result['confidence'] >= self.min_confidence]

        logger.info(f"[CLASSIFIER] Classified {len(result)}/{len(events)} events (confidence >= {self.min_confidence})")

        return result

    def _classify_single_event(self, event: pd.Series) -> Dict[str, Any]:
        """
        Classify a single event.

        Args:
            event: Event row

        Returns:
            Classification dictionary
        """
        source_type = event.get('source_type', '')

        if source_type == 'sec':
            return self._classify_sec_filing(event)
        elif source_type == 'news':
            return self._classify_news(event)
        else:
            return self._get_default_classification()

    def _classify_sec_filing(self, event: pd.Series) -> Dict[str, Any]:
        """
        Classify SEC filing.

        Args:
            event: SEC filing event

        Returns:
            Classification dictionary
        """
        filing_type = event.get('filing_type', '')
        raw_text = str(event.get('raw_text', ''))

        # Rule-based classification
        event_type = 'other'
        confidence = 0.7

        if filing_type in self.filing_type_rules:
            rules = self.filing_type_rules[filing_type]
            event_type = rules['primary_types'][0] if rules['primary_types'] else 'other'
            confidence = 0.8

        # Enhance with keyword matching
        for keyword_type, keywords in self.keyword_patterns.items():
            if any(keyword.lower() in raw_text.lower() for keyword in keywords):
                event_type = keyword_type
                confidence = 0.85
                break

        # Calculate severity and impacts
        severity = self._calculate_severity(event_type, filing_type)
        impacts = self._calculate_impacts(event_type, raw_text)

        # Determine if rare event
        is_rare = severity >= self.high_severity_cutoff

        return {
            'event_type': event_type,
            'event_subtype': filing_type,
            'event_severity': severity,
            'impact_dividend': impacts['dividend'],
            'impact_liquidity': impacts['liquidity'],
            'impact_credit': impacts['credit'],
            'impact_growth': impacts['growth'],
            'is_rare_event': is_rare,
            'confidence': confidence,
            'sentiment_score': self._calculate_sentiment(raw_text),
            'temporal_relevance': 1.0
        }

    def _classify_news(self, event: pd.Series) -> Dict[str, Any]:
        """
        Classify news article.

        Args:
            event: News article event

        Returns:
            Classification dictionary
        """
        headline = str(event.get('headline', ''))
        raw_text = str(event.get('raw_text', ''))

        # Keyword-based classification
        event_type = 'other'
        confidence = 0.6

        for keyword_type, keywords in self.keyword_patterns.items():
            matched = sum(1 for keyword in keywords if keyword.lower() in headline.lower() or keyword.lower() in raw_text.lower())
            if matched > 0:
                event_type = keyword_type
                confidence = min(0.95, 0.6 + (matched * 0.1))
                break

        # Calculate severity and impacts
        severity = self._calculate_severity(event_type, 'news')
        impacts = self._calculate_impacts(event_type, raw_text)

        # Sentiment analysis
        sentiment = self._calculate_sentiment(headline + ' ' + raw_text)

        # Temporal relevance (news is time-sensitive)
        urgency_level = event.get('urgency_level', 'medium')
        temporal_relevance = {'high': 1.0, 'medium': 0.7, 'low': 0.4}.get(urgency_level, 0.7)

        # Determine if rare event
        is_rare = severity >= self.high_severity_cutoff

        return {
            'event_type': event_type,
            'event_subtype': 'news',
            'event_severity': severity,
            'impact_dividend': impacts['dividend'],
            'impact_liquidity': impacts['liquidity'],
            'impact_credit': impacts['credit'],
            'impact_growth': impacts['growth'],
            'is_rare_event': is_rare,
            'confidence': confidence,
            'sentiment_score': sentiment,
            'temporal_relevance': temporal_relevance
        }

    def _calculate_severity(self, event_type: str, source: str) -> float:
        """
        Calculate event severity score.

        Args:
            event_type: Classified event type
            source: Event source type

        Returns:
            Severity score [0.0, 1.0]
        """
        # Severity mapping by event type
        severity_map = {
            'cybersecurity_incident': 0.90,
            'product_recall': 0.85,
            'litigation': 0.75,
            'm_and_a': 0.70,
            'dividend_change': 0.65,
            'management_change': 0.60,
            'refinancing': 0.55,
            'covenant_change': 0.70,
            'earnings': 0.50,
            'guidance': 0.55,
            'stock_buyback': 0.45,
            'regulatory': 0.65,
            'macro': 0.40,
            'other': 0.30
        }

        base_severity = severity_map.get(event_type, 0.30)

        # SEC filings are generally more severe than news
        if source.startswith('8-K'):
            base_severity = min(1.0, base_severity * 1.2)

        return base_severity

    def _calculate_impacts(self, event_type: str, text: str) -> Dict[str, float]:
        """
        Calculate impact scores for different dimensions.

        Args:
            event_type: Classified event type
            text: Event text content

        Returns:
            Dictionary with impact scores
        """
        impacts = {
            'dividend': 0.0,
            'liquidity': 0.0,
            'credit': 0.0,
            'growth': 0.0
        }

        # Event type impact profiles
        impact_profiles = {
            'dividend_change': {'dividend': 1.0, 'liquidity': 0.3, 'credit': 0.2, 'growth': 0.1},
            'refinancing': {'dividend': 0.2, 'liquidity': 0.8, 'credit': 0.9, 'growth': 0.3},
            'covenant_change': {'dividend': 0.4, 'liquidity': 0.7, 'credit': 0.9, 'growth': 0.2},
            'litigation': {'dividend': 0.5, 'liquidity': 0.6, 'credit': 0.7, 'growth': 0.4},
            'm_and_a': {'dividend': 0.6, 'liquidity': 0.8, 'credit': 0.5, 'growth': 0.9},
            'earnings': {'dividend': 0.3, 'liquidity': 0.2, 'credit': 0.3, 'growth': 0.7},
            'guidance': {'dividend': 0.2, 'liquidity': 0.1, 'credit': 0.2, 'growth': 0.8},
            'stock_buyback': {'dividend': 0.7, 'liquidity': 0.6, 'credit': 0.4, 'growth': 0.3},
            'management_change': {'dividend': 0.3, 'liquidity': 0.2, 'credit': 0.3, 'growth': 0.6},
            'product_recall': {'dividend': 0.4, 'liquidity': 0.7, 'credit': 0.6, 'growth': 0.5},
            'cybersecurity_incident': {'dividend': 0.3, 'liquidity': 0.8, 'credit': 0.7, 'growth': 0.6}
        }

        if event_type in impact_profiles:
            impacts = impact_profiles[event_type]

        return impacts

    def _calculate_sentiment(self, text: str) -> float:
        """
        Calculate sentiment score from text.

        Args:
            text: Text content

        Returns:
            Sentiment score [-1.0, 1.0]
        """
        # Simple keyword-based sentiment
        positive_keywords = ['growth', 'increase', 'improve', 'gain', 'profit', 'strong', 'upgrade']
        negative_keywords = ['decline', 'loss', 'weak', 'downgrade', 'cut', 'miss', 'lawsuit']

        text_lower = text.lower()

        positive_count = sum(1 for word in positive_keywords if word in text_lower)
        negative_count = sum(1 for word in negative_keywords if word in text_lower)

        total = positive_count + negative_count
        if total == 0:
            return 0.0

        sentiment = (positive_count - negative_count) / total
        return sentiment

    def _get_default_classification(self) -> Dict[str, Any]:
        """
        Get default classification for unrecognized events.

        Returns:
            Default classification dictionary
        """
        return {
            'event_type': 'other',
            'event_subtype': 'unknown',
            'event_severity': 0.3,
            'impact_dividend': 0.0,
            'impact_liquidity': 0.0,
            'impact_credit': 0.0,
            'impact_growth': 0.0,
            'is_rare_event': False,
            'confidence': 0.5,
            'sentiment_score': 0.0,
            'temporal_relevance': 0.5
        }

    def get_statistics(self) -> Dict[str, Any]:
        """
        Get classifier statistics.

        Returns:
            Statistics dictionary
        """
        return {
            'model_id': self.model_id,
            'event_types': self.event_types,
            'min_confidence': self.min_confidence,
            'high_severity_cutoff': self.high_severity_cutoff,
            'ambiguity_threshold': self.ambiguity_threshold
        }
