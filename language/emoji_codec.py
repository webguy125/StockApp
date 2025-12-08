"""
Emoji Codec - Encoder/Decoder for Multi-Agent Communication
Translates between emoji shorthand and structured JSON
"""

import json
from datetime import datetime
from typing import List, Dict, Any


class EmojiCodec:
    """
    Handles encoding and decoding of emoji shorthand for agent communication.

    Grammar: SYMBOL | EMOJIS | SCORE | CONFIDENCE | TIMESTAMP
    Example: AAPL | 🙂📈💵📰 | 82 | 0.78 | 2025-10-27
    """

    def __init__(self, blueprint_path: str = "C:/StockApp/language/emoji_blueprint.json"):
        """Load emoji lexicon from blueprint"""
        with open(blueprint_path, 'r', encoding='utf-8') as f:
            blueprint = json.load(f)

        self.lexicon = blueprint['system_blueprint']['emoji_language']['emoji_lexicon']

        # Build reverse lookup (emoji -> meaning)
        self.emoji_to_meaning = {}
        for category, mappings in self.lexicon.items():
            for emoji, meaning in mappings.items():
                self.emoji_to_meaning[emoji] = {
                    'category': category,
                    'meaning': meaning
                }

    def encode_to_shorthand(self,
                           symbol: str,
                           emojis: List[str],
                           score: int,
                           confidence: float,
                           timestamp: str = None) -> str:
        """
        Encode structured data into emoji shorthand.

        Args:
            symbol: Stock symbol (e.g., 'AAPL')
            emojis: List of emoji symbols
            score: Numeric score (0-100)
            confidence: Confidence level (0.0-1.0)
            timestamp: ISO timestamp (defaults to now)

        Returns:
            Shorthand string: "AAPL | 🙂📈💵 | 82 | 0.78 | 2025-10-27"
        """
        if timestamp is None:
            timestamp = datetime.now().strftime('%Y-%m-%d')

        emoji_str = ''.join(emojis)
        return f"{symbol} | {emoji_str} | {score} | {confidence:.2f} | {timestamp}"

    def decode_from_shorthand(self, line: str) -> Dict[str, Any]:
        """
        Decode emoji shorthand into structured JSON.

        Args:
            line: Shorthand string (e.g., "AAPL | 🙂📈💵 | 82 | 0.78 | 2025-10-27")

        Returns:
            Dict with keys: symbol, emojis, emoji_meanings, score, confidence, timestamp
        """
        parts = [p.strip() for p in line.split('|')]

        if len(parts) != 5:
            raise ValueError(f"Invalid shorthand format: {line}")

        symbol, emoji_str, score_str, confidence_str, timestamp = parts

        # Parse emojis and look up meanings
        emojis = list(emoji_str)
        emoji_meanings = []
        for emoji in emojis:
            if emoji in self.emoji_to_meaning:
                emoji_meanings.append({
                    'emoji': emoji,
                    'category': self.emoji_to_meaning[emoji]['category'],
                    'meaning': self.emoji_to_meaning[emoji]['meaning']
                })
            else:
                emoji_meanings.append({
                    'emoji': emoji,
                    'category': 'unknown',
                    'meaning': 'unknown'
                })

        return {
            'symbol': symbol,
            'emojis': emojis,
            'emoji_meanings': emoji_meanings,
            'score': int(score_str),
            'confidence': float(confidence_str),
            'timestamp': timestamp
        }

    def validate_emojis(self, emojis: List[str]) -> bool:
        """Check if all emojis are in the lexicon"""
        return all(emoji in self.emoji_to_meaning for emoji in emojis)

    def get_emojis_by_category(self, category: str) -> Dict[str, str]:
        """Get all emojis for a specific category"""
        return self.lexicon.get(category, {})


# Example usage
if __name__ == "__main__":
    import sys
    import io

    # Fix Windows console encoding for emojis
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')

    try:
        codec = EmojiCodec()
        print("✅ Emoji codec initialized successfully!")

        # Test encoding
        shorthand = codec.encode_to_shorthand(
            symbol="BTC-USD",
            emojis=["🙂", "📈", "💵", "📰"],
            score=82,
            confidence=0.78
        )
        print(f"Encoded: {shorthand}")

        # Test decoding
        decoded = codec.decode_from_shorthand(shorthand)
        print(f"Decoded: {json.dumps(decoded, indent=2, ensure_ascii=False)}")

        # Test validation
        valid = codec.validate_emojis(["📈", "💵"])
        print(f"Emojis valid: {valid}")

        # Test categories
        outcomes = codec.get_emojis_by_category("outcomes")
        print(f"Outcome emojis: {outcomes}")

        print("\n✅ All emoji codec tests passed!")

    except Exception as e:
        print(f"❌ Error: {e}")
        import traceback
        traceback.print_exc()
