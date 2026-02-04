"""
Sector Commentary Generation Module

Generates human-readable dynamic commentary for sector sentiment analysis.
All logic is deterministic and contamination-proof - HOLD signals never influence directional sentiment.
"""


def generate_sector_commentary(
    sector_name: str,
    buy_count: int,
    sell_count: int,
    hold_count: int,
    avg_buy_conf: float,
    avg_sell_conf: float,
    sector_sentiment: str,
    news_context: str = None
) -> str:
    """
    Generate dynamic commentary for a sector based on signal distribution and confidence levels.

    Args:
        sector_name: Name of the sector (e.g., "Technology", "Healthcare")
        buy_count: Number of BUY signals in this sector
        sell_count: Number of SELL signals in this sector
        hold_count: Number of HOLD signals in this sector
        avg_buy_conf: Average confidence of BUY signals (0.0 to 1.0)
        avg_sell_conf: Average confidence of SELL signals (0.0 to 1.0)
        sector_sentiment: Sector sentiment tag ("BULLISH", "BEARISH", "NEUTRAL")
        news_context: Optional news summary for this sector

    Returns:
        Human-readable commentary string explaining the sector's current state
    """
    total_directional = buy_count + sell_count
    delta = buy_count - sell_count

    commentary = ''

    # =========================
    # NEUTRAL COMMENTARY LOGIC
    # =========================
    if sector_sentiment == 'NEUTRAL':

        # Case 1: HOLD dominance
        if hold_count > total_directional:
            commentary += (
                f"The {sector_name} sector is neutral overall, with most stocks "
                f"({hold_count}) in consolidation. Directional signals exist "
                f"({buy_count} BUY vs {sell_count} SELL), but they are not strong "
                f"enough to drive a clear trend. This environment favors neutral "
                f"strategies such as iron condors or calendars. "
            )

        # Case 2: Weak directional imbalance (<2)
        else:
            commentary += (
                f"The {sector_name} sector shows mixed signals with {buy_count} BUY "
                f"and {sell_count} SELL alerts. The imbalance is not strong enough "
                f"to establish a directional trend, keeping the sector in a neutral "
                f"stance. Traders may prefer to wait for clearer momentum or use "
                f"range-bound strategies. "
            )

        # Append news if available (UI-only)
        if news_context:
            commentary += f" Recent headline: {news_context}"

        return commentary

    # =========================
    # BULLISH COMMENTARY LOGIC
    # =========================
    if sector_sentiment == 'BULLISH':
        commentary += (
            f"The {sector_name} sector leans bullish with {buy_count} BUY signals "
            f"outnumbering {sell_count} SELL signals. "
        )

        if avg_buy_conf > avg_sell_conf:
            commentary += (
                f"Average BUY confidence ({avg_buy_conf*100:.1f}%) exceeds SELL "
                f"confidence ({avg_sell_conf*100:.1f}%), supporting the upward bias. "
            )

        if hold_count > total_directional:
            commentary += (
                f"However, the {hold_count} HOLD signals indicate many stocks remain "
                f"range-bound, which may limit momentum. "
            )

        # Append news if available (UI-only)
        if news_context:
            commentary += f" Recent headline: {news_context}"

        return commentary

    # =========================
    # BEARISH COMMENTARY LOGIC
    # =========================
    if sector_sentiment == 'BEARISH':
        commentary += (
            f"The {sector_name} sector leans bearish with {sell_count} SELL signals "
            f"outnumbering {buy_count} BUY signals. "
        )

        if avg_sell_conf > avg_buy_conf:
            commentary += (
                f"Average SELL confidence ({avg_sell_conf*100:.1f}%) exceeds BUY "
                f"confidence ({avg_buy_conf*100:.1f}%), reinforcing the downside bias. "
            )

        if hold_count > total_directional:
            commentary += (
                f"Still, the {hold_count} HOLD signals show that many stocks are "
                f"consolidating, which may dampen momentum. "
            )

        # Append news if available (UI-only)
        if news_context:
            commentary += f" Recent headline: {news_context}"

        return commentary

    return commentary

# END generate_sector_commentary
