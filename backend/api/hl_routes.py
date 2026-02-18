"""
HL API Routes - Read-Only Endpoints

Exposes HL (Higher-Level) analytics via REST API.

IMPORTANT CONSTRAINTS:
- All endpoints are READ-ONLY
- No endpoints trigger trading actions
- No endpoints modify database state
- HL is advisory-only and informational
"""

from flask import Blueprint, jsonify
from backend.turbomode.hl import build_hl_output
from backend.turbomode.analyzer import analyze_symbol

hl_bp = Blueprint('hl', __name__, url_prefix='/api/hl')


@hl_bp.route('/<symbol>', methods=['GET'])
def get_hl_output(symbol: str):
    """
    Get HL (Higher-Level) analytics output for a symbol.

    Args:
        symbol: Stock symbol (e.g., 'AAPL')

    Returns:
        JSON object with complete HL output including:
        - hl_bias: 'bullish', 'bearish', or 'neutral'
        - hl_confidence: 0.0 - 1.0
        - trend: Trend analytics block
        - volume: Volume analytics block
        - volatility: Volatility analytics block
        - probability_drift: Probability drift block
        - structure: Market structure block
        - hl_summary: Narrative summary

    Constraints:
        - READ-ONLY endpoint
        - Does NOT trigger any trading actions
        - Does NOT modify any database state
    """

    try:
        symbol = symbol.upper()

        # Fetch analytics from existing analyzer (or use stubs if not available)
        # For now, we'll use a placeholder signal_type since HL is advisory-only
        # In production, this could be enhanced to accept signal_type as a query param
        signal_type = 'BUY'  # Placeholder - HL doesn't depend on this

        # Get analytics from THE ANALYZER
        try:
            analytics = analyze_symbol(symbol, signal_type)
        except Exception as e:
            # If analyzer fails, use empty context
            print(f"[HL API] Analyzer failed for {symbol}: {e}")
            analytics = {}

        # Build context dictionary for HL service
        context = {
            'trend': {
                'trend': analytics.get('trend'),
                'trend_strength': analytics.get('trend_strength', 50),
                'multi_tf_alignment': analytics.get('with_trend') == 1,
                'ema_alignment': _infer_ema_alignment(analytics.get('trend')),
                'swing_structure': _infer_swing_structure(analytics.get('trend')),
                'trend_commentary': analytics.get('trend_commentary', 'No trend data')
            },
            'volume': {
                'ord_initial_volume': analytics.get('ord_initial_volume'),
                'ord_correction_volume': analytics.get('ord_correction_volume'),
                'ord_retest_volume': analytics.get('ord_retest_volume'),
                'ord_retest_strength_pct': analytics.get('ord_retest_strength_pct'),
                'ord_classification': analytics.get('ord_classification'),
                'ord_commentary': analytics.get('ord_commentary', 'No volume data')
            },
            'volatility': {
                'atr_current': None,  # Placeholder - would need to fetch from market data
                'atr_percentile': 0.5,  # Placeholder
                'volatility_state': 'stable',  # Placeholder
                'commentary': 'Volatility data pending integration'
            },
            'probability': {
                'initial_probability': 0.5,  # Placeholder - would need signal probabilities
                'live_probability': 0.5,  # Placeholder
                'commentary': 'Probability drift pending integration'
            },
            'structure': {
                'current_price': None,  # Placeholder
                'key_support': None,  # Placeholder
                'key_resistance': None,  # Placeholder
                'structure_state': 'holding',  # Placeholder
                'commentary': 'Structure data pending integration'
            }
        }

        # Build HL output
        hl_output = build_hl_output(symbol, context)

        return jsonify(hl_output.to_dict())

    except Exception as e:
        import traceback
        traceback.print_exc()
        return jsonify({'error': str(e)}), 500


def _infer_ema_alignment(trend: str) -> str:
    """Infer EMA alignment from trend direction"""
    if trend == 'uptrend':
        return 'bullish'
    elif trend == 'downtrend':
        return 'bearish'
    else:
        return 'mixed'


def _infer_swing_structure(trend: str) -> str:
    """Infer swing structure from trend direction"""
    if trend == 'uptrend':
        return 'HH/HL'  # Higher Highs / Higher Lows
    elif trend == 'downtrend':
        return 'LL/LH'  # Lower Lows / Lower Highs
    else:
        return 'choppy'
