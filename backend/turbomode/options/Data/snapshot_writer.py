import os
import sqlite3
from datetime import datetime

BASE_DIR = os.path.dirname(__file__)
DB_PATH = os.path.join(BASE_DIR, 'options_intel.db')


def write_snapshot(symbol, enriched_data):
    """
    Write enriched signal intelligence to the database.

    Args:
        symbol: Stock symbol
        enriched_data: Dictionary containing all enriched signal fields

    Returns:
        True if successful, False otherwise
    """
    try:
        conn = sqlite3.connect(DB_PATH)
        cursor = conn.cursor()

        # Extract all fields from enriched_data
        cursor.execute('''
            INSERT OR REPLACE INTO signal_intelligence (
                symbol, signal_type, current_price, stop_upper, stop_lower,
                confidence, prob_buy, prob_sell, prob_hold, created_at, sector,
                short_call_strike, short_put_strike, long_call_strike, long_put_strike,
                expiration, max_credit, max_profit, max_loss, pnl,
                drift, pressure_index, volatility_regime, quality_score,
                regime_state, transition_state, transition_strength,
                signal_strength, confirmation_status, risk_level, regime_alignment, timing_phase,
                signal_quality_score, age_days, model_probability,
                implied_volatility, delta, gamma, theta, vega,
                last_updated, data_source
            ) VALUES (
                ?, ?, ?, ?, ?,
                ?, ?, ?, ?, ?, ?,
                ?, ?, ?, ?,
                ?, ?, ?, ?, ?,
                ?, ?, ?, ?,
                ?, ?, ?,
                ?, ?, ?, ?, ?,
                ?, ?, ?,
                ?, ?, ?, ?, ?,
                ?, ?
            )
        ''', (
            enriched_data.get('symbol'),
            enriched_data.get('base_signal'),
            enriched_data.get('current_price'),
            enriched_data.get('stop_upper'),
            enriched_data.get('stop_lower'),
            enriched_data.get('model_confidence'),
            enriched_data.get('prob_buy'),
            enriched_data.get('prob_sell'),
            enriched_data.get('prob_hold'),
            enriched_data.get('created_at'),
            enriched_data.get('sector'),
            enriched_data.get('short_call_strike'),
            enriched_data.get('short_put_strike'),
            enriched_data.get('long_call_strike'),
            enriched_data.get('long_put_strike'),
            enriched_data.get('expiration'),
            enriched_data.get('max_credit'),
            enriched_data.get('max_profit'),
            enriched_data.get('max_loss'),
            enriched_data.get('pnl'),
            enriched_data.get('drift'),
            enriched_data.get('pressure_index'),
            enriched_data.get('volatility_regime'),
            enriched_data.get('quality_score'),
            enriched_data.get('regime_state'),
            enriched_data.get('transition_state'),
            enriched_data.get('transition_strength'),
            enriched_data.get('signal_strength'),
            enriched_data.get('confirmation_status'),
            enriched_data.get('risk_level'),
            enriched_data.get('regime_alignment'),
            enriched_data.get('timing_phase'),
            enriched_data.get('signal_quality_score'),
            enriched_data.get('age_days'),
            enriched_data.get('model_probability'),
            enriched_data.get('implied_volatility'),
            enriched_data.get('delta'),
            enriched_data.get('gamma'),
            enriched_data.get('theta'),
            enriched_data.get('vega'),
            datetime.now().isoformat(),
            enriched_data.get('data_source', 'tradier')
        ))

        conn.commit()
        conn.close()
        return True

    except Exception as e:
        print(f"[SNAPSHOT_WRITER] Error writing snapshot for {symbol}: {e}")
        return False


def write_snapshots_batch(enriched_signals_list):
    """
    Write multiple enriched signals to database in a single transaction.

    Args:
        enriched_signals_list: List of enriched signal dictionaries

    Returns:
        Number of successfully written snapshots
    """
    try:
        conn = sqlite3.connect(DB_PATH)
        cursor = conn.cursor()

        count = 0
        for enriched_data in enriched_signals_list:
            try:
                cursor.execute('''
                    INSERT OR REPLACE INTO signal_intelligence (
                        symbol, signal_type, current_price, stop_upper, stop_lower,
                        confidence, prob_buy, prob_sell, prob_hold, created_at, sector,
                        short_call_strike, short_put_strike, long_call_strike, long_put_strike,
                        expiration, max_credit, max_profit, max_loss, pnl,
                        drift, pressure_index, volatility_regime, quality_score,
                        regime_state, transition_state, transition_strength,
                        signal_strength, confirmation_status, risk_level, regime_alignment, timing_phase,
                        signal_quality_score, age_days, model_probability,
                        implied_volatility, delta, gamma, theta, vega,
                        last_updated, data_source
                    ) VALUES (
                        ?, ?, ?, ?, ?,
                        ?, ?, ?, ?, ?, ?,
                        ?, ?, ?, ?,
                        ?, ?, ?, ?, ?,
                        ?, ?, ?, ?,
                        ?, ?, ?,
                        ?, ?, ?, ?, ?,
                        ?, ?, ?,
                        ?, ?, ?, ?, ?,
                        ?, ?
                    )
                ''', (
                    enriched_data.get('symbol'),
                    enriched_data.get('base_signal'),
                    enriched_data.get('current_price'),
                    enriched_data.get('stop_upper'),
                    enriched_data.get('stop_lower'),
                    enriched_data.get('model_confidence'),
                    enriched_data.get('prob_buy'),
                    enriched_data.get('prob_sell'),
                    enriched_data.get('prob_hold'),
                    enriched_data.get('created_at'),
                    enriched_data.get('sector'),
                    enriched_data.get('short_call_strike'),
                    enriched_data.get('short_put_strike'),
                    enriched_data.get('long_call_strike'),
                    enriched_data.get('long_put_strike'),
                    enriched_data.get('expiration'),
                    enriched_data.get('max_credit'),
                    enriched_data.get('max_profit'),
                    enriched_data.get('max_loss'),
                    enriched_data.get('pnl'),
                    enriched_data.get('drift'),
                    enriched_data.get('pressure_index'),
                    enriched_data.get('volatility_regime'),
                    enriched_data.get('quality_score'),
                    enriched_data.get('regime_state'),
                    enriched_data.get('transition_state'),
                    enriched_data.get('transition_strength'),
                    enriched_data.get('signal_strength'),
                    enriched_data.get('confirmation_status'),
                    enriched_data.get('risk_level'),
                    enriched_data.get('regime_alignment'),
                    enriched_data.get('timing_phase'),
                    enriched_data.get('signal_quality_score'),
                    enriched_data.get('age_days'),
                    enriched_data.get('model_probability'),
                    enriched_data.get('implied_volatility'),
                    enriched_data.get('delta'),
                    enriched_data.get('gamma'),
                    enriched_data.get('theta'),
                    enriched_data.get('vega'),
                    datetime.now().isoformat(),
                    enriched_data.get('data_source', 'tradier')
                ))
                count += 1
            except Exception as e:
                print(f"[SNAPSHOT_WRITER] Error writing {enriched_data.get('symbol')}: {e}")
                continue

        conn.commit()
        conn.close()
        return count

    except Exception as e:
        print(f"[SNAPSHOT_WRITER] Batch write error: {e}")
        return 0
