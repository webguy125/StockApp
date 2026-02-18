"""
Clean skip reasons for Iron Condor pricing failures
No exceptions thrown - always return None gracefully
"""

class SkipReason:
    """Skip reasons for logging only - never raise exceptions"""
    MARKET_CLOSED = "market_closed"
    NO_OPTION_CHAIN = "no_option_chain"
    NO_VALID_EXPIRATION = "no_valid_expiration"
    NO_OTM_WINGS = "no_otm_wings"
    INVALID_MID_PRICE = "invalid_mid_price"
    IBKR_CONNECTION_FAILED = "ibkr_connection_failed"
    NOT_HOLD_SIGNAL = "not_hold_signal"
    MISSING_STOP_BOUNDS = "missing_stop_bounds"
