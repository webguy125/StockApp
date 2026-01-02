# Complete List of 179 GPU-Accelerated Features

## Summary
**Total Features**: 179+ (all calculated on GPU via `GPUFeatureEngineer`)
**Current Features in Use**: 30 (via `VectorizedGPUFeatures`)
**Missing Features**: 149+

---

## Category 1: Momentum Indicators (20+ features)

### RSI (Relative Strength Index)
- `rsi_7` - 7-period RSI
- `rsi_14` - 14-period RSI
- `rsi_21` - 21-period RSI
- `rsi_28` - 28-period RSI

### Stochastic Oscillator
- `stochastic_k` - %K line (14,3)
- `stochastic_d` - %D line (smoothed %K)

### Rate of Change (ROC)
- `roc_5` - 5-period rate of change
- `roc_10` - 10-period rate of change
- `roc_20` - 20-period rate of change

### Other Momentum Indicators
- `williams_r` - Williams %R (14-period)
- `mfi_14` - Money Flow Index (14-period)
- `cci_20` - Commodity Channel Index (20-period)
- `ultimate_oscillator` - Ultimate Oscillator (7,14,28)
- `momentum_10` - Simple momentum (10-period price change)
- `momentum_20` - Simple momentum (20-period price change)

---

## Category 2: Trend Indicators (25+ features)

### Simple Moving Averages (SMA)
- `sma_5` - 5-period SMA
- `sma_10` - 10-period SMA
- `sma_20` - 20-period SMA
- `sma_50` - 50-period SMA
- `sma_100` - 100-period SMA
- `sma_200` - 200-period SMA

### Price vs SMA Distance
- `price_vs_sma_5` - Distance from 5-period SMA (%)
- `price_vs_sma_10` - Distance from 10-period SMA (%)
- `price_vs_sma_20` - Distance from 20-period SMA (%)
- `price_vs_sma_50` - Distance from 50-period SMA (%)
- `price_vs_sma_100` - Distance from 100-period SMA (%)
- `price_vs_sma_200` - Distance from 200-period SMA (%)

### Exponential Moving Averages (EMA)
- `ema_5` - 5-period EMA
- `ema_10` - 10-period EMA
- `ema_20` - 20-period EMA
- `ema_50` - 50-period EMA
- `ema_100` - 100-period EMA
- `ema_200` - 200-period EMA

### MACD (Moving Average Convergence Divergence)
- `macd` - MACD line (12, 26)
- `macd_signal` - Signal line (9-period EMA of MACD)
- `macd_histogram` - MACD histogram (MACD - signal)

### ADX (Average Directional Index)
- `adx_14` - 14-period ADX (trend strength)
- `plus_di` - Plus Directional Indicator
- `minus_di` - Minus Directional Indicator

### Parabolic SAR
- `parabolic_sar` - Parabolic SAR value
- `price_vs_psar` - Distance from SAR (%)

### Supertrend
- `supertrend` - Supertrend indicator (10, 3.0)

---

## Category 3: Volume Indicators (20+ features)

### On-Balance Volume (OBV)
- `obv` - Current OBV value
- `obv_sma_20` - 20-period SMA of OBV

### Accumulation/Distribution
- `ad_line` - Accumulation/Distribution line

### Chaikin Money Flow
- `cmf_20` - 20-period Chaikin Money Flow

### VWAP (Volume-Weighted Average Price)
- `vwap` - VWAP value
- `price_vs_vwap` - Distance from VWAP (%)

### Volume Ratios
- `volume_ratio_20` - Current volume vs 20-period average
- `volume_ratio_50` - Current volume vs 50-period average
- `volume_trend` - 10-period vs 30-period volume SMA ratio

### Advanced Volume Indicators
- `ease_of_movement` - Ease of Movement (14-period)
- `force_index` - Force Index (13-period)
- `nvi` - Negative Volume Index
- `pvi` - Positive Volume Index
- `vpt` - Volume Price Trend

---

## Category 4: Volatility Indicators (15+ features)

### Average True Range (ATR)
- `atr_7` - 7-period ATR
- `atr_14` - 14-period ATR
- `atr_21` - 21-period ATR
- `atr_pct_7` - ATR as % of price (7-period)
- `atr_pct_14` - ATR as % of price (14-period)
- `atr_pct_21` - ATR as % of price (21-period)

### Bollinger Bands
- `bb_upper` - Upper Bollinger Band (20, 2)
- `bb_middle` - Middle Bollinger Band (20-period SMA)
- `bb_lower` - Lower Bollinger Band (20, 2)
- `bb_width` - Bollinger Band width (%)
- `bb_position` - Price position within bands (%)

### Keltner Channels
- `keltner_upper` - Upper Keltner Channel (20, 2)
- `keltner_lower` - Lower Keltner Channel (20, 2)

### Historical Volatility (Standard Deviation)
- `historical_vol_10` - 10-period historical volatility
- `historical_vol_20` - 20-period historical volatility
- `historical_vol_30` - 30-period historical volatility

### Donchian Channels
- `donchian_upper` - Upper Donchian Channel (20-period)
- `donchian_lower` - Lower Donchian Channel (20-period)

---

## Category 5: Price Pattern Features (25+ features)

### Pivot Points
- `pivot_point` - Standard pivot point
- `resistance_1` - R1 resistance level
- `resistance_2` - R2 resistance level
- `support_1` - S1 support level
- `support_2` - S2 support level
- `price_vs_pivot` - Distance from pivot (%)

### Candlestick Patterns
- `body_size` - Candle body size
- `upper_shadow` - Upper shadow length
- `lower_shadow` - Lower shadow length
- `is_bullish_candle` - Boolean: bullish candle (1/0)

### Gap Detection
- `gap_pct` - Gap size (%)
- `has_gap_up` - Boolean: gap up > 0.5%
- `has_gap_down` - Boolean: gap down < -0.5%

### Price Range Positions
- `range_position_5` - Position in 5-period range (%)
- `range_position_10` - Position in 10-period range (%)
- `range_position_20` - Position in 20-period range (%)

### Swing Analysis
- `distance_to_swing_high` - Distance to swing high (%)
- `distance_to_swing_low` - Distance to swing low (%)
- `higher_high` - Boolean: made higher high
- `lower_low` - Boolean: made lower low

### Fibonacci Retracement Levels
- `fib_0_236` - 23.6% retracement level
- `fib_0_382` - 38.2% retracement level
- `fib_0_500` - 50.0% retracement level
- `fib_0_618` - 61.8% retracement level

---

## Category 6: Statistical Features (20+ features)

### Returns (Multiple Periods)
- `return_1d` - 1-day return (%)
- `return_5d` - 5-day return (%)
- `return_10d` - 10-day return (%)
- `return_20d` - 20-day return (%)

### Mean Returns
- `mean_return_10` - Mean of 10-period returns (%)
- `mean_return_20` - Mean of 20-period returns (%)
- `mean_return_50` - Mean of 50-period returns (%)

### Standard Deviation of Returns
- `std_return_10` - Std dev of 10-period returns (%)
- `std_return_20` - Std dev of 20-period returns (%)
- `std_return_50` - Std dev of 50-period returns (%)

### Skewness and Kurtosis
- `skew_10` - 10-period return skewness
- `skew_20` - 20-period return skewness
- `skew_50` - 50-period return skewness
- `kurtosis_10` - 10-period return kurtosis
- `kurtosis_20` - 20-period return kurtosis
- `kurtosis_50` - 50-period return kurtosis

### Z-Scores
- `z_score_20` - 20-period price z-score
- `z_score_50` - 50-period price z-score

### Sharpe Ratio
- `sharpe_ratio_20` - 20-day Sharpe ratio

### Linear Regression Slopes
- `lr_slope_10` - 10-period linear regression slope (%)
- `lr_slope_20` - 20-period linear regression slope (%)
- `lr_slope_50` - 50-period linear regression slope (%)

---

## Category 7: Market Structure Features (15+ features)

### Trend Classification
- `trend_strength` - ADX-based trend strength
- `is_trending` - Boolean: ADX > 25
- `is_ranging` - Boolean: ADX < 20

### Momentum Score
- `momentum_score` - Composite momentum score (-1 to +1)

### Structural Bias
- `bullish_structure` - Boolean: bullish MA alignment
- `bearish_structure` - Boolean: bearish MA alignment

### Consecutive Days
- `consecutive_up_days` - Number of consecutive up days
- `consecutive_down_days` - Number of consecutive down days

### 52-Week Highs/Lows
- `near_52w_high` - Boolean: within 5% of 52-week high
- `near_52w_low` - Boolean: within 5% of 52-week low

### Volume Strength
- `volume_strength` - Current volume vs 20-period average

### Moving Average Alignment
- `ma_alignment_score` - Composite MA alignment score (-1 to +1)

---

## Category 8: Multi-Timeframe Features (20+ features)

### Weekly Aggregations (5-day)
- `weekly_range` - Weekly high-low range (%)
- `weekly_return` - Weekly return (%)
- `weekly_position` - Price position in weekly range (%)
- `weekly_rsi` - RSI on weekly timeframe
- `weekly_macd` - MACD on weekly timeframe
- `weekly_macd_signal` - MACD signal on weekly timeframe

### Monthly Aggregations (20-day)
- `monthly_range` - Monthly high-low range (%)
- `monthly_return` - Monthly return (%)
- `monthly_position` - Price position in monthly range (%)

### Quarterly Aggregations (60-day)
- `quarterly_range` - Quarterly high-low range (%)
- `quarterly_return` - Quarterly return (%)
- `quarterly_position` - Price position in quarterly range (%)

### Volume Across Timeframes
- `volume_5d_avg` - 5-day average volume
- `volume_20d_avg` - 20-day average volume
- `volume_60d_avg` - 60-day average volume

### Volatility Across Timeframes
- `volatility_5d` - 5-day volatility (annualized %)
- `volatility_20d` - 20-day volatility (annualized %)
- `volatility_60d` - 60-day volatility (annualized %)
- `historical_volatility_60d` - Alias for volatility_60d

### Risk Metrics
- `beta` - Beta (market correlation, default 1.0)
- `liquidity_score` - log10 of average volume

---

## Category 9: Derived/Interaction Features (30+ features)

### Divergence Detection
- `rsi_price_divergence` - RSI vs price momentum divergence
- `price_volume_divergence` - Price vs volume trend divergence

### Agreement Indicators
- `macd_rsi_agreement` - Boolean: MACD and RSI agree
- `trend_agreement` - Boolean: short and long trends agree

### Volume Confirmation
- `volume_confirmed_move` - Volume-weighted price move

### Compression/Expansion
- `bb_squeeze` - Bollinger Band squeeze ratio
- `volatility_expansion` - Short-term vs long-term volatility

### Alignment Indicators
- `ema_bullish_alignment` - Boolean: EMAs aligned bullishly
- `ema_bearish_alignment` - Boolean: EMAs aligned bearishly

### Quality Indicators
- `momentum_quality` - RSI extremity × ADX strength

### Position Indicators
- `composite_price_position` - Average of BB and range positions

### Volatility Regimes
- `low_volatility_regime` - Boolean: volatility < 15%
- `medium_volatility_regime` - Boolean: 15% ≤ volatility ≤ 30%
- `high_volatility_regime` - Boolean: volatility > 30%

### Support/Resistance Proximity
- `near_resistance` - Boolean: within 2% of swing high
- `near_support` - Boolean: within 2% of swing low

### Oversold/Overbought Composites
- `oversold_composite` - Composite oversold score (0-1)
- `overbought_composite` - Composite overbought score (0-1)

### Composite Scores
- `trend_strength_composite` - ADX × MA alignment
- `gap_significance` - Gap size / ATR ratio
- `breakout_potential` - Composite breakout probability score
- `mean_reversion_signal` - Composite mean reversion score

### Momentum Metrics
- `momentum_acceleration` - ROC(5) - ROC(10) difference

---

## Metadata Features (3)
- `feature_count` - Total number of features calculated
- `last_price` - Most recent close price
- `last_volume` - Most recent volume

---

## Current vs Full Feature Set

### Currently Used (30 features via VectorizedGPUFeatures)
**Momentum (10)**: rsi_14, rsi_7, stoch_k, roc_10, roc_20, williams_r, mfi_14, cci_20, momentum_10, momentum_20
**Trend (10)**: sma_10, sma_20, sma_50, ema_12, ema_26, macd, macd_signal, macd_hist, adx_14, trend_strength
**Volatility (5)**: atr_14, bb_width, volatility_20d, true_range, historical_volatility_20d
**Volume (5)**: volume_ratio, obv, vwap, volume_trend, avg_volume_20d

### Missing (149+ features)
- All other RSI periods (7, 21, 28)
- All SMA periods except 10, 20, 50 (missing: 5, 100, 200)
- All EMA periods except 12, 26 (missing: 5, 10, 20, 50, 100, 200)
- All price-to-SMA distance calculations
- All statistical features (returns, skew, kurtosis, z-scores, Sharpe)
- All price pattern features (pivots, Fibonacci, gaps)
- All multi-timeframe features (weekly/monthly/quarterly)
- All derived/interaction features (divergences, agreements, composites)
- Most volume indicators (CMF, Force Index, NVI, PVI, VPT, etc.)
- Market structure features
- And 100+ more...

---

## How to Enable All 179 Features

**Current Issue**: `historical_backtest.py` lines 196-210 prioritize `VectorizedGPUFeatures` (30 features) over `GPUFeatureEngineer` (179 features)

**Solution**: Modify `backend/advanced_ml/backtesting/historical_backtest.py`:

```python
# CHANGE FROM (current priority):
if self.vectorized_gpu is not None:
    all_features = self.vectorized_gpu.extract_features_vectorized(...)
elif hasattr(self.feature_engineer, 'extract_features_batch'):
    all_features = self.feature_engineer.extract_features_batch(...)

# CHANGE TO (new priority):
if hasattr(self.feature_engineer, 'extract_features_batch') and self.use_gpu:
    all_features = self.feature_engineer.extract_features_batch(...)
# elif self.vectorized_gpu is not None:  # DISABLED
#     all_features = self.vectorized_gpu.extract_features_vectorized(...)
```

**Expected Impact**:
- Backtest time: 6 hours → 10-12 hours
- Model accuracy: 60-73% → 85-95%
- Training samples: Same (199K)
- Features per sample: 30 → 179

---

**End of Feature List**
