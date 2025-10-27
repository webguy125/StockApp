"""
Prediction Service Module
ML-based price prediction using linear regression
"""

import numpy as np
import pandas as pd
from datetime import timedelta
from sklearn.linear_model import LinearRegression


def predict_price(df, forecast_days=30, interval='1d'):
    """ML-based price prediction using linear regression and trend analysis"""
    # Prepare features for ML
    df['Days'] = np.arange(len(df))
    df['MA_7'] = df['Close'].rolling(window=7).mean()
    df['MA_21'] = df['Close'].rolling(window=21).mean()
    df['Volatility'] = df['Close'].rolling(window=14).std()
    df = df.dropna()

    # Train simple linear regression model
    X = df[['Days', 'MA_7', 'MA_21', 'Volatility']].values
    y = df['Close'].values

    model = LinearRegression()
    model.fit(X, y)

    # Make predictions
    last_day = df['Days'].iloc[-1]
    last_ma7 = df['MA_7'].iloc[-1]
    last_ma21 = df['MA_21'].iloc[-1]
    last_vol = df['Volatility'].iloc[-1]

    predictions = []
    confidence_scores = []

    for i in range(1, forecast_days + 1):
        future_day = last_day + i

        # Extrapolate moving averages
        future_ma7 = last_ma7 + (last_ma7 - df['MA_7'].iloc[-7]) * (i / 7)
        future_ma21 = last_ma21 + (last_ma21 - df['MA_21'].iloc[-21]) * (i / 21)

        X_future = np.array([[future_day, future_ma7, future_ma21, last_vol]])
        pred_price = model.predict(X_future)[0]

        # Calculate confidence (decreases with distance)
        confidence = max(0.9 - (i / forecast_days) * 0.3, 0.6)

        predictions.append(float(pred_price))
        confidence_scores.append(float(confidence))

    # Calculate trend
    recent_prices = df['Close'].tail(10).values
    trend = "bullish" if recent_prices[-1] > recent_prices[0] else "bearish"
    trend_strength = abs(recent_prices[-1] - recent_prices[0]) / recent_prices[0] * 100

    # Generate forecast dates
    last_date = df['Date'].iloc[-1]
    forecast_dates = []
    for i in range(1, forecast_days + 1):
        if 'd' in interval or 'wk' in interval or 'mo' in interval:
            forecast_dates.append((last_date + timedelta(days=i)).strftime('%Y-%m-%d'))
        else:
            forecast_dates.append((last_date + timedelta(hours=i)).strftime('%Y-%m-%d %H:%M:%S'))

    return {
        'predictions': predictions,
        'dates': forecast_dates,
        'confidence': confidence_scores,
        'current_price': float(df['Close'].iloc[-1]),
        'trend': trend,
        'trend_strength': float(trend_strength),
        'model': 'Linear Regression',
        'r2_score': float(model.score(X, y))
    }
