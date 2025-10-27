"""
Configuration file for API keys and settings
"""

# Alpha Vantage API Configuration
# Get your API key from: https://www.alphavantage.co/support/#api-key
ALPHA_VANTAGE_API_KEY = "YOUR_API_KEY_HERE"  # Replace with your actual API key

# Data source selection
USE_ALPHA_VANTAGE = False  # Set to True once you have your API key
USE_YFINANCE = True        # Set to False when switching to Alpha Vantage

# Cache settings
CACHE_DURATION_MINUTES = 5

# Rate limiting
ALPHA_VANTAGE_REQUESTS_PER_MINUTE = 75  # Free tier: 5, Premium: 75, Ultra: 600
