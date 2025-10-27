@echo off
cd StockApp

REM Create requirements.txt
echo flask> requirements.txt
echo flask-cors>> requirements.txt
echo yfinance>> requirements.txt
echo pandas>> requirements.txt
echo plotly>> requirements.txt

REM Create README.md
echo # StockApp Project > README.md
echo This project pulls stock data from Yahoo Finance and saves it to CSV. >> README.md

REM Create backend/api_server.py
echo import yfinance as yf> backend\api_server.py
echo import pandas as pd>> backend\api_server.py
echo >> backend\api_server.py
echo def fetch_to_csv(symbols, filename="data/stocks.csv"):>> backend\api_server.py
echo.    data = yf.download(symbols, group_by="ticker")>> backend\api_server.py
echo.    data.to_csv(filename)>> backend\api_server.py
echo >> backend\api_server.py
echo if __name__ == "__main__":>> backend\api_server.py
echo.    fetch_to_csv(["AAPL","MSFT","GOOG"])>> backend\api_server.py
echo.    print("Data saved to data/stocks.csv")>> backend\api_server.py

echo Files created successfully.