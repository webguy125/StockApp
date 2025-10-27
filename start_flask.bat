@echo off
cd /d C:\StockApp
call venv\Scripts\activate
cd backend
python api_server.py
pause