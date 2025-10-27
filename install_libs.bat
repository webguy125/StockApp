@echo off
cd StockApp

REM Create virtual environment
python -m venv venv

REM Activate venv and install libraries
call venv\Scripts\activate
pip install -r requirements.txt

echo Libraries installed successfully.