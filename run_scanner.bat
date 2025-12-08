@echo off
echo ========================================
echo  Comprehensive Scanner
echo ========================================
echo.

cd /d "%~dp0"
cd agents

echo Activating virtual environment...
call ..\venv\Scripts\activate.bat

echo.
echo Running comprehensive scanner...
python comprehensive_scanner.py

echo.
echo ========================================
echo Scanner complete!
echo Output: agents\repository\scanner_output.json
echo ========================================
echo.

pause
