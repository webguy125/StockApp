@echo off
REM This batch file is designed for Windows Task Scheduler
REM It runs without pausing and logs output to a file

cd /d "%~dp0"
cd agents

REM Create logs directory if it doesn't exist
if not exist "logs" mkdir logs

REM Set log file with timestamp
set LOG_FILE=logs\scanner_%date:~-4,4%%date:~-10,2%%date:~-7,2%_%time:~0,2%%time:~3,2%%time:~6,2%.log
set LOG_FILE=%LOG_FILE: =0%

REM Activate virtual environment and run scanner
call ..\venv\Scripts\activate.bat

echo ======================================== >> %LOG_FILE% 2>&1
echo Scheduled Scanner Run >> %LOG_FILE% 2>&1
echo Started: %date% %time% >> %LOG_FILE% 2>&1
echo ======================================== >> %LOG_FILE% 2>&1

python comprehensive_scanner.py >> %LOG_FILE% 2>&1

echo. >> %LOG_FILE% 2>&1
echo ======================================== >> %LOG_FILE% 2>&1
echo Completed: %date% %time% >> %LOG_FILE% 2>&1
echo ======================================== >> %LOG_FILE% 2>&1

REM Optional: Run fusion agent after scanner
echo. >> %LOG_FILE% 2>&1
echo Running Fusion Agent... >> %LOG_FILE% 2>&1
python fusion_agent.py >> %LOG_FILE% 2>&1

exit /b 0
