@echo off
setlocal EnableDelayedExpansion
cd /d C:\StockApp

echo ============================================
echo  Claude Session Launcher with Full Continuity
echo ============================================
echo.

REM ========================================
REM SESSION FILE DIRECTORY
REM ========================================
set SESSION_DIR=C:\StockApp\session_files
if not exist "%SESSION_DIR%" mkdir "%SESSION_DIR%"

REM ========================================
REM GET TODAY'S DATE
REM ========================================
for /f "tokens=2 delims==" %%I in ('wmic os get localdatetime /value') do set datetime=%%I
set TODAY=%datetime:~0,4%-%datetime:~4,2%-%datetime:~6,2%
set SESSION_START=%TODAY% %datetime:~8,2%:%datetime:~10,2%

REM ========================================
REM CREATE TODAY'S SESSION FILE IF MISSING
REM ========================================
set TODAY_FILE=session_notes_%TODAY%.md
set TODAY_PATH=%SESSION_DIR%\%TODAY_FILE%

if not exist "%TODAY_PATH%" (
    echo Creating new session file for today...
    echo SESSION STARTED AT: %SESSION_START% >> "%TODAY_PATH%"
    echo. >> "%TODAY_PATH%"
)

REM ========================================
REM FIND MOST RECENT EXISTING FILE BEFORE TODAY
REM ========================================
set LAST_KNOWN_FILE=
for %%F in (%SESSION_DIR%\session_notes_*.md) do (
    set FILENAME=%%~nxF
    set FILEDATE=!FILENAME:~14,10!
    if "!FILEDATE!" lss "%TODAY%" (
        set LAST_KNOWN_FILE=%%F
    )
)

if not defined LAST_KNOWN_FILE (
    echo ERROR: No previous session file found before today.
    echo Skipping preload creation.
    pause
    exit /b 1
)

REM ========================================
REM BUILD PRELOAD FILE FOR CLAUDE
REM ========================================
if exist preload.txt del preload.txt

>> preload.txt echo Hi Claude - please read the previous working session file for context only:
>> preload.txt echo.
>> preload.txt echo   Previous file (READ ONLY): %LAST_KNOWN_FILE%
>> preload.txt echo.
>> preload.txt echo CRITICAL: All session updates MUST be APPENDED to today's file:
>> preload.txt echo   %TODAY_PATH%
>> preload.txt echo.
>> preload.txt echo DO NOT create new files like "_part2" or "_continuation".
>> preload.txt echo DO NOT overwrite. ALWAYS APPEND using the Edit tool.
>> preload.txt echo.
>> preload.txt echo ============================================
>> preload.txt echo CANONICAL SESSION FILE RULES (READ DAILY)
>> preload.txt echo ============================================
>> preload.txt echo.
>> preload.txt echo 1. You must use exactly ONE session file per day.
>> preload.txt echo    Filename format: session_notes_YYYY-MM-DD.md
>> preload.txt echo    Directory: C:\StockApp\session_files
>> preload.txt echo.
>> preload.txt echo 2. If today's file does not exist, create it immediately.
>> preload.txt echo    Include a header:
>> preload.txt echo    SESSION STARTED AT: YYYY-MM-DD HH:MM
>> preload.txt echo.
>> preload.txt echo 3. All updates must be appended to the current working file for the day.
>> preload.txt echo    NEVER EVER create files like session_notes_2026-01-01_part2.md
>> preload.txt echo    NEVER EVER create files like session_notes_2026-01-01_continuation.md
>> preload.txt echo    ONE FILE PER DAY. APPEND ONLY. Use the Edit tool to append.
>> preload.txt echo.
>> preload.txt echo 4. Every update must include a timestamp:
>> preload.txt echo    [YYYY-MM-DD HH:MM] ^<update text^>
>> preload.txt echo.
>> preload.txt echo 5. Never delete or rename old session files.
>> preload.txt echo    They form the permanent historical archive.
>> preload.txt echo.
>> preload.txt echo 6. Always save files to:
>> preload.txt echo    C:\StockApp\session_files
>> preload.txt echo.
>> preload.txt echo ============================================
>> preload.txt echo END OF RULES
>> preload.txt echo ============================================
>> preload.txt echo.
>> preload.txt echo SESSION START TIME: %SESSION_START%

echo Context loader created: preload.txt
echo.
echo Previous working file:
echo   - %LAST_KNOWN_FILE%
echo.

REM ========================================
REM COPY CONTEXT TO CLIPBOARD
REM ========================================
powershell -NoProfile -Command "Get-Content 'preload.txt' -Raw | Set-Clipboard"

if errorlevel 1 (
  echo ERROR: Failed to copy to clipboard
  echo.
  echo Please manually paste this into Claude:
  echo.
  type preload.txt
  echo.
  pause
  exit /b 1
)

echo Context copied to clipboard!
echo.
echo INSTRUCTIONS:
echo   1. The context prompt is now in your clipboard
echo   2. Press CTRL+V to paste it into Claude
echo   3. Press ENTER to send
echo.
echo Starting Claude...
echo.

REM ========================================
REM LAUNCH CLAUDE
REM ========================================
cmd /k "claude"