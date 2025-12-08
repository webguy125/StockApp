# PowerShell Script to Set Up Scanner Automation
# Run this as Administrator

Write-Host "========================================" -ForegroundColor Cyan
Write-Host " StockApp Scanner Automation Setup" -ForegroundColor Cyan
Write-Host "========================================" -ForegroundColor Cyan
Write-Host ""

# Check if running as Administrator
$isAdmin = ([Security.Principal.WindowsPrincipal] [Security.Principal.WindowsIdentity]::GetCurrent()).IsInRole([Security.Principal.WindowsBuiltInRole]::Administrator)

if (-not $isAdmin) {
    Write-Host "ERROR: This script must be run as Administrator!" -ForegroundColor Red
    Write-Host ""
    Write-Host "Right-click PowerShell and select 'Run as Administrator'" -ForegroundColor Yellow
    Write-Host ""
    pause
    exit 1
}

# Create logs directory
Write-Host "Creating logs directory..." -ForegroundColor Green
$logsPath = "C:\StockApp\agents\logs"
if (-not (Test-Path $logsPath)) {
    New-Item -ItemType Directory -Path $logsPath -Force | Out-Null
    Write-Host "  Created: $logsPath" -ForegroundColor Gray
} else {
    Write-Host "  Already exists: $logsPath" -ForegroundColor Gray
}

Write-Host ""

# Create the scheduled task
Write-Host "Creating scheduled task..." -ForegroundColor Green

$taskName = "StockApp Scanner"
$taskPath = "C:\StockApp\run_scanner_scheduled.bat"
$startTime = "00:00"  # Midnight

# Delete existing task if it exists
$existingTask = schtasks /query /tn "$taskName" 2>$null
if ($LASTEXITCODE -eq 0) {
    Write-Host "  Removing existing task..." -ForegroundColor Yellow
    schtasks /delete /tn "$taskName" /f | Out-Null
}

# Create new task
$result = schtasks /create /tn "$taskName" /tr "$taskPath" /sc daily /st $startTime /rl limited /f

if ($LASTEXITCODE -eq 0) {
    Write-Host "  Task created successfully!" -ForegroundColor Green
    Write-Host "  Task Name: $taskName" -ForegroundColor Gray
    Write-Host "  Schedule: Daily at $startTime (midnight)" -ForegroundColor Gray
    Write-Host "  Script: $taskPath" -ForegroundColor Gray
} else {
    Write-Host "  ERROR: Failed to create task!" -ForegroundColor Red
    Write-Host "  $result" -ForegroundColor Red
    pause
    exit 1
}

Write-Host ""
Write-Host "========================================" -ForegroundColor Cyan
Write-Host " Setup Complete!" -ForegroundColor Cyan
Write-Host "========================================" -ForegroundColor Cyan
Write-Host ""

# Ask if user wants to test now
$response = Read-Host "Would you like to test the scanner now? (y/n)"

if ($response -eq 'y' -or $response -eq 'Y') {
    Write-Host ""
    Write-Host "Running scanner test..." -ForegroundColor Green
    Write-Host ""

    schtasks /run /tn "$taskName"

    Write-Host ""
    Write-Host "Task started! Check the logs folder in a few minutes:" -ForegroundColor Yellow
    Write-Host "  $logsPath" -ForegroundColor Gray
    Write-Host ""

    # Wait a moment
    Start-Sleep -Seconds 3

    # Ask if user wants to open logs folder
    $openLogs = Read-Host "Open logs folder now? (y/n)"
    if ($openLogs -eq 'y' -or $openLogs -eq 'Y') {
        explorer $logsPath
    }
}

Write-Host ""
Write-Host "========================================" -ForegroundColor Cyan
Write-Host " Task Scheduler Status" -ForegroundColor Cyan
Write-Host "========================================" -ForegroundColor Cyan
Write-Host ""

# Show task details
schtasks /query /tn "$taskName" /fo LIST /v | Select-String -Pattern "TaskName|Next Run Time|Last Run Time|Status|Last Result"

Write-Host ""
Write-Host "========================================" -ForegroundColor Cyan
Write-Host " Next Steps" -ForegroundColor Cyan
Write-Host "========================================" -ForegroundColor Cyan
Write-Host ""
Write-Host "1. The scanner will run automatically every night at midnight" -ForegroundColor White
Write-Host "2. Check logs after each run: $logsPath" -ForegroundColor White
Write-Host "3. View results: http://127.0.0.1:5000/heatmap" -ForegroundColor White
Write-Host ""
Write-Host "To manage the task:" -ForegroundColor White
Write-Host "  View tasks:   taskschd.msc" -ForegroundColor Gray
Write-Host "  Run now:      schtasks /run /tn ""$taskName""" -ForegroundColor Gray
Write-Host "  Disable:      schtasks /change /tn ""$taskName"" /disable" -ForegroundColor Gray
Write-Host "  Enable:       schtasks /change /tn ""$taskName"" /enable" -ForegroundColor Gray
Write-Host "  Delete:       schtasks /delete /tn ""$taskName"" /f" -ForegroundColor Gray
Write-Host ""

pause
