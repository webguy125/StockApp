"""
Send test stock picks email
"""
import sys
sys.path.insert(0, 'C:\\StockApp')

from backend.turbomode.task_monitor import send_daily_report

print("Sending stock picks email...")
result = send_daily_report()

if result:
    print("SUCCESS: Emails sent!")
else:
    print("FAILED: Check logs for errors")
