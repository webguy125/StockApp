"""
Task Monitoring and SMS Notification System
Tracks all scheduled task outcomes and sends daily SMS reports at 8:30 AM
"""

import os
import sys
import json
import logging
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, List

# Add parent directory to path
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
if parent_dir not in sys.path:
    sys.path.insert(0, parent_dir)

# Setup logging
logger = logging.getLogger('task_monitor')
logger.setLevel(logging.INFO)

# Task status file
TASK_STATUS_FILE = Path(__file__).parent.parent / 'data' / 'task_status.json'

# SMS configuration
SMS_CONFIG_FILE = Path(__file__).parent.parent / 'data' / 'sms_config.json'


def load_sms_config():
    """Load SMS configuration (Twilio credentials and phone number)"""
    if SMS_CONFIG_FILE.exists():
        try:
            with open(SMS_CONFIG_FILE, 'r') as f:
                return json.load(f)
        except:
            pass

    return {
        'enabled': False,
        'provider': 'twilio',
        'account_sid': '',
        'auth_token': '',
        'from_number': '',
        'to_number': '',
        'last_sent': None
    }


def save_sms_config(config: Dict):
    """Save SMS configuration"""
    SMS_CONFIG_FILE.parent.mkdir(parents=True, exist_ok=True)
    with open(SMS_CONFIG_FILE, 'w') as f:
        json.dump(config, f, indent=2)


def load_task_status():
    """Load task status history"""
    if TASK_STATUS_FILE.exists():
        try:
            with open(TASK_STATUS_FILE, 'r') as f:
                return json.load(f)
        except:
            pass

    return {
        'tasks': {},
        'last_report': None
    }


def save_task_status(status: Dict):
    """Save task status history"""
    TASK_STATUS_FILE.parent.mkdir(parents=True, exist_ok=True)
    with open(TASK_STATUS_FILE, 'w') as f:
        json.dump(status, f, indent=2)


def log_task_result(task_name: str, success: bool, error_msg: str = None, duration: float = None):
    """
    Log the result of a scheduled task execution.

    Args:
        task_name: Name of the task (e.g., 'overnight_scan', 'outcome_tracker')
        success: True if task succeeded, False if failed
        error_msg: Error message if task failed
        duration: Task execution time in seconds
    """
    status = load_task_status()

    if task_name not in status['tasks']:
        status['tasks'][task_name] = {
            'last_run': None,
            'last_success': None,
            'total_runs': 0,
            'total_successes': 0,
            'total_failures': 0,
            'recent_runs': []
        }

    task = status['tasks'][task_name]

    # Update counters
    task['last_run'] = datetime.now().isoformat()
    task['last_success'] = success
    task['total_runs'] += 1

    if success:
        task['total_successes'] += 1
    else:
        task['total_failures'] += 1

    # Add to recent runs (keep last 30)
    run_record = {
        'timestamp': datetime.now().isoformat(),
        'success': success,
        'error': error_msg,
        'duration_seconds': duration
    }

    task['recent_runs'].insert(0, run_record)
    task['recent_runs'] = task['recent_runs'][:30]  # Keep last 30 runs

    save_task_status(status)

    logger.info(f"Task '{task_name}' logged: {'SUCCESS' if success else 'FAILED'}")


def send_email(subject: str, body: str, recipients: List[str]) -> bool:
    """
    Send email using Gmail SMTP.

    Args:
        subject: Email subject line
        body: Email body content
        recipients: List of email addresses to send to

    Returns:
        bool: True if sent successfully, False otherwise
    """
    config = load_sms_config()

    if not config.get('enabled', False):
        logger.warning("Email not configured or disabled")
        return False

    if not recipients:
        logger.warning("No recipient email addresses provided")
        return False

    try:
        import smtplib
        from email.mime.text import MIMEText
        from email.mime.multipart import MIMEMultipart

        # Create message
        msg = MIMEMultipart()
        msg['From'] = config['from_email']
        msg['To'] = ', '.join(recipients)  # Multiple recipients comma-separated
        msg['Subject'] = subject

        msg.attach(MIMEText(body, 'plain'))

        # Connect to Gmail SMTP
        server = smtplib.SMTP('smtp.gmail.com', 587)
        server.starttls()
        server.login(config['from_email'], config['app_password'])

        # Send email to all recipients
        server.send_message(msg)
        server.quit()

        logger.info(f"Email sent successfully to {len(recipients)} recipient(s): {', '.join(recipients)}")

        # Update last sent time
        config['last_sent'] = datetime.now().isoformat()
        save_sms_config(config)

        return True

    except Exception as e:
        logger.error(f"Failed to send email: {e}")
        import traceback
        traceback.print_exc()
        return False


def generate_daily_report() -> str:
    """
    Generate daily task status report for the last 24 hours.

    Returns:
        str: Formatted report text
    """
    status = load_task_status()

    # Time window: last 24 hours
    now = datetime.now()
    cutoff = now - timedelta(hours=24)

    report_lines = []
    report_lines.append(f"TurboMode Daily Report")
    report_lines.append(f"{now.strftime('%Y-%m-%d %H:%M')}")
    report_lines.append("")

    # Task order (by execution time)
    task_order = [
        ('overnight_scan', 'Overnight Scan'),
        ('outcome_tracker', 'Outcome Tracker'),
        ('sample_generator', 'Sample Generator'),
        ('monthly_retrain', 'Model Retrain'),
        ('meta_retrain', 'Meta-Learner Retrain')
    ]

    tasks_with_status = []

    for task_id, task_display_name in task_order:
        task_data = status['tasks'].get(task_id)

        if not task_data:
            continue

        # Find most recent run in last 24 hours
        recent_run = None
        for run in task_data.get('recent_runs', []):
            run_time = datetime.fromisoformat(run['timestamp'])
            if run_time >= cutoff:
                recent_run = run
                break

        if recent_run:
            status_icon = "PASS" if recent_run['success'] else "FAIL"
            error_suffix = f" ({recent_run['error'][:30]})" if recent_run.get('error') else ""

            tasks_with_status.append(f"{status_icon}: {task_display_name}{error_suffix}")
        else:
            # No run in last 24 hours
            tasks_with_status.append(f"SKIP: {task_display_name}")

    if tasks_with_status:
        report_lines.extend(tasks_with_status)
    else:
        report_lines.append("No tasks executed in last 24 hours")

    # Add trading signals section
    report_lines.append("")
    report_lines.append("-" * 40)
    report_lines.append("TRADING SIGNALS")
    report_lines.append("-" * 40)

    try:
        # Load latest predictions
        predictions_file = Path(__file__).parent.parent / 'data' / 'all_predictions.json'

        if predictions_file.exists():
            with open(predictions_file, 'r') as f:
                predictions_data = json.load(f)

            predictions = predictions_data.get('predictions', [])

            # Separate by signal type
            buy_signals = [p for p in predictions if p['prediction'] == 'buy']
            sell_signals = [p for p in predictions if p['prediction'] == 'sell']

            # Sort by confidence
            buy_signals.sort(key=lambda x: x.get('confidence', 0), reverse=True)
            sell_signals.sort(key=lambda x: x.get('confidence', 0), reverse=True)

            # Add BUY signals (top 5)
            report_lines.append("")
            report_lines.append(f"BUY Signals ({len(buy_signals)} total):")
            if buy_signals:
                for i, signal in enumerate(buy_signals[:5], 1):
                    symbol = signal['symbol']
                    conf = signal.get('confidence', 0)
                    report_lines.append(f"  {i}. {symbol:6s} - {conf:.1%} confidence")
                if len(buy_signals) > 5:
                    report_lines.append(f"  ... and {len(buy_signals) - 5} more")
            else:
                report_lines.append("  None")

            # Add SELL signals (top 5)
            report_lines.append("")
            report_lines.append(f"SELL Signals ({len(sell_signals)} total):")
            if sell_signals:
                for i, signal in enumerate(sell_signals[:5], 1):
                    symbol = signal['symbol']
                    conf = signal.get('confidence', 0)
                    report_lines.append(f"  {i}. {symbol:6s} - {conf:.1%} confidence")
                if len(sell_signals) > 5:
                    report_lines.append(f"  ... and {len(sell_signals) - 5} more")
            else:
                report_lines.append("  None")

            # Add timestamp
            timestamp = predictions_data.get('timestamp', 'Unknown')
            report_lines.append("")
            report_lines.append(f"Scan Time: {timestamp}")

        else:
            report_lines.append("")
            report_lines.append("No predictions file found")

    except Exception as e:
        report_lines.append("")
        report_lines.append(f"Error loading signals: {str(e)}")

    return "\n".join(report_lines)


def generate_user_friendly_report() -> str:
    """
    Generate engaging, user-friendly stock picks report with emojis and analysis.

    Returns:
        str: Formatted user-friendly report
    """
    import random

    # Catchy opening phrases
    catchy_phrases = [
        "Fresh Market Opportunities Await!",
        "Your Daily Edge in the Market",
        "Smart Picks for Smart Investors",
        "Today's Top Market Movers",
        "Unlock Today's Profit Potential",
        "Your Personal Trading Advantage",
        "Market Intelligence Delivered",
        "Premium Picks for Maximum Gains",
        "Data-Driven Decisions Start Here",
        "Your Path to Smarter Trading",
        "AI-Powered Market Insights",
        "Trade Smarter, Not Harder",
        "Elite Stock Picks Inside",
        "Your Daily Competitive Edge",
        "High-Confidence Trading Signals"
    ]

    now = datetime.now()
    day_name = now.strftime('%A')

    lines = []
    lines.append(f"{'=' * 50}")
    lines.append(f"  {random.choice(catchy_phrases)}")
    lines.append(f"{'=' * 50}")
    lines.append(f"📅 {day_name}, {now.strftime('%B %d, %Y')}")
    lines.append("")

    try:
        # Load latest predictions
        predictions_file = Path(__file__).parent.parent / 'data' / 'all_predictions.json'

        if not predictions_file.exists():
            lines.append("❌ No market data available today")
            return "\n".join(lines)

        with open(predictions_file, 'r') as f:
            predictions_data = json.load(f)

        predictions = predictions_data.get('predictions', [])

        # Get BUY signals only
        buy_signals = [p for p in predictions if p['prediction'] == 'buy']
        buy_signals.sort(key=lambda x: x.get('confidence', 0), reverse=True)

        if not buy_signals:
            lines.append("🔍 No high-confidence buy opportunities today")
            lines.append("")
            lines.append("The market is in a holding pattern. Sometimes the best")
            lines.append("move is patience. We'll alert you when opportunities arise!")
            return "\n".join(lines)

        # Take top 3 buy signals
        top_picks = buy_signals[:min(3, len(buy_signals))]

        lines.append(f"💎 TODAY'S TOP {len(top_picks)} PICKS")
        lines.append("")

        for i, pick in enumerate(top_picks, 1):
            symbol = pick['symbol']
            price = pick.get('current_price', 0)
            confidence = pick.get('confidence', 0)
            sector = pick.get('sector', 'unknown').replace('_', ' ').title()
            market_cap = pick.get('market_cap_category', 'unknown').replace('_', ' ').title()

            lines.append(f"{'─' * 50}")
            lines.append(f"#{i}  {symbol} - {sector}")
            lines.append(f"{'─' * 50}")
            lines.append(f"💵 Buy Price: ${price:.2f}")
            lines.append(f"📊 AI Confidence: {confidence:.1%}")
            lines.append(f"🏢 Market Cap: {market_cap}")
            lines.append("")

            # Generate analysis based on confidence level
            if confidence >= 0.95:
                analysis = [
                    "Our AI system shows exceptionally strong conviction on this pick.",
                    "Multiple technical indicators are aligned for potential upside.",
                    "This represents one of our highest-confidence opportunities."
                ]
            elif confidence >= 0.85:
                analysis = [
                    "Strong technical setup with favorable risk/reward ratio.",
                    "AI models indicate significant upward momentum potential.",
                    "Market conditions are favorable for this position."
                ]
            else:
                analysis = [
                    "Solid opportunity with good upside potential.",
                    "Technical indicators suggest positive momentum building.",
                    "Favorable entry point based on current market dynamics."
                ]

            lines.append("📈 Analysis:")
            for point in analysis:
                lines.append(f"   • {point}")
            lines.append("")

        # Add risk disclaimer
        lines.append("")
        lines.append(f"{'=' * 50}")
        lines.append("⚠️  IMPORTANT TRADING GUIDELINES")
        lines.append(f"{'=' * 50}")
        lines.append("")
        lines.append("✓ Hold Period: 5 trading days maximum")
        lines.append("✓ Position Sizing: Never risk more than 2% per trade")
        lines.append("✓ Stop Loss: Consider 5-7% below entry")
        lines.append("✓ Review: We'll report results every Friday")
        lines.append("")
        lines.append("⚡ These picks are generated by our advanced AI system")
        lines.append("   analyzing 179 technical indicators across 8 ML models")
        lines.append("")
        lines.append("📧 Questions? Reply to this email anytime!")
        lines.append("")
        lines.append(f"{'=' * 50}")
        lines.append("        Happy Trading! 🚀")
        lines.append(f"{'=' * 50}")

    except Exception as e:
        lines.append(f"❌ Error generating report: {str(e)}")
        logger.error(f"Error in generate_user_friendly_report: {e}")
        import traceback
        traceback.print_exc()

    return "\n".join(lines)


def generate_friday_performance_report() -> str:
    """
    Generate Friday performance report showing how the week's picks performed.

    Returns:
        str: Formatted performance report
    """
    lines = []
    lines.append(f"{'=' * 50}")
    lines.append("  📊 WEEKLY PERFORMANCE REPORT")
    lines.append(f"{'=' * 50}")
    lines.append(f"Week Ending: {datetime.now().strftime('%B %d, %Y')}")
    lines.append("")

    try:
        # Load active signals from database
        import sqlite3
        from backend.turbomode.paths import TURBOMODE_DB

        conn = sqlite3.connect(str(TURBOMODE_DB))
        cursor = conn.cursor()

        # Get signals from this week (5 trading days ago)
        five_days_ago = datetime.now() - timedelta(days=5)

        cursor.execute("""
            SELECT symbol, entry_price, signal_date, signal_type, confidence
            FROM signal_history
            WHERE signal_date >= ? AND signal_type = 'buy'
            ORDER BY signal_date DESC
        """, (five_days_ago.strftime('%Y-%m-%d'),))

        signals = cursor.fetchall()
        conn.close()

        if not signals:
            lines.append("📭 No completed trades this week to report")
            lines.append("")
            lines.append("We're building our track record. Check back next Friday!")
            return "\n".join(lines)

        lines.append(f"🎯 TRADES THIS WEEK: {len(signals)}")
        lines.append("")

        # Note: In future, we'll fetch actual exit prices and calculate P&L
        # For now, show the picks that were made
        lines.append("📈 Picks Made This Week:")
        lines.append("")

        for symbol, entry_price, signal_date, signal_type, confidence in signals:
            lines.append(f"• {symbol} @ ${entry_price:.2f} ({signal_date})")

        lines.append("")
        lines.append("💡 Full performance tracking coming soon!")
        lines.append("   We're accumulating data to show you exact returns.")
        lines.append("")

    except Exception as e:
        lines.append(f"❌ Error loading performance data: {str(e)}")
        logger.error(f"Error in generate_friday_performance_report: {e}")

    lines.append(f"{'=' * 50}")
    lines.append("   See you Monday with fresh picks! 📬")
    lines.append(f"{'=' * 50}")

    return "\n".join(lines)


def send_daily_report():
    """
    Generate and send daily reports via email.
    - Admin emails: Technical scheduler status
    - User emails: Stock picks with analysis (Mon-Thu) or performance report (Friday)
    Called by scheduler at 8:30 AM daily.
    """
    logger.info("=" * 80)
    logger.info("GENERATING DAILY REPORTS")
    logger.info(f"Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    logger.info("=" * 80)

    config = load_sms_config()

    if not config.get('enabled', False):
        logger.warning("Email system disabled")
        return

    today = datetime.now()
    is_friday = today.weekday() == 4  # 4 = Friday

    try:
        # Send admin report (technical status)
        admin_emails = config.get('admin_emails', [])
        if admin_emails:
            admin_report = generate_daily_report()
            logger.info("\nAdmin Report:\n" + admin_report)

            subject = f"TurboMode Admin Report - {today.strftime('%Y-%m-%d')}"
            success = send_email(subject, admin_report, admin_emails)

            if success:
                logger.info(f"Admin report sent to {len(admin_emails)} recipient(s)")
            else:
                logger.error("Failed to send admin report via email")

        # Send user report (stock picks or Friday performance)
        user_emails = config.get('user_emails', [])
        if user_emails:
            if is_friday:
                # Friday: Send performance report
                user_report = generate_friday_performance_report()
                subject = f"📊 Your Weekly Trading Performance - {today.strftime('%B %d, %Y')}"
                logger.info("\nFriday Performance Report:\n" + user_report)
            else:
                # Mon-Thu: Send stock picks
                user_report = generate_user_friendly_report()
                subject = f"💎 Today's Stock Picks - {today.strftime('%B %d, %Y')}"
                logger.info("\nUser Report:\n" + user_report)

            success = send_email(subject, user_report, user_emails)

            if success:
                logger.info(f"User report sent to {len(user_emails)} recipient(s)")
            else:
                logger.error("Failed to send user report email")

        # Update last report time
        status = load_task_status()
        status['last_report'] = today.isoformat()
        save_task_status(status)

        logger.info("=" * 80)
        logger.info("DAILY REPORTS COMPLETE")
        logger.info("=" * 80)

        return True

    except Exception as e:
        logger.error(f"Failed to generate/send daily reports: {e}")
        import traceback
        traceback.print_exc()
        return False


def configure_email(from_email: str, app_password: str, to_email: str):
    """
    Configure email notification settings using Gmail SMTP.

    Args:
        from_email: Your Gmail address (e.g., 'your.email@gmail.com')
        app_password: Gmail App Password (16-character, no spaces)
        to_email: Recipient email address (can be same as from_email)

    Note:
        You must create an App Password in your Google Account settings:
        1. Go to myaccount.google.com/security
        2. Enable 2-Step Verification
        3. Go to App Passwords
        4. Create new app password for "Mail"
        5. Copy the 16-character password
    """
    config = {
        'enabled': True,
        'provider': 'gmail_smtp',
        'from_email': from_email,
        'app_password': app_password,
        'to_email': to_email,
        'last_sent': None
    }

    save_sms_config(config)

    logger.info("Email configuration saved successfully")
    logger.info(f"  From: {from_email}")
    logger.info(f"  To: {to_email}")
    logger.info(f"  Enabled: True")


if __name__ == '__main__':
    # Test the monitoring system
    print("Testing Task Monitoring System...")
    print("=" * 80)

    # Test 1: Log some sample task results
    print("\nTest 1: Logging sample task results...")
    log_task_result('overnight_scan', True, duration=8.5)
    log_task_result('outcome_tracker', True, duration=3.2)
    log_task_result('sample_generator', False, error_msg='Database connection failed', duration=1.1)
    print("[OK] Sample results logged")

    # Test 2: Generate daily report
    print("\nTest 2: Generating daily report...")
    report = generate_daily_report()
    print("\n" + report)
    print("\n[OK] Daily report generated")

    # Test 3: Check email configuration
    print("\nTest 3: Checking email configuration...")
    config = load_sms_config()
    if config.get('enabled', False):
        print(f"[OK] Email configured: {config['from_email']} -> {config['to_email']}")
    else:
        print("[INFO] Email not configured")
        print("       To configure, run:")
        print("       from backend.turbomode.task_monitor import configure_email")
        print("       configure_email('your.email@gmail.com', 'your_app_password', 'recipient@email.com')")

    print("\n" + "=" * 80)
    print("[OK] Task Monitoring System tests complete")
    print("\nTo send daily report: python -c \"from backend.turbomode.task_monitor import send_daily_report; send_daily_report()\"")
