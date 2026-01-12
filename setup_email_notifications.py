"""
Easy Email Notification Setup Script
Configures Gmail SMTP for daily TurboMode task reports
"""

import sys
import os

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from backend.turbomode.task_monitor import configure_email, send_daily_report, load_sms_config


def main():
    print("=" * 80)
    print("TURBOMODE - EMAIL NOTIFICATION SETUP")
    print("=" * 80)
    print("\nThis script will configure daily email reports for scheduled task status.")
    print("You'll receive an email every morning at 8:30 AM with PASS/FAIL status.\n")

    # Check if already configured
    existing_config = load_sms_config()
    if existing_config.get('enabled', False):
        print("EXISTING CONFIGURATION FOUND:")
        print(f"  From: {existing_config.get('from_email', 'N/A')}")
        print(f"  To: {existing_config.get('to_email', 'N/A')}")
        print()

        reconfigure = input("Do you want to reconfigure? (y/n): ").lower()
        if reconfigure != 'y':
            print("\nKeeping existing configuration.")

            # Offer to send test email
            test = input("\nSend a test email now? (y/n): ").lower()
            if test == 'y':
                print("\nSending test report...")
                success = send_daily_report()
                if success:
                    print("\n✅ Test email sent! Check your inbox.")
                else:
                    print("\n❌ Failed to send test email. Check the error above.")
            return

    print("\n" + "-" * 80)
    print("STEP 1: GET GMAIL APP PASSWORD")
    print("-" * 80)
    print("\nYou need a Gmail App Password (NOT your regular password):")
    print("  1. Go to: https://myaccount.google.com/security")
    print("  2. Enable '2-Step Verification' (if not already enabled)")
    print("  3. Go to: https://myaccount.google.com/apppasswords")
    print("  4. Select 'Mail' and 'Windows Computer' (or Other)")
    print("  5. Click 'Generate'")
    print("  6. Copy the 16-character password\n")

    input("Press ENTER when you have your App Password ready...")

    print("\n" + "-" * 80)
    print("STEP 2: ENTER CREDENTIALS")
    print("-" * 80)

    # Get Gmail address
    from_email = input("\nEnter your Gmail address: ").strip()

    # Validate email format
    if '@' not in from_email or '.' not in from_email:
        print("\n❌ Invalid email format. Please try again.")
        return

    # Get app password
    print("\nEnter your Gmail App Password (16 characters, no spaces):")
    print("(It will be hidden for security)")

    # Hide password input
    import getpass
    app_password = getpass.getpass("App Password: ").strip().replace(' ', '')

    if len(app_password) != 16:
        print(f"\n⚠️  WARNING: App password should be 16 characters (you entered {len(app_password)})")
        print("    It might still work, continuing anyway...")

    # Get recipient email
    print(f"\nWhere should reports be sent?")
    to_email = input(f"Recipient email (press ENTER for {from_email}): ").strip()

    if not to_email:
        to_email = from_email

    print("\n" + "-" * 80)
    print("STEP 3: SAVING CONFIGURATION")
    print("-" * 80)

    try:
        configure_email(from_email, app_password, to_email)

        print("\n✅ Configuration saved successfully!")
        print(f"   From: {from_email}")
        print(f"   To: {to_email}")
        print(f"   Saved to: backend/data/sms_config.json")

    except Exception as e:
        print(f"\n❌ Failed to save configuration: {e}")
        return

    print("\n" + "-" * 80)
    print("STEP 4: SEND TEST EMAIL")
    print("-" * 80)

    test = input("\nSend a test email now to verify it works? (y/n): ").lower()

    if test == 'y':
        print("\nSending test report...")

        try:
            success = send_daily_report()

            if success:
                print("\n" + "=" * 80)
                print("✅ SUCCESS! Test email sent.")
                print("=" * 80)
                print(f"\nCheck your inbox at: {to_email}")
                print("Subject: TurboMode Daily Report - [today's date]")
                print("\nDaily reports will be sent automatically at 8:30 AM.")
            else:
                print("\n" + "=" * 80)
                print("❌ FAILED to send test email")
                print("=" * 80)
                print("\nCommon issues:")
                print("  1. App Password incorrect (should be 16 characters)")
                print("  2. 2-Step Verification not enabled")
                print("  3. 'Less secure app access' disabled (not needed for App Passwords)")
                print("\nCheck the error message above for details.")

        except Exception as e:
            print(f"\n❌ Error sending test email: {e}")
            import traceback
            traceback.print_exc()

    else:
        print("\n" + "=" * 80)
        print("✅ SETUP COMPLETE")
        print("=" * 80)
        print("\nEmail notifications configured successfully!")
        print("Daily reports will be sent automatically at 8:30 AM.")
        print("\nTo test later, run:")
        print("  python -c \"from backend.turbomode.task_monitor import send_daily_report; send_daily_report()\"")

    print("\n" + "=" * 80)
    print("ADDITIONAL INFORMATION")
    print("=" * 80)
    print("\nScheduled Email Time: 8:30 AM daily")
    print("Report Content: PASS/FAIL status of all tasks from last 24 hours")
    print("Configuration File: backend/data/sms_config.json")
    print("\nTo disable notifications:")
    print("  Set 'enabled': false in backend/data/sms_config.json")
    print("\nTo reconfigure:")
    print("  Run this script again: python setup_email_notifications.py")
    print("\n" + "=" * 80)


if __name__ == '__main__':
    try:
        main()
    except KeyboardInterrupt:
        print("\n\nSetup cancelled by user.")
        sys.exit(0)
    except Exception as e:
        print(f"\n\n❌ Unexpected error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
