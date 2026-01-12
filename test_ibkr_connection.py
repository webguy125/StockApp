"""
Quick test to verify IBKR Gateway connection
Run this after launching IB Gateway and logging in
"""

from ib_insync import IB, Stock
import sys

def test_connection():
    print("="*60)
    print("IBKR Gateway Connection Test")
    print("="*60)
    print("\nMake sure IB Gateway is running and you're logged in!")
    print("Paper trading typically uses port 4002")
    print("Live trading uses port 7496")
    print()

    # Create connection
    ib = IB()

    try:
        # Connect to paper trading gateway (port 4002)
        print("[STEP 1] Connecting to IB Gateway (port 4002 - paper trading)...")
        ib.connect('127.0.0.1', 4002, clientId=1)
        print("[OK] Connected successfully!")
        print()

        # Test getting account info
        print("[STEP 2] Getting account info...")
        accounts = ib.managedAccounts()
        print(f"[OK] Found accounts: {accounts}")
        print()

        # Test fetching a quote
        print("[STEP 3] Testing market data - fetching AAPL quote...")
        contract = Stock('AAPL', 'SMART', 'USD')
        ib.qualifyContracts(contract)

        # Request market data
        ticker = ib.reqMktData(contract, '', False, False)
        ib.sleep(2)  # Wait for data

        print(f"[OK] AAPL Last Price: ${ticker.last}")
        print(f"     Bid: ${ticker.bid} | Ask: ${ticker.ask}")
        print(f"     Spread: ${ticker.ask - ticker.bid:.2f}")
        print()

        # Test options chain
        print("[STEP 4] Testing options data - fetching AAPL option chain...")
        chains = ib.reqSecDefOptParams(contract.symbol, '', contract.secType, contract.conId)

        if chains:
            chain = chains[0]
            print(f"[OK] Found {len(chain.expirations)} expirations")
            print(f"     Next 5 expirations: {sorted(chain.expirations)[:5]}")
            print(f"     Available strikes: {len(chain.strikes)} strikes")
            print()

        print("="*60)
        print("SUCCESS! IBKR Gateway is working perfectly!")
        print("="*60)
        print()
        print("Ready to integrate with curation script.")
        print("Speed: 50 req/sec = 3,000/min (vs yfinance ~10/min)")
        print("That's 300x faster!")

        return True

    except ConnectionRefusedError:
        print("[ERROR] Connection refused!")
        print()
        print("Troubleshooting:")
        print("1. Make sure IB Gateway is running")
        print("2. Check you're logged in to paper trading account")
        print("3. Verify port 7497 is being used (paper trading)")
        print("4. In IB Gateway settings, enable API connections:")
        print("   - File > Global Configuration > API > Settings")
        print("   - Check 'Enable ActiveX and Socket Clients'")
        print("   - Check 'Read-Only API'")
        return False

    except Exception as e:
        print(f"[ERROR] {type(e).__name__}: {e}")
        print()
        print("If you see market data errors, you may need to:")
        print("1. Subscribe to market data in IBKR (free with account)")
        print("2. Or use snapshot data ($1/month credit)")
        return False

    finally:
        ib.disconnect()
        print("\n[OK] Disconnected from IB Gateway")


if __name__ == '__main__':
    success = test_connection()
    sys.exit(0 if success else 1)
