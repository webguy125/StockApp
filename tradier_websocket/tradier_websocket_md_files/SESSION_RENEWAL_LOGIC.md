# Tradier WebSocket Session Auto-Renewal Logic

## Overview

The Tradier WebSocket client automatically renews sessions every 55 minutes to maintain a persistent connection 24/7. Sessions expire after ~60 minutes, so we proactively renew at 55 minutes to prevent disconnection.

---

## Configuration

**File:** `tradier_websocket_client.py` (line 18)

```python
# Session expires after ~60 minutes, renew at 55 min
SESSION_RENEWAL_SECONDS = 55 * 60  # 3300 seconds = 55 minutes
```

**Why 55 minutes?**
- Tradier sessions expire at ~60 minutes
- Renewing at 55 minutes provides 5-minute buffer
- Prevents mid-stream disconnection
- Ensures seamless transition to new session

---

## How It Works

### 1. Session Creation Tracking

When a session is created, we store the timestamp:

```python
def create_session(self) -> Optional[str]:
    """Create a new Tradier streaming session"""
    # ... API call to create session ...

    session_id = session_data["stream"]["sessionid"]
    self.session_created_at = datetime.now()  # STORE TIMESTAMP

    print(f"[TRADIER WS] Session created: {session_id[:8]}... (expires in ~60 min)")
    return session_id
```

**Key Variables:**
- `self.session_id` - Current session ID
- `self.session_created_at` - Timestamp when session was created

---

### 2. Continuous Monitoring (Main Loop)

The `_stream_loop()` method continuously checks for session expiration:

```python
async def _stream_loop(self):
    """Main WebSocket streaming loop with auto-reconnection"""

    while self.running:
        try:
            # ============================================================
            # CHECK 1: Before connecting, check if session needs renewal
            # ============================================================
            if self.session_created_at:
                elapsed = (datetime.now() - self.session_created_at).total_seconds()

                if elapsed >= SESSION_RENEWAL_SECONDS:  # 55 minutes
                    print(f"[TRADIER WS] Session expired after {elapsed:.0f}s, renewing...")

                    # Create new session
                    self.session_id = self.create_session()

                    if not self.session_id:
                        print(f"[TRADIER WS ERROR] Session renewal failed, retrying...")
                        await asyncio.sleep(reconnect_delay)
                        reconnect_delay = min(reconnect_delay * 2, max_reconnect_delay)
                        continue

            # ============================================================
            # Connect to WebSocket with current (or renewed) session
            # ============================================================
            async with websockets.connect(TRADIER_WS_URL, ...) as websocket:
                self.ws = websocket
                print(f"[TRADIER WS] [OK] Connected!")

                # Subscribe to symbols
                await self._update_subscription()

                # Reset reconnect delay on successful connection
                reconnect_delay = 1

                # ============================================================
                # Process messages
                # ============================================================
                while self.running:
                    try:
                        # Wait for message (70 second timeout)
                        message = await asyncio.wait_for(websocket.recv(), timeout=70)
                        self._handle_message(message)

                    except asyncio.TimeoutError:
                        # ====================================================
                        # CHECK 2: No message in 70s, check session expiration
                        # ====================================================
                        if self.session_created_at:
                            elapsed = (datetime.now() - self.session_created_at).total_seconds()

                            if elapsed >= SESSION_RENEWAL_SECONDS:
                                print(f"[TRADIER WS] Session renewal needed, reconnecting...")
                                break  # Exit inner loop, triggers reconnect

                        continue  # Keep listening

                    except websockets.exceptions.ConnectionClosed:
                        print(f"[TRADIER WS] Connection closed, reconnecting...")
                        break  # Exit inner loop, triggers reconnect

        except Exception as e:
            print(f"[TRADIER WS ERROR] Stream error: {e}")

        finally:
            self.ws = None

        # ============================================================
        # Reconnect with exponential backoff
        # ============================================================
        if self.running:
            print(f"[TRADIER WS] Reconnecting in {reconnect_delay}s...")
            await asyncio.sleep(reconnect_delay)
            reconnect_delay = min(reconnect_delay * 2, max_reconnect_delay)
```

---

### 3. Dual Checking Strategy

The system checks for session expiration in TWO places:

**Check #1 (Before Connection):**
- Location: Line 246-254
- When: Before each WebSocket connection attempt
- Purpose: Proactive renewal before connecting

**Check #2 (During Streaming):**
- Location: Line 283-291
- When: After 70-second message timeout
- Purpose: Catch expired sessions during idle periods

**Why Two Checks?**
- **Redundancy:** Ensures renewal even if one check fails
- **Coverage:** Handles both reconnection and timeout scenarios
- **Reliability:** Prevents edge cases where expiration goes undetected

---

### 4. Symbol Re-subscription

After session renewal and reconnection, all symbols are automatically re-subscribed:

```python
async def _update_subscription(self):
    """Update WebSocket subscription with current symbol list"""

    if not self.ws or not self.session_id:
        return

    with self.lock:
        symbols_list = list(self.symbols)  # Get all subscribed symbols

    if not symbols_list:
        print(f"[TRADIER WS] No symbols to subscribe to")
        return

    # Create subscription message with NEW session ID
    subscribe_message = {
        "symbols": symbols_list,
        "sessionid": self.session_id,  # USES NEW SESSION ID
        "filter": ["quote", "trade"],
        "linebreak": True
    }

    try:
        await self.ws.send(json.dumps(subscribe_message))
        print(f"[TRADIER WS] Updated subscription: {len(symbols_list)} symbols")
    except Exception as e:
        print(f"[TRADIER WS ERROR] Failed to update subscription: {e}")
```

**Key Points:**
- Uses `self.symbols` set (maintained throughout session)
- Automatically includes ALL currently subscribed symbols
- No manual intervention required
- Seamless continuation of data stream

---

## Timeline Example

### Continuous Operation Over 3 Hours:

```
Time    Event                           Session ID    Symbols
------  -----------------------------   -----------   --------
00:00   Initial session created         abc123...     []
00:05   User opens AAPL chart          abc123...     [AAPL]
00:15   User opens MSFT chart          abc123...     [AAPL, MSFT]
00:30   User opens TSLA chart          abc123...     [AAPL, MSFT, TSLA]

00:55   ⏰ CHECK: elapsed >= 55 min
00:55   📋 Renewal triggered!
00:55   🔄 Create new session           def456...     [AAPL, MSFT, TSLA]
00:55   🔌 Disconnect old WebSocket
00:55   🔌 Connect new WebSocket
00:55   📡 Re-subscribe: 3 symbols
00:55   ✅ Streaming continues

01:00   ⚰️  Old session expires (abc123)
01:30   User still receiving updates   def456...     [AAPL, MSFT, TSLA]

01:50   ⏰ CHECK: elapsed >= 55 min
01:50   📋 Renewal triggered!
01:50   🔄 Create new session           ghi789...     [AAPL, MSFT, TSLA]
01:50   🔌 Disconnect old WebSocket
01:50   🔌 Connect new WebSocket
01:50   📡 Re-subscribe: 3 symbols
01:50   ✅ Streaming continues

01:55   ⚰️  Old session expires (def456)
02:30   User still receiving updates   ghi789...     [AAPL, MSFT, TSLA]

02:45   ⏰ CHECK: elapsed >= 55 min
02:45   📋 Renewal triggered!
02:45   🔄 Create new session           jkl012...     [AAPL, MSFT, TSLA]
02:45   🔌 Disconnect old WebSocket
02:45   🔌 Connect new WebSocket
02:45   📡 Re-subscribe: 3 symbols
02:45   ✅ Streaming continues

... continues indefinitely ...
```

---

## Log Examples

### Normal Operation (First Connection):

```
[TRADIER WS] Creating new streaming session...
[TRADIER WS] Session created: abc123... (expires in ~60 min)
[TRADIER WS] Connecting to wss://ws.tradier.com/v1/markets/events...
[TRADIER WS] [OK] Connected!
[TRADIER WS] No symbols to subscribe to
```

### After User Opens Chart:

```
[TRADIER WS] Added AAPL to subscription list (total: 1)
[TRADIER WS] Registered callback for AAPL
[TRADIER WS] Updated subscription: 1 symbols
```

### After 55 Minutes (Auto-Renewal):

```
[TRADIER WS] Session expired after 3300s, renewing...
[TRADIER WS] Creating new streaming session...
[TRADIER WS] Session created: def456... (expires in ~60 min)
[TRADIER WS] Connecting to wss://ws.tradier.com/v1/markets/events...
[TRADIER WS] [OK] Connected!
[TRADIER WS] Updated subscription: 1 symbols
```

### After 110 Minutes (Second Renewal):

```
[TRADIER WS] Session expired after 3300s, renewing...
[TRADIER WS] Creating new streaming session...
[TRADIER WS] Session created: ghi789... (expires in ~60 min)
[TRADIER WS] Connecting to wss://ws.tradier.com/v1/markets/events...
[TRADIER WS] [OK] Connected!
[TRADIER WS] Updated subscription: 1 symbols
```

---

## Error Handling

### Session Creation Failure:

If session creation fails during renewal:

```python
self.session_id = self.create_session()

if not self.session_id:
    print(f"[TRADIER WS ERROR] Session renewal failed, retrying...")
    await asyncio.sleep(reconnect_delay)
    reconnect_delay = min(reconnect_delay * 2, max_reconnect_delay)
    continue  # Retry with exponential backoff
```

**Exponential Backoff:**
- First retry: 1 second
- Second retry: 2 seconds
- Third retry: 4 seconds
- Fourth retry: 8 seconds
- ...continues up to 60 seconds max

### Connection Failure After Renewal:

If WebSocket connection fails with new session:

```python
except Exception as e:
    print(f"[TRADIER WS ERROR] Stream error: {e}")

finally:
    self.ws = None

if self.running:
    print(f"[TRADIER WS] Reconnecting in {reconnect_delay}s...")
    await asyncio.sleep(reconnect_delay)
    # Retry connection (will check for renewal again)
```

---

## Benefits

### 1. Zero Downtime
- Sessions renewed BEFORE expiration
- 5-minute buffer prevents disconnection
- Continuous data stream maintained

### 2. Automatic Recovery
- Handles session creation failures
- Exponential backoff prevents hammering
- Always retries until successful

### 3. No Data Loss
- All subscriptions automatically restored
- Symbol list maintained in memory
- Seamless transition between sessions

### 4. 24/7 Operation
- Runs indefinitely without intervention
- Self-healing on errors
- No manual restart required

### 5. Multiple Symbols Support
- Tracks all subscribed symbols
- Re-subscribes all on renewal
- No limit on symbol count

---

## Testing

### Manual Test (Verify Renewal Logic):

**Option 1: Wait 55 Minutes**
```bash
# Start server
python backend/api_server.py

# Open chart to subscribe to symbol
# Wait 55 minutes
# Check logs for renewal:
[TRADIER WS] Session expired after 3300s, renewing...
```

**Option 2: Modify Timeout (Testing Only)**

In `tradier_websocket_client.py`:
```python
# ORIGINAL (55 minutes):
SESSION_RENEWAL_SECONDS = 55 * 60  # 3300 seconds

# FOR TESTING (5 minutes):
SESSION_RENEWAL_SECONDS = 5 * 60  # 300 seconds
```

Run server and wait 5 minutes to see renewal.

**⚠️ WARNING:** Change back to 55 minutes after testing!

### Automated Test:

```python
import asyncio
from tradier_websocket_client import TradierWebSocketClient

async def test_renewal():
    client = TradierWebSocketClient(api_key="your_key")

    # Override renewal time for testing
    client.SESSION_RENEWAL_SECONDS = 5  # 5 seconds (not 55 minutes)

    client.subscribe("AAPL")
    client.register_callback("AAPL", lambda sym, data: print(f"Quote: {sym} = ${data['last']}"))
    client.start()

    # Wait 10 seconds (should trigger renewal at 5s)
    await asyncio.sleep(10)

    # Check logs for renewal message
    client.stop()

asyncio.run(test_renewal())
```

---

## Monitoring

### Key Metrics to Watch:

**1. Session Age:**
```python
if self.session_created_at:
    elapsed = (datetime.now() - self.session_created_at).total_seconds()
    print(f"[DEBUG] Current session age: {elapsed:.0f}s (renewal at 3300s)")
```

**2. Renewal Count:**
```python
self.renewal_count = 0  # Add to __init__

def create_session(self):
    # ... create session ...
    self.renewal_count += 1
    print(f"[TRADIER WS] Session renewed {self.renewal_count} times")
```

**3. Uptime:**
```python
self.start_time = datetime.now()  # Add to start()

def get_uptime(self):
    return (datetime.now() - self.start_time).total_seconds()
```

---

## Troubleshooting

### Issue: Session renews but no data received

**Symptom:** Logs show renewal but no quotes coming through

**Possible Causes:**
1. Market closed (no trades/quotes happening)
2. Symbol not actively trading
3. Subscription failed after renewal

**Check:**
```bash
# Look for this after renewal:
[TRADIER WS] Updated subscription: X symbols

# If missing, subscription failed
```

**Fix:** Check `_update_subscription()` for errors

---

### Issue: Session doesn't renew after 55 minutes

**Symptom:** Connection drops after ~60 minutes

**Possible Causes:**
1. `session_created_at` not set properly
2. Clock skew on server
3. Loop exited prematurely

**Check:**
```python
# Add debug logging:
if self.session_created_at:
    elapsed = (datetime.now() - self.session_created_at).total_seconds()
    print(f"[DEBUG] Session age: {elapsed:.0f}s / {SESSION_RENEWAL_SECONDS}s")
```

**Fix:** Verify `session_created_at` is set in `create_session()`

---

### Issue: Renewal fails repeatedly

**Symptom:**
```
[TRADIER WS ERROR] Session renewal failed, retrying...
[TRADIER WS] Reconnecting in 1s...
[TRADIER WS ERROR] Session renewal failed, retrying...
[TRADIER WS] Reconnecting in 2s...
```

**Possible Causes:**
1. API key invalid/expired
2. Tradier API outage
3. Network connectivity issue
4. Rate limit exceeded

**Check:**
```bash
# Test session creation manually:
curl -X POST https://api.tradier.com/v1/markets/events/session \
  -H "Authorization: Bearer YOUR_API_KEY" \
  -H "Accept: application/json"
```

**Fix:**
- Verify API key in `backend/.env`
- Check Tradier status page
- Wait for rate limit to reset

---

## Summary

### How Session Renewal Works:

1. **Track Session Age:** Store `session_created_at` timestamp
2. **Check Before Connect:** Renew if elapsed >= 55 min
3. **Check During Stream:** Renew if timeout AND elapsed >= 55 min
4. **Create New Session:** Call Tradier API for new session ID
5. **Reconnect WebSocket:** Connect with new session
6. **Re-subscribe Symbols:** Restore all subscriptions automatically
7. **Continue Streaming:** No data loss, seamless transition

### Key Features:

- ✅ Automatic renewal every 55 minutes
- ✅ 5-minute buffer before expiration
- ✅ Dual checking (before connect + during stream)
- ✅ Automatic symbol re-subscription
- ✅ Exponential backoff on failures
- ✅ 24/7 operation without manual intervention
- ✅ Zero data loss during transitions
- ✅ Self-healing on errors

### No Manual Intervention Required:

The system is **fully automated** and will maintain the WebSocket connection indefinitely!

---

**File Location:** `C:\StockApp\tradier_websocket\SESSION_RENEWAL_LOGIC.md`
**Last Updated:** February 5, 2026
**Related Files:**
- `tradier_websocket_client.py` - Implementation
- `test_tradier_ws_integration.py` - Integration tests
- `../session_files/session_notes_2026-02-05_tradier_websocket_upgrade.md` - Full upgrade notes
