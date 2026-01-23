# Schwab API Security Guide
**Protecting Your Account Information**

---

## Overview

Your Schwab account credentials (username/password) will **NEVER** be stored in code or configuration files. The OAuth 2.0 flow ensures your login info only goes to Schwab's official website, not through our API.

---

## How OAuth 2.0 Works (Simplified)

```
┌─────────────────────────────────────────────────────────────┐
│ Step 1: Initial Setup (One-Time, Browser-Based)            │
└─────────────────────────────────────────────────────────────┘

You (in browser) → Schwab Login Page → Enter username/password
                     ↓
                 Schwab verifies YOU
                     ↓
                 "Authorize this app?" → You click YES
                     ↓
       Schwab gives app: Access Token + Refresh Token
                     ↓
         Tokens saved to schwab_token.json (encrypted)


┌─────────────────────────────────────────────────────────────┐
│ Step 2: Normal Operation (Automatic, No Login Needed)       │
└─────────────────────────────────────────────────────────────┘

App needs data → Uses Access Token → Schwab API
                     ↓
            (Access token expires in 30 min)
                     ↓
  App uses Refresh Token → Get new Access Token → Continue
                     ↓
         (Refresh token lasts 7 days)
                     ↓
   After 7 days → Browser login again (one-time) → New tokens
```

---

## What Gets Stored (and What Doesn't)

### ✅ Stored Locally (Safe)

**File: `backend/data/schwab_config.json`**
```json
{
  "api_key": "YOUR_API_KEY_FROM_DEVELOPER_PORTAL",
  "app_secret": "YOUR_APP_SECRET_FROM_DEVELOPER_PORTAL",
  "callback_url": "http://localhost:8182/callback"
}
```
- **api_key**: Public identifier for your app (not sensitive)
- **app_secret**: Secret key (like a password for the app, not your account)
- **callback_url**: Where Schwab sends you after login

**File: `backend/data/schwab_token.json`** (auto-generated)
```json
{
  "access_token": "encrypted_string_here",
  "refresh_token": "encrypted_string_here",
  "expires_at": "2026-01-03T15:30:00Z"
}
```
- Auto-created by schwab-py library
- Encrypted by the library
- Auto-refreshed when needed

### ❌ NEVER Stored Anywhere

- ❌ Your Schwab username
- ❌ Your Schwab password
- ❌ Your account number (unless you explicitly save it)
- ❌ Your personal information

---

## Security Best Practices

### 1. Protect Configuration Files

**Add to `.gitignore` immediately:**
```gitignore
# Schwab API credentials - NEVER commit these
backend/data/schwab_config.json
backend/data/schwab_token.json
backend/data/schwab_api.log
```

**Set file permissions (Windows):**
```powershell
# Make files read/write only for you
icacls "backend\data\schwab_config.json" /inheritance:r /grant:r "%USERNAME%:F"
icacls "backend\data\schwab_token.json" /inheritance:r /grant:r "%USERNAME%:F"
```

### 2. Use Environment Variables (Alternative Method)

Instead of storing in `schwab_config.json`, you can use environment variables:

**Create `.env` file (also add to .gitignore):**
```env
SCHWAB_API_KEY=your_api_key_here
SCHWAB_APP_SECRET=your_app_secret_here
SCHWAB_CALLBACK_URL=http://localhost:8182/callback
```

**Load in code:**
```python
import os
from dotenv import load_dotenv

load_dotenv()

api_key = os.getenv('SCHWAB_API_KEY')
app_secret = os.getenv('SCHWAB_APP_SECRET')
```

**Pros:**
- No hardcoded credentials
- Easy to rotate keys
- Industry standard

**Cons:**
- Need to install `python-dotenv` package

### 3. Token Refresh Strategy

**Automatic (recommended):**
```python
from schwab import auth

client = auth.client_from_token_file(
    token_path='backend/data/schwab_token.json',
    api_key=api_key,
    app_secret=app_secret,
    asyncio=False
)

# schwab-py auto-refreshes token when needed
# No manual intervention required
```

**Manual check (optional):**
```python
import json
from datetime import datetime

with open('backend/data/schwab_token.json') as f:
    token_data = json.load(f)

expires_at = datetime.fromisoformat(token_data['expires_at'])
if datetime.now() >= expires_at:
    print("Token expired - will auto-refresh on next API call")
```

### 4. Rotate API Keys Regularly

**Every 90 days:**
1. Go to https://developer.schwab.com/
2. Generate new API Key + App Secret
3. Update `schwab_config.json` (or `.env`)
4. Delete old `schwab_token.json` (forces re-auth)
5. Run OAuth flow again

### 5. Monitor API Access

**Log all API calls (optional but recommended):**
```python
import logging

logging.basicConfig(
    filename='backend/data/schwab_api.log',
    level=logging.INFO,
    format='%(asctime)s | %(levelname)s | %(message)s'
)

# Log each API call
logging.info(f"API call: get_option_chain({symbol})")
```

**Review logs regularly:**
```bash
# Check for suspicious activity
tail -f backend/data/schwab_api.log
```

---

## Initial OAuth Setup (Step-by-Step)

### What Happens During First Run:

1. **You run the auth script:**
   ```bash
   python backend/data/schwab_auth_setup.py
   ```

2. **Script opens your browser** to Schwab's login page:
   ```
   https://api.schwabapi.com/v1/oauth/authorize?...
   ```

3. **You log in on Schwab's official website:**
   - Enter your username/password **directly on schwab.com**
   - App never sees your credentials
   - Schwab verifies it's really you

4. **Schwab asks for permission:**
   ```
   "Allow StockApp to access your account?"

   This app will be able to:
   - Read market data
   - Read account information
   - Place trades (if enabled)

   [Approve]  [Deny]
   ```

5. **You click Approve**

6. **Schwab redirects to callback URL:**
   ```
   http://localhost:8182/callback?code=AUTH_CODE_HERE
   ```

7. **schwab-py exchanges code for tokens:**
   - Sends AUTH_CODE to Schwab
   - Receives Access Token + Refresh Token
   - Saves to `schwab_token.json`

8. **Done! No more manual login needed** (until refresh token expires in 7 days)

---

## What If Tokens Are Compromised?

### Immediate Actions:

1. **Revoke tokens:**
   - Log into developer.schwab.com
   - Go to "My Apps" → Your app → "Revoke Tokens"

2. **Delete local token file:**
   ```bash
   del backend\data\schwab_token.json
   ```

3. **Rotate API keys:**
   - Generate new API Key + App Secret
   - Update configuration

4. **Re-authenticate:**
   ```bash
   python backend/data/schwab_auth_setup.py
   ```

### Prevention:

- ✅ Never commit `schwab_token.json` to git
- ✅ Don't share token files via email/Slack
- ✅ Use file permissions to restrict access
- ✅ Use encrypted disk storage if possible

---

## Token Expiration Timeline

```
Day 0: Initial OAuth
       ├─ Access Token valid for 30 minutes
       └─ Refresh Token valid for 7 days

Day 0-7: Normal operation
         ├─ Access token auto-refreshes every 30 min
         └─ No manual intervention needed

Day 7: Refresh token expires
       └─ Browser login required (one-time)
              └─ New tokens issued for another 7 days
```

---

## Advanced Security: IP Whitelisting

**If you want extra protection:**

1. **Check if Schwab supports IP restrictions** (as of 2026-01-03, check developer docs)
2. **Whitelist your home/office IP:**
   - Find your IP: https://whatismyip.com
   - Add to Schwab developer portal
   - Only that IP can use your API keys

**Pros:**
- Tokens useless if stolen (wrong IP)

**Cons:**
- Breaks if your IP changes
- Not practical for dynamic IPs

---

## Comparison: Config File vs Environment Variables

| Method | Security | Convenience | Best For |
|--------|----------|-------------|----------|
| **schwab_config.json** | Good (if in .gitignore) | Easy setup | Single developer, local use |
| **.env file** | Better (industry standard) | Easy rotation | Production, team environments |
| **Environment variables** | Best (no files) | Harder to manage | Cloud deployment, CI/CD |

**Recommended for you:** Start with `schwab_config.json`, migrate to `.env` later if needed.

---

## Audit Checklist

Before going live, verify:

- [ ] `schwab_config.json` added to `.gitignore`
- [ ] `schwab_token.json` added to `.gitignore`
- [ ] No credentials in git history (`git log --all --full-history --source -- '*schwab*'`)
- [ ] File permissions set (read/write for you only)
- [ ] API key rotated in last 90 days
- [ ] Tested token auto-refresh (wait 30 min, make API call)
- [ ] Logged out and re-authenticated successfully
- [ ] API call logging enabled (optional)

---

## Example: Complete Auth Setup Script

**File: `backend/data/schwab_auth_setup.py`**

```python
"""
One-time OAuth setup for Schwab API
Run this once to authenticate, then tokens auto-refresh
"""

import json
from pathlib import Path
from schwab import auth

# Load config
config_path = Path(__file__).parent / 'schwab_config.json'
with open(config_path) as f:
    config = json.load(f)

# Token storage location
token_path = Path(__file__).parent / 'schwab_token.json'

print("="*60)
print("SCHWAB API AUTHENTICATION SETUP")
print("="*60)
print("\nThis will open your browser to log into Schwab.")
print("Your username/password are NEVER stored by this app.")
print("\nAfter logging in, Schwab will redirect to localhost.")
print("The browser will show 'Cannot connect' - THIS IS NORMAL.")
print("Copy the FULL URL from the browser and paste it here.")
print("\n" + "="*60)

try:
    # This opens browser and waits for callback
    client = auth.client_from_manual_flow(
        api_key=config['api_key'],
        app_secret=config['app_secret'],
        callback_url=config['callback_url'],
        token_path=str(token_path)
    )

    print("\n✅ SUCCESS! Authentication complete.")
    print(f"✅ Tokens saved to: {token_path}")
    print("\nYou won't need to log in again for 7 days.")
    print("Tokens will auto-refresh during normal operation.")

    # Test API call
    print("\nTesting API connection...")
    quote = client.get_quote('SPY')
    print(f"✅ API working! SPY price: ${quote.json()['SPY']['quote']['lastPrice']}")

except Exception as e:
    print(f"\n❌ ERROR: {e}")
    print("\nTroubleshooting:")
    print("1. Check that API Key/Secret are correct")
    print("2. Verify callback URL matches developer portal")
    print("3. Make sure you approved the authorization request")

```

**To run:**
```bash
python backend/data/schwab_auth_setup.py
```

---

## FAQ

**Q: Can Schwab see my StockApp code?**
A: No. They only see API requests (symbol lookups, option chains, etc.).

**Q: Can someone steal my tokens and drain my account?**
A: Tokens give **read-only access by default**. Trading requires explicit permission (which you don't grant).

**Q: What if I accidentally commit schwab_token.json to GitHub?**
A:
1. Immediately revoke tokens on developer.schwab.com
2. Delete from git history: `git filter-branch --index-filter 'git rm --cached --ignore-unmatch backend/data/schwab_token.json'`
3. Force push (if repo is public)
4. Generate new tokens

**Q: Do I need a funded account?**
A: No! API access works with $0 balance. You only need a valid account.

**Q: Can I use paper trading / sandbox?**
A: Check Schwab docs - as of 2026, they may offer sandbox accounts.

**Q: How often do I need to manually log in?**
A: Every 7 days (when refresh token expires). Can request 90-day tokens with special approval.

---

## Next Steps

1. **Create Schwab account** → schwab.com
2. **Register app** → developer.schwab.com
3. **Save credentials** → Create `schwab_config.json` (add to .gitignore first!)
4. **Run OAuth setup** → `python backend/data/schwab_auth_setup.py`
5. **Test API** → Fetch a quote
6. **Integrate** → Replace yfinance calls

---

**Your account credentials stay safe because:**
- OAuth separates authentication (you + Schwab) from authorization (app + tokens)
- Tokens can be revoked instantly without changing your password
- schwab-py library handles encryption automatically
- Best practice: treat tokens like passwords (never commit, rotate regularly)
