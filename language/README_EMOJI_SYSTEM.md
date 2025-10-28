# Emoji Shorthand System – Multi-Agent Blueprint (Crypto-Focused)

This document explains how the emoji-based shorthand language integrates into the multi-agent crypto analysis system. All agents must read and follow this guide before execution.

## 1. Purpose
- Reduce token usage by using emojis as compact communication symbols.
- Maintain auditability by expanding shorthand into JSON for storage.
- Ensure consistency across all agents with a shared lexicon and grammar.

## 2. Emoji Lexicon
The core vocabulary is defined in `emoji_blueprint.json` under `emoji_language.emoji_lexicon`.

### Outcomes
- 🙂 = good / success
- 🙁 = bad / failure
- 💵 = profit
- 💸 = loss
- ⚖️ = neutral / breakeven

### Technicals
- 📈 = uptrend
- 📉 = downtrend
- 🔄 = retest
- 📊 = consolidation / sideways
- ⚡ = breakout / momentum
- 🌀 = volatility spike

### Volume & Flow
- 📦 = accumulation
- 🏃 = distribution / exit
- 💧 = liquidity zone
- 🔥 = exhaustion

### Fundamentals & News
- 📰 = positive news
- 🗞️ = negative news
- 📅 = token unlock / event
- 🏦 = on-chain strong
- 🏚️ = on-chain weak

### Actions
- ✅ = criteria pass
- ❌ = criteria fail
- ⏸️ = hold / pending
- ⬆️ = promote / whitelist
- ⬇️ = drop / blacklist
- 📂 = archive / remember

## 3. Grammar
Shorthand lines follow this structure:
SYMBOL | EMOJIS | SCORE | CONFIDENCE | TIMESTAMP

Example:
BTCUSDT | 🙂📈💵📰 | 82 | 0.78 | 2025-10-27

Meaning: BTCUSDT trade was good, uptrend, profitable, positive news, score 82, confidence 78%.

## 4. Encoder/Decoder
- File: `emoji_codec.py`
- Location: `C:\stockapp\language\emoji_codec.py`
- Functions:
  - encode_to_shorthand(symbol, emojis, score, confidence, timestamp)
  - decode_from_shorthand(line)

Agents must use these functions for all shorthand ↔ JSON translations.

## 5. Policy
- Runtime: Agents communicate in emoji shorthand.
- Storage: Expanded JSON persisted to disk (free, auditable).
- Bridge: Encoder/decoder ensures consistency.
- Governance: Schema versioning, audit trail, rollback, and manual overrides apply.

## 6. Workflow Integration
1. Scanner produces candidate crypto pairs.
2. Agents analyze and output shorthand lines.
3. Shorthand is decoded into JSON and saved.
4. Supreme Leader reviews JSON verdicts.
5. Worker Bee publishes results.

## 7. Deployment
- Blueprint JSON: `C:\stockapp\language\emoji_blueprint.json`
- Codec: `C:\stockapp\language\emoji_codec.py`
- This README: `C:\stockapp\language\README_EMOJI_SYSTEM.md`

All agents must read this file before starting work.

## 8. Quick Start Examples
Tracker Agent:
BTCUSDT | 🙂📈✅ | 75 | 0.72 | 2025-10-27T14:00Z
Meaning: BTC is good, trending up, criteria passed, score 75, confidence 72%.

Criteria Auditor:
ETHUSDT | 📊❌ | 40 | 0.65 | 2025-10-27T15:00Z
Meaning: ETH consolidated sideways, failed criteria, score 40, confidence 65%.

Evaluator:
SOLUSDT | 💵⚡✅ | 88 | 0.80 | 2025-10-27T18:30Z
Meaning: SOL profitable with momentum breakout, criteria passed, score 88, confidence 80%.

Supreme Leader:
BTCUSDT | 🙂⬆️📂 | 90 | 0.86 | 2025-10-27T19:00Z
Meaning: BTC marked good, promoted to whitelist, archived, score 90, confidence 86%.

Worker Bee:
ETHUSDT | 🙂📈💵⬆️ | 92 | 0.90 | 2025-10-27T22:00Z
Meaning: ETH good, uptrend, profitable, promoted, score 92, confidence 90%.
