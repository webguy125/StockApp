# Emoji Shorthand System – Multi-Agent Blueprint (Crypto-Focused)

This document explains how the emoji-based shorthand language integrates into the multi-agent crypto analysis system. All agents must read and follow this guide before execution.

## 1. Purpose
- Reduce token usage by using emojis as compact communication symbols.
- Maintain auditability by expanding shorthand into JSON for storage.
- Ensure consistency across all agents with a shared lexicon and grammar.
- Enable a feedback loop so the system can learn and improve over time.

## 2. Emoji Lexicon
Defined in `emoji_blueprint.json` under `emoji_language.emoji_lexicon`.

### Outcomes
🙂 = good / success
🙁 = bad / failure
💵 = profit
💸 = loss
⚖️ = neutral / breakeven

### Technicals
📈 = uptrend
📉 = downtrend
🔄 = retest
📊 = consolidation / sideways
⚡ = breakout / momentum
🌀 = volatility spike

### Volume & Flow
📦 = accumulation
🏃 = distribution / exit
💧 = liquidity zone
🔥 = exhaustion

### Fundamentals & News
📰 = positive news
🗞️ = negative news
📅 = token unlock / event
🏦 = on-chain strong
🏚️ = on-chain weak

### Actions
✅ = criteria pass
❌ = criteria fail
⏸️ = hold / pending
⬆️ = promote / whitelist
⬇️ = drop / blacklist
📂 = archive / remember

## 3. Grammar
Format: SYMBOL | EMOJIS | SCORE | CONFIDENCE | TIMESTAMP

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
6. Archivist stores outcomes in `archivist.json`.
7. Trainer consumes `archivist.json` when drift or performance drop is detected, fine-tunes models, and saves updates to `updated_models/`.
8. Supreme Leader consumes improved models/rules for governance.

## 7. Deployment
- Blueprint JSON: `C:\StockApp\language\emoji_blueprint.json`
- Codec: `C:\StockApp\language\emoji_codec.py`
- Archivist Log: `C:\StockApp\agents\repository\archivist.json`
- Updated Models: `C:\StockApp\agents\repository\updated_models\`
- This README: `C:\StockApp\language\README_EMOJI_SYSTEM.md`

### Updated Models Directory Structure
The `updated_models/` directory stores calibrated parameters after Trainer retraining events:

- **`weights.json`** - Fusion agent scoring weights (technical, volume, fundamentals, news percentages)
- **`thresholds.json`** - Win/loss criteria (return %, drawdown %, confidence minimums)
- **`constraints.json`** - Supreme Leader governance rules (max positions, blacklists, score cutoffs)
- **`metadata.json`** - Training history (timestamp, trigger reason, win_rate before/after, records used)

When the Trainer agent completes retraining, it writes updated parameters to timestamped files (e.g., `weights_2025-10-28T02-00Z.json`) and creates symlinks to `weights.json` for the latest version. All agents read from the non-timestamped files for current parameters.

All agents must read this file before starting work.

## 8. Quick Start Examples

### Tracker Agent
BTCUSDT | 🙂📈✅ | 75 | 0.72 | 2025-10-27T14:00Z
Meaning: BTC is good, trending up, criteria passed, score 75, confidence 72%.

### Criteria Auditor
ETHUSDT | 📊❌ | 40 | 0.65 | 2025-10-27T15:00Z
Meaning: ETH consolidated sideways, failed criteria, score 40, confidence 65%.

### Evaluator
SOLUSDT | 💵⚡✅ | 88 | 0.80 | 2025-10-27T18:30Z
Meaning: SOL profitable with momentum breakout, criteria passed, score 88, confidence 80%.

### Supreme Leader
BTCUSDT | 🙂⬆️📂 | 90 | 0.86 | 2025-10-27T19:00Z
Meaning: BTC marked good, promoted to whitelist, archived, score 90, confidence 86%.

### Worker Bee
ETHUSDT | 🙂📈💵⬆️ | 92 | 0.90 | 2025-10-27T22:00Z
Meaning: ETH good, uptrend, profitable, promoted, score 92, confidence 90%.

### Archivist (Feedback Loop)
SYSTEM | 📂⚖️🌀 | 0 | 1.00 | 2025-10-28T00:00Z
Meaning: Archivist archived outcomes, detected drift (volatility spike), flagged for calibration. Score 0 (system-level event), confidence 100%.

This shorthand signals that the Archivist has identified a learning opportunity and will feed data back into fine-tuning and rule calibration.

### Trainer (Retraining Event)
SYSTEM | 📂✅⚡ | 0 | 1.00 | 2025-10-28T02:00Z
Meaning: Trainer consumed Archivist logs, retraining completed successfully, models updated. Score 0 (system-level event), confidence 100%.

### Scribe (Documentation Update)
SYSTEM | 📂✅📝 | 0 | 1.00 | 2025-10-28T03:00Z
Meaning: Scribe updated README_EMOJI_SYSTEM.md with approved changes, documentation synced. Score 0 (system-level event), confidence 100%.

## 9. Constraints
- Maximum emoji length per shorthand line: 4
- Enforced by: Supreme Leader
- Archivist proposals must not exceed 4 emojis.
- If a proposal is rejected, Supreme Leader provides a reason. Archivist may resubmit a corrected version within the 4‑emoji limit.
- Purpose: Keep shorthand concise, consistent, and decodable.

## 10. Documentation Agent (Scribe)
- **Name:** Scribe
- **Role:** Automatically updates `README_EMOJI_SYSTEM.md` when changes are approved by the Supreme Leader.
- **Inputs:** `emoji_blueprint.json`, `archivist.json`, `supreme_leader.json`
- **Output:** Updated `README_EMOJI_SYSTEM.md`
- **Trigger:** Supreme Leader sets `update_readme: true`
- **Constraints:**
  - Only merges approved changes.
  - Maintains formatting consistency.
  - Enforces maximum emoji length in examples.
