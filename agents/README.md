# Multi-Agent Stock Analysis System

This directory contains the repository structure for the multi-agent prediction system.

## Directory Structure

```
agents/
└── repository/           # Shared state store for all agents
    ├── scanner_output.json              # Filtered S&P500 candidates
    ├── tick_agent/<symbol>.json         # Tick bar analysis
    ├── volume_agent/<symbol>.json       # Ord/Weis wave analysis
    ├── indicators_agent/<symbol>.json   # RSI, MACD, MAs, ATR
    ├── fundamentals_agent/<symbol>.json # Growth, quality, value, earnings
    ├── news_agent/<symbol>.json         # Sentiment classification
    ├── fusion_agent/<symbol>.json       # Combined weighted scoring
    ├── supreme_leader.json              # Final approval list
    ├── tracker_agent/<symbol>.json      # Post-prediction monitoring
    ├── criteria_auditor/<symbol>.json   # Criteria verification
    ├── evaluator/<symbol>.json          # Win/loss verdict
    └── archivist.json                   # Learning archive
```

## Agent Roles

### Pre-Prediction Agents
1. **Scanner**: Filter S&P500 for liquid, volatile candidates
2. **Tick Agent**: Build 50-tick and 100-tick bars
3. **Volume Agent**: Ord/Weis wave analysis (accumulation/distribution)
4. **Indicators Agent**: RSI, MACD, Moving Averages, ATR
5. **Fundamentals Agent**: Growth, quality, value, earnings metrics
6. **News Agent**: Sentiment classification from news/social
7. **Fusion Agent**: Weighted scoring combining all signals
8. **Supreme Leader**: Apply constraints, approve final list

### Post-Prediction Agents
9. **Worker Bee**: Update frontend, push to GitHub
10. **Tracker Agent**: Monitor outcomes (1h, EOD, 3d, 10d checkpoints)
11. **Criteria Auditor**: Verify original criteria still hold
12. **Evaluator**: Assign win/loss verdict, calibrate forecasts
13. **Archivist**: Store features + outcomes for model learning

## Scoring System

### Weights
- Technical: 40%
- Volume: 25%
- Fundamentals: 20%
- News: 15%

### Thresholds
- **Win**: Return >= 0.5% with drawdown >= -0.5%
- **Loss**: Return < 0.5% OR criteria fail

## Communication Protocol

Agents use emoji shorthand for efficient communication.
See: `C:\StockApp\language\README_EMOJI_SYSTEM.md`

## Governance

- **Schema Versioning**: Track blueprint changes
- **Audit Trail**: Git commits + JSON logs
- **Manual Override**: Supreme Leader can pause publishing
- **Rollback**: Snapshots before Worker Bee pushes

## Integration with StockApp

This multi-agent system will integrate with the existing StockApp tick chart infrastructure:
- Real-time Coinbase WebSocket feeds
- Tick bar accumulation (already implemented)
- Canvas rendering for visualization
- Backend API for data persistence
