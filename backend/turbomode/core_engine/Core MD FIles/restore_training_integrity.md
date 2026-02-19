Step 1: Get full visibility
Go ahead and paste:
• 	The main training orchestration file (whatever calls the models and loops over data)
• 	Plus any config file that defines:
• 	14D window
• 	±6% regime
• 	paths/checkpoints
Once I see that, I’ll:
• 	Annotate what it’s doing now
• 	Mark where integrity is broken or ambiguous
• 	Propose minimal, surgical fixes

Step 2: Restore core integrity guarantees
These are the invariants I’ll be aiming for in your code:
• 	Single source of truth for labels and regimes
• 	14D window and ±6% band defined in one place
• 	Same logic used for training and backtesting
• 	Deterministic data loading
• 	Explicit sort order
• 	No hidden shuffling before label creation
• 	Clear separation: load → transform → label → train
• 	Checkpoint discipline
• 	Clear rule: when to resume vs when to start fresh
• 	Logging that states exactly which path was taken
• 	No silent reuse of stale checkpoints
• 	Model/run logging
• 	Regime, window, feature count, sample count
• 	Train/val split definition
• 	Seed values for any randomness

Step 3: Re‑establish training–backtest alignment
We’ll verify in code that:
• 	The same feature pipeline is used in:
• 	training
• 	backtesting
• 	any live/scan mode
• 	The same regime filters are applied:
• 	directional vs neutrality
• 	±6% band logic
• 	The same label construction is used everywhere
If anything diverges, we fix it at the source.

Step 4: Only then—layer in pruning and feature versions
Once integrity is restored and we have a clean, trusted training loop, we’ll:
• 	Add a feature version file (e.g., )
• 	Wire the training system to obey that list
• 	Design the pruning process as a separate, explicit step
But we park that until the core is solid.

Drop the training orchestration file when you’re ready, and we’ll do a line‑by‑line integrity pass and lock this system back into something you can trust.