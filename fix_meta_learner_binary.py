"""
Fix meta_learner.py for binary classification
"""
import re
from pathlib import Path

filepath = Path('backend/advanced_ml/models/meta_learner.py')
content = filepath.read_text()
original = content

# Fix 1: prepare_meta_features docstring example (line 89-90)
content = re.sub(
    r"'random_forest': \{'buy_prob': 0\.3, 'hold_prob': 0\.5, 'sell_prob': 0\.2\},\s*\n\s*'xgboost': \{'buy_prob': 0\.4, 'hold_prob': 0\.4, 'sell_prob': 0\.2\}",
    "'random_forest': {'buy_prob': 0.6, 'sell_prob': 0.4},\n                                 'xgboost': {'buy_prob': 0.7, 'sell_prob': 0.3}",
    content
)

# Fix 2: prepare_meta_features implementation (lines 102-106)
content = re.sub(
    r"meta_features\.extend\(\[\s*\n\s*pred\.get\('buy_prob', 0\.33\),\s*\n\s*pred\.get\('hold_prob', 0\.34\),\s*\n\s*pred\.get\('sell_prob', 0\.33\)\s*\n\s*\]\)",
    "meta_features.extend([\n                pred.get('buy_prob', 0.50),\n                pred.get('sell_prob', 0.50)\n            ])",
    content
)

# Fix 3: train docstring (line 118)
content = re.sub(
    r"y_true: True labels \(n_samples,\) - 0=Buy, 1=Hold, 2=Sell",
    "y_true: True labels (n_samples,) - 0=Buy, 1=Sell",
    content
)

# Fix 4: feature_names in train method (lines 167-171)
content = re.sub(
    r"feature_names\.extend\(\[\s*\n\s*f'\{model_name\}_buy_prob',\s*\n\s*f'\{model_name\}_hold_prob',\s*\n\s*f'\{model_name\}_sell_prob'\s*\n\s*\]\)",
    "feature_names.extend([\n                    f'{model_name}_buy_prob',\n                    f'{model_name}_sell_prob'\n                ])",
    content
)

# Fix 5: predict method class_labels (line 245)
content = re.sub(
    r"class_labels = \['buy', 'hold', 'sell'\]",
    "class_labels = ['buy', 'sell']",
    content,
    count=1
)

# Fix 6: predict method return (lines 254-256)
content = re.sub(
    r"'buy_prob': float\(probabilities\[0\]\),\s*\n\s*'hold_prob': float\(probabilities\[1\]\),\s*\n\s*'sell_prob': float\(probabilities\[2\]\),",
    "'buy_prob': float(probabilities[0]),\n            'sell_prob': float(probabilities[1]),",
    content
)

# Fix 7: predict_batch class_labels (line 288)
content = re.sub(
    r"class_labels = \['buy', 'hold', 'sell'\]",
    "class_labels = ['buy', 'sell']",
    content
)

# Fix 8: predict_batch return (lines 297-299)
content = re.sub(
    r"'buy_prob': float\(probs\[0\]\),\s*\n\s*'hold_prob': float\(probs\[1\]\),\s*\n\s*'sell_prob': float\(probs\[2\]\),",
    "'buy_prob': float(probs[0]),\n                'sell_prob': float(probs[1]),",
    content
)

# Fix 9: _simple_average untrained fallback (lines 319-322)
content = re.sub(
    r"'prediction': 'hold',\s*\n\s*'buy_prob': 0\.33,\s*\n\s*'hold_prob': 0\.34,\s*\n\s*'sell_prob': 0\.33,",
    "'prediction': 'buy',\n                'buy_prob': 0.50,\n                'sell_prob': 0.50,",
    content
)

# Fix 10: _simple_average implementation (lines 328-334)
content = re.sub(
    r"buy_probs = \[p\.get\('buy_prob', 0\.33\) for p in base_predictions\.values\(\)\]\s*\n\s*hold_probs = \[p\.get\('hold_prob', 0\.34\) for p in base_predictions\.values\(\)\]\s*\n\s*sell_probs = \[p\.get\('sell_prob', 0\.33\) for p in base_predictions\.values\(\)\]\s*\n\s*\n\s*avg_buy = np\.mean\(buy_probs\)\s*\n\s*avg_hold = np\.mean\(hold_probs\)\s*\n\s*avg_sell = np\.mean\(sell_probs\)",
    "buy_probs = [p.get('buy_prob', 0.50) for p in base_predictions.values()]\n        sell_probs = [p.get('sell_prob', 0.50) for p in base_predictions.values()]\n\n        avg_buy = np.mean(buy_probs)\n        avg_sell = np.mean(sell_probs)",
    content
)

# Fix 11: _simple_average probs array (line 337)
content = re.sub(
    r"probs = np\.array\(\[avg_buy, avg_hold, avg_sell\]\)",
    "probs = np.array([avg_buy, avg_sell])",
    content
)

# Fix 12: _simple_average class_labels (line 339)
content = re.sub(
    r"class_labels = \['buy', 'hold', 'sell'\]",
    "class_labels = ['buy', 'sell']",
    content
)

# Fix 13: _simple_average return (lines 343-345)
content = re.sub(
    r"'buy_prob': float\(avg_buy\),\s*\n\s*'hold_prob': float\(avg_hold\),\s*\n\s*'sell_prob': float\(avg_sell\),",
    "'buy_prob': float(avg_buy),\n            'sell_prob': float(avg_sell),",
    content
)

# Fix 14: Test code - rf_hold and xgb_hold calculations (lines 454, 459)
content = re.sub(
    r"rf_hold = 1 - rf_buy - rf_sell",
    "# rf_hold = 1 - rf_buy - rf_sell  # Binary classification",
    content
)
content = re.sub(
    r"xgb_hold = 1 - xgb_buy - xgb_sell",
    "# xgb_hold = 1 - xgb_buy - xgb_sell  # Binary classification",
    content
)

# Fix 15: Test code - rf_predictions dict (lines 461-465)
content = re.sub(
    r"rf_predictions\.append\(\{\s*\n\s*'buy_prob': rf_buy,\s*\n\s*'hold_prob': rf_hold,\s*\n\s*'sell_prob': rf_sell\s*\n\s*\}\)",
    "rf_predictions.append({\n            'buy_prob': rf_buy,\n            'sell_prob': rf_sell\n        })",
    content
)

# Fix 16: Test code - xgb_predictions dict (lines 467-471)
content = re.sub(
    r"xgb_predictions\.append\(\{\s*\n\s*'buy_prob': xgb_buy,\s*\n\s*'hold_prob': xgb_hold,\s*\n\s*'sell_prob': xgb_sell\s*\n\s*\}\)",
    "xgb_predictions.append({\n            'buy_prob': xgb_buy,\n            'sell_prob': xgb_sell\n        })",
    content
)

# Fix 17: Test code - avg_hold calculation and probs array (lines 476, 478)
content = re.sub(
    r"avg_hold = \(rf_hold \+ xgb_hold\) / 2\s*\n\s*\n\s*probs = np\.array\(\[avg_buy, avg_hold, avg_sell\]\)",
    "# avg_hold = (rf_hold + xgb_hold) / 2  # Binary classification\n\n        probs = np.array([avg_buy, avg_sell])",
    content
)

# Fix 18: Test code - print statement (line 511)
content = re.sub(
    r"print\(f\"  Hold: \{prediction\['hold_prob'\]:.3f\}\"\)",
    "# print(f\"  Hold: {prediction['hold_prob']:.3f}\")  # Binary classification",
    content
)

if content != original:
    filepath.write_text(content)
    print(f"[FIXED] {filepath}")
else:
    print(f"[NO CHANGE] {filepath}")

print("\n[DONE] Meta-learner fixed for binary classification")
