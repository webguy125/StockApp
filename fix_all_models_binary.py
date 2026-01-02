"""
Quick script to fix all remaining models for binary classification
"""
import re
from pathlib import Path

# Models to fix
models_to_fix = [
    'backend/advanced_ml/models/lightgbm_model.py',
    'backend/advanced_ml/models/pytorch_nn_model.py',
    'backend/advanced_ml/models/xgboost_linear_model.py',
]

for model_path in models_to_fix:
    filepath = Path(model_path)
    if not filepath.exists():
        print(f"[SKIP] {model_path} not found")
        continue

    content = filepath.read_text()
    original = content

    # Fix 1: Change untrained prediction from 3-class to 2-class
    content = re.sub(
        r"'prediction': 'hold', 'buy_prob': 0\.33, 'hold_prob': 0\.34, 'sell_prob': 0\.33",
        "'prediction': 'buy', 'buy_prob': 0.50, 'sell_prob': 0.50",
        content
    )

    # Fix 2: Change class_labels from 3 to 2
    content = re.sub(
        r"class_labels = \['buy', 'hold', 'sell'\]",
        "class_labels = ['buy', 'sell']",
        content
    )

    # Fix 3: Remove hold_prob from return statements
    content = re.sub(
        r"'buy_prob': float\(probabilities\[0\]\),\s*\n\s*'hold_prob': float\(probabilities\[1\]\),\s*\n\s*'sell_prob': float\(probabilities\[2\]\),",
        "'buy_prob': float(probabilities[0]),\n            'sell_prob': float(probabilities[1]),",
        content
    )

    # Fix 4: Remove hold_prob from batch predictions
    content = re.sub(
        r"'buy_prob': float\(probabilities\[i\]\[0\]\),\s*\n\s*'hold_prob': float\(probabilities\[i\]\[1\]\),\s*\n\s*'sell_prob': float\(probabilities\[i\]\[2\]\),",
        "'buy_prob': float(probabilities[i][0]),\n                'sell_prob': float(probabilities[i][1]),",
        content
    )

    # Fix 5: predict_proba return for untrained
    content = re.sub(
        r"return np\.full\(\(X\.shape\[0\], 3\), 1/3\)",
        "return np.full((X.shape[0], 2), 0.5)  # Binary classification",
        content
    )

    if content != original:
        filepath.write_text(content)
        print(f"[FIXED] {model_path}")
    else:
        print(f"[NO CHANGE] {model_path}")

print("\n[DONE] All models fixed for binary classification")
