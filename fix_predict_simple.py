"""
Simple string replacement to fix predict() methods in all 8 models.
"""

import os

models_and_fixes = [
    ('xgboost_model.py', 'xgboost'),
    ('lightgbm_model.py', 'lightgbm'),
    ('catboost_model.py', 'catboost_gpu'),
    ('xgboost_hist_model.py', 'xgboost_hist_gpu'),
    ('xgboost_dart_model.py', 'xgboost_dart_gpu'),
    ('xgboost_gblinear_model.py', 'xgboost_gblinear_gpu'),
    ('xgboost_approx_model.py', 'xgboost_approx_gpu')
]

base_path = r'C:\StockApp\backend\advanced_ml\models'

for filename, model_name in models_and_fixes:
    filepath = os.path.join(base_path, filename)

    with open(filepath, 'r', encoding='utf-8') as f:
        content = f.read()

    # Find and replace the specific lines
    # First, add int() cast to prediction_class if missing
    if 'prediction_class = self.model.predict(X_scaled)[0]' in content:
        content = content.replace(
            'prediction_class = self.model.predict(X_scaled)[0]',
            'prediction_class = int(self.model.predict(X_scaled)[0])'
        )

    # Replace the hardcoded class_labels
    content = content.replace(
        "        class_labels = ['buy', 'sell']\n        prediction_label = class_labels[prediction_class]",
        """        # Auto-detect number of classes from probability array
        n_classes = len(probabilities)
        if n_classes == 2:
            class_labels = ['buy', 'sell']
        else:
            class_labels = ['buy', 'hold', 'sell']

        prediction_label = class_labels[prediction_class]"""
    )

    # Replace the return block
    old_return = f"""        return {{
            'prediction': prediction_label,
            'buy_prob': float(probabilities[0]),
            'sell_prob': float(probabilities[1]),
            'confidence': confidence,
            'model': '{model_name}'
        }}"""

    new_return = f"""        result = {{
            'prediction': prediction_label,
            'confidence': confidence,
            'model': '{model_name}'
        }}

        # Add probability for each class
        for i, label in enumerate(class_labels):
            result[f'{{label}}_prob'] = float(probabilities[i])

        return result"""

    if old_return in content:
        content = content.replace(old_return, new_return)

        # Write back
        with open(filepath, 'w', encoding='utf-8') as f:
            f.write(content)

        print(f"[OK] Fixed {filename}")
    else:
        print(f"[SKIP] {filename} - pattern not found (maybe already fixed)")

print("\n[DONE] All models processed")
