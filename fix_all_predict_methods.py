"""
Fix predict() methods in all 8 TurboMode models to support multi-class classification.
This script updates the hardcoded binary classification labels to auto-detect the number of classes.
"""

import os
import re

# Models that need fixing (the 8 models used in TurboMode training)
models_to_fix = [
    'xgboost_model.py',
    'xgboost_et_model.py',  # Already fixed, but will re-fix for consistency
    'lightgbm_model.py',
    'catboost_model.py',
    'xgboost_hist_model.py',
    'xgboost_dart_model.py',
    'xgboost_gblinear_model.py',
    'xgboost_approx_model.py'
]

base_path = r'C:\StockApp\backend\advanced_ml\models'

# The old pattern (binary classification)
old_pattern = r"""    def predict\(self, features.*?\) -> Dict\[str, Any\]:
        if not self\.is_trained:
            return \{'prediction': 'buy', 'buy_prob': 0\.50, 'sell_prob': 0\.50, 'confidence': 0\.0, 'model': '.*?'\}

        X = self\.prepare_features\(features.*?\)
        X_scaled = self\.scaler\.transform\(X\)

        prediction_class = (?:int\()?self\.model\.predict\(X_scaled\)\[0\](?:\))?
        probabilities = self\.model\.predict_proba\(X_scaled\)\[0\]

        class_labels = \['buy', 'sell'\]
        prediction_label = class_labels\[prediction_class\]
        confidence = float\(np\.max\(probabilities\)\)

        return \{
            'prediction': prediction_label,
            'buy_prob': float\(probabilities\[0\]\),
            'sell_prob': float\(probabilities\[1\]\),
            'confidence': confidence,
            'model': '.*?'
        \}"""

# The new pattern (multi-class classification with auto-detection)
new_template = """    def predict(self, {param_name}: Dict[str, Any]) -> Dict[str, Any]:
        if not self.is_trained:
            return {{'prediction': 'buy', 'buy_prob': 0.50, 'sell_prob': 0.50, 'confidence': 0.0, 'model': '{model_name}_untrained'}}

        X = self.prepare_features({param_name})
        X_scaled = self.scaler.transform(X)

        prediction_class = int(self.model.predict(X_scaled)[0])
        probabilities = self.model.predict_proba(X_scaled)[0]

        # Auto-detect number of classes from probability array
        n_classes = len(probabilities)
        if n_classes == 2:
            class_labels = ['buy', 'sell']
        else:
            class_labels = ['buy', 'hold', 'sell']

        prediction_label = class_labels[prediction_class]
        confidence = float(np.max(probabilities))

        result = {{
            'prediction': prediction_label,
            'confidence': confidence,
            'model': '{model_name}'
        }}

        # Add probability for each class
        for i, label in enumerate(class_labels):
            result[f'{{label}}_prob'] = float(probabilities[i])

        return result"""

fixed_count = 0

for model_file in models_to_fix:
    file_path = os.path.join(base_path, model_file)

    if not os.path.exists(file_path):
        print(f"[SKIP] {model_file} not found")
        continue

    with open(file_path, 'r', encoding='utf-8') as f:
        content = f.read()

    # Extract model name from file name
    model_name = model_file.replace('_model.py', '').replace('_', '_')
    if 'xgboost' in model_name:
        if model_name == 'xgboost':
            display_name = 'xgboost'
        else:
            display_name = model_name
    elif model_name == 'lightgbm':
        display_name = 'lightgbm'
    elif model_name == 'catboost':
        display_name = 'catboost_gpu'
    else:
        display_name = model_name

    # Try to find and replace the predict method
    # First, find the def predict line to get the parameter name
    param_match = re.search(r'def predict\(self, (.*?): Dict\[str, Any\]\)', content)
    if not param_match:
        print(f"[SKIP] {model_file} - couldn't find predict method signature")
        continue

    param_name = param_match.group(1)

    # Look for the exact pattern (with variations)
    patterns_to_try = [
        # Pattern 1: with int() cast
        (r"    def predict\(self, " + re.escape(param_name) + r": Dict\[str, Any\]\) -> Dict\[str, Any\]:\s+"
         r"if not self\.is_trained:\s+"
         r"return \{'prediction': 'buy', 'buy_prob': 0\.50, 'sell_prob': 0\.50, 'confidence': 0\.0, 'model': '[^']+'\}\s+"
         r"\s+X = self\.prepare_features\(" + re.escape(param_name) + r"\)\s+"
         r"X_scaled = self\.scaler\.transform\(X\)\s+"
         r"\s+prediction_class = int\(self\.model\.predict\(X_scaled\)\[0\]\)\s+"
         r"probabilities = self\.model\.predict_proba\(X_scaled\)\[0\]\s+"
         r"\s+class_labels = \['buy', 'sell'\]\s+"
         r"prediction_label = class_labels\[prediction_class\]\s+"
         r"confidence = float\(np\.max\(probabilities\)\)\s+"
         r"\s+return \{\s+"
         r"'prediction': prediction_label,\s+"
         r"'buy_prob': float\(probabilities\[0\]\),\s+"
         r"'sell_prob': float\(probabilities\[1\]\),\s+"
         r"'confidence': confidence,\s+"
         r"'model': '[^']+'\s+"
         r"\}"),
        # Pattern 2: without int() cast
        (r"    def predict\(self, " + re.escape(param_name) + r": Dict\[str, Any\]\) -> Dict\[str, Any\]:\s+"
         r"if not self\.is_trained:\s+"
         r"return \{'prediction': 'buy', 'buy_prob': 0\.50, 'sell_prob': 0\.50, 'confidence': 0\.0, 'model': '[^']+'\}\s+"
         r"\s+X = self\.prepare_features\(" + re.escape(param_name) + r"\)\s+"
         r"X_scaled = self\.scaler\.transform\(X\)\s+"
         r"\s+prediction_class = self\.model\.predict\(X_scaled\)\[0\]\s+"
         r"probabilities = self\.model\.predict_proba\(X_scaled\)\[0\]\s+"
         r"\s+class_labels = \['buy', 'sell'\]\s+"
         r"prediction_label = class_labels\[prediction_class\]\s+"
         r"confidence = float\(np\.max\(probabilities\)\)\s+"
         r"\s+return \{\s+"
         r"'prediction': prediction_label,\s+"
         r"'buy_prob': float\(probabilities\[0\]\),\s+"
         r"'sell_prob': float\(probabilities\[1\]\),\s+"
         r"'confidence': confidence,\s+"
         r"'model': '[^']+'\s+"
         r"\}")
    ]

    replaced = False
    for pattern in patterns_to_try:
        if re.search(pattern, content, re.DOTALL):
            new_code = new_template.format(param_name=param_name, model_name=display_name)
            content = re.sub(pattern, new_code, content, flags=re.DOTALL)
            replaced = True
            break

    if not replaced:
        print(f"[SKIP] {model_file} - pattern not found (might already be fixed)")
        continue

    # Write back
    with open(file_path, 'w', encoding='utf-8') as f:
        f.write(content)

    fixed_count += 1
    print(f"[OK] Fixed {model_file}")

print(f"\n[SUCCESS] Fixed {fixed_count} out of {len(models_to_fix)} models")
