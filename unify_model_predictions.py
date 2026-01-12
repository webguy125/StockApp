"""
Unify Model Prediction Layer
Updates all TurboMode models to use the canonical 3-class prediction schema
"""

import os
import re

# The 8 models actively used in TurboMode training
TURBOMODE_MODELS = [
    ('xgboost_model.py', 'xgboost'),
    ('xgboost_et_model.py', 'xgboost_et_gpu'),
    ('lightgbm_model.py', 'lightgbm'),
    ('catboost_model.py', 'catboost_gpu'),
    ('xgboost_hist_model.py', 'xgboost_hist_gpu'),
    ('xgboost_dart_model.py', 'xgboost_dart_gpu'),
    ('xgboost_gblinear_model.py', 'xgboost_gblinear_gpu'),
    ('xgboost_approx_model.py', 'xgboost_approx_gpu')
]

BASE_PATH = r'C:\StockApp\backend\advanced_ml\models'

# Import statement to add
IMPORT_STATEMENT = "from backend.turbomode.shared.prediction_utils import format_prediction"

def add_import_if_missing(content):
    """Add the shared prediction utils import if not already present"""
    if 'from backend.turbomode.shared.prediction_utils import format_prediction' in content:
        print("    [SKIP] Import already present")
        return content

    # Find the last import statement
    import_lines = []
    other_lines = []
    in_imports = True

    for line in content.split('\n'):
        if in_imports and (line.startswith('import ') or line.startswith('from ') or line.strip() == '' or line.startswith('#')):
            import_lines.append(line)
        else:
            in_imports = False
            other_lines.append(line)

    # Add our import after existing imports
    import_lines.append(IMPORT_STATEMENT)
    import_lines.append('')

    print("    [OK] Added import statement")
    return '\n'.join(import_lines + other_lines)


def update_predict_method(content, model_name):
    """
    Update the predict() method to use format_prediction()

    Strategy:
    1. Find the predict() method
    2. Replace the manual class_labels and result construction with format_prediction() call
    """

    # Pattern to find and replace the predict() method's return logic
    # This pattern looks for the auto-detect block we added previously and replaces it

    pattern = r"""        # Auto-detect number of classes from probability array
        n_classes = len\(probabilities\)
        if n_classes == 2:
            class_labels = \['buy', 'sell'\]
        else:
            class_labels = \['buy', 'hold', 'sell'\]

        prediction_label = class_labels\[prediction_class\]
        confidence = float\(np\.max\(probabilities\)\)

        result = \{
            'prediction': prediction_label,
            'confidence': confidence,
            'model': '[^']+'\n        \}

        # Add probability for each class
        for i, label in enumerate\(class_labels\):
            result\[f'\{label\}_prob'\] = float\(probabilities\[i\]\)

        return result"""

    replacement = f"""        # Use unified prediction layer
        return format_prediction(probabilities, prediction_class, '{model_name}')"""

    if re.search(pattern, content, re.DOTALL):
        content = re.sub(pattern, replacement, content, flags=re.DOTALL)
        print(f"    [OK] Updated predict() method to use format_prediction()")
        return content
    else:
        print(f"    [WARNING] Could not find pattern to replace in predict()")
        return content


def update_predict_batch_method(content, model_name):
    """
    Update the predict_batch() method to use format_prediction()

    This ensures batch predictions also use the unified schema
    """

    # Pattern for the predict_batch result construction
    pattern = r"""        for i in range\(len\(predictions\)\):
            pred_class = int\(predictions\[i\]\)
            probs = probabilities\[i\]

            result = \{
                'prediction': class_labels\[pred_class\],
                'confidence': float\(np\.max\(probs\)\),
                'model': '[^']+'\n            \}

            # Add probability for each class
            for j, label in enumerate\(class_labels\):
                result\[f'\{label\}_prob'\] = float\(probs\[j\]\)

            results\.append\(result\)"""

    replacement = f"""        for i in range(len(predictions)):
            pred_class = int(predictions[i])
            probs = probabilities[i]

            # Use unified prediction layer
            result = format_prediction(probs, pred_class, '{model_name}')
            results.append(result)"""

    if re.search(pattern, content, re.DOTALL):
        content = re.sub(pattern, replacement, content, flags=re.DOTALL)
        print(f"    [OK] Updated predict_batch() method to use format_prediction()")
        return content
    else:
        print(f"    [INFO] No predict_batch() pattern found (may not exist or already updated)")
        return content


def process_model_file(filename, model_name):
    """Process a single model file"""
    filepath = os.path.join(BASE_PATH, filename)

    print(f"\n[Processing] {filename}")

    if not os.path.exists(filepath):
        print(f"  [ERROR] File not found: {filepath}")
        return False

    # Read the file
    with open(filepath, 'r', encoding='utf-8') as f:
        content = f.read()

    # Apply transformations
    original_content = content
    content = add_import_if_missing(content)
    content = update_predict_method(content, model_name)
    content = update_predict_batch_method(content, model_name)

    # Only write if changes were made
    if content != original_content:
        with open(filepath, 'w', encoding='utf-8') as f:
            f.write(content)
        print(f"  [SUCCESS] File updated")
        return True
    else:
        print(f"  [SKIP] No changes needed")
        return False


def main():
    """Main execution"""
    print("="*80)
    print("UNIFYING MODEL PREDICTION LAYER")
    print("="*80)
    print(f"\nTargeting {len(TURBOMODE_MODELS)} TurboMode models")
    print(f"Base path: {BASE_PATH}\n")

    updated_count = 0

    for filename, model_name in TURBOMODE_MODELS:
        if process_model_file(filename, model_name):
            updated_count += 1

    print("\n" + "="*80)
    print(f"SUMMARY: Updated {updated_count}/{len(TURBOMODE_MODELS)} models")
    print("="*80)

    if updated_count > 0:
        print("\n[NEXT STEP] Run training to verify unified prediction layer:")
        print("  cd C:\\StockApp\\backend\\turbomode")
        print("  python train_turbomode_models.py")
    else:
        print("\n[INFO] All models already use unified prediction layer")


if __name__ == '__main__':
    main()
