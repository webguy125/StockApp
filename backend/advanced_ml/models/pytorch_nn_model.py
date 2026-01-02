"""
PyTorch GPU Neural Network Model for Trading Signals
GPU-accelerated Neural Network using PyTorch
Replaces: Neural Network (sklearn MLPClassifier)
"""

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.preprocessing import StandardScaler
import joblib
import os
from typing import Dict, Any, List, Optional
from datetime import datetime
import json


class TurboModeNN(nn.Module):
    """PyTorch Neural Network Architecture"""

    def __init__(self, input_size: int = 179, hidden_sizes: List[int] = None, num_classes: int = 2):
        super().__init__()
        if hidden_sizes is None:
            hidden_sizes = [64, 32, 16]  # Reduced: [128,64,32] → [64,32,16]

        self.fc1 = nn.Linear(input_size, hidden_sizes[0])
        self.fc2 = nn.Linear(hidden_sizes[0], hidden_sizes[1])
        self.fc3 = nn.Linear(hidden_sizes[1], hidden_sizes[2])
        self.fc4 = nn.Linear(hidden_sizes[2], num_classes)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(0.5)  # Increased: 0.3 → 0.5
        self.batch_norm1 = nn.BatchNorm1d(hidden_sizes[0])
        self.batch_norm2 = nn.BatchNorm1d(hidden_sizes[1])
        self.batch_norm3 = nn.BatchNorm1d(hidden_sizes[2])

    def forward(self, x):
        x = self.batch_norm1(self.dropout(self.relu(self.fc1(x))))
        x = self.batch_norm2(self.dropout(self.relu(self.fc2(x))))
        x = self.batch_norm3(self.dropout(self.relu(self.fc3(x))))
        return self.fc4(x)


class PyTorchNNModel:
    """
    PyTorch GPU Neural Network classifier for trading signal prediction

    GPU Advantages:
    - 5-20x faster than sklearn MLPClassifier
    - GPU parallelization for batch processing
    - Better gradient descent with GPU acceleration
    - More control over architecture
    """

    def __init__(self, model_path: str = "backend/data/ml_models/pytorch_nn"):
        self.model_path = model_path
        self.model: Optional[TurboModeNN] = None
        self.scaler: Optional[StandardScaler] = None
        self.feature_names: List[str] = []
        self.is_trained = False
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        # Training hyperparameters with anti-overfitting
        self.hyperparameters = {
            'hidden_sizes': [64, 32, 16],   # Reduced: [128,64,32] → [64,32,16]
            'learning_rate': 0.0005,        # Reduced: 0.001 → 0.0005
            'batch_size': 256,
            'epochs': 100,
            'early_stopping_patience': 7    # Reduced: 10 → 7
        }

        self.training_metrics = {}
        self.feature_importance = {}
        os.makedirs(self.model_path, exist_ok=True)
        self.load()

    def prepare_features(self, features_dict: Dict[str, Any]) -> np.ndarray:
        exclude_keys = ['feature_count', 'symbol', 'last_price', 'last_volume', 'timestamp', 'error']
        feature_values = []
        feature_names = []

        for key, value in sorted(features_dict.items()):
            if key not in exclude_keys:
                if isinstance(value, (int, float)):
                    if np.isnan(value) or np.isinf(value):
                        value = 0.0
                    feature_values.append(float(value))
                    feature_names.append(key)

        if not self.feature_names:
            self.feature_names = feature_names

        return np.array(feature_values).reshape(1, -1)

    def train(self, X: np.ndarray, y: np.ndarray, validate: bool = True, sample_weight: np.ndarray = None) -> Dict[str, Any]:
        print(f"\n[TRAIN] PyTorch GPU Neural Network Model")
        print(f"  Samples: {X.shape[0]}")
        print(f"  Features: {X.shape[1]}")
        print(f"  Classes: {len(np.unique(y))}")
        print(f"  Using GPU: {torch.cuda.is_available()}")
        print(f"  Device: {self.device}")

        # Initialize scaler
        self.scaler = StandardScaler()
        X_scaled = self.scaler.fit_transform(X)

        # Initialize model
        input_size = X.shape[1]
        self.model = TurboModeNN(input_size=input_size, hidden_sizes=self.hyperparameters['hidden_sizes']).to(self.device)

        # Prepare data
        X_tensor = torch.FloatTensor(X_scaled).to(self.device)
        y_tensor = torch.LongTensor(y).to(self.device)

        # Loss and optimizer
        criterion = nn.CrossEntropyLoss()
        optimizer = optim.Adam(self.model.parameters(), lr=self.hyperparameters['learning_rate'])

        # Training loop
        print("  Training on GPU...")
        batch_size = self.hyperparameters['batch_size']
        epochs = self.hyperparameters['epochs']

        self.model.train()
        for epoch in range(epochs):
            total_loss = 0
            num_batches = 0

            # Mini-batch training
            for i in range(0, len(X_tensor), batch_size):
                batch_X = X_tensor[i:i+batch_size]
                batch_y = y_tensor[i:i+batch_size]

                # Forward pass
                optimizer.zero_grad()
                outputs = self.model(batch_X)
                loss = criterion(outputs, batch_y)

                # Backward pass
                loss.backward()
                optimizer.step()

                total_loss += loss.item()
                num_batches += 1

            avg_loss = total_loss / num_batches
            if (epoch + 1) % 20 == 0:
                print(f"    Epoch {epoch+1}/{epochs}, Loss: {avg_loss:.4f}")

        # Calculate training accuracy
        self.model.eval()
        with torch.no_grad():
            outputs = self.model(X_tensor)
            _, predicted = torch.max(outputs, 1)
            train_score = (predicted == y_tensor).sum().item() / len(y_tensor)

        # Store metrics
        self.training_metrics = {
            'train_accuracy': float(train_score),
            'n_samples': int(X.shape[0]),
            'n_features': int(X.shape[1]),
            'timestamp': datetime.now().isoformat(),
            'gpu_enabled': torch.cuda.is_available()
        }

        self.is_trained = True

        # Print results
        print(f"\n  Training Accuracy: {train_score:.4f}")

        # Auto-save
        self.save()

        return self.training_metrics

    def predict(self, features: Dict[str, Any]) -> Dict[str, Any]:
        if not self.is_trained:
            return {'prediction': 'buy', 'buy_prob': 0.50, 'sell_prob': 0.50, 'confidence': 0.0, 'model': 'pytorch_nn_untrained'}

        X = self.prepare_features(features)
        X_scaled = self.scaler.transform(X)
        X_tensor = torch.FloatTensor(X_scaled).to(self.device)

        self.model.eval()
        with torch.no_grad():
            outputs = self.model(X_tensor)
            probabilities = torch.softmax(outputs, dim=1).cpu().numpy()[0]
            prediction_class = probabilities.argmax()

        class_labels = ['buy', 'sell']
        prediction_label = class_labels[prediction_class]
        confidence = float(probabilities.max())

        return {
            'prediction': prediction_label,
            'buy_prob': float(probabilities[0]),
            'sell_prob': float(probabilities[1]),
            'confidence': confidence,
            'model': 'pytorch_nn_gpu'
        }

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        if not self.is_trained:
            return np.full((X.shape[0], 2), 0.5)  # Binary classification

        X_scaled = self.scaler.transform(X)
        X_tensor = torch.FloatTensor(X_scaled).to(self.device)

        self.model.eval()
        with torch.no_grad():
            outputs = self.model(X_tensor)
            probabilities = torch.softmax(outputs, dim=1).cpu().numpy()

        return probabilities

    def predict_batch(self, features_list: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Batch prediction for multiple samples"""
        if not self.is_trained:
            return [self.predict(features) for features in features_list]

        # Prepare all features
        X_list = [self.prepare_features(features) for features in features_list]
        X = np.vstack(X_list)
        X_scaled = self.scaler.transform(X)

        # Predict on GPU
        X_tensor = torch.FloatTensor(X_scaled).to(self.device)
        self.model.eval()
        with torch.no_grad():
            outputs = self.model(X_tensor)
            probabilities = torch.softmax(outputs, dim=1).cpu().numpy()
            predictions = np.argmax(probabilities, axis=1)

        # Format results
        class_labels = ['buy', 'sell']
        results = []
        for i in range(len(predictions)):
            results.append({
                'prediction': class_labels[predictions[i]],
                'buy_prob': float(probabilities[i][0]),
                'sell_prob': float(probabilities[i][1]),
                'confidence': float(np.max(probabilities[i])),
                'model': 'pytorch_nn_gpu'
            })
        return results

    def evaluate(self, X: np.ndarray, y: np.ndarray) -> Dict[str, Any]:
        """Evaluate model on test data"""
        if not self.is_trained:
            return {'error': 'Model not trained'}

        X_scaled = self.scaler.transform(X)
        X_tensor = torch.FloatTensor(X_scaled).to(self.device)

        self.model.eval()
        with torch.no_grad():
            outputs = self.model(X_tensor)
            y_pred = torch.argmax(outputs, dim=1).cpu().numpy()

        accuracy = float(np.mean(y_pred == y))

        class_labels = ['buy', 'sell']
        class_accuracies = {}
        for i, label in enumerate(class_labels):
            mask = (y == i)
            if np.sum(mask) > 0:
                class_acc = float(np.mean(y_pred[mask] == i))
                class_accuracies[f'{label}_accuracy'] = class_acc

        return {
            'accuracy': accuracy,
            **class_accuracies,
            'model': 'pytorch_nn_gpu'
        }

    def save(self) -> bool:
        if not self.is_trained:
            return False
        try:
            model_file = os.path.join(self.model_path, "model.pt")
            scaler_file = os.path.join(self.model_path, "scaler.pkl")
            metadata_file = os.path.join(self.model_path, "metadata.json")

            torch.save(self.model.state_dict(), model_file)
            joblib.dump(self.scaler, scaler_file)

            metadata = {
                'feature_names': self.feature_names,
                'training_metrics': self.training_metrics,
                'feature_importance': self.feature_importance,
                'is_trained': self.is_trained,
                'model_version': '1.0.0',
                'input_size': len(self.feature_names),
                'hidden_sizes': self.hyperparameters['hidden_sizes'],
                'saved_at': datetime.now().isoformat()
            }

            with open(metadata_file, 'w') as f:
                json.dump(metadata, f, indent=2)

            print(f"[OK] Model saved to {self.model_path}")
            return True
        except Exception as e:
            print(f"[ERROR] Failed to save model: {e}")
            return False

    def load(self) -> bool:
        try:
            model_file = os.path.join(self.model_path, "model.pt")
            scaler_file = os.path.join(self.model_path, "scaler.pkl")
            metadata_file = os.path.join(self.model_path, "metadata.json")

            if not all(os.path.exists(f) for f in [model_file, scaler_file, metadata_file]):
                return False

            with open(metadata_file, 'r') as f:
                metadata = json.load(f)

            # Recreate model architecture
            input_size = metadata['input_size']
            hidden_sizes = metadata['hidden_sizes']
            self.model = TurboModeNN(input_size=input_size, hidden_sizes=hidden_sizes).to(self.device)
            self.model.load_state_dict(torch.load(model_file, map_location=self.device))
            self.model.eval()

            self.scaler = joblib.load(scaler_file)
            self.feature_names = metadata['feature_names']
            self.training_metrics = metadata['training_metrics']
            self.feature_importance = metadata['feature_importance']
            self.is_trained = metadata['is_trained']

            print(f"[OK] Model loaded from {self.model_path}")
            print(f"  Trained on {self.training_metrics.get('n_samples', 0)} samples")
            print(f"  Accuracy: {self.training_metrics.get('train_accuracy', 0):.4f}")

            return True
        except Exception as e:
            return False
