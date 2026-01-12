"""
PyTorch GPU LSTM Model for Trading Signals with Temporal Context
Uses sequence of 10-20 candles to predict price direction
"""

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.preprocessing import StandardScaler
import joblib
import os
from typing import Dict, Any, List, Optional, Tuple
from datetime import datetime
import json


class TurboModeLSTM(nn.Module):
    """LSTM Architecture for temporal pattern recognition"""

    def __init__(self, input_size: int = 176, hidden_size: int = 64, num_layers: int = 2, num_classes: int = 2):
        super().__init__()

        # LSTM layers for temporal processing (reduced hidden size to prevent overfitting)
        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=0.5 if num_layers > 1 else 0  # Increased dropout: 0.3 → 0.5
        )

        # Fully connected layers for classification (smaller network)
        self.fc1 = nn.Linear(hidden_size, 32)  # Reduced: 64 → 32
        self.fc2 = nn.Linear(32, 16)  # Reduced: 32 → 16
        self.fc3 = nn.Linear(16, num_classes)

        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(0.5)  # Increased dropout: 0.3 → 0.5
        self.batch_norm1 = nn.BatchNorm1d(32)
        self.batch_norm2 = nn.BatchNorm1d(16)

    def forward(self, x):
        """
        Forward pass through LSTM

        Args:
            x: Tensor of shape (batch_size, sequence_length, input_size)

        Returns:
            Output logits of shape (batch_size, num_classes)
        """
        # LSTM processes the sequence
        lstm_out, (h_n, c_n) = self.lstm(x)

        # Use the last hidden state
        last_hidden = h_n[-1]  # Shape: (batch_size, hidden_size)

        # Fully connected layers
        x = self.dropout(self.relu(self.fc1(last_hidden)))
        x = self.batch_norm1(x)
        x = self.dropout(self.relu(self.fc2(x)))
        x = self.batch_norm2(x)
        return self.fc3(x)


class LSTMModel:
    """
    PyTorch GPU LSTM classifier for trading signal prediction with temporal context

    Key Features:
    - Processes sequences of 10-20 candles (not just single point)
    - Learns temporal patterns and trends
    - GPU-accelerated training and inference
    - Better at capturing momentum and trend reversals

    GPU Advantages:
    - 10-30x faster than CPU LSTM
    - Parallel processing of sequences
    - Efficient backpropagation through time (BPTT)
    """

    def __init__(self, model_path: str = "backend/data/ml_models/lstm", sequence_length: int = 15):
        self.model_path = model_path
        self.model: Optional[TurboModeLSTM] = None
        self.scaler: Optional[StandardScaler] = None
        self.feature_names: List[str] = []
        self.is_trained = False
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.sequence_length = sequence_length  # Number of candles to look back

        # Training hyperparameters (reduced to prevent overfitting)
        self.hyperparameters = {
            'hidden_size': 64,  # Reduced: 128 → 64
            'num_layers': 2,
            'learning_rate': 0.0001,  # Reduced: 0.0005 → 0.0001
            'batch_size': 128,
            'epochs': 50,
            'early_stopping_patience': 5,  # Reduced: 8 → 5
            'sequence_length': sequence_length
        }

        self.training_metrics = {}
        self.feature_importance = {}
        os.makedirs(self.model_path, exist_ok=True)
        self.load()

    def prepare_features(self, features_dict: Dict[str, Any]) -> np.ndarray:
        """Prepare features from dict (single timestep)"""
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

    def create_sequences(self, X: np.ndarray, y: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        Create sequences from time series data

        Args:
            X: Feature array of shape (n_samples, n_features)
            y: Labels of shape (n_samples,)

        Returns:
            X_seq: Sequences of shape (n_sequences, sequence_length, n_features)
            y_seq: Labels of shape (n_sequences,)
        """
        X_seq = []
        y_seq = []

        for i in range(self.sequence_length, len(X)):
            # Get last sequence_length samples
            X_seq.append(X[i-self.sequence_length:i])
            y_seq.append(y[i])

        return np.array(X_seq), np.array(y_seq)

    def train(self, X: np.ndarray, y: np.ndarray, validate: bool = True, sample_weight: np.ndarray = None) -> Dict[str, Any]:
        print(f"\n[TRAIN] PyTorch GPU LSTM Model")
        print(f"  Original samples: {X.shape[0]}")
        print(f"  Features: {X.shape[1]}")
        print(f"  Classes: {len(np.unique(y))}")
        print(f"  Sequence length: {self.sequence_length} candles")
        print(f"  Using GPU: {torch.cuda.is_available()}")

        # Initialize scaler
        self.scaler = StandardScaler()
        X_scaled = self.scaler.fit_transform(X)

        # Create sequences
        X_seq, y_seq = self.create_sequences(X_scaled, y)
        print(f"  Sequences created: {X_seq.shape[0]} (reduced from {X.shape[0]})")

        # Convert to PyTorch tensors
        X_train = torch.FloatTensor(X_seq).to(self.device)
        y_train = torch.LongTensor(y_seq).to(self.device)

        # Initialize model
        num_classes = len(np.unique(y_seq))
        self.model = TurboModeLSTM(
            input_size=X.shape[1],
            hidden_size=self.hyperparameters['hidden_size'],
            num_layers=self.hyperparameters['num_layers'],
            num_classes=num_classes
        ).to(self.device)

        # Loss and optimizer
        criterion = nn.CrossEntropyLoss()
        optimizer = optim.Adam(self.model.parameters(), lr=self.hyperparameters['learning_rate'])

        # Training loop
        batch_size = self.hyperparameters['batch_size']
        epochs = self.hyperparameters['epochs']

        print(f"\n  Training on GPU...")
        best_loss = float('inf')
        patience_counter = 0

        for epoch in range(epochs):
            self.model.train()
            epoch_loss = 0
            num_batches = 0

            # Mini-batch training
            for i in range(0, len(X_train), batch_size):
                batch_X = X_train[i:i+batch_size]
                batch_y = y_train[i:i+batch_size]

                # Forward pass
                outputs = self.model(batch_X)
                loss = criterion(outputs, batch_y)

                # Backward pass
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                epoch_loss += loss.item()
                num_batches += 1

            avg_loss = epoch_loss / num_batches

            # Early stopping
            if avg_loss < best_loss:
                best_loss = avg_loss
                patience_counter = 0
            else:
                patience_counter += 1

            if (epoch + 1) % 10 == 0:
                print(f"    Epoch {epoch+1}/{epochs}, Loss: {avg_loss:.4f}")

            if patience_counter >= self.hyperparameters['early_stopping_patience']:
                print(f"    Early stopping at epoch {epoch+1}")
                break

        # Calculate final accuracy
        self.model.eval()
        with torch.no_grad():
            outputs = self.model(X_train)
            _, predicted = torch.max(outputs, 1)
            train_acc = (predicted == y_train).sum().item() / len(y_train)

        # Store metrics
        self.training_metrics = {
            'train_accuracy': float(train_acc),
            'n_samples': int(len(X_seq)),
            'n_original_samples': int(X.shape[0]),
            'n_features': int(X.shape[1]),
            'sequence_length': int(self.sequence_length),
            'final_loss': float(best_loss),
            'epochs_trained': epoch + 1,
            'timestamp': datetime.now().isoformat(),
            'gpu_enabled': torch.cuda.is_available()
        }

        self.is_trained = True

        # Print results
        print(f"\n  Training Accuracy: {train_acc:.4f}")
        print(f"  Final Loss: {best_loss:.4f}")
        print(f"  Epochs: {epoch + 1}")

        # Auto-save
        self.save()

        return self.training_metrics

    def predict(self, features: Dict[str, Any]) -> Dict[str, Any]:
        """
        Predict on single feature dict (requires sequence history)

        Note: For LSTM, single predictions require historical context.
        This returns neutral prediction for now. Use predict_batch for sequences.
        """
        if not self.is_trained:
            return {'prediction': 'buy', 'buy_prob': 0.50, 'sell_prob': 0.50, 'confidence': 0.0, 'model': 'lstm_untrained'}

        # For single prediction without context, return neutral
        return {
            'prediction': 'buy',
            'buy_prob': 0.50,
            'sell_prob': 0.50,
            'confidence': 0.0,
            'model': 'lstm_requires_sequence'
        }

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """Predict probabilities for feature array"""
        if not self.is_trained:
            return np.full((X.shape[0], 2), 0.5)

        # Scale features
        X_scaled = self.scaler.transform(X)

        # Create sequences
        if len(X_scaled) < self.sequence_length:
            # Not enough data for sequence
            return np.full((X.shape[0], 2), 0.5)

        X_seq, _ = self.create_sequences(X_scaled, np.zeros(len(X_scaled)))

        # Convert to tensor
        X_tensor = torch.FloatTensor(X_seq).to(self.device)

        # Predict
        self.model.eval()
        with torch.no_grad():
            outputs = self.model(X_tensor)
            probabilities = torch.softmax(outputs, dim=1).cpu().numpy()

        # Pad beginning with neutral predictions (sequences lost)
        padding = np.full((self.sequence_length, 2), 0.5)
        probabilities = np.vstack([padding, probabilities])

        return probabilities[:X.shape[0]]  # Trim to original length

    def predict_batch(self, features_list: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Batch prediction (requires sequence)"""
        if not self.is_trained:
            return [self.predict(features) for features in features_list]

        # Prepare all features
        X_list = [self.prepare_features(features) for features in features_list]
        X = np.vstack(X_list)

        # Get probabilities
        probabilities = self.predict_proba(X)

        # Format results (binary classification)
        class_labels = ['buy', 'sell']
        results = []
        for i in range(len(probabilities)):
            pred_class = np.argmax(probabilities[i])
            results.append({
                'prediction': class_labels[pred_class],
                'buy_prob': float(probabilities[i][0]),
                'sell_prob': float(probabilities[i][1]),
                'confidence': float(np.max(probabilities[i])),
                'model': 'lstm_gpu'
            })
        return results

    def evaluate(self, X: np.ndarray, y: np.ndarray) -> Dict[str, Any]:
        """Evaluate model on test data"""
        if not self.is_trained:
            return {'error': 'Model not trained'}

        # Scale and create sequences
        X_scaled = self.scaler.transform(X)
        X_seq, y_seq = self.create_sequences(X_scaled, y)

        # Convert to tensors
        X_tensor = torch.FloatTensor(X_seq).to(self.device)
        y_tensor = torch.LongTensor(y_seq).to(self.device)

        # Predict
        self.model.eval()
        with torch.no_grad():
            outputs = self.model(X_tensor)
            _, predicted = torch.max(outputs, 1)
            probabilities = torch.softmax(outputs, dim=1).cpu().numpy()

        predicted = predicted.cpu().numpy()
        y_seq = y_tensor.cpu().numpy()

        accuracy = float(np.mean(predicted == y_seq))

        # Per-class accuracy
        class_labels = ['buy', 'sell']
        class_accuracies = {}
        for i, label in enumerate(class_labels):
            mask = (y_seq == i)
            if np.sum(mask) > 0:
                class_acc = float(np.mean(predicted[mask] == i))
                class_accuracies[f'{label}_accuracy'] = class_acc

        return {
            'accuracy': accuracy,
            **class_accuracies,
            'model': 'lstm_gpu',
            'n_sequences_evaluated': len(X_seq)
        }

    def save(self) -> bool:
        """Save model to disk"""
        if not self.is_trained:
            return False
        try:
            model_file = os.path.join(self.model_path, "model.pt")
            scaler_file = os.path.join(self.model_path, "scaler.pkl")
            metadata_file = os.path.join(self.model_path, "metadata.json")

            # Save PyTorch model
            torch.save({
                'model_state_dict': self.model.state_dict(),
                'hyperparameters': self.hyperparameters,
                'input_size': self.model.lstm.input_size,
                'hidden_size': self.model.lstm.hidden_size,
                'num_layers': self.model.lstm.num_layers,
                'num_classes': self.model.fc3.out_features
            }, model_file)

            joblib.dump(self.scaler, scaler_file)

            metadata = {
                'feature_names': self.feature_names,
                'training_metrics': self.training_metrics,
                'is_trained': self.is_trained,
                'model_version': '1.0.0',
                'saved_at': datetime.now().isoformat()
            }

            with open(metadata_file, 'w') as f:
                json.dump(metadata, f, indent=2)

            print(f"[OK] LSTM model saved to {self.model_path}")
            return True
        except Exception as e:
            print(f"[ERROR] Failed to save LSTM model: {e}")
            return False

    def load(self) -> bool:
        """Load model from disk"""
        try:
            model_file = os.path.join(self.model_path, "model.pt")
            scaler_file = os.path.join(self.model_path, "scaler.pkl")
            metadata_file = os.path.join(self.model_path, "metadata.json")

            if not all(os.path.exists(f) for f in [model_file, scaler_file, metadata_file]):
                return False

            # Load checkpoint
            checkpoint = torch.load(model_file, map_location=self.device)

            # Recreate model
            self.model = TurboModeLSTM(
                input_size=checkpoint['input_size'],
                hidden_size=checkpoint['hidden_size'],
                num_layers=checkpoint['num_layers'],
                num_classes=checkpoint['num_classes']
            ).to(self.device)

            self.model.load_state_dict(checkpoint['model_state_dict'])
            self.hyperparameters = checkpoint['hyperparameters']
            self.sequence_length = self.hyperparameters['sequence_length']

            self.scaler = joblib.load(scaler_file)

            with open(metadata_file, 'r') as f:
                metadata = json.load(f)

            self.feature_names = metadata['feature_names']
            self.training_metrics = metadata['training_metrics']
            self.is_trained = metadata['is_trained']

            print(f"[OK] LSTM model loaded from {self.model_path}")
            print(f"  Trained on {self.training_metrics.get('n_sequences_evaluated', 0)} sequences")
            print(f"  Accuracy: {self.training_metrics.get('train_accuracy', 0):.4f}")

            return True
        except Exception as e:
            return False
