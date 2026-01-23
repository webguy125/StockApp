import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Dict, Any, Optional
import warnings

from torch.utils.data import TensorDataset, DataLoader
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.cuda.amp import GradScaler, autocast

from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score

from backend.turbomode.shared.prediction_utils import format_prediction

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

class TurboCoreNN(nn.Module):
    def __init__(self, feat_dim: int = 179, seq_len: int = 1, recurrent_type: str = 'lstm', recurrent_hidden: int = 256, hidden1: int = 512, hidden2: int = 256, hidden3: int = 128, num_classes: int = 3, dropout_rate: float = 0.2, bidirectional: bool = True):
        super().__init__()
        self.seq_len = seq_len
        self.feat_dim = feat_dim
        self.recurrent_type = recurrent_type.lower()
        self.bidirectional = bidirectional
        self.num_directions = 2 if bidirectional else 1
        self.recurrent_out_dim = recurrent_hidden * self.num_directions

        if self.recurrent_type == 'lstm':
            self.rnn = nn.LSTM(input_size=feat_dim, hidden_size=recurrent_hidden, num_layers=1, batch_first=True, bidirectional=bidirectional)
        elif self.recurrent_type == 'gru':
            self.rnn = nn.GRU(input_size=feat_dim, hidden_size=recurrent_hidden, num_layers=1, batch_first=True, bidirectional=bidirectional)
        else:
            raise ValueError(f'Unsupported recurrent_type: {recurrent_type}. Must be "lstm" or "gru".')

        # Apply custom initialization for RNN layers
        if self.recurrent_type == 'gru':
            self._init_gru_weights()
        elif self.recurrent_type == 'lstm':
            self._init_lstm_weights()

        self.bn0 = nn.BatchNorm1d(self.recurrent_out_dim)
        self.dropout0 = nn.Dropout(dropout_rate)
        self.fc1 = nn.Linear(self.recurrent_out_dim, hidden1)
        self.bn1 = nn.BatchNorm1d(hidden1)
        self.dropout1 = nn.Dropout(dropout_rate)
        self.fc2 = nn.Linear(hidden1, hidden2)
        self.bn2 = nn.BatchNorm1d(hidden2)
        self.dropout2 = nn.Dropout(dropout_rate)
        self.fc3 = nn.Linear(hidden2, hidden3)
        self.bn3 = nn.BatchNorm1d(hidden3)
        self.dropout3 = nn.Dropout(dropout_rate)
        self.out = nn.Linear(hidden3, num_classes)

    def _init_gru_weights(self):
        """Custom initialization for GRU (Xavier uniform for input, Orthogonal for recurrent)"""
        for name, param in self.rnn.named_parameters():
            if 'weight_ih' in name:  # Input-hidden weights
                nn.init.xavier_uniform_(param.data)
            elif 'weight_hh' in name:  # Hidden-hidden (recurrent) weights
                nn.init.orthogonal_(param.data)
            elif 'bias' in name:  # Bias terms
                param.data.fill_(0.0)

    def _init_lstm_weights(self):
        """Custom initialization for LSTM (Xavier uniform for input, Orthogonal for recurrent)"""
        for name, param in self.rnn.named_parameters():
            if 'weight_ih' in name:  # Input-hidden weights
                nn.init.xavier_uniform_(param.data)
            elif 'weight_hh' in name:  # Hidden-hidden (recurrent) weights
                nn.init.orthogonal_(param.data)
            elif 'bias' in name:  # Bias terms
                param.data.fill_(0.0)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        batch_size = x.size(0)
        x = x.view(batch_size, self.seq_len, self.feat_dim)
        if self.recurrent_type == 'lstm':
            _, (hn, _) = self.rnn(x)
        elif self.recurrent_type == 'gru':
            _, hn = self.rnn(x)
        if self.bidirectional:
            x = torch.cat((hn[0], hn[1]), dim=1)
        else:
            x = hn[0]
        x = self.bn0(x)
        x = F.gelu(x)
        x = self.dropout0(x)
        x = self.fc1(x)
        x = self.bn1(x)
        x = F.gelu(x)
        x = self.dropout1(x)
        x = self.fc2(x)
        x = self.bn2(x)
        x = F.gelu(x)
        x = self.dropout2(x)
        x = self.fc3(x)
        x = self.bn3(x)
        x = F.gelu(x)
        x = self.dropout3(x)
        x = self.out(x)
        return x

class TurboCoreNNWrapper:
    def __init__(self, input_dim: int = 179, num_classes: int = 3, model_name: str = 'tc_nn_1', dropout_rate: float = 0.1, seq_len: int = 1, recurrent_hidden: int = 256, bidirectional: bool = True, recurrent_type: str = 'lstm'):
        if input_dim % seq_len != 0:
            raise ValueError(f'input_dim {input_dim} must be divisible by seq_len {seq_len}')
        feat_dim = input_dim // seq_len
        self.model_name = model_name
        self.input_dim = input_dim
        self.num_classes = num_classes
        self.model = TurboCoreNN(feat_dim=feat_dim, seq_len=seq_len, recurrent_type=recurrent_type, recurrent_hidden=recurrent_hidden, num_classes=num_classes, dropout_rate=dropout_rate, bidirectional=bidirectional).to(DEVICE)
        self.is_trained = False

    def fit(self, X_train: np.ndarray, y_train: np.ndarray, X_val: np.ndarray, y_val: np.ndarray, epochs: int = 50, batch_size: int = 4096, lr: float = 3e-3, weight_decay: float = 1e-4, patience: int = 10, use_class_weights: bool = True, monitor_metric: str = 'loss'):
        if not isinstance(X_train, np.ndarray) or not isinstance(y_train, np.ndarray):
            raise ValueError('Inputs must be numpy arrays.')

        # FORCE 3-class mode (down=0, neutral=1, up=2)
        n_classes = len(np.unique(y_train))
        if n_classes != 3:
            raise ValueError(f"Expected 3 classes, got {n_classes}. Labels must be 0=down, 1=neutral, 2=up")
        if self.num_classes != 3:
            raise ValueError(f"Model must be initialized with num_classes=3, got {self.num_classes}")

        valid_metrics = ['loss', 'acc', 'f1']
        if monitor_metric not in valid_metrics:
            raise ValueError(f'monitor_metric must be one of {valid_metrics}')
        minimize = monitor_metric == 'loss'
        best_val = float('inf') if minimize else float('-inf')
        compare = lambda current, best: current < best if minimize else current > best

        self.model.train()
        optimizer = torch.optim.AdamW(self.model.parameters(), lr=lr, weight_decay=weight_decay)

        weights = None
        if use_class_weights:
            class_counts = np.bincount(y_train, minlength=self.num_classes)
            weights = len(y_train) / (self.num_classes * class_counts.astype(float) + 1e-6)
            weights = torch.tensor(weights, dtype=torch.float32, device=DEVICE)
        criterion = nn.CrossEntropyLoss(weight=weights)

        X_train_t = torch.from_numpy(X_train).float().to(DEVICE)
        y_train_t = torch.from_numpy(y_train).long().to(DEVICE)
        X_val_t = torch.from_numpy(X_val).float().to(DEVICE)
        y_val_t = torch.from_numpy(y_val).long().to(DEVICE)

        train_dataset = TensorDataset(X_train_t, y_train_t)
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=0)

        scheduler = ReduceLROnPlateau(optimizer, mode='min' if minimize else 'max', factor=0.5, patience=2, verbose=True)
        scaler = GradScaler(enabled='cuda' in DEVICE.type)

        best_state = None
        no_improve = 0

        for epoch in range(epochs):
            self.model.train()
            total_loss = 0.0
            train_correct = 0
            n_samples = 0
            for xb, yb in train_loader:
                optimizer.zero_grad()
                with autocast(enabled='cuda' in DEVICE.type):
                    logits = self.model(xb)
                    loss = criterion(logits, yb)
                scaler.scale(loss).backward()
                # Gradient clipping to prevent exploding gradients
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
                scaler.step(optimizer)
                scaler.update()
                total_loss += loss.item() * xb.size(0)
                preds = logits.argmax(dim=1)
                train_correct += (preds == yb).sum().item()
                n_samples += xb.size(0)

            avg_train_loss = total_loss / n_samples
            avg_train_acc = train_correct / n_samples

            self.model.eval()
            with torch.no_grad():
                val_logits = self.model(X_val_t)
                val_loss = criterion(val_logits, y_val_t).item()
                val_pred = val_logits.argmax(dim=1).cpu().numpy()
                val_y = y_val_t.cpu().numpy()
                val_acc = accuracy_score(val_y, val_pred)
                val_f1 = f1_score(val_y, val_pred, average='weighted')
                val_prec = precision_score(val_y, val_pred, average='weighted', zero_division=0)
                val_rec = recall_score(val_y, val_pred, average='weighted', zero_division=0)

            print(f'Epoch {epoch+1}/{epochs}: Train Loss {avg_train_loss:.4f}, Train Acc {avg_train_acc:.4f}, Val Loss {val_loss:.4f}, Val Acc {val_acc:.4f}, Val F1 {val_f1:.4f}, Val Prec {val_prec:.4f}, Val Rec {val_rec:.4f}')

            if monitor_metric == 'loss':
                monitor_val = val_loss
            elif monitor_metric == 'acc':
                monitor_val = val_acc
            else:
                monitor_val = val_f1
            scheduler.step(monitor_val if minimize else -monitor_val)

            if compare(monitor_val, best_val):
                best_val = monitor_val
                best_state = self.model.state_dict()
                no_improve = 0
            else:
                no_improve += 1
            if no_improve >= patience:
                print(f'Early stopping at epoch {epoch+1} based on {monitor_metric}')
                break

        if best_state is not None:
            self.model.load_state_dict(best_state)
        self.is_trained = True

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        if not self.is_trained:
            warnings.warn('Model not trained; returning uniform probabilities.')
            return np.full((X.shape[0], self.num_classes), 1.0 / self.num_classes, dtype=np.float32)

        self.model.eval()
        X_t = torch.from_numpy(X).float().to(DEVICE)
        with torch.no_grad(), autocast(enabled='cuda' in DEVICE.type):
            logits = self.model(X_t)
            probs = torch.softmax(logits, dim=1)
        return probs.cpu().numpy()

    def predict(self, X: np.ndarray) -> np.ndarray:
        """
        Predict 3-class probabilities for single sample.

        Args:
            X: Single feature vector (shape [1, input_dim] or [input_dim])

        Returns:
            3-class probability array: [prob_down, prob_neutral, prob_up]
        """
        # Handle both 1D and 2D input
        if X.ndim == 1:
            X = X.reshape(1, -1)
        elif X.ndim == 2 and X.shape[0] != 1:
            raise ValueError('predict() expects a single sample. Use predict_batch() for multiple.')

        probs = self.predict_proba(X)[0]

        if len(probs) != 3:
            raise ValueError(f"Expected 3-class output, got {len(probs)} classes")

        return probs  # [prob_down, prob_neutral, prob_up]

    def predict_batch(self, X) -> list:
        """
        Make predictions for multiple samples

        Args:
            X: Either numpy array (N, input_dim) or list of feature dictionaries

        Returns:
            List of prediction dictionaries
        """
        # Handle list of dictionaries (from meta-learner training)
        if isinstance(X, list) and len(X) > 0 and isinstance(X[0], dict):
            # Convert list of dicts to numpy array
            # Assume feature names are 'feature_0', 'feature_1', ... 'feature_N'
            feature_names = [f'feature_{i}' for i in range(self.input_dim)]
            X = np.array([[feat.get(name, 0.0) for name in feature_names] for feat in X])

        probs = self.predict_proba(X)
        preds = np.argmax(probs, axis=1)
        return [format_prediction(probs[i], preds[i], self.model_name) for i in range(X.shape[0])]

    def evaluate(self, X, y):
        preds = self.predict_batch(X)
        preds = np.argmax(preds, axis=1)
        accuracy = float((preds == y).mean())
        return {"accuracy": accuracy, "n_samples": len(y)}

    def save(self, path: str):
        torch.save(self.model.state_dict(), path)

    def load(self, path: str):
        state = torch.load(path, map_location=DEVICE)
        self.model.load_state_dict(state)
        self.model.to(DEVICE)
        self.is_trained = True
