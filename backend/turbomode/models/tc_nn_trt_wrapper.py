import os
import numpy as np
import torch
import warnings
import logging
from typing import Dict, Any, Optional, List

from backend.turbomode.models.tc_nn_model import TurboCoreNNWrapper
from backend.turbomode.models.tc_nn_export import export_tc_nn_to_onnx, build_tensorrt_engine_from_onnx, TensorRTInferenceEngine
from backend.turbomode.shared.prediction_utils import format_prediction

logger = logging.getLogger(__name__)

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

class TurboCoreNNTRTWrapper:
    """
    Enhanced TensorRT inference wrapper with PyTorch fallback.
    Automatically exports to ONNX and builds TensorRT engine if missing.
    Supports FP16 (default), FP8 (if hardware allows), and explicit quantization (via pre-quantized ONNX with Q/DQ layers).
    Validates TRT vs PyTorch outputs with mode-specific tolerance.
    Handles large batches by splitting.
    Fully compatible with recurrent_type ('lstm' or 'gru') and seq_len.
    """
    def __init__(
        self,
        input_dim: int = 179,
        num_classes: int = 3,
        model_name: str = 'tc_nn_1',
        engine_path: Optional[str] = None,
        onnx_path: Optional[str] = None,
        trained_state_dict_path: Optional[str] = None,
        use_trt: bool = True,
        max_batch_size: int = 8192,
        dropout_rate: float = 0.2,
        seq_len: int = 1,
        recurrent_hidden: int = 256,
        bidirectional: bool = True,
        recurrent_type: str = 'lstm',
        build_if_missing: bool = True,
        validate_outputs: bool = True,
        quantization_mode: str = 'fp16'  # 'fp16', 'fp8', 'explicit'
    ):
        self.model_name = model_name
        self.input_dim = input_dim
        self.num_classes = num_classes
        self.use_trt = use_trt
        self.max_batch_size = max_batch_size
        self.quantization_mode = quantization_mode

        self.torch_wrapper = TurboCoreNNWrapper(
            input_dim=input_dim,
            num_classes=num_classes,
            model_name=model_name,
            dropout_rate=dropout_rate,
            seq_len=seq_len,
            recurrent_hidden=recurrent_hidden,
            bidirectional=bidirectional,
            recurrent_type=recurrent_type
        )

        if trained_state_dict_path and os.path.exists(trained_state_dict_path):
            self.torch_wrapper.load(trained_state_dict_path)
            logger.info(f'Loaded trained weights from {trained_state_dict_path} for {model_name}')

        if onnx_path is None:
            onnx_path = f'models/{model_name}.onnx'
        if engine_path is None:
            engine_path = f'models/{model_name}.{quantization_mode}.engine'

        self.trt_engine = None
        if use_trt:
            if not os.path.exists(engine_path):
                if build_if_missing:
                    logger.info(f'TensorRT engine not found at {engine_path}. Attempting to build...')
                    if quantization_mode == 'explicit':
                        logger.info('Explicit mode: Ensure ONNX contains Q/DQ layers (use TensorRT Model Optimizer for quantization).')
                    if not os.path.exists(onnx_path):
                        if trained_state_dict_path is None:
                            raise RuntimeError('Trained weights must be loaded for ONNX export.')
                        logger.info(f'ONNX not found at {onnx_path}. Exporting from PyTorch model...')
                        export_tc_nn_to_onnx(self.torch_wrapper.model, input_dim, onnx_path)
                    build_tensorrt_engine_from_onnx(
                        onnx_path,
                        engine_path,
                        max_batch_size=max_batch_size,
                        quantization_mode=quantization_mode
                    )
                else:
                    warnings.warn(f'Engine not found at {engine_path} and build_if_missing=False. Disabling TRT inference.')
                    self.use_trt = False

            if os.path.exists(engine_path):
                try:
                    self.trt_engine = TensorRTInferenceEngine(engine_path)
                    logger.info(f'TensorRT engine ({quantization_mode}) loaded successfully from {engine_path}.')
                    if validate_outputs:
                        self._validate_trt_vs_torch()
                except Exception as e:
                    warnings.warn(f'Failed to load TensorRT engine: {e}. Falling back to PyTorch inference.')
                    self.trt_engine = None
                    self.use_trt = False

    def _validate_trt_vs_torch(self):
        num_samples = 64
        dummy_input = np.random.randn(num_samples, self.input_dim).astype(np.float32)
        torch_probs = self.torch_wrapper.predict_proba(dummy_input)
        trt_logits = self.trt_engine.infer(dummy_input)
        trt_probs = self._softmax(trt_logits)

        mae = np.mean(np.abs(torch_probs - trt_probs))
        if self.quantization_mode == 'fp8':
            tol = 0.02
        elif self.quantization_mode == 'explicit':
            tol = 0.05
        else:
            tol = 0.005

        if mae > tol:
            warnings.warn(f'TRT vs PyTorch validation failed: MAE={mae:.6f} > tolerance {tol} (mode={self.quantization_mode}). Possible accuracy drop.')
        else:
            logger.info(f'TRT vs PyTorch validation passed: MAE={mae:.6f} (mode={self.quantization_mode}).')

    def _softmax(self, logits: np.ndarray) -> np.ndarray:
        logits_max = np.max(logits, axis=1, keepdims=True)
        exp_logits = np.exp(logits - logits_max)
        return exp_logits / np.sum(exp_logits, axis=1, keepdims=True)

    @property
    def is_trained(self) -> bool:
        return self.torch_wrapper.is_trained

    def fit(self, X_train: np.ndarray, y_train: np.ndarray, X_val: np.ndarray, y_val: np.ndarray, **kwargs):
        """Training proxy to underlying PyTorch wrapper"""
        return self.torch_wrapper.fit(X_train, y_train, X_val, y_val, **kwargs)

    def save(self, path: str):
        """Save proxy to underlying PyTorch wrapper"""
        return self.torch_wrapper.save(path)

    def load(self, path: str):
        """Load proxy to underlying PyTorch wrapper"""
        return self.torch_wrapper.load(path)

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        if X.ndim != 2 or X.shape[1] != self.input_dim:
            raise ValueError(f'Expected input shape [N, {self.input_dim}], got {X.shape}')

        if self.trt_engine is not None and self.use_trt:
            if X.shape[0] > self.max_batch_size:
                probs_list = []
                for i in range(0, X.shape[0], self.max_batch_size):
                    batch = X[i:i + self.max_batch_size].astype(np.float32)
                    logits = self.trt_engine.infer(batch)
                    probs_list.append(self._softmax(logits))
                return np.concatenate(probs_list, axis=0).astype(np.float32)
            else:
                logits = self.trt_engine.infer(X.astype(np.float32))
                return self._softmax(logits).astype(np.float32)

        return self.torch_wrapper.predict_proba(X)

    def predict(self, X: np.ndarray) -> Dict[str, Any]:
        if X.ndim != 2 or X.shape[0] != 1:
            raise ValueError('predict() expects a single sample (shape [1, input_dim]). Use predict_batch() for multiple.')
        probs = self.predict_proba(X)[0]
        pred_class = int(np.argmax(probs))
        return format_prediction(probs, pred_class, self.model_name)

    def predict_batch(self, X) -> List[Dict[str, Any]]:
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
        return [format_prediction(probs[i], int(preds[i]), self.model_name) for i in range(X.shape[0])]

    def evaluate(self, X_test: np.ndarray, y_test: np.ndarray) -> Dict[str, Any]:
        """
        Evaluate model on test set

        Args:
            X_test: Test features (n_samples, n_features)
            y_test: Test labels (n_samples,)

        Returns:
            Evaluation metrics dictionary
        """
        # Get predictions
        y_pred = np.argmax(self.predict_proba(X_test), axis=1)
        probabilities = self.predict_proba(X_test)

        # Calculate accuracy
        accuracy = float(np.mean(y_pred == y_test))

        # Calculate mean confidence
        confidence = float(np.max(probabilities, axis=1).mean())

        # Per-class accuracy
        class_accuracies = {}
        for cls in range(self.num_classes):
            mask = y_test == cls
            if mask.sum() > 0:
                class_acc = float(np.mean(y_pred[mask] == y_test[mask]))
                class_accuracies[cls] = class_acc

        return {
            'accuracy': accuracy,
            'mean_confidence': confidence,
            'n_samples': int(len(X_test)),
            'class_accuracies': class_accuracies
        }
