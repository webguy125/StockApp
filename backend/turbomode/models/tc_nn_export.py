"""
TurboCoreNN Export Utilities
ONNX and TensorRT export with explicit quantization support

Supports FP16 (default), FP8 (Hopper+), and explicit INT8/FP8 quantization.
Implicit INT8 calibration removed due to TensorRT deprecation.

For explicit quantization, use NVIDIA TensorRT Model Optimizer to generate
pre-quantized ONNX models with Q/DQ layers via PTQ or QAT.
"""

import os
import torch
import numpy as np
import time
import logging
from typing import Optional, Tuple
from pathlib import Path

logger = logging.getLogger(__name__)

# Optional ONNX imports
try:
    import onnx
    import onnxruntime as ort
    ONNX_AVAILABLE = True
except ImportError:
    ONNX_AVAILABLE = False
    onnx = None
    ort = None

# Optional TensorRT imports
try:
    import tensorrt as trt
    TENSORRT_AVAILABLE = True
except ImportError:
    TENSORRT_AVAILABLE = False
    trt = None

# Optional PyCUDA imports
try:
    import pycuda.driver as cuda
    import pycuda.autoinit  # noqa: F401
    PYCUDA_AVAILABLE = True
except ImportError:
    PYCUDA_AVAILABLE = False
    cuda = None


def get_gpu_arch() -> Optional[str]:
    """
    Get GPU compute capability (SM version) for engine naming

    Returns:
        GPU arch string (e.g., 'sm89' for Ada, 'sm90' for Hopper) or None
    """
    if not torch.cuda.is_available():
        return None

    try:
        # Get compute capability
        major, minor = torch.cuda.get_device_capability()
        return f"sm{major}{minor}"
    except:
        return None


def export_tc_nn_to_onnx(
    model: torch.nn.Module,
    input_size: int,
    output_path: str,
    opset_version: int = 18  # Updated for better quantization support
) -> bool:
    """
    Export TurboCoreNN model to ONNX format

    Args:
        model: Trained TurboCoreNN model
        input_size: Number of input features
        output_path: Path to save ONNX file
        opset_version: ONNX opset version (18+ recommended for quantization)

    Returns:
        True if successful
    """
    if not ONNX_AVAILABLE:
        logger.error("ONNX not available. Install with: pip install onnx onnxruntime")
        return False

    try:
        # Set model to evaluation mode
        model.eval()

        # Create dummy input
        dummy_input = torch.randn(1, input_size)

        # Export to ONNX
        torch.onnx.export(
            model,
            dummy_input,
            output_path,
            export_params=True,
            opset_version=opset_version,
            do_constant_folding=True,
            input_names=['input'],
            output_names=['output'],
            dynamic_axes={
                'input': {0: 'batch_size'},
                'output': {0: 'batch_size'}
            }
        )

        # Verify ONNX model
        onnx_model = onnx.load(output_path)
        onnx.checker.check_model(onnx_model)

        logger.info(f"Model exported to ONNX: {output_path}")
        return True

    except Exception as e:
        logger.error(f"Failed to export to ONNX: {e}")
        return False


def build_tensorrt_engine_from_onnx(
    onnx_path: str,
    engine_path: str,
    max_batch_size: int = 8192,
    quantization_mode: str = 'fp16'  # 'fp16', 'fp8', 'explicit'
) -> bool:
    """
    Build TensorRT engine from ONNX model with explicit quantization support

    **Quantization Modes:**
    - 'fp16': FP16 precision (default, best accuracy/speed balance)
    - 'fp8': FP8 precision (requires Hopper H100+ GPUs, experimental)
    - 'explicit': Explicit INT8/FP8 via Q/DQ layers in ONNX (use TensorRT Model Optimizer)

    **Hardware Requirements:**
    - FP16: Any modern NVIDIA GPU (Pascal+)
    - FP8: NVIDIA Hopper architecture (H100, H800) or newer
    - Explicit INT8: Any GPU with INT8 tensor cores (Turing+)

    **For Explicit Quantization:**
    Use NVIDIA TensorRT Model Optimizer to generate pre-quantized ONNX:
    ```
    pip install nvidia-modelopt[torch]
    # Then use PTQ or QAT to create quantized ONNX with Q/DQ layers
    ```

    Args:
        onnx_path: Path to ONNX model file
        engine_path: Path to save TensorRT engine
        max_batch_size: Maximum batch size for inference
        quantization_mode: Quantization mode ('fp16', 'fp8', 'explicit')

    Returns:
        True if successful
    """
    if not TENSORRT_AVAILABLE:
        logger.error("TensorRT is not available")
        return False

    # Validate quantization mode
    if quantization_mode.startswith('implicit'):
        raise ValueError(
            "Implicit quantization (INT8 calibration) is deprecated in recent TensorRT versions. "
            "Use 'explicit' mode with pre-quantized ONNX (Q/DQ layers) instead. "
            "Generate quantized ONNX using TensorRT Model Optimizer."
        )

    logger_trt = trt.Logger(trt.Logger.WARNING)
    builder = trt.Builder(logger_trt)
    network_flags = 1 << int(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH)
    network = builder.create_network(network_flags)
    parser = trt.OnnxParser(network, logger_trt)

    # Parse ONNX model
    with open(onnx_path, 'rb') as f:
        if not parser.parse(f.read()):
            logger.error("Failed to parse ONNX model:")
            for i in range(parser.num_errors):
                logger.error(f"  {parser.get_error(i)}")
            return False

    config = builder.create_builder_config()
    config.set_memory_pool_limit(trt.MemoryPoolType.WORKSPACE, 1 << 30)  # 1GB

    # Configure quantization mode
    if quantization_mode == 'fp16':
        config.set_flag(trt.BuilderFlag.FP16)
        logger.info("Building with FP16 precision")

    elif quantization_mode == 'fp8':
        # Check FP8 support
        try:
            if hasattr(trt.BuilderFlag, 'FP8'):
                config.set_flag(trt.BuilderFlag.FP8)
                logger.info("Building with FP8 precision (requires Hopper H100+)")
            else:
                logger.warning(
                    "FP8 not supported in this TensorRT version. "
                    "Falling back to FP16. Upgrade to TensorRT 8.6+ for FP8 support."
                )
                config.set_flag(trt.BuilderFlag.FP16)
        except AttributeError:
            logger.warning("FP8 not supported on this hardware. Falling back to FP16.")
            config.set_flag(trt.BuilderFlag.FP16)

    elif quantization_mode == 'explicit':
        # No flags needed for explicit quantization
        # Quantization controlled by Q/DQ layers in ONNX model
        logger.info(
            "Building with explicit quantization (Q/DQ layers in ONNX). "
            "Ensure ONNX was generated with TensorRT Model Optimizer for best results."
        )
    else:
        raise ValueError(
            f"Unsupported quantization_mode: {quantization_mode}. "
            f"Use 'fp16', 'fp8', or 'explicit'."
        )

    # Common optimizations
    config.set_flag(trt.BuilderFlag.PREFER_PRECISION_CONSTRAINTS)

    # Set optimization profile for dynamic batch sizes
    profile = builder.create_optimization_profile()
    input_name = network.get_input(0).name
    input_shape = network.get_input(0).shape
    _, input_dim = input_shape

    profile.set_shape(
        input_name,
        (1, input_dim),                      # min shape
        (max_batch_size // 4, input_dim),    # opt shape
        (max_batch_size, input_dim)          # max shape
    )
    config.add_optimization_profile(profile)

    # Build engine
    logger.info(f"Building TensorRT engine (mode={quantization_mode}, this may take a few minutes)...")
    serialized_engine = builder.build_serialized_network(network, config)

    if serialized_engine is None:
        logger.error("Failed to build TensorRT engine")
        return False

    # Save engine
    engine_path = Path(engine_path)
    engine_path.parent.mkdir(parents=True, exist_ok=True)

    with open(engine_path, 'wb') as f:
        f.write(serialized_engine)

    logger.info(f"TensorRT engine ({quantization_mode}) built successfully: {engine_path}")
    return True


class TensorRTInferenceEngine:
    """
    TensorRT inference engine for TurboCoreNN

    Provides optimized inference using TensorRT engine built from ONNX model.
    Supports FP16, FP8, and explicit INT8 quantization.
    """

    def __init__(self, engine_path: str):
        """
        Initialize TensorRT inference engine

        Args:
            engine_path: Path to TensorRT engine file
        """
        if not TENSORRT_AVAILABLE:
            raise ImportError("TensorRT not available. Install TensorRT Python bindings.")

        if not PYCUDA_AVAILABLE:
            raise ImportError("PyCUDA not available. Install with: pip install pycuda")

        self.engine_path = engine_path
        self.engine = None
        self.context = None
        self.logger = trt.Logger(trt.Logger.WARNING)

        self._load_engine()

    def _load_engine(self):
        """Load TensorRT engine from file"""
        try:
            with open(self.engine_path, 'rb') as f:
                runtime = trt.Runtime(self.logger)
                self.engine = runtime.deserialize_cuda_engine(f.read())

            if self.engine is None:
                raise RuntimeError("Failed to deserialize engine")

            self.context = self.engine.create_execution_context()

            logger.info(f"TensorRT engine loaded: {self.engine_path}")

        except Exception as e:
            raise RuntimeError(f"Failed to load TensorRT engine: {e}")

    def infer(self, input_data: np.ndarray) -> np.ndarray:
        """
        Run inference with TensorRT engine

        Args:
            input_data: Input features (batch_size, n_features)

        Returns:
            Output predictions (batch_size, num_classes)
        """
        # Allocate device memory
        batch_size = input_data.shape[0]
        d_input = cuda.mem_alloc(input_data.nbytes)
        output_shape = (batch_size, 3)  # 3 classes
        output = np.empty(output_shape, dtype=np.float32)
        d_output = cuda.mem_alloc(output.nbytes)

        # Copy input to device
        cuda.memcpy_htod(d_input, input_data.astype(np.float32))

        # Run inference
        bindings = [int(d_input), int(d_output)]
        self.context.execute_v2(bindings=bindings)

        # Copy output from device
        cuda.memcpy_dtoh(output, d_output)

        return output

    def benchmark(self, input_data: np.ndarray, num_runs: int = 100) -> dict:
        """
        Benchmark inference speed

        Args:
            input_data: Sample input for benchmarking
            num_runs: Number of inference runs

        Returns:
            Dictionary with timing statistics
        """
        # Warmup
        for _ in range(10):
            self.infer(input_data)

        # Benchmark
        times = []
        for _ in range(num_runs):
            start = time.perf_counter()
            self.infer(input_data)
            end = time.perf_counter()
            times.append((end - start) * 1000)  # ms

        times = np.array(times)

        return {
            'mean_ms': float(np.mean(times)),
            'std_ms': float(np.std(times)),
            'min_ms': float(np.min(times)),
            'max_ms': float(np.max(times)),
            'throughput_samples_per_sec': float(input_data.shape[0] / (np.mean(times) / 1000))
        }

    def __del__(self):
        """Cleanup resources"""
        if self.context:
            del self.context
        if self.engine:
            del self.engine


def export_tc_nn_full_pipeline(
    model: torch.nn.Module,
    input_size: int,
    output_dir: str,
    model_name: str = "tc_nn",
    quantization_mode: str = 'fp16',
    max_batch_size: int = 8192,
    build_tensorrt: bool = False
) -> Tuple[bool, bool]:
    """
    Export TurboCoreNN through full pipeline: PyTorch -> ONNX -> TensorRT

    Args:
        model: Trained TurboCoreNN model
        input_size: Number of input features
        output_dir: Directory to save exported models
        model_name: Base name for exported files
        quantization_mode: 'fp16', 'fp8', or 'explicit'
        max_batch_size: Maximum batch size for TensorRT engine
        build_tensorrt: Build TensorRT engine (requires CUDA)

    Returns:
        (onnx_success, tensorrt_success)
    """
    os.makedirs(output_dir, exist_ok=True)

    # Export to ONNX
    onnx_path = os.path.join(output_dir, f"{model_name}.onnx")
    onnx_success = export_tc_nn_to_onnx(model, input_size, onnx_path)

    tensorrt_success = False
    if onnx_success and build_tensorrt:
        # Get GPU architecture for engine naming
        gpu_arch = get_gpu_arch()
        arch_suffix = f".{gpu_arch}" if gpu_arch else ""

        # Build TensorRT engine with GPU-specific naming
        engine_path = os.path.join(
            output_dir,
            f"{model_name}.{quantization_mode}{arch_suffix}.engine"
        )

        if quantization_mode == 'explicit':
            logger.info(
                "For explicit quantization, ensure ONNX was pre-quantized with "
                "TensorRT Model Optimizer (pip install nvidia-modelopt[torch]). "
                "ONNX should contain Q/DQ layers from PTQ or QAT."
            )

        tensorrt_success = build_tensorrt_engine_from_onnx(
            onnx_path,
            engine_path,
            max_batch_size,
            quantization_mode
        )

    return onnx_success, tensorrt_success
