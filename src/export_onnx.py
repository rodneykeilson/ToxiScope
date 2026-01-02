#!/usr/bin/env python3
"""
ToxiScope: ONNX Export for Chrome Extension Deployment

Exports trained DeBERTa/RoBERTa model to ONNX format for efficient browser inference.

Features:
- Dynamic input shapes
- FP16/INT8 quantization options
- Inference validation
- Performance benchmarking

Usage:
    python export_onnx.py --model outputs/models/deberta_base/best
    python export_onnx.py --model outputs/models/deberta_base/best --quantize int8
"""

import argparse
import json
import os
import sys
import time
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import torch
import torch.nn as nn

# ONNX
try:
    import onnx
    import onnxruntime as ort
    from onnxruntime.quantization import quantize_dynamic, QuantType
    ONNX_AVAILABLE = True
except ImportError:
    ONNX_AVAILABLE = False
    print("Warning: onnx/onnxruntime not installed. Install with: pip install onnx onnxruntime")

# HuggingFace
from transformers import AutoTokenizer, AutoModelForSequenceClassification


class ONNXWrapper(nn.Module):
    """
    Wrapper for ONNX export that returns only logits.
    """
    
    def __init__(self, model):
        super().__init__()
        self.model = model
    
    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
    ) -> torch.Tensor:
        outputs = self.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
        )
        return outputs.logits


def export_to_onnx(
    model_dir: str,
    output_path: str,
    max_length: int = 256,
    opset_version: int = 14,
) -> str:
    """
    Export HuggingFace model to ONNX format.
    
    Args:
        model_dir: Path to saved model directory
        output_path: Path for output ONNX file
        max_length: Maximum sequence length
        opset_version: ONNX opset version
        
    Returns:
        Path to exported ONNX file
    """
    if not ONNX_AVAILABLE:
        raise ImportError("onnx and onnxruntime required for export")
    
    model_path = Path(model_dir)
    if not model_path.exists():
        raise FileNotFoundError(f"Model directory not found: {model_path}")
    
    if (model_path / "best_model").exists():
        model_path = model_path / "best_model"
    
    print(f"Loading model from {model_path}...")
    
    # Load model using HuggingFace standard API
    model = AutoModelForSequenceClassification.from_pretrained(str(model_path))
    model.eval()
    
    # Wrap for ONNX export (returns only logits)
    wrapper = ONNXWrapper(model)
    wrapper.eval()
    
    # Load tokenizer with fallback for tiktoken compatibility
    try:
        tokenizer = AutoTokenizer.from_pretrained(str(model_path))
    except (ValueError, AttributeError):
        tokenizer = AutoTokenizer.from_pretrained(str(model_path), use_fast=False)
    
    # Create dummy input
    dummy_text = "This is a sample sentence for exporting the model."
    dummy_input = tokenizer(
        dummy_text,
        return_tensors="pt",
        max_length=max_length,
        padding="max_length",
        truncation=True,
    )
    
    input_ids = dummy_input["input_ids"]
    attention_mask = dummy_input["attention_mask"]
    
    print(f"Input shape: {input_ids.shape}")
    print(f"Exporting to ONNX (opset {opset_version})...")
    
    # Ensure output directory exists
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    
    # Export
    torch.onnx.export(
        wrapper,
        (input_ids, attention_mask),
        output_path,
        input_names=["input_ids", "attention_mask"],
        output_names=["logits"],
        dynamic_axes={
            "input_ids": {0: "batch_size", 1: "sequence_length"},
            "attention_mask": {0: "batch_size", 1: "sequence_length"},
            "logits": {0: "batch_size"},
        },
        opset_version=opset_version,
        do_constant_folding=True,
    )
    
    print(f"Exported to {output_path}")
    
    # Validate
    print("Validating ONNX model...")
    onnx_model = onnx.load(output_path)
    onnx.checker.check_model(onnx_model)
    print("✓ ONNX model is valid!")
    
    # Print model size
    size_mb = Path(output_path).stat().st_size / (1024 * 1024)
    print(f"Model size: {size_mb:.2f} MB")
    
    return output_path


def quantize_model(
    onnx_path: str,
    output_path: str,
    quant_type: str = "int8",
) -> str:
    """
    Apply dynamic quantization to ONNX model.
    
    Args:
        onnx_path: Path to input ONNX file
        output_path: Path for quantized output
        quant_type: Quantization type ('int8' or 'uint8')
        
    Returns:
        Path to quantized model
    """
    if not ONNX_AVAILABLE:
        raise ImportError("onnxruntime required for quantization")
    
    print(f"Quantizing model ({quant_type})...")
    
    # Determine quantization type
    qtype = QuantType.QInt8 if quant_type == "int8" else QuantType.QUInt8
    
    # Apply dynamic quantization
    quantize_dynamic(
        onnx_path,
        output_path,
        weight_type=qtype,
    )
    
    # Print size comparison
    orig_size = Path(onnx_path).stat().st_size / (1024 * 1024)
    quant_size = Path(output_path).stat().st_size / (1024 * 1024)
    reduction = (1 - quant_size / orig_size) * 100
    
    print(f"Original size: {orig_size:.2f} MB")
    print(f"Quantized size: {quant_size:.2f} MB")
    print(f"Size reduction: {reduction:.1f}%")
    
    return output_path


def benchmark_inference(
    onnx_path: str,
    tokenizer_path: str,
    num_samples: int = 100,
    max_length: int = 256,
) -> Dict[str, float]:
    """
    Benchmark ONNX inference speed.
    
    Args:
        onnx_path: Path to ONNX model
        tokenizer_path: Path to tokenizer
        num_samples: Number of samples to benchmark
        max_length: Maximum sequence length
        
    Returns:
        Dictionary with timing statistics
    """
    print(f"\nBenchmarking inference ({num_samples} samples)...")
    
    # Load ONNX model
    session = ort.InferenceSession(onnx_path, providers=["CPUExecutionProvider"])
    
    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_path)
    
    # Sample texts
    test_texts = [
        "You're such a noob, uninstall the game",
        "Great play! That was amazing",
        "This team is garbage, ff at 15",
        "Can you help me understand this mechanic?",
        "Kill yourself you worthless piece of trash",
    ] * (num_samples // 5 + 1)
    test_texts = test_texts[:num_samples]
    
    # Warm up
    for text in test_texts[:5]:
        inputs = tokenizer(text, return_tensors="np", max_length=max_length, 
                          padding="max_length", truncation=True)
        session.run(None, {
            "input_ids": inputs["input_ids"],
            "attention_mask": inputs["attention_mask"],
        })
    
    # Benchmark
    times = []
    for text in test_texts:
        inputs = tokenizer(text, return_tensors="np", max_length=max_length,
                          padding="max_length", truncation=True)
        
        start = time.perf_counter()
        session.run(None, {
            "input_ids": inputs["input_ids"],
            "attention_mask": inputs["attention_mask"],
        })
        end = time.perf_counter()
        
        times.append((end - start) * 1000)  # Convert to ms
    
    # Statistics
    stats = {
        "mean_ms": np.mean(times),
        "std_ms": np.std(times),
        "min_ms": np.min(times),
        "max_ms": np.max(times),
        "p50_ms": np.percentile(times, 50),
        "p95_ms": np.percentile(times, 95),
        "p99_ms": np.percentile(times, 99),
        "throughput": 1000 / np.mean(times),  # samples/sec
    }
    
    print(f"Mean latency: {stats['mean_ms']:.2f} ms")
    print(f"P95 latency: {stats['p95_ms']:.2f} ms")
    print(f"Throughput: {stats['throughput']:.1f} samples/sec")
    
    return stats


def validate_outputs(
    pytorch_model_path: str,
    onnx_path: str,
    test_texts: Optional[List[str]] = None,
    rtol: float = 1e-3,
    atol: float = 1e-5,
) -> bool:
    """
    Validate ONNX outputs match PyTorch outputs.
    
    Args:
        pytorch_model_path: Path to PyTorch model
        onnx_path: Path to ONNX model
        test_texts: Optional list of test texts
        rtol: Relative tolerance
        atol: Absolute tolerance
        
    Returns:
        True if outputs match
    """
    print("\nValidating output consistency...")
    
    model_path = Path(pytorch_model_path)
    if (model_path / "best_model").exists():
        model_path = model_path / "best_model"
    
    # Load PyTorch model
    model = AutoModelForSequenceClassification.from_pretrained(str(model_path))
    model.eval()
    
    tokenizer = AutoTokenizer.from_pretrained(str(model_path))
    
    # Load ONNX model
    session = ort.InferenceSession(onnx_path, providers=["CPUExecutionProvider"])
    
    # Default test texts
    if test_texts is None:
        test_texts = [
            "You're such a noob, uninstall the game",
            "Great play! That was amazing",
            "This team is garbage",
            "Can you help me understand?",
            "Kill yourself you trash",
        ]
    
    all_close = True
    max_diff = 0.0
    
    for text in test_texts[:10]:
        inputs = tokenizer(
            text,
            return_tensors="pt",
            max_length=256,
            padding="max_length",
            truncation=True,
        )
        
        # PyTorch inference
        with torch.no_grad():
            pt_outputs = model(
                input_ids=inputs["input_ids"],
                attention_mask=inputs["attention_mask"],
            )
            pt_logits = pt_outputs.logits.numpy()
        
        # ONNX inference
        onnx_logits = session.run(
            None,
            {
                "input_ids": inputs["input_ids"].numpy(),
                "attention_mask": inputs["attention_mask"].numpy(),
            }
        )[0]
        
        # Compare
        diff = np.abs(pt_logits - onnx_logits).max()
        max_diff = max(max_diff, diff)
        
        if not np.allclose(pt_logits, onnx_logits, rtol=rtol, atol=atol):
            all_close = False
            print(f"  ⚠ Mismatch for: {text[:50]}...")
            print(f"    PyTorch: {pt_logits[0][:3]}...")
            print(f"    ONNX:    {onnx_logits[0][:3]}...")
    
    print(f"Maximum difference: {max_diff:.6f}")
    
    if all_close:
        print("✓ All outputs match within tolerance!")
    else:
        print("⚠ Some outputs differ (may still be acceptable)")
    
    return all_close


def export_for_extension(
    model_dir: str,
    output_dir: str,
    quantize: bool = True,
    max_length: int = 256,
) -> Dict[str, str]:
    """
    Complete export pipeline for Chrome extension deployment.
    
    Args:
        model_dir: Path to trained model
        output_dir: Output directory for extension assets
        quantize: Whether to apply INT8 quantization
        max_length: Maximum sequence length
        
    Returns:
        Dictionary with paths to exported files
    """
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    print("="*60)
    print("ToxiScope Export for Chrome Extension")
    print("="*60)
    
    # Paths
    onnx_path = output_path / "model.onnx"
    quant_path = output_path / "model_quantized.onnx"
    
    # 1. Export to ONNX
    print("\n[1/4] Exporting to ONNX...")
    export_to_onnx(model_dir, str(onnx_path), max_length=max_length)
    
    # 2. Quantize (optional)
    final_model_path = str(onnx_path)
    if quantize:
        print("\n[2/4] Quantizing model...")
        quantize_model(str(onnx_path), str(quant_path))
        final_model_path = str(quant_path)
    else:
        print("\n[2/4] Skipping quantization")
    
    # 3. Validate
    print("\n[3/4] Validating outputs...")
    validate_outputs(model_dir, final_model_path)
    
    # 4. Benchmark
    print("\n[4/4] Benchmarking...")
    model_path = Path(model_dir)
    if (model_path / "best_model").exists():
        tokenizer_path = str(model_path / "best_model")
    else:
        tokenizer_path = str(model_path)
    
    stats = benchmark_inference(final_model_path, tokenizer_path)
    
    # Save metadata
    metadata = {
        "model_path": final_model_path,
        "max_length": max_length,
        "quantized": quantize,
        "benchmark": stats,
    }
    
    # Copy tokenizer files needed for extension
    print("\nCopying tokenizer files...")
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_path)
    tokenizer.save_pretrained(str(output_path))
    
    # Save metadata
    with open(output_path / "metadata.json", "w") as f:
        json.dump(metadata, f, indent=2)
    
    print("\n" + "="*60)
    print("Export complete!")
    print(f"Output directory: {output_path}")
    print(f"Model: {final_model_path}")
    print(f"Mean latency: {stats['mean_ms']:.2f} ms")
    print("="*60)
    
    return {
        "model": final_model_path,
        "tokenizer": str(output_path),
        "metadata": str(output_path / "metadata.json"),
    }


def main():
    """CLI interface."""
    parser = argparse.ArgumentParser(description="ToxiScope ONNX Export")
    parser.add_argument("--model", type=str, required=True,
                        help="Path to trained model directory")
    parser.add_argument("--output", type=str, default="outputs/onnx",
                        help="Output directory")
    parser.add_argument("--quantize", action="store_true",
                        help="Apply INT8 quantization")
    parser.add_argument("--max-length", type=int, default=256,
                        help="Maximum sequence length")
    parser.add_argument("--benchmark", action="store_true",
                        help="Run benchmark only (requires existing ONNX)")
    parser.add_argument("--validate", action="store_true",
                        help="Validate only (requires existing ONNX)")
    
    args = parser.parse_args()
    
    if not ONNX_AVAILABLE:
        print("Error: onnx and onnxruntime required")
        print("Install with: pip install onnx onnxruntime")
        sys.exit(1)
    
    if args.benchmark:
        onnx_path = Path(args.output) / "model_quantized.onnx"
        if not onnx_path.exists():
            onnx_path = Path(args.output) / "model.onnx"
        benchmark_inference(str(onnx_path), args.model)
    elif args.validate:
        onnx_path = Path(args.output) / "model_quantized.onnx"
        if not onnx_path.exists():
            onnx_path = Path(args.output) / "model.onnx"
        validate_outputs(args.model, str(onnx_path))
    else:
        export_for_extension(
            args.model,
            args.output,
            quantize=args.quantize,
            max_length=args.max_length,
        )


if __name__ == "__main__":
    main()
