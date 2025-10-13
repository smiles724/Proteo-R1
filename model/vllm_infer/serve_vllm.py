"""
vLLM Server for PLLM Model

This script starts a vLLM server with OpenAI-compatible API for the PLLM model.
Since PLLM has protein encoders as prefix, we need a custom model wrapper.

Usage:
    python serve_vllm.py --model-path ./pllm --port 30000
"""

import argparse
from vllm import LLM, SamplingParams
from vllm.entrypoints.openai.api_server import run_server
import torch
from typing import List, Optional, Dict, Any


def main():
    parser = argparse.ArgumentParser(description="Serve PLLM model with vLLM")
    parser.add_argument("--model-path", type=str, default="./pllm",
                        help="Path to PLLM model")
    parser.add_argument("--host", type=str, default="0.0.0.0",
                        help="Host to bind to")
    parser.add_argument("--port", type=int, default=30000,
                        help="Port to bind to")
    parser.add_argument("--dtype", type=str, default="bfloat16",
                        choices=["float16", "bfloat16", "float32"],
                        help="Data type for model")
    parser.add_argument("--tensor-parallel-size", type=int, default=1,
                        help="Number of GPUs for tensor parallelism")
    parser.add_argument("--gpu-memory-utilization", type=float, default=0.90,
                        help="GPU memory utilization (0.0-1.0)")
    parser.add_argument("--max-model-len", type=int, default=None,
                        help="Maximum model length")
    
    args = parser.parse_args()
    
    print("="*80)
    print("Starting vLLM Server for PLLM")
    print("="*80)
    print(f"Model path: {args.model_path}")
    print(f"Host: {args.host}")
    print(f"Port: {args.port}")
    print(f"Dtype: {args.dtype}")
    print(f"Tensor parallel size: {args.tensor_parallel_size}")
    print(f"GPU memory utilization: {args.gpu_memory_utilization}")
    print("="*80)
    
    # Note: For PLLM with protein encoders, we need to use the base LLM
    # The protein encoding will be handled separately in the client
    
    # Get the base LLM path from PLLM
    import os
    llm_path = os.path.join(args.model_path, "llm")
    
    if not os.path.exists(llm_path):
        print(f"Error: LLM not found at {llm_path}")
        print("PLLM model should have an 'llm' subdirectory with the base model")
        return
    
    print(f"Loading base LLM from: {llm_path}")
    print()
    print("Note: This serves the base LLM only.")
    print("Protein encoding must be handled client-side.")
    print("For full PLLM integration, use the client wrapper.")
    print("="*80)
    
    # Start vLLM server with the base LLM
    # The protein encoding will be prepended as a prefix in the client
    import subprocess
    import sys
    
    cmd = [
        sys.executable, "-m", "vllm.entrypoints.openai.api_server",
        "--model", llm_path,
        "--host", args.host,
        "--port", str(args.port),
        "--dtype", args.dtype,
        "--tensor-parallel-size", str(args.tensor_parallel_size),
        "--gpu-memory-utilization", str(args.gpu_memory_utilization),
    ]
    
    if args.max_model_len:
        cmd.extend(["--max-model-len", str(args.max_model_len)])
    
    print("Starting vLLM server...")
    print(f"Command: {' '.join(cmd)}")
    print("="*80)
    
    subprocess.run(cmd)


if __name__ == "__main__":
    main()

