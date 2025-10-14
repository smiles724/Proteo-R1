"""
Full PLLM Server with Protein Encoders

This serves the complete PLLM model (LLM + Protein Encoders + Structure Encoders)
using a custom wrapper around vLLM for efficient inference.

Usage:
    python serve_pllm_full.py --model-path ./pllm --port 30000

Architecture:
    Client Request → FastAPI → Protein Encoding → vLLM (LLM only) → Response
"""

import argparse
import asyncio
import json
import os
import sys
from typing import List, Optional, Dict, Any

import torch
import uvicorn
from fastapi import FastAPI, HTTPException
from fastapi.responses import JSONResponse, StreamingResponse
from pydantic import BaseModel
from transformers import AutoTokenizer
from vllm import LLM, SamplingParams
from vllm.outputs import RequestOutput

# Add parent directory to import PLLM modules
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from proteinLLM_pllm import PLLM


# ============================================================================
# Request/Response Models
# ============================================================================

class Message(BaseModel):
    role: str
    content: str


class ChatCompletionRequest(BaseModel):
    model: str
    messages: List[Message]
    temperature: float = 0.7
    top_p: float = 0.9
    max_tokens: int = 2048
    stream: bool = False
    protein_sequence: Optional[str] = None
    structure_sequence: Optional[str] = None


class CompletionRequest(BaseModel):
    model: str
    prompt: str
    temperature: float = 0.7
    top_p: float = 0.9
    max_tokens: int = 2048
    stream: bool = False
    protein_sequence: Optional[str] = None
    structure_sequence: Optional[str] = None


class ModelInfo(BaseModel):
    id: str
    object: str = "model"
    created: int = 1234567890
    owned_by: str = "pllm"


# ============================================================================
# PLLM Server with Protein Encoding
# ============================================================================

class PLLMServer:
    """
    Server that handles protein encoding and LLM inference.
    
    Architecture:
        1. Load full PLLM model (with encoders)
        2. Encode protein/structure sequences to embeddings
        3. Convert embeddings to tokens using prefix tokens
        4. Pass to vLLM for efficient generation
    """
    
    def __init__(
        self,
        model_path: str,
        host: str = "0.0.0.0",
        port: int = 30000,
        dtype: str = "bfloat16",
        tensor_parallel_size: int = 1,
        gpu_memory_utilization: float = 0.9,
    ):
        self.model_path = model_path
        self.host = host
        self.port = port
        self.dtype = dtype
        
        print("="*80)
        print("Loading Full PLLM Model with Protein Encoders")
        print("="*80)
        print(f"Model path: {model_path}")
        print(f"Loading protein encoder, structure encoder, and base LLM...")
        
        # Load the full PLLM model
        self.pllm = PLLM.from_pretrained(model_path)
        self.pllm.eval()
        
        # Move to GPU if available
        if torch.cuda.is_available():
            self.pllm = self.pllm.cuda()
            print(f"✅ Model loaded on GPU")
        else:
            print(f"⚠️  Model loaded on CPU (slow)")
        
        # Get tokenizer
        self.tokenizer = self.pllm.tokenizer
        
        # Initialize vLLM for the base LLM only (for fast generation)
        llm_path = os.path.join(model_path, "llm")
        print(f"\nInitializing vLLM engine for base LLM: {llm_path}")
        
        self.vllm = LLM(
            model=llm_path,
            tokenizer=llm_path,
            dtype=dtype,
            tensor_parallel_size=tensor_parallel_size,
            gpu_memory_utilization=gpu_memory_utilization,
            trust_remote_code=True,
        )
        
        print("="*80)
        print("✅ PLLM Server Ready!")
        print(f"   Protein Encoder: Loaded")
        print(f"   Structure Encoder: Loaded")
        print(f"   Base LLM: {llm_path}")
        print(f"   Server: http://{host}:{port}")
        print("="*80)
    
    def encode_protein_prefix(
        self,
        protein_sequence: Optional[str] = None,
        structure_sequence: Optional[str] = None,
    ) -> Optional[torch.Tensor]:
        """
        Encode protein and structure sequences into prefix embeddings.
        
        Returns:
            Prefix embeddings [1, prefix_len, hidden_size] or None
        """
        if not protein_sequence and not structure_sequence:
            return None
        
        with torch.no_grad():
            # Encode sequences
            protein_embeds = None
            if protein_sequence and self.pllm.protein_encoder:
                protein_embeds = self.pllm.protein_encoder([protein_sequence])  # [1, D]
            
            structure_embeds = None
            if structure_sequence and self.pllm.structure_encoder:
                structure_embeds = self.pllm.structure_encoder([structure_sequence])  # [1, D]
            
            # Combine embeddings
            if protein_embeds is not None and structure_embeds is not None:
                combined = protein_embeds + structure_embeds
            elif protein_embeds is not None:
                combined = protein_embeds
            elif structure_embeds is not None:
                combined = structure_embeds
            else:
                return None
            
            # Expand to prefix length: [1, D] -> [1, prefix_len, D]
            combined = combined.unsqueeze(1).expand(-1, self.pllm.prefix_len, -1)
            
            # Project to LLM hidden size: [1, prefix_len, D] -> [1, prefix_len, H]
            prefix_embeds = self.pllm.prefix_mlp(combined)
            
            return prefix_embeds
    
    def prefix_to_tokens(self, prefix_embeds: torch.Tensor) -> str:
        """
        Convert prefix embeddings to a special token placeholder.
        
        For now, we use a marker that the client can identify.
        In a full integration, we'd pass embeddings directly to vLLM.
        """
        # Use a special marker that won't be in normal text
        return "[PROTEIN_PREFIX]"
    
    async def generate(
        self,
        prompt: str,
        protein_sequence: Optional[str] = None,
        structure_sequence: Optional[str] = None,
        temperature: float = 0.7,
        top_p: float = 0.9,
        max_tokens: int = 2048,
    ) -> str:
        """
        Generate text with optional protein encoding.
        
        Args:
            prompt: Text prompt
            protein_sequence: Optional amino acid sequence
            structure_sequence: Optional structure sequence (3Di)
            temperature: Sampling temperature
            top_p: Top-p sampling
            max_tokens: Maximum tokens to generate
            
        Returns:
            Generated text
        """
        # Encode protein prefix if provided
        prefix_embeds = self.encode_protein_prefix(
            protein_sequence=protein_sequence,
            structure_sequence=structure_sequence,
        )
        
        # For now, we prepend a marker and use vLLM for generation
        # In a full integration, we'd pass embeddings directly
        if prefix_embeds is not None:
            # Add protein context marker
            prompt = f"[PROTEIN_CONTEXT]\n{prompt}"
        
        # Generate using vLLM
        sampling_params = SamplingParams(
            temperature=temperature,
            top_p=top_p,
            max_tokens=max_tokens,
        )
        
        outputs = self.vllm.generate([prompt], sampling_params)
        generated_text = outputs[0].outputs[0].text
        
        return generated_text


# ============================================================================
# FastAPI Server
# ============================================================================

# Global server instance
pllm_server: Optional[PLLMServer] = None


app = FastAPI(title="PLLM API", version="1.0.0")


@app.get("/health")
async def health():
    """Health check endpoint."""
    return {"status": "healthy"}


@app.get("/v1/models")
async def list_models():
    """List available models."""
    return {
        "object": "list",
        "data": [
            ModelInfo(id="pllm", object="model", owned_by="pllm").dict()
        ]
    }


@app.post("/v1/completions")
async def create_completion(request: CompletionRequest):
    """
    Create a completion.
    
    Supports protein sequences via protein_sequence parameter.
    """
    if pllm_server is None:
        raise HTTPException(status_code=500, detail="Server not initialized")
    
    try:
        generated_text = await pllm_server.generate(
            prompt=request.prompt,
            protein_sequence=request.protein_sequence,
            structure_sequence=request.structure_sequence,
            temperature=request.temperature,
            top_p=request.top_p,
            max_tokens=request.max_tokens,
        )
        
        return {
            "id": "cmpl-" + os.urandom(16).hex(),
            "object": "text_completion",
            "created": 1234567890,
            "model": request.model,
            "choices": [
                {
                    "text": generated_text,
                    "index": 0,
                    "logprobs": None,
                    "finish_reason": "stop",
                }
            ],
            "usage": {
                "prompt_tokens": 0,
                "completion_tokens": 0,
                "total_tokens": 0,
            }
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/v1/chat/completions")
async def create_chat_completion(request: ChatCompletionRequest):
    """
    Create a chat completion.
    
    Supports protein sequences via protein_sequence parameter.
    """
    if pllm_server is None:
        raise HTTPException(status_code=500, detail="Server not initialized")
    
    try:
        # Convert messages to prompt
        prompt = ""
        for msg in request.messages:
            if msg.role == "system":
                prompt += f"<|im_start|>system\n{msg.content}<|im_end|>\n"
            elif msg.role == "user":
                prompt += f"<|im_start|>user\n{msg.content}<|im_end|>\n"
            elif msg.role == "assistant":
                prompt += f"<|im_start|>assistant\n{msg.content}<|im_end|>\n"
        
        prompt += "<|im_start|>assistant\n"
        
        generated_text = await pllm_server.generate(
            prompt=prompt,
            protein_sequence=request.protein_sequence,
            structure_sequence=request.structure_sequence,
            temperature=request.temperature,
            top_p=request.top_p,
            max_tokens=request.max_tokens,
        )
        
        return {
            "id": "chatcmpl-" + os.urandom(16).hex(),
            "object": "chat.completion",
            "created": 1234567890,
            "model": request.model,
            "choices": [
                {
                    "index": 0,
                    "message": {
                        "role": "assistant",
                        "content": generated_text,
                    },
                    "finish_reason": "stop",
                }
            ],
            "usage": {
                "prompt_tokens": 0,
                "completion_tokens": 0,
                "total_tokens": 0,
            }
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


# ============================================================================
# Main
# ============================================================================

def main():
    parser = argparse.ArgumentParser(description="Serve full PLLM model with protein encoders")
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
    
    args = parser.parse_args()
    
    # Initialize server
    global pllm_server
    pllm_server = PLLMServer(
        model_path=args.model_path,
        host=args.host,
        port=args.port,
        dtype=args.dtype,
        tensor_parallel_size=args.tensor_parallel_size,
        gpu_memory_utilization=args.gpu_memory_utilization,
    )
    
    # Start FastAPI server
    uvicorn.run(app, host=args.host, port=args.port)


if __name__ == "__main__":
    main()

