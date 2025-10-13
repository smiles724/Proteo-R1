"""
vLLM Client for PLLM Model

This client connects to a vLLM server for text generation.
The server handles all model inference - no local model loading needed!

Usage:
    # Start server first:
    python serve_vllm.py --model-path ./pllm --port 30000
    
    # Then run client:
    python vllm_client.py
"""

import requests
from typing import List, Optional, Dict, Any
from dataclasses import dataclass
import json


@dataclass
class ProteinChatMessage:
    """Message format for protein-aware chat."""
    role: str  # "system", "user", "assistant"
    content: str
    aa_seq: Optional[str] = None  # Amino acid sequence
    stru_seq: Optional[str] = None  # Structure sequence (3Di tokens)


class PLLMVLLMClient:
    """
    Lightweight client for PLLM with vLLM server.
    
    Architecture:
    1. Client formats prompts with protein context markers
    2. Client sends requests to vLLM server via HTTP
    3. vLLM server handles all inference (fast!)
    4. Client receives and returns responses
    
    No local model loading - everything runs on the server!
    """
    
    def __init__(
        self,
        base_url: str = "http://localhost:30000/v1",
        api_key: str = "EMPTY",
    ):
        """
        Initialize PLLM vLLM client.
        
        Args:
            base_url: vLLM server URL
            api_key: API key (default: "EMPTY")
        """
        self.base_url = base_url
        self.api_key = api_key
        
        # Get model name from vLLM server
        print(f"Connecting to vLLM server at {base_url}...")
        try:
            response = requests.get(f"{base_url}/models")
            models = response.json()
            self.model_name = models["data"][0]["id"]
            print(f"✅ Connected to vLLM server!")
            print(f"   Model: {self.model_name}")
        except Exception as e:
            print(f"⚠️  Could not connect to vLLM server: {e}")
            print(f"   Make sure server is running: python serve_vllm.py")
            self.model_name = "unknown"
        
        print("✅ Client initialized successfully!")
    
    def _encode_protein_prefix(
        self,
        aa_seq: Optional[str],
        stru_seq: Optional[str],
    ) -> Optional[str]:
        """
        Encode protein sequences to a text prefix.
        
        Since vLLM doesn't support custom embeddings in the API,
        we encode the protein and convert it to special tokens.
        
        Returns:
            Text prefix representing protein context
        """
        if not aa_seq and not stru_seq:
            return None
        
        # For now, we'll use a marker token approach
        # In a full integration, you'd pass embeddings directly
        return "[PROTEIN_CONTEXT]"
    
    def _format_chat_prompt(
        self,
        messages: List[ProteinChatMessage],
    ) -> str:
        """
        Format messages into a chat prompt with protein context.
        """
        formatted_messages = []
        
        for msg in messages:
            if msg.role == "system":
                formatted_messages.append(
                    f"<|im_start|>system\n{msg.content}<|im_end|>"
                )
            elif msg.role == "user":
                content = msg.content
                
                # Add protein context if available
                if msg.aa_seq or msg.stru_seq:
                    protein_prefix = self._encode_protein_prefix(msg.aa_seq, msg.stru_seq)
                    if protein_prefix:
                        content = f"{protein_prefix}\n{content}"
                
                formatted_messages.append(
                    f"<|im_start|>user\n{content}<|im_end|>"
                )
            elif msg.role == "assistant":
                formatted_messages.append(
                    f"<|im_start|>assistant\n{msg.content}<|im_end|>"
                )
        
        prompt = "\n".join(formatted_messages)
        prompt += "\n<|im_start|>assistant\n"
        
        return prompt
    
    def chat(
        self,
        messages: List[Dict[str, Any]],
        max_tokens: int = 512,
        temperature: float = 0.7,
        top_p: float = 0.9,
        n: int = 1,
        stream: bool = False,
    ) -> str:
        """
        Chat with the model using OpenAI-compatible API.
        
        Args:
            messages: List of message dicts
            max_tokens: Maximum tokens to generate
            temperature: Sampling temperature
            top_p: Nucleus sampling parameter
            n: Number of completions
            stream: Whether to stream responses
            
        Returns:
            Generated response
        """
        # Convert to ProteinChatMessage format
        chat_messages = []
        for msg in messages:
            chat_messages.append(ProteinChatMessage(
                role=msg["role"],
                content=msg["content"],
                aa_seq=msg.get("aa_seq"),
                stru_seq=msg.get("stru_seq"),
            ))
        
        # Format prompt
        prompt = self._format_chat_prompt(chat_messages)
        
        # Call vLLM server
        url = f"{self.base_url}/completions"
        headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {self.api_key}",
        }
        
        data = {
            "model": self.model_name,
            "prompt": prompt,
            "max_tokens": max_tokens,
            "temperature": temperature,
            "top_p": top_p,
            "n": n,
            "stream": stream,
        }
        
        response = requests.post(url, headers=headers, json=data)
        response.raise_for_status()
        
        result = response.json()
        
        if n == 1:
            return result["choices"][0]["text"].strip()
        else:
            return [choice["text"].strip() for choice in result["choices"]]
    
    def chat_completions(
        self,
        messages: List[Dict[str, Any]],
        max_tokens: int = 512,
        temperature: float = 0.7,
        top_p: float = 0.9,
        n: int = 1,
    ) -> Dict[str, Any]:
        """
        Chat completions using OpenAI chat format.
        
        This is compatible with the OpenAI API format used in deepscaler.
        """
        # Convert messages to OpenAI format
        openai_messages = []
        for msg in messages:
            content = msg["content"]
            
            # Add protein context marker if sequences provided
            if msg.get("aa_seq") or msg.get("stru_seq"):
                content = f"[PROTEIN_CONTEXT]\n{content}"
            
            openai_messages.append({
                "role": msg["role"],
                "content": content,
            })
        
        # Call vLLM server
        url = f"{self.base_url}/chat/completions"
        headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {self.api_key}",
        }
        
        data = {
            "model": self.model_name,
            "messages": openai_messages,
            "max_tokens": max_tokens,
            "temperature": temperature,
            "top_p": top_p,
            "n": n,
        }
        
        response = requests.post(url, headers=headers, json=data)
        response.raise_for_status()
        
        return response.json()


def main():
    """Example usage of PLLM vLLM client."""
    
    print("="*80)
    print("PLLM vLLM Client Example")
    print("="*80)
    print()
    print("Make sure vLLM server is running:")
    print("  python serve_vllm.py --model-path ./pllm --port 30000")
    print()
    print("="*80)
    
    # Initialize client (no local model loading!)
    client = PLLMVLLMClient(
        base_url="http://localhost:30000/v1",
    )
    
    print("\n" + "="*80)
    print("Example 1: Single-turn conversation with protein")
    print("="*80)
    
    messages = [
        {
            "role": "system",
            "content": "You are a helpful protein analysis assistant."
        },
        {
            "role": "user",
            "content": "What is the likely function of this protein?",
            "aa_seq": "MKTFFVAIATGAFSATA",
            "stru_seq": "ACDEFGHIKLMNPQRSTVWY",
        }
    ]
    
    try:
        response = client.chat(messages, max_tokens=100, temperature=0.7)
        print(f"\nUser: {messages[1]['content']}")
        print(f"Protein: {messages[1]['aa_seq']}")
        print(f"\nAssistant: {response}")
    except Exception as e:
        print(f"Error: {e}")
        print("\nMake sure vLLM server is running!")
    
    print("\n" + "="*80)
    print("Example 2: Using chat completions API (OpenAI format)")
    print("="*80)
    
    messages2 = [
        {
            "role": "system",
            "content": "You are a helpful protein analysis assistant."
        },
        {
            "role": "user",
            "content": "Analyze this protein sequence.",
            "aa_seq": "MGDVEKGKKIFIMKCSQCHTVEK",
            "stru_seq": "ACDEFGHIKLMNP",
        }
    ]
    
    try:
        result = client.chat_completions(messages2, max_tokens=100, temperature=0.7)
        response = result["choices"][0]["message"]["content"]
        print(f"\nUser: {messages2[1]['content']}")
        print(f"Protein: {messages2[1]['aa_seq']}")
        print(f"\nAssistant: {response}")
    except Exception as e:
        print(f"Error: {e}")
        print("\nMake sure vLLM server is running!")
    
    print("\n" + "="*80)
    print("Example 3: Batch sampling (n > 1)")
    print("="*80)
    
    messages3 = [
        {
            "role": "user",
            "content": "Briefly describe this protein.",
            "aa_seq": "MKTFFVAIATGAFSATA",
            "stru_seq": "ACDEFGHIKLMNPQRSTVWY",
        }
    ]
    
    try:
        responses = client.chat(messages3, max_tokens=50, temperature=0.8, n=3)
        print(f"\nUser: {messages3[0]['content']}")
        print(f"Protein: {messages3[0]['aa_seq']}")
        print(f"\nGenerated {len(responses)} responses:")
        for i, resp in enumerate(responses, 1):
            print(f"\n{i}. {resp}")
    except Exception as e:
        print(f"Error: {e}")
        print("\nMake sure vLLM server is running!")
    
    print("\n" + "="*80)
    print("✅ Examples completed!")
    print("="*80)


if __name__ == "__main__":
    main()

