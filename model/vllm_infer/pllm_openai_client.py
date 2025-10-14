"""
Custom OpenAI-compatible client for PLLM that passes protein sequences.

This extends the standard OpenAI client to support protein-specific parameters.
"""

import os
from typing import Any, Dict, List, Optional

import openai
from openai import AsyncOpenAI


class PLLMOpenAIClient:
    """
    OpenAI-compatible client that supports protein sequences.
    
    Usage:
        client = PLLMOpenAIClient(base_url="http://localhost:30000/v1")
        response = await client.completions.create(
            model="pllm",
            prompt="Analyze this protein:",
            protein_sequence="MALVFV...",
            max_tokens=100
        )
    """
    
    def __init__(
        self,
        base_url: str = "http://localhost:30000/v1",
        api_key: str = "EMPTY",
    ):
        self.base_url = base_url
        self.api_key = api_key
        self.client = AsyncOpenAI(base_url=base_url, api_key=api_key)
    
    async def create_completion(
        self,
        model: str,
        prompt: str,
        protein_sequence: Optional[str] = None,
        structure_sequence: Optional[str] = None,
        **kwargs
    ) -> Any:
        """
        Create a completion with optional protein sequences.
        
        Args:
            model: Model name
            prompt: Text prompt
            protein_sequence: Optional amino acid sequence
            structure_sequence: Optional structure sequence (3Di)
            **kwargs: Additional OpenAI parameters
            
        Returns:
            Completion response
        """
        # Build request body
        body = {
            "model": model,
            "prompt": prompt,
            **kwargs
        }
        
        # Add protein sequences if provided
        if protein_sequence:
            body["protein_sequence"] = protein_sequence
        if structure_sequence:
            body["structure_sequence"] = structure_sequence
        
        # Make request
        response = await self.client.completions.create(**body)
        return response
    
    async def create_chat_completion(
        self,
        model: str,
        messages: List[Dict[str, str]],
        protein_sequence: Optional[str] = None,
        structure_sequence: Optional[str] = None,
        **kwargs
    ) -> Any:
        """
        Create a chat completion with optional protein sequences.
        
        Args:
            model: Model name
            messages: Chat messages
            protein_sequence: Optional amino acid sequence
            structure_sequence: Optional structure sequence (3Di)
            **kwargs: Additional OpenAI parameters
            
        Returns:
            Chat completion response
        """
        # Build request body
        body = {
            "model": model,
            "messages": messages,
            **kwargs
        }
        
        # Add protein sequences if provided
        if protein_sequence:
            body["protein_sequence"] = protein_sequence
        if structure_sequence:
            body["structure_sequence"] = structure_sequence
        
        # Make request
        response = await self.client.chat.completions.create(**body)
        return response

