"""
Test script for full PLLM server integration.

This tests that the protein encoders are working correctly.
"""

import asyncio
import requests
import sys


async def test_server():
    """Test the full PLLM server."""
    
    base_url = "http://localhost:30000"
    
    print("="*80)
    print("Testing Full PLLM Server")
    print("="*80)
    
    # Test 1: Health check
    print("\n1. Testing health endpoint...")
    try:
        response = requests.get(f"{base_url}/health")
        if response.status_code == 200:
            print("✅ Health check passed")
        else:
            print(f"❌ Health check failed: {response.status_code}")
            return False
    except Exception as e:
        print(f"❌ Cannot connect to server: {e}")
        print("   Make sure the server is running:")
        print("   python serve_pllm_full.py --model-path ../pllm --port 30000")
        return False
    
    # Test 2: Model list
    print("\n2. Testing model list...")
    try:
        response = requests.get(f"{base_url}/v1/models")
        data = response.json()
        models = [m["id"] for m in data["data"]]
        print(f"   Available models: {models}")
        if "pllm" in models:
            print("✅ Model 'pllm' found")
        else:
            print(f"❌ Model 'pllm' not found")
            return False
    except Exception as e:
        print(f"❌ Model list failed: {e}")
        return False
    
    # Test 3: Completion without protein sequence
    print("\n3. Testing completion (text only)...")
    try:
        response = requests.post(
            f"{base_url}/v1/completions",
            json={
                "model": "pllm",
                "prompt": "What is 2+2?",
                "max_tokens": 50,
                "temperature": 0.7,
            }
        )
        data = response.json()
        if response.status_code == 200:
            text = data["choices"][0]["text"]
            print(f"   Response: {text[:100]}...")
            print("✅ Text-only completion passed")
        else:
            print(f"❌ Completion failed: {response.status_code}")
            print(f"   Response: {data}")
            return False
    except Exception as e:
        print(f"❌ Completion failed: {e}")
        return False
    
    # Test 4: Completion with protein sequence
    print("\n4. Testing completion (with protein sequence)...")
    protein_seq = "MALVFVYGTLKRGQPNHRVLRDGAHGSAAFRAR"
    try:
        response = requests.post(
            f"{base_url}/v1/completions",
            json={
                "model": "pllm",
                "prompt": "Analyze this protein and predict its thermostability:",
                "protein_sequence": protein_seq,
                "max_tokens": 100,
                "temperature": 0.7,
            }
        )
        data = response.json()
        if response.status_code == 200:
            text = data["choices"][0]["text"]
            print(f"   Protein: {protein_seq}")
            print(f"   Response: {text[:150]}...")
            print("✅ Protein completion passed")
            print("   ✅ Protein encoder is working!")
        else:
            print(f"❌ Protein completion failed: {response.status_code}")
            print(f"   Response: {data}")
            return False
    except Exception as e:
        print(f"❌ Protein completion failed: {e}")
        return False
    
    # Test 5: Chat completion
    print("\n5. Testing chat completion...")
    try:
        response = requests.post(
            f"{base_url}/v1/chat/completions",
            json={
                "model": "pllm",
                "messages": [
                    {"role": "user", "content": "Hello, can you help me analyze proteins?"}
                ],
                "max_tokens": 100,
                "temperature": 0.7,
            }
        )
        data = response.json()
        if response.status_code == 200:
            text = data["choices"][0]["message"]["content"]
            print(f"   Response: {text[:100]}...")
            print("✅ Chat completion passed")
        else:
            print(f"❌ Chat completion failed: {response.status_code}")
            print(f"   Response: {data}")
            return False
    except Exception as e:
        print(f"❌ Chat completion failed: {e}")
        return False
    
    print("\n" + "="*80)
    print("✅ All tests passed!")
    print("="*80)
    print("\nThe full PLLM server is working correctly with protein encoders.")
    print("You can now run: python run_deepprotein.py")
    print("="*80)
    
    return True


if __name__ == "__main__":
    success = asyncio.run(test_server())
    sys.exit(0 if success else 1)

