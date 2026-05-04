"""Quick environment check for enhanced testing."""
import sys
print(f"Python: {sys.version}")
print(f"Path: {sys.executable}")

try:
    import torch
    print(f"PyTorch: {torch.__version__}")
    print(f"CUDA available: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"GPU: {torch.cuda.get_device_name(0)}")
except ImportError:
    print("PyTorch: NOT INSTALLED")

try:
    import openai
    print(f"OpenAI SDK: {openai.__version__}")
except ImportError:
    print("OpenAI SDK: NOT INSTALLED")

# Test DeepSeek R1 API connectivity
try:
    from openai import OpenAI
    client = OpenAI(
        api_key="sk-b896056753ec440cb735873f0179bb67",
        base_url="https://api.deepseek.com",
    )
    resp = client.chat.completions.create(
        model="deepseek-reasoner",
        messages=[{"role": "user", "content": "Reply with exactly: OK"}],
        max_tokens=16,
    )
    msg = resp.choices[0].message
    content = msg.content or ""
    reasoning = getattr(msg, "reasoning_content", "") or ""
    print(f"DeepSeek R1 API test: SUCCESS")
    print(f"  content: {content[:100]}")
    print(f"  reasoning snippet: {reasoning[:100]}")
except Exception as e:
    print(f"DeepSeek R1 API test: FAILED — {e}")
