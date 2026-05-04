"""Test DeepSeek R1 response structure."""
from openai import OpenAI

client = OpenAI(
    api_key="sk-b896056753ec440cb735873f0179bb67",
    base_url="https://api.deepseek.com",
)
resp = client.chat.completions.create(
    model="deepseek-reasoner",
    messages=[{"role": "user", "content": "Reply with exactly one word: Hello"}],
    max_tokens=128,
)
msg = resp.choices[0].message
print(f"content type: {type(msg.content)}")
print(f"content value: [{msg.content}]")
print(f"reasoning_content type: {type(getattr(msg, 'reasoning_content', None))}")
rc = getattr(msg, "reasoning_content", "") or ""
print(f"reasoning_content: [{rc[:200]}]")
print(f"finish_reason: {resp.choices[0].finish_reason}")
print(f"model: {resp.model}")
print(f"usage: {resp.usage}")
