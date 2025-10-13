import os
from typing import List, Dict, Tuple
from openai import OpenAI

# ✅ 兼容两种环境变量，优先使用 OPENAI_API_BASE
BASE_URL = (
    os.environ.get("OPENAI_API_BASE") or
    os.environ.get("OPENAI_BASE_URL") or
    "https://api.openai.com/v1"
)
API_KEY = os.environ.get("OPENAI_API_KEY")

if not API_KEY:
    raise RuntimeError("❌ 请先 export OPENAI_API_KEY='你的key'，并设置 OPENAI_API_BASE 或 OPENAI_BASE_URL")

client = OpenAI(api_key=API_KEY, base_url=BASE_URL)

def get_multiple_completion(
    batch_of_dialogs: List[List[Dict[str, str]]],
    model_name: str = "gpt-4o-mini",
    temperature: float = 0.0,
    max_tokens: int = 100
) -> Tuple[List[str], float]:
    """与本地 llm_adapter 同名接口：给一批对话，返回一批文本"""
    results = []
    total_cost = 0.0
    for dialogs in batch_of_dialogs:
        resp = client.chat.completions.create(
            model=model_name,
            messages=dialogs,
            temperature=temperature,
            max_tokens=max_tokens,
        )
        results.append(resp.choices[0].message.content)
    return results, total_cost
