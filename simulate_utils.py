import sys
import numpy as np
import matplotlib.pyplot as plt
import yaml
import pandas as pd
import seaborn as sns
import re
import os
import multiprocessing
import scipy

save_path = './qwen_result/'

brackets = list(np.array([0, 97, 394.75, 842, 1607.25, 2041, 5103])*100/12)
quantiles = [0, 0.25, 0.5, 0.75, 1.0]

from datetime import datetime
world_start_time = datetime.strptime('2001.01', '%Y.%m')

prompt_cost_1k, completion_cost_1k = 0.001, 0.002
# Try to read API key from environment first, then fallback to config.yaml
def _load_openai_api_key():
    api_key = os.environ.get('OPENAI_API_KEY')
    if api_key:
        return api_key
    try:
        import os as _os
        _cfg_path = _os.path.join(_os.path.dirname(__file__), 'config.yaml')
        with open(_cfg_path, 'r') as f:
            cfg = yaml.safe_load(f) or {}
        # support either top-level or nested under 'openai'
        if 'OPENAI_API_KEY' in cfg:
            return cfg.get('OPENAI_API_KEY')
        if 'openai_api_key' in cfg:
            return cfg.get('openai_api_key')
        if isinstance(cfg.get('openai'), dict):
            maybe = cfg['openai'].get('api_key') or cfg['openai'].get('OPENAI_API_KEY')
            if maybe:
                return maybe
    except Exception:
        pass
    return None

_OPENAI_API_KEY = _load_openai_api_key()

def prettify_document(document: str) -> str:
    # Remove sequences of whitespace characters (including newlines)
    cleaned = re.sub(r'\s+', ' ', document).strip()
    return cleaned


def get_multiple_completion(dialogs, num_cpus=15, temperature=0, max_tokens=100, model_name=None):
    processes = max(1, min(num_cpus, len(dialogs)))
    args = [(d, temperature, max_tokens, model_name) for d in dialogs]
    with multiprocessing.Pool(processes=processes) as pool:
        results = pool.starmap(get_completion, args)
    total_cost = sum(cost for _, cost in results)
    return [resp for resp, _ in results], total_cost

def get_completion(dialogs, temperature=0, max_tokens=100, model_name=None):
    import os
    import time
    from openai import OpenAI

    # ✅ 设置默认的 DashScope OpenAI 兼容接口
    api_base = os.getenv(
        "OPENAI_API_BASE",
        "https://dashscope.aliyuncs.com/compatible-mode/v1"
    )
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key or api_key == "EMPTY":
        raise ValueError("❌ OPENAI_API_KEY 未设置，请先 export 你的百炼 API Key")

    # ✅ 正确的 Qwen 模型名称
    model = model_name or os.getenv("OPENAI_MODEL", "qwen2.5-7b-instruct")

    # ✅ 初始化客户端
    client = OpenAI(
        base_url=api_base,
        api_key=api_key
    )

    max_retries = 20
    for i in range(max_retries):
        try:
            response = client.chat.completions.create(
                model=model,
                messages=dialogs,
                temperature=temperature,
                max_tokens=max_tokens,
            )
            prompt_tokens = getattr(response.usage, 'prompt_tokens', 0) or 0
            completion_tokens = getattr(response.usage, 'completion_tokens', 0) or 0

            # ⚠️ prompt_cost_1k、completion_cost_1k 需在外部定义（或在这里设置默认值）
            this_cost = prompt_tokens / 1000 * prompt_cost_1k + completion_tokens / 1000 * completion_cost_1k
            content = response.choices[0].message.content
            return content, this_cost

        except Exception as e:
            if i < max_retries - 1:
                time.sleep(6)
            else:
                print(f"❌ An error of type {type(e).__name__} occurred: {e}")
                return "Error", 0.0

def format_numbers(numbers):
    return '[' + ', '.join('{:.2f}'.format(num) for num in numbers) + ']'

def format_percentages(numbers):
    return '[' + ', '.join('{:.2%}'.format(num) for num in numbers) + ']'
