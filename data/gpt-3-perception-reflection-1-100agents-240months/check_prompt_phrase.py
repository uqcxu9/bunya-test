#!/usr/bin/env python3
import os, re, sys, pickle, glob

# 目标短语（用正则容许任意换行/空白差异）
PHRASE_RE = re.compile(
    r"after\s+collection,\s+the\s+government\s+evenly\s+redistributes\s+"
    r"the\s+tax\s+revenue\s+back\s+to\s+all\s+citizens,\s+irrespective\s+of\s+their\s+earnings\.",
    re.IGNORECASE | re.DOTALL,
)

def contains_phrase(text: str) -> bool:
    return bool(PHRASE_RE.search(text or ""))

def check_dialog_pickles(base_dir):
    results = []
    for fn in sorted(glob.glob(os.path.join(base_dir, "dialog_*.pkl"))):
        try:
            with open(fn, "rb") as f:
                dialogs = pickle.load(f)  # list(deque(...)) length = n_agents
        except Exception as e:
            print(f"[WARN] 读取失败: {fn} ({e})")
            continue

        total_user = 0
        matched = 0
        for agent_idx, dq in enumerate(dialogs):
            # dq 是一串 dict: {'role': 'user'|'assistant'|'system', 'content': '...'}
            for msg in dq:
                if msg.get("role") == "user":
                    total_user += 1
                    if contains_phrase(str(msg.get("content",""))):
                        matched += 1
                    else:
                        results.append(("PKL", fn, agent_idx, msg.get("content","")[:120].replace("\n"," ")))

        yield fn, total_user, matched, results

def check_dialog_textfolder(base_dir):
    # 文本日志：data/.../dialogs/每个智能体一个文件（之前代码里保存的）
    folder = os.path.join(base_dir, "dialogs")
    if not os.path.isdir(folder):
        return

    missed = []
    total_user = 0
    matched = 0
    for root, _, files in os.walk(folder):
        for name in files:
            path = os.path.join(root, name)
            try:
                with open(path, "r", encoding="utf-8", errors="ignore") as f:
                    content = f.read()
            except Exception as e:
                print(f"[WARN] 读取失败: {path} ({e})")
                continue

            # 粗略切分：保存时格式是 >>>>>>>>>role: content
            # 我们只挑 'user:' 段落检查
            for block in content.split(">>>>>>>>"):
                if block.lstrip().lower().startswith("user:"):
                    total_user += 1
                    text = block.split(":",1)[1] if ":" in block else block
                    if contains_phrase(text):
                        matched += 1
                    else:
                        missed.append(("TXT", path, text[:120].replace("\n"," ")))

    yield folder, total_user, matched, missed

def main():
    if len(sys.argv) < 2:
        print("用法: python3 check_prompt_phrase.py <数据目录>")
        sys.exit(1)

    base = sys.argv[1]
    if not os.path.isdir(base):
        print(f"目录不存在: {base}")
        sys.exit(1)

    grand_total = 0
    grand_match = 0
    misses = []

    # 检查 dialog_*.pkl
    for fn, tot, ok, missed in check_dialog_pickles(base):
        grand_total += tot
        grand_match += ok
        # missed 列表里积累了缺失样本（类型, 文件, agent_idx, 片段）
        # 但为了不重复加入，在这里不追加；等循环结束后统一收集
    # 为了拿到 missed，重新跑一次更简洁（也可在上面就地收集）：
    for fn in sorted(glob.glob(os.path.join(base, "dialog_*.pkl"))):
        try:
            with open(fn, "rb") as f:
                dialogs = pickle.load(f)
        except:
            continue
        for agent_idx, dq in enumerate(dialogs):
            for msg in dq:
                if msg.get("role") == "user":
                    if not contains_phrase(str(msg.get("content",""))):
                        misses.append(("PKL", f"{fn}#agent{agent_idx}", str(msg.get("content",""))[:120].replace("\n"," ")))

    # 检查文本 dialogs/
    for folder, tot, ok, missed in check_dialog_textfolder(base) or []:
        grand_total += tot
        grand_match += ok
        misses.extend(missed or [])

    # 汇总
    print("\n=== 检查结果汇总 ===")
    print(f"总共发现 user prompts：{grand_total}")
    print(f"其中包含目标短语的：{grand_match}")
    ratio = (grand_match / grand_total * 100) if grand_total else 0.0
    print(f"覆盖率：{ratio:.2f}%")

    if misses:
        print("\n以下 user prompt **未**匹配到目标短语（列出最多前 50 条）：")
        for i, m in enumerate(misses[:50], 1):
            kind, where, snippet = m
            print(f"{i:2d}. [{kind}] {where}  |  预览: {snippet}")

if __name__ == "__main__":
    main()
