import re

def extract_last_answer(text):
    # 优先匹配结构化格式
    for pattern in [
        r"<answer>\s*([A-D])\s*</answer>",
        r'boxed\{\s*([A-D])\s*\}',
    ]:
        matches = re.findall(pattern, text, re.IGNORECASE)
        if matches:
            return matches[-1].strip(), "first"

    # 退化到模糊匹配（仅在无结构化标签时）
    for pattern in [
        r'answer:\s*([A-D])\b',  # answer: A
        r'([A-D]):',  # A:
        r'([A-D])\.',  # B.
        r'\(([A-D])\)',  # (C)
        r'\[([A-D])\]',  # [C]
        r'\{([A-D])\}',  # {D}
        r'"([A-D])"',  # "A"
        r"'([A-D])'",  # 'B'
        r'\*\*([A-D])\*\*',       # **D**
    ]:
        matches = re.findall(pattern, text)
        if matches:
            return matches[-1].strip(), "second"

    return None, "error"
