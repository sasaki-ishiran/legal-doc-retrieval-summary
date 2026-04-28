"""文本处理辅助函数。"""

from __future__ import annotations

import math
import re
from collections import Counter
from typing import Iterable


CHINESE_STOPWORDS = {
    "的",
    "了",
    "和",
    "与",
    "及",
    "或",
    "在",
    "对",
    "其",
    "并",
    "被",
    "为",
    "以",
    "因",
    "后",
    "中",
    "将",
    "已",
    "于",
    "由",
    "是",
    "本",
    "该",
    "等",
    "及其",
    "进行",
    "认为",
    "法院",
    "判决",
    "如下",
    "原告",
    "被告",
    "被告人",
}


def clean_text(text: object) -> str:
    """清洗文本：去除 HTML、合并空白字符。"""

    if text is None:
        return ""
    value = str(text)
    value = re.sub(r"<[^>]+>", "", value)
    value = value.replace("\u3000", " ").replace("&nbsp;", " ")
    value = re.sub(r"\s+", " ", value)
    return value.strip()


def truncate_text(text: object, max_length: int = 220) -> str:
    """按字符长度截断文本，适合界面摘要展示。"""

    value = clean_text(text)
    if len(value) <= max_length:
        return value
    return value[:max_length].rstrip() + "..."


def tokenize(text: object) -> list[str]:
    """轻量级中英文混合分词。

    不额外引入中文分词依赖，适合当前毕业设计原型。中文部分按连续 2 到 6 字窗口提取，英文数字按词提取。
    """

    value = clean_text(text).lower()
    if not value:
        return []

    tokens: list[str] = []
    tokens.extend(re.findall(r"[a-z0-9_]+", value))

    chinese_blocks = re.findall(r"[\u4e00-\u9fff]+", value)
    for block in chinese_blocks:
        if len(block) <= 1:
            continue
        max_window = min(6, len(block))
        for size in range(2, max_window + 1):
            for start in range(0, len(block) - size + 1):
                token = block[start : start + size]
                if token not in CHINESE_STOPWORDS:
                    tokens.append(token)

    return tokens


def extract_query_terms(query: object) -> list[str]:
    """从查询中提取更适合展示的关键词。"""

    value = clean_text(query)
    if not value:
        return []

    candidates = re.findall(r"[\u4e00-\u9fff]{2,}|[a-zA-Z0-9_]+", value)
    terms: list[str] = []
    for item in candidates:
        item = item.strip().lower()
        if not item or item in CHINESE_STOPWORDS:
            continue
        if item not in terms:
            terms.append(item)

    if terms:
        return terms

    return list(dict.fromkeys(tokenize(value)))[:8]


def hit_terms(query: object, text: object) -> list[str]:
    """返回查询词中出现在文本里的词。"""

    content = clean_text(text).lower()
    hits: list[str] = []
    for term in extract_query_terms(query):
        if term and term.lower() in content and term not in hits:
            hits.append(term)
    return hits


def keyword_overlap_score(query: object, text: object) -> float:
    """计算查询与文本之间的关键词重合分数，范围约为 0 到 1。"""

    query_tokens = tokenize(query)
    text_tokens = tokenize(text)
    if not query_tokens or not text_tokens:
        return 0.0

    query_counter = Counter(query_tokens)
    text_counter = Counter(text_tokens)
    overlap = 0
    for token, count in query_counter.items():
        overlap += min(count, text_counter.get(token, 0))

    denominator = math.sqrt(sum(query_counter.values())) * math.sqrt(sum(text_counter.values()))
    if denominator == 0:
        return 0.0

    return min(overlap / denominator, 1.0)


def normalize_scores(values: Iterable[float]) -> list[float]:
    """将分数线性归一化到 0 到 1。"""

    scores = [float(v) for v in values]
    if not scores:
        return []

    min_score = min(scores)
    max_score = max(scores)
    if abs(max_score - min_score) < 1e-12:
        return [1.0 if max_score > 0 else 0.0 for _ in scores]

    return [(score - min_score) / (max_score - min_score) for score in scores]


def build_snippet(text: object, query: object = "", max_length: int = 220) -> str:
    """生成围绕命中词的内容片段。"""

    content = clean_text(text)
    if not content:
        return ""

    for term in extract_query_terms(query):
        idx = content.lower().find(term.lower())
        if idx >= 0:
            start = max(0, idx - max_length // 3)
            end = min(len(content), start + max_length)
            prefix = "..." if start > 0 else ""
            suffix = "..." if end < len(content) else ""
            return prefix + content[start:end].strip() + suffix

    return truncate_text(content, max_length)


def build_reason(query: object, case: dict, semantic_score: float = 0.0, keyword_score: float = 0.0) -> str:
    """生成检索推荐理由。"""

    combined_text = f"{case.get('case_name', '')} {case.get('content', '')} {case.get('judgement', '')}"
    hits = hit_terms(query, combined_text)
    parts = []
    if hits:
        parts.append("命中关键词：" + "、".join(hits[:6]))
    if semantic_score > 0:
        parts.append(f"语义相似度：{semantic_score:.3f}")
    if keyword_score > 0:
        parts.append(f"关键词相关度：{keyword_score:.3f}")
    if not parts:
        parts.append("根据文本内容综合相关性排序")
    return "；".join(parts)
