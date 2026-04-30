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


LEGAL_SECTION_KEYWORDS = {
    "经审理查明": 0.18,
    "本院查明": 0.16,
    "本院认为": 0.18,
    "法院认为": 0.16,
    "判决如下": 0.18,
    "裁判如下": 0.16,
    "依照": 0.10,
    "诉称": 0.08,
    "辩称": 0.08,
    "证据": 0.08,
    "认定": 0.06,
    "事实": 0.06,
    "法律依据": 0.10,
}


def split_text_into_chunks(text: object, chunk_size: int = 900, overlap: int = 160) -> list[dict]:
    """将长文本按句子边界切分为带重叠窗口的分片。"""

    value = clean_text(text)
    if not value:
        return []

    chunk_size = max(100, int(chunk_size or 900))
    overlap = max(0, min(int(overlap or 0), chunk_size // 2))
    if len(value) <= chunk_size:
        return [{"chunk_id": 1, "text": value, "length": len(value)}]

    chunks: list[dict] = []

    def append_chunk(chunk_text: str) -> None:
        cleaned = clean_text(chunk_text)
        if not cleaned:
            return
        if chunks and chunks[-1].get("text") == cleaned:
            return
        chunks.append({"chunk_id": len(chunks) + 1, "text": cleaned, "length": len(cleaned)})

    sentences = re.findall(r"[^。！？!?；;]+[。！？!?；;]?", value)
    if not sentences:
        sentences = [value]

    current_parts: list[str] = []
    current_length = 0

    def flush_current() -> None:
        nonlocal current_parts, current_length
        if current_parts:
            append_chunk("".join(current_parts))
            current_parts = []
            current_length = 0

    for sentence in sentences:
        sentence = sentence.strip()
        if not sentence:
            continue

        if len(sentence) > chunk_size:
            flush_current()
            start = 0
            while start < len(sentence):
                end = min(len(sentence), start + chunk_size)
                append_chunk(sentence[start:end])
                if end >= len(sentence):
                    break
                next_start = end - overlap
                start = next_start if next_start > start else end
            continue

        if current_parts and current_length + len(sentence) > chunk_size:
            previous_text = "".join(current_parts)
            append_chunk(previous_text)
            overlap_text = previous_text[-overlap:] if overlap else ""
            current_parts = [overlap_text] if overlap_text else []
            current_length = len(overlap_text)

        current_parts.append(sentence)
        current_length += len(sentence)

    flush_current()
    return chunks


def _chunk_section_boost(text: str) -> float:
    """根据法律文书典型段落标题给分片增加轻量权重。"""

    score = 0.0
    for keyword, weight in LEGAL_SECTION_KEYWORDS.items():
        if keyword in text:
            score += weight
    return min(score, 0.5)


def rank_text_chunks(query: object, chunks: list[dict], top_k: int = 5) -> list[dict]:
    """按查询词、法律段落特征和首尾位置对分片排序。"""

    retrieval_query = clean_text(query)
    total_chunks = len(chunks)
    ranked: list[dict] = []

    for position, chunk in enumerate(chunks):
        chunk_text = clean_text(chunk.get("text", ""))
        if not chunk_text:
            continue

        keyword_score = keyword_overlap_score(retrieval_query, chunk_text) if retrieval_query else 0.0
        section_score = _chunk_section_boost(chunk_text)
        boundary_score = 0.0
        if position == 0:
            boundary_score += 0.08
        if total_chunks > 1 and position == total_chunks - 1:
            boundary_score += 0.08

        item = dict(chunk)
        item["score"] = keyword_score + section_score + boundary_score
        item["keyword_score"] = keyword_score
        item["section_score"] = section_score
        item["boundary_score"] = boundary_score
        ranked.append(item)

    ranked.sort(key=lambda item: (float(item.get("score", 0.0)), -int(item.get("chunk_id", 0))), reverse=True)
    if top_k and top_k > 0:
        return ranked[:top_k]
    return ranked


def build_rag_context(
    text: object,
    query: object,
    chunk_size: int = 900,
    overlap: int = 160,
    top_k: int = 5,
    max_chars: int = 4200,
) -> tuple[str, list[dict]]:
    """基于分片排序构建可送入大模型的证据上下文。"""

    chunks = split_text_into_chunks(text, chunk_size=chunk_size, overlap=overlap)
    if not chunks:
        return "", []

    top_k = max(1, int(top_k or 1))
    max_chars = max(300, int(max_chars or 300))
    ranked_all = rank_text_chunks(query, chunks, top_k=0)
    ranked_by_id = {int(item.get("chunk_id", 0)): item for item in ranked_all}

    selected_by_id: dict[int, dict] = {}
    first_id = int(chunks[0].get("chunk_id", 1))
    last_id = int(chunks[-1].get("chunk_id", first_id))
    selected_by_id[first_id] = ranked_by_id.get(first_id, chunks[0])
    selected_by_id[last_id] = ranked_by_id.get(last_id, chunks[-1])

    target_count = min(top_k, len(chunks))
    for item in ranked_all:
        chunk_id = int(item.get("chunk_id", 0))
        if chunk_id <= 0:
            continue
        selected_by_id[chunk_id] = item
        if len(selected_by_id) >= target_count:
            break

    selected_chunks = sorted(
        selected_by_id.values(),
        key=lambda item: (-float(item.get("score", 0.0) or 0.0), int(item.get("chunk_id", 0) or 0)),
    )
    context_parts: list[str] = []
    final_chunks: list[dict] = []
    used_chars = 0
    total_chunks = len(chunks)

    for item in selected_chunks:
        chunk_text = clean_text(item.get("text", ""))
        if not chunk_text:
            continue

        score = float(item.get("score", 0.0) or 0.0)
        header = "【片段 {}/{}，相关度 {:.3f}】\n".format(item.get("chunk_id"), total_chunks, score)
        remaining = max_chars - used_chars - len(header) - 2
        if remaining <= 80:
            break
        if len(chunk_text) > remaining:
            chunk_text = chunk_text[:remaining].rstrip() + "..."

        part = header + chunk_text
        context_parts.append(part)
        used_chars += len(part) + 2

        copied = dict(item)
        copied["text"] = chunk_text
        copied["total_chunks"] = total_chunks
        final_chunks.append(copied)

    return "\n\n".join(context_parts), final_chunks


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
