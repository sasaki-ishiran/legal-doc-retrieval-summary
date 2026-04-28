"""检索模块。

提供关键词检索、语义向量检索结果包装和混合检索融合逻辑。
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Dict, List, Optional, Union

from config import RETRIEVAL_CONFIG
from text_utils import build_reason, build_snippet, keyword_overlap_score, normalize_scores, truncate_text


@dataclass
class SearchResult:
    """统一的检索结果结构。"""

    id: Union[int, str]
    title: str
    content: str
    judgement: str
    score: float
    semantic_score: float = 0.0
    keyword_score: float = 0.0
    reason: str = ""
    snippet: str = ""
    source: str = ""

    def to_dict(self) -> Dict:
        """转换为 Gradio 状态可保存的字典。"""

        return {
            "id": self.id,
            "title": self.title,
            "case_name": self.title,
            "content": self.content,
            "judgement": self.judgement,
            "score": float(self.score),
            "semantic_score": float(self.semantic_score),
            "keyword_score": float(self.keyword_score),
            "reason": self.reason,
            "snippet": self.snippet,
            "source": self.source,
            "display_label": "【综合分 {:.3f}】{}".format(self.score, self.title),
        }


def normalize_case(raw_case: Dict) -> Dict:
    """统一数据库案例和演示案例字段。"""

    return {
        "id": raw_case.get("id"),
        "case_name": raw_case.get("case_name") or raw_case.get("title") or "未知案件",
        "content": raw_case.get("content") or "",
        "judgement": raw_case.get("judgement") or "",
        "aj_id": raw_case.get("aj_id") or "",
        "writ_id": raw_case.get("writ_id") or "",
    }


def case_search_text(case: Dict) -> str:
    """拼接用于关键词检索的文本。"""

    normalized = normalize_case(case)
    return "{} {} {}".format(normalized["case_name"], normalized["content"], normalized["judgement"])


def keyword_search(query: str, cases: List[Dict], top_k: Optional[int] = None) -> List[Dict]:
    """基于轻量关键词重合度的检索。"""

    top_k = top_k or RETRIEVAL_CONFIG.default_top_k
    scored_results: List[SearchResult] = []

    for case in cases:
        normalized = normalize_case(case)
        search_text = case_search_text(normalized)
        keyword_score = keyword_overlap_score(query, search_text)
        if keyword_score <= 0:
            continue

        result = SearchResult(
            id=normalized["id"],
            title=normalized["case_name"],
            content=normalized["content"],
            judgement=normalized["judgement"],
            score=keyword_score,
            keyword_score=keyword_score,
            reason=build_reason(query, normalized, keyword_score=keyword_score),
            snippet=build_snippet(normalized["content"], query, RETRIEVAL_CONFIG.snippet_length),
            source="keyword",
        )
        scored_results.append(result)

    scored_results.sort(key=lambda item: item.score, reverse=True)
    return [item.to_dict() for item in scored_results[:top_k]]


def wrap_semantic_results(query: str, raw_results: List[Dict], source: str = "semantic") -> List[Dict]:
    """为语义检索结果补充片段、推荐理由和统一字段。"""

    results: List[Dict] = []
    for raw in raw_results:
        normalized = normalize_case(raw)
        semantic_score = float(raw.get("score", raw.get("semantic_score", 0.0)) or 0.0)
        keyword_score = float(raw.get("keyword_score", keyword_overlap_score(query, case_search_text(normalized))) or 0.0)
        score = float(raw.get("score", semantic_score) or 0.0)
        result = SearchResult(
            id=normalized["id"],
            title=normalized["case_name"],
            content=normalized["content"],
            judgement=normalized["judgement"],
            score=score,
            semantic_score=semantic_score,
            keyword_score=keyword_score,
            reason=raw.get("reason") or build_reason(query, normalized, semantic_score, keyword_score),
            snippet=raw.get("snippet") or build_snippet(normalized["content"], query, RETRIEVAL_CONFIG.snippet_length),
            source=source,
        )
        results.append(result.to_dict())
    return results


def hybrid_fusion(
    query: str,
    semantic_results: List[Dict],
    keyword_results: List[Dict],
    top_k: Optional[int] = None,
    semantic_weight: Optional[float] = None,
    keyword_weight: Optional[float] = None,
) -> List[Dict]:
    """融合语义检索和关键词检索结果。"""

    top_k = top_k or RETRIEVAL_CONFIG.default_top_k
    semantic_weight = RETRIEVAL_CONFIG.semantic_weight if semantic_weight is None else semantic_weight
    keyword_weight = RETRIEVAL_CONFIG.keyword_weight if keyword_weight is None else keyword_weight

    merged: Dict[str, Dict] = {}

    for item in semantic_results:
        normalized = normalize_case(item)
        key = str(normalized["id"])
        merged[key] = {
            **item,
            "semantic_score": float(item.get("semantic_score", item.get("score", 0.0)) or 0.0),
            "keyword_score": float(item.get("keyword_score", 0.0) or 0.0),
        }

    for item in keyword_results:
        normalized = normalize_case(item)
        key = str(normalized["id"])
        if key not in merged:
            merged[key] = {
                **item,
                "semantic_score": float(item.get("semantic_score", 0.0) or 0.0),
                "keyword_score": float(item.get("keyword_score", item.get("score", 0.0)) or 0.0),
            }
        else:
            merged[key]["keyword_score"] = max(
                float(merged[key].get("keyword_score", 0.0) or 0.0),
                float(item.get("keyword_score", item.get("score", 0.0)) or 0.0),
            )

    items = list(merged.values())
    semantic_norm = normalize_scores([float(item.get("semantic_score", 0.0) or 0.0) for item in items])
    keyword_norm = normalize_scores([float(item.get("keyword_score", 0.0) or 0.0) for item in items])

    fused_results: List[Dict] = []
    for index, item in enumerate(items):
        normalized = normalize_case(item)
        semantic_score = semantic_norm[index]
        keyword_score = keyword_norm[index]
        final_score = semantic_weight * semantic_score + keyword_weight * keyword_score
        item["score"] = final_score
        item["semantic_score"] = float(item.get("semantic_score", 0.0) or 0.0)
        item["keyword_score"] = float(item.get("keyword_score", 0.0) or 0.0)
        item["reason"] = build_reason(query, normalized, item["semantic_score"], item["keyword_score"])
        item["snippet"] = item.get("snippet") or build_snippet(
            normalized["content"], query, RETRIEVAL_CONFIG.snippet_length
        )
        item["source"] = "hybrid"
        item["display_label"] = "【综合分 {:.3f}】{}".format(final_score, normalized["case_name"])
        fused_results.append(item)

    fused_results.sort(key=lambda result: float(result.get("score", 0.0)), reverse=True)
    return fused_results[:top_k]


def search_with_mode(
    query: str,
    mode: str,
    cases_provider: Callable[[], List[Dict]],
    semantic_provider: Optional[Callable[[str, int], List[Dict]]] = None,
    top_k: Optional[int] = None,
) -> List[Dict]:
    """按模式执行检索。

    mode 支持：关键词检索、语义检索、混合检索。
    """

    top_k = top_k or RETRIEVAL_CONFIG.default_top_k
    mode = mode or "混合检索"

    keyword_candidates = keyword_search(query, cases_provider(), top_k=max(top_k * 2, top_k))

    if mode == "关键词检索":
        return keyword_candidates[:top_k]

    semantic_candidates: List[Dict] = []
    if semantic_provider is not None:
        semantic_candidates = wrap_semantic_results(query, semantic_provider(query, max(top_k * 2, top_k)))

    if mode == "语义检索":
        return semantic_candidates[:top_k]

    return hybrid_fusion(query, semantic_candidates, keyword_candidates, top_k=top_k)


def build_results_table(results: List[Dict]) -> List[List]:
    """构建 Gradio Dataframe 可展示的数据。"""

    table = []
    for rank, item in enumerate(results, start=1):
        table.append(
            [
                rank,
                item.get("id"),
                item.get("title") or item.get("case_name"),
                round(float(item.get("score", 0.0) or 0.0), 4),
                round(float(item.get("semantic_score", 0.0) or 0.0), 4),
                round(float(item.get("keyword_score", 0.0) or 0.0), 4),
                truncate_text(item.get("reason", ""), 80),
            ]
        )
    return table
