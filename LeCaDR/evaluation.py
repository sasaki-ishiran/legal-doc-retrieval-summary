"""检索评价指标。

用于论文实验章节，可比较关键词检索、语义检索和混合检索的效果。
"""

from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Iterable, List, Dict, Union


def _to_set(values: Iterable) -> set:
    return {str(value) for value in values}


def precision_at_k(retrieved_ids: List, relevant_ids: Iterable, k: int) -> float:
    """计算 Precision@K。"""

    if k <= 0:
        return 0.0
    retrieved_top_k = [str(item) for item in retrieved_ids[:k]]
    if not retrieved_top_k:
        return 0.0
    relevant = _to_set(relevant_ids)
    hit_count = sum(1 for item in retrieved_top_k if item in relevant)
    return hit_count / len(retrieved_top_k)


def recall_at_k(retrieved_ids: List, relevant_ids: Iterable, k: int) -> float:
    """计算 Recall@K。"""

    relevant = _to_set(relevant_ids)
    if not relevant or k <= 0:
        return 0.0
    retrieved_top_k = [str(item) for item in retrieved_ids[:k]]
    hit_count = sum(1 for item in retrieved_top_k if item in relevant)
    return hit_count / len(relevant)


def mrr(retrieved_ids: List, relevant_ids: Iterable) -> float:
    """计算单个查询的 Reciprocal Rank。"""

    relevant = _to_set(relevant_ids)
    if not relevant:
        return 0.0
    for index, item in enumerate(retrieved_ids, start=1):
        if str(item) in relevant:
            return 1.0 / index
    return 0.0


def ndcg_at_k(retrieved_ids: List, relevant_ids: Iterable, k: int) -> float:
    """计算二值相关性版本的 nDCG@K。"""

    if k <= 0:
        return 0.0
    relevant = _to_set(relevant_ids)
    if not relevant:
        return 0.0

    dcg = 0.0
    for index, item in enumerate(retrieved_ids[:k], start=1):
        rel = 1.0 if str(item) in relevant else 0.0
        if rel:
            dcg += rel / math.log2(index + 1)

    ideal_hits = min(len(relevant), k)
    idcg = sum(1.0 / math.log2(index + 1) for index in range(1, ideal_hits + 1))
    if idcg == 0:
        return 0.0
    return dcg / idcg


def evaluate_single_query(retrieved_ids: List, relevant_ids: Iterable, k: int = 10) -> Dict:
    """计算单个查询的常用检索指标。"""

    return {
        "Precision@{}".format(k): precision_at_k(retrieved_ids, relevant_ids, k),
        "Recall@{}".format(k): recall_at_k(retrieved_ids, relevant_ids, k),
        "MRR": mrr(retrieved_ids, relevant_ids),
        "nDCG@{}".format(k): ndcg_at_k(retrieved_ids, relevant_ids, k),
    }


def average_metrics(metrics_list: List[Dict]) -> Dict:
    """对多个查询的指标取平均。"""

    if not metrics_list:
        return {}

    keys = metrics_list[0].keys()
    return {
        key: sum(float(item.get(key, 0.0)) for item in metrics_list) / len(metrics_list)
        for key in keys
    }


def load_evaluation_cases(path: Union[str, Path]) -> List[Dict]:
    """读取评测查询模板。"""

    file_path = Path(path)
    if not file_path.exists():
        return []

    with file_path.open("r", encoding="utf-8") as file:
        data = json.load(file)

    if not isinstance(data, list):
        raise ValueError("评测文件必须是 JSON 数组")
    return data
