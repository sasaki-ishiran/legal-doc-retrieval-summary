"""智能分析工具。

优先调用本地 Ollama 兼容 OpenAI 接口；调用失败时使用规则方法兜底，确保无大模型环境也能演示。
"""

from __future__ import annotations

import json
import re
from typing import Dict, Tuple

from config import MODEL_CONFIG, RETRIEVAL_CONFIG
from text_utils import build_rag_context, clean_text, truncate_text


LEGAL_ANALYSIS_QUERY = (
    "案由 法院 当事人 原告 被告 被告人 事实 争议焦点 本院认为 法院认为 "
    "证据 法律条款 依照 判决如下 裁判如下 判决结果 罪名 量刑 民事责任"
)

COT_PROMPT_STRATEGIES = {"cot", "rag_cot", "structured_cot", "evidence_cot"}

LEGAL_SYSTEM_PROMPT = """
你是一个严谨的法律文书分析助手。回答必须坚持证据约束：只能依据用户提供的证据片段生成摘要和实体，不得根据经验、常识或模板补全证据中没有出现的信息。
""".strip()

LEGAL_COT_SYSTEM_PROMPT = """
你是一个严谨的法律文书分析助手。请采用“内部逐步推理 + 证据核验”的工作流，但最终不要输出推理链或分析过程。

内部工作流：
1. 先逐个阅读证据片段，识别案件事实、争议焦点、法院认定、法律依据和裁判结果。
2. 再为每个待抽取字段寻找证据片段中的直接依据，不能确认的字段写“未识别”。
3. 最后基于已核验信息生成摘要，摘要必须覆盖事实、争点、法院认定和裁判结果中的可确认内容。
4. 输出前自检 JSON 是否合法、字段是否完整、是否存在证据外推断。

输出要求：只输出最终 JSON，不输出思考过程、推理链、草稿或 Markdown 标记。
""".strip()


def build_legal_analysis_context(content: str, judgement: str = "") -> Tuple[str, list[dict]]:
    """按 RAG 分片思想构建法律文书分析上下文。"""

    text = clean_text(content)
    judgement_text = clean_text(judgement)
    combined = "【正文】{} 【判决结果】{}".format(text, judgement_text).strip()
    return build_rag_context(
        combined,
        LEGAL_ANALYSIS_QUERY,
        chunk_size=RETRIEVAL_CONFIG.rag_chunk_size,
        overlap=RETRIEVAL_CONFIG.rag_chunk_overlap,
        top_k=RETRIEVAL_CONFIG.rag_top_k,
        max_chars=RETRIEVAL_CONFIG.rag_context_max_chars,
    )


SUMMARY_KEYWORDS = {
    "判决如下": 1.0,
    "裁判如下": 1.0,
    "本院认为": 0.9,
    "法院认为": 0.9,
    "经审理查明": 0.8,
    "本院查明": 0.8,
    "争议": 0.5,
    "构成": 0.4,
    "成立": 0.4,
    "证据": 0.3,
    "借条": 0.3,
    "转账": 0.3,
}


def split_summary_sentences(text: str) -> list[str]:
    """按中文标点拆分摘要候选句。"""

    return [sentence.strip() for sentence in re.findall(r"[^。！？!?；;]+[。！？!?；;]?", clean_text(text)) if sentence.strip()]


def build_rule_summary(text: str, judgement_text: str, selected_chunks: list[dict]) -> str:
    """从 RAG 片段中优先抽取裁判要点，作为无模型兜底摘要。"""

    chunk_text = " ".join(chunk.get("text", "") for chunk in selected_chunks)
    sentences = split_summary_sentences(chunk_text)
    if judgement_text not in chunk_text:
        sentences.extend(split_summary_sentences(judgement_text))
    scored_sentences = []
    seen = set()
    for index, sentence in enumerate(sentences):
        normalized = clean_text(sentence)
        if not normalized or normalized in seen:
            continue
        seen.add(normalized)
        score = sum(weight for keyword, weight in SUMMARY_KEYWORDS.items() if keyword in normalized)
        if score <= 0 and index < 3:
            score = 0.05
        scored_sentences.append((score, index, normalized))

    scored_sentences.sort(key=lambda item: (-item[0], item[1]))
    selected = []
    for score, _, sentence in scored_sentences:
        if score <= 0 and selected:
            continue
        selected.append(sentence)
        if len("".join(selected)) >= 190 or len(selected) >= 3:
            break

    summary_source = "".join(selected) or chunk_text or text
    return truncate_text(summary_source, 200) if summary_source else "暂无可分析内容"


def normalize_party_matches(matches: list[str]) -> list[str]:
    """过滤规则抽取中明显不是当事人名称的短语。"""

    parties = []
    noise_words = ["提交", "认为", "诉称", "辩称", "主张", "请求", "没有", "对部分", "长期", "于本"]
    for match in matches:
        if any(word in match for word in noise_words):
            continue
        short_match = re.match(r"((?:原告|被告人|被告|申请人|被申请人)[\u4e00-\u9fffA-Za-z0-9]{0,4}某)", match)
        normalized = short_match.group(1) if short_match else match
        if normalized and normalized not in parties:
            parties.append(normalized)
    return parties


def rule_based_analysis(content: str, judgement: str = "") -> Tuple[str, str]:
    """无大模型环境下的规则分析兜底。"""

    text = clean_text(content)
    judgement_text = clean_text(judgement)
    combined = "{} {}".format(text, judgement_text).strip()
    _, selected_chunks = build_legal_analysis_context(content, judgement)

    summary = build_rule_summary(text, judgement_text, selected_chunks)

    court_match = re.search(r"([\u4e00-\u9fff]{2,30}(?:人民法院|法院))", combined)
    cause_match = re.search(r"构成([\u4e00-\u9fff、，,]{2,30}(?:罪|纠纷))", combined)
    if not cause_match:
        cause_match = re.search(r"(民间借贷纠纷|劳动合同纠纷|交通事故责任纠纷|危险驾驶罪|合同诈骗罪|传销活动罪)", combined)

    law_matches = re.findall(r"《[^》]+》第[一二三四五六七八九十百千万零〇0-9条款款项之、]+", combined)
    raw_party_matches = re.findall(r"(?:原告|被告人|被告|申请人|被申请人)[\u4e00-\u9fffA-Za-z0-9某]{1,12}", combined)
    party_matches = normalize_party_matches(raw_party_matches)

    entities: Dict[str, str] = {
        "案由": cause_match.group(1) if cause_match else "未识别",
        "法院": court_match.group(1) if court_match else "未识别",
        "当事人": "、".join(dict.fromkeys(party_matches[:8])) if party_matches else "未识别",
        "法律条款": "、".join(dict.fromkeys(law_matches[:5])) if law_matches else "未识别",
        "判决结果": truncate_text(judgement_text or text, 160),
    }

    return summary, json.dumps(entities, ensure_ascii=False, indent=2)


def build_llm_messages(source_text: str, selected_chunks: list[dict]) -> list[dict]:
    """构建支持提示词实验的消息列表。"""

    use_cot_prompt = MODEL_CONFIG.llm_prompt_strategy in COT_PROMPT_STRATEGIES
    system_prompt = LEGAL_COT_SYSTEM_PROMPT if use_cot_prompt else LEGAL_SYSTEM_PROMPT
    cot_note = "已启用内部逐步推理与证据核验提示。" if use_cot_prompt else "使用基础证据约束提示。"

    user_prompt = f"""
下面的法律文书可能篇幅很长，系统已先按 RAG 思想进行分片，并从全文中选取了 {len(selected_chunks)} 个证据片段。{cot_note}

任务：
1. 生成 200 字以内的案情摘要。
2. 提取关键实体：案由、法院、当事人、法律条款、判决结果。

约束：
1. 只能依据给定证据片段生成结论，不要补全未知事实。
2. 证据片段没有明确出现的信息，一律写“未识别”。
3. 摘要重点覆盖案情事实、争议焦点、法院认定和裁判结果。
4. 请务必只输出一段合法 JSON 文本，不要包含 Markdown 标记。

输出格式如下：
{{
  "summary": "这里写摘要内容",
  "entities": {{
    "案由": "...",
    "法院": "...",
    "当事人": "...",
    "法律条款": "...",
    "判决结果": "..."
  }}
}}

证据片段：
{source_text}
"""
    return [{"role": "system", "content": system_prompt}, {"role": "user", "content": user_prompt}]


def llm_process(content: str, judgement: str = "") -> Tuple[str, str]:
    """调用本地大模型进行摘要与信息抽取，失败时自动降级到规则方法。"""

    if not clean_text(content):
        return "请先选择一篇文书", "{}"

    if not MODEL_CONFIG.enable_llm:
        return rule_based_analysis(content, judgement)

    try:
        from openai import OpenAI
    except Exception:
        return rule_based_analysis(content, judgement)

    client = OpenAI(api_key=MODEL_CONFIG.llm_api_key, base_url=MODEL_CONFIG.llm_base_url)
    source_text, selected_chunks = build_legal_analysis_context(content, judgement)
    if not source_text:
        return rule_based_analysis(content, judgement)

    messages = build_llm_messages(source_text, selected_chunks)

    try:
        response = client.chat.completions.create(
            model=MODEL_CONFIG.llm_model_name,
            messages=messages,
            temperature=0.1,
            response_format={"type": "json_object"},
        )
        res_text = response.choices[0].message.content or "{}"
        data = json.loads(res_text)
        return data.get("summary", "无摘要"), json.dumps(data.get("entities", {}), indent=2, ensure_ascii=False)
    except Exception:
        return rule_based_analysis(content, judgement)
