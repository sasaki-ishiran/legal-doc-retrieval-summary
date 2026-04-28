"""智能分析工具。

优先调用本地 Ollama 兼容 OpenAI 接口；调用失败时使用规则方法兜底，确保无大模型环境也能演示。
"""

from __future__ import annotations

import json
import re
from typing import Dict, Tuple

from config import MODEL_CONFIG
from text_utils import clean_text, truncate_text


def rule_based_analysis(content: str, judgement: str = "") -> Tuple[str, str]:
    """无大模型环境下的规则分析兜底。"""

    text = clean_text(content)
    judgement_text = clean_text(judgement)
    combined = "{} {}".format(text, judgement_text).strip()

    summary = truncate_text(text, 180) if text else "暂无可分析内容"

    court_match = re.search(r"([\u4e00-\u9fff]{2,30}(?:人民法院|法院))", combined)
    cause_match = re.search(r"构成([\u4e00-\u9fff、，,]{2,30}(?:罪|纠纷))", combined)
    if not cause_match:
        cause_match = re.search(r"(民间借贷纠纷|劳动合同纠纷|交通事故责任纠纷|危险驾驶罪|合同诈骗罪|传销活动罪)", combined)

    law_matches = re.findall(r"《[^》]+》第[一二三四五六七八九十百千万零〇0-9条款款项之、]+", combined)
    party_matches = re.findall(r"(?:原告|被告人|被告|申请人|被申请人)[\u4e00-\u9fffA-Za-z0-9某]{1,12}", combined)

    entities: Dict[str, str] = {
        "案由": cause_match.group(1) if cause_match else "未识别",
        "法院": court_match.group(1) if court_match else "未识别",
        "当事人": "、".join(dict.fromkeys(party_matches[:8])) if party_matches else "未识别",
        "法律条款": "、".join(dict.fromkeys(law_matches[:5])) if law_matches else "未识别",
        "判决结果": truncate_text(judgement_text or text, 160),
    }

    return summary, json.dumps(entities, ensure_ascii=False, indent=2)


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
    source_text = clean_text("{} {}".format(content, judgement))[:2800]

    prompt = """
你是一个法律助手。请分析下面的法律文书内容。

任务：
1. 生成 200 字以内的案情摘要。
2. 提取关键实体：案由、法院、当事人、法律条款、判决结果。

请务必只输出一段合法 JSON 文本，不要包含 Markdown 标记，格式如下：
{
  "summary": "这里写摘要内容",
  "entities": {
    "案由": "...",
    "法院": "...",
    "当事人": "...",
    "法律条款": "...",
    "判决结果": "..."
  }
}

文书内容：
{}
""".format(source_text)

    try:
        response = client.chat.completions.create(
            model=MODEL_CONFIG.llm_model_name,
            messages=[{"role": "user", "content": prompt}],
            temperature=0.1,
            response_format={"type": "json_object"},
        )
        res_text = response.choices[0].message.content
        data = json.loads(res_text)
        return data.get("summary", "无摘要"), json.dumps(data.get("entities", {}), indent=2, ensure_ascii=False)
    except Exception:
        return rule_based_analysis(content, judgement)
