"""法律文书智能检索与信息抽取系统。

支持真实数据模式和演示模式：
1. 真实数据模式：使用 MySQL + FAISS + 语义模型完成检索。
2. 演示模式：当数据库、索引或模型不可用时，使用内置样例数据保证界面可运行。
"""

from __future__ import annotations

from typing import Dict, List, Optional, Tuple

import gradio as gr

from config import APP_CONFIG, DB_CONFIG, MODEL_CONFIG, PATH_CONFIG, RETRIEVAL_CONFIG
from evaluation import average_metrics, evaluate_single_query, load_evaluation_cases
from llm_utils import llm_process
from retrieval import build_results_table, hybrid_fusion, keyword_search, wrap_semantic_results
from sample_data import get_sample_cases
from text_utils import build_reason, build_snippet, keyword_overlap_score


try:
    import faiss
except Exception:
    faiss = None

try:
    import numpy as np
except Exception:
    np = None

try:
    import pymysql
except Exception:
    pymysql = None

try:
    from modelscope import snapshot_download
    from sentence_transformers import SentenceTransformer
except Exception:
    snapshot_download = None
    SentenceTransformer = None


class RuntimeState:
    """保存系统启动时加载的资源。"""

    def __init__(self) -> None:
        self.demo_mode: bool = True
        self.status: str = "系统尚未初始化"
        self.embed_model = None
        self.search_index = None
        self.db_ids_map = None
        self.sample_cases: List[Dict] = get_sample_cases()


RUNTIME = RuntimeState()


def connect_db():
    """连接数据库。"""

    if pymysql is None:
        raise RuntimeError("未安装 pymysql，无法连接 MySQL")
    return pymysql.connect(**DB_CONFIG.to_pymysql_kwargs())


def fetch_all_cases(limit: Optional[int] = None) -> List[Dict]:
    """从数据库读取案例，用于关键词检索和混合检索。"""

    if RUNTIME.demo_mode:
        return RUNTIME.sample_cases

    sql = "SELECT id, aj_id, writ_id, case_name, content, judgement FROM cases"
    if limit:
        sql += " LIMIT %s"

    conn = connect_db()
    cursor = conn.cursor()
    try:
        if limit:
            cursor.execute(sql, (int(limit),))
        else:
            cursor.execute(sql)
        rows = cursor.fetchall()
        return [
            {
                "id": row[0],
                "aj_id": row[1],
                "writ_id": row[2],
                "case_name": row[3],
                "content": row[4],
                "judgement": row[5],
            }
            for row in rows
        ]
    finally:
        cursor.close()
        conn.close()


def initialize_runtime() -> None:
    """初始化真实检索资源；失败时自动切换演示模式。"""

    if not APP_CONFIG.auto_demo_mode:
        RUNTIME.demo_mode = False

    required_modules_ready = all([faiss is not None, np is not None, pymysql is not None, SentenceTransformer is not None])
    index_ready = PATH_CONFIG.index_path.exists() and PATH_CONFIG.id_map_path.exists()

    if not required_modules_ready:
        RUNTIME.demo_mode = True
        RUNTIME.status = "演示模式：依赖库未完全安装，使用内置样例数据。"
        return

    if not index_ready:
        RUNTIME.demo_mode = True
        RUNTIME.status = "演示模式：未发现向量索引文件，使用内置样例数据。"
        return

    try:
        conn = connect_db()
        cursor = conn.cursor()
        cursor.execute("SELECT COUNT(*) FROM cases")
        total = cursor.fetchone()[0]
        cursor.close()
        conn.close()
        if total == 0:
            raise RuntimeError("数据库 cases 表为空")

        model_dir = snapshot_download(MODEL_CONFIG.embedding_model_id)
        RUNTIME.embed_model = SentenceTransformer(model_dir)
        RUNTIME.search_index = faiss.read_index(str(PATH_CONFIG.index_path))
        RUNTIME.db_ids_map = np.load(str(PATH_CONFIG.id_map_path))
        RUNTIME.demo_mode = False
        RUNTIME.status = "真实数据模式：已连接 MySQL，加载 FAISS 索引 {} 条向量。".format(
            RUNTIME.search_index.ntotal
        )
    except Exception as exc:
        RUNTIME.demo_mode = True
        RUNTIME.status = "演示模式：真实数据资源加载失败，原因：{}。".format(exc)


def semantic_search(query: str, top_k: int) -> List[Dict]:
    """执行语义检索。真实模式使用 FAISS，演示模式用关键词相关度模拟语义结果。"""

    if RUNTIME.demo_mode or RUNTIME.embed_model is None or RUNTIME.search_index is None:
        demo_results = []
        for case in RUNTIME.sample_cases:
            combined = "{} {} {}".format(case.get("case_name", ""), case.get("content", ""), case.get("judgement", ""))
            score = keyword_overlap_score(query, combined)
            demo_results.append(
                {
                    "id": case["id"],
                    "case_name": case["case_name"],
                    "content": case["content"],
                    "judgement": case["judgement"],
                    "score": score,
                    "semantic_score": score,
                }
            )
        demo_results.sort(key=lambda item: float(item.get("score", 0.0)), reverse=True)
        return demo_results[:top_k]

    query_vec = RUNTIME.embed_model.encode([query])
    faiss.normalize_L2(query_vec)
    distances, ann_indices = RUNTIME.search_index.search(query_vec, top_k)

    conn = connect_db()
    cursor = conn.cursor()
    results: List[Dict] = []
    try:
        for index, ann_idx in enumerate(ann_indices[0]):
            if ann_idx == -1:
                continue
            try:
                real_db_id = int(RUNTIME.db_ids_map[ann_idx])
            except Exception:
                continue

            cursor.execute(
                "SELECT id, case_name, content, judgement FROM cases WHERE id = %s",
                (real_db_id,),
            )
            row = cursor.fetchone()
            if not row:
                continue

            score = float(distances[0][index])
            results.append(
                {
                    "id": row[0],
                    "case_name": row[1],
                    "content": row[2],
                    "judgement": row[3],
                    "score": score,
                    "semantic_score": score,
                }
            )
    finally:
        cursor.close()
        conn.close()

    return results


def normalize_top_k(top_k: int) -> int:
    """限制 Top K 范围，避免界面输入过大。"""

    try:
        value = int(top_k)
    except Exception:
        value = RETRIEVAL_CONFIG.default_top_k
    return max(1, min(value, RETRIEVAL_CONFIG.max_top_k))


def search_engine(query: str, mode: str = "混合检索", top_k: int = RETRIEVAL_CONFIG.default_top_k) -> Tuple[List[Dict], str]:
    """统一检索入口。"""

    if not query or not query.strip():
        return [], "请输入查询内容"

    top_k = normalize_top_k(top_k)
    mode = mode or "混合检索"

    try:
        if mode == "关键词检索":
            results = keyword_search(query, fetch_all_cases(), top_k=top_k)
        elif mode == "语义检索":
            raw_semantic = semantic_search(query, top_k=top_k)
            results = wrap_semantic_results(query, raw_semantic, source="semantic")
        else:
            raw_semantic = semantic_search(query, top_k=max(top_k * 2, top_k))
            semantic_results = wrap_semantic_results(query, raw_semantic, source="semantic")
            keyword_results = keyword_search(query, fetch_all_cases(), top_k=max(top_k * 2, top_k))
            results = hybrid_fusion(query, semantic_results, keyword_results, top_k=top_k)

        for item in results:
            item["reason"] = item.get("reason") or build_reason(
                query,
                item,
                float(item.get("semantic_score", 0.0) or 0.0),
                float(item.get("keyword_score", 0.0) or 0.0),
            )
            item["snippet"] = item.get("snippet") or build_snippet(
                item.get("content", ""), query, RETRIEVAL_CONFIG.snippet_length
            )

        status_prefix = "演示模式" if RUNTIME.demo_mode else "真实数据模式"
        return results, "{}：{}完成，共返回 {} 条结果。".format(status_prefix, mode, len(results))
    except Exception as exc:
        if not RUNTIME.demo_mode:
            fallback = keyword_search(query, RUNTIME.sample_cases, top_k=top_k)
            return fallback, "真实检索失败，已使用演示数据兜底：{}".format(exc)
        return [], "检索失败：{}".format(exc)


def on_click_search(query: str, mode: str, top_k: int):
    """检索按钮事件。"""

    results, msg = search_engine(query, mode, top_k)
    choices = [(item["display_label"], str(item["id"])) for item in results]
    table = build_results_table(results)
    if not choices:
        return gr.update(choices=[], value=None), table, msg, [], "", "", "", ""

    first_id = choices[0][1]
    first_case = results[0]
    return (
        gr.update(choices=choices, value=first_id),
        table,
        msg,
        results,
        first_case.get("content", ""),
        first_case.get("judgement", ""),
        first_case.get("reason", ""),
        first_case.get("snippet", ""),
    )


def on_select_case(case_id: str, results_list: List[Dict]):
    """选择案例事件。"""

    for item in results_list or []:
        if str(item.get("id")) == str(case_id):
            return (
                item.get("content", ""),
                item.get("judgement", ""),
                item.get("reason", ""),
                item.get("snippet", ""),
                "",
                "",
            )
    return "", "", "", "", "", ""


def on_click_analyze(content: str, judgement: str):
    """智能分析按钮事件。"""

    return llm_process(content, judgement)


def run_demo_evaluation(k: int = 5) -> List[List]:
    """运行三种检索模式对比评测。

    演示模式下使用内置案例编号；真实论文实验应替换为真实数据的人工标注评测集。
    """

    k = normalize_top_k(k)
    eval_cases = load_evaluation_cases(PATH_CONFIG.evaluation_cases_path)
    if not eval_cases:
        eval_cases = [
            {"query": "组织传销 发展下线 层级", "relevant_ids": [6]},
            {"query": "酒后驾驶 机动车 醉酒", "relevant_ids": [3]},
            {"query": "借条 转账 民间借贷", "relevant_ids": [2]},
        ]

    rows = []
    metrics_by_mode: Dict[str, List[Dict]] = {
        "关键词检索": [],
        "语义检索": [],
        "混合检索": [],
    }
    for item in eval_cases:
        query = item.get("query", "")
        relevant_ids = item.get("relevant_ids", [])
        for mode in metrics_by_mode.keys():
            results, _ = search_engine(query, mode, k)
            retrieved_ids = [result.get("id") for result in results]
            metrics = evaluate_single_query(retrieved_ids, relevant_ids, k=k)
            metrics_by_mode[mode].append(metrics)
            rows.append(
                [
                    mode,
                    query,
                    ",".join(str(v) for v in relevant_ids),
                    ",".join(str(v) for v in retrieved_ids),
                    round(metrics.get("Precision@{}".format(k), 0.0), 4),
                    round(metrics.get("Recall@{}".format(k), 0.0), 4),
                    round(metrics.get("MRR", 0.0), 4),
                    round(metrics.get("nDCG@{}".format(k), 0.0), 4),
                ]
            )

    for mode, metrics_list in metrics_by_mode.items():
        avg = average_metrics(metrics_list)
        if not avg:
            continue
        rows.append(
            [
                mode,
                "平均值",
                "-",
                "-",
                round(avg.get("Precision@{}".format(k), 0.0), 4),
                round(avg.get("Recall@{}".format(k), 0.0), 4),
                round(avg.get("MRR", 0.0), 4),
                round(avg.get("nDCG@{}".format(k), 0.0), 4),
            ]
        )
    return rows


def build_interface() -> gr.Blocks:
    """构建 Gradio 界面。"""

    mode_text = "演示模式" if RUNTIME.demo_mode else "真实数据模式"
    description = "当前运行状态：{}。{}".format(mode_text, RUNTIME.status)

    with gr.Blocks(title="法律文书智能检索系统") as demo:
        gr.Markdown("# ⚖️ 法律文书智能检索与信息抽取系统")
        gr.Markdown("面向毕业设计的法律文书检索、融合排序、辅助分析与实验评测原型。")
        gr.Markdown(description)

        state_results = gr.State([])

        with gr.Row():
            with gr.Column(scale=5):
                txt_query = gr.Textbox(
                    label="请输入案情描述或关键词",
                    placeholder="例如：组织传销活动 发展下线层级",
                    lines=2,
                )
            with gr.Column(scale=2):
                radio_mode = gr.Radio(
                    label="检索模式",
                    choices=["混合检索", "语义检索", "关键词检索"],
                    value="混合检索",
                )
                slider_top_k = gr.Slider(
                    label="返回结果 Top K",
                    minimum=1,
                    maximum=RETRIEVAL_CONFIG.max_top_k,
                    value=RETRIEVAL_CONFIG.default_top_k,
                    step=1,
                )
                btn_search = gr.Button("智能检索", variant="primary")

        lbl_status = gr.Markdown("等待检索...")

        with gr.Row():
            with gr.Column(scale=5):
                dropdown_cases = gr.Dropdown(label="检索结果", interactive=True)
            with gr.Column(scale=5):
                txt_reason = gr.Textbox(label="推荐理由", lines=3, interactive=False)

        result_table = gr.Dataframe(
            headers=["排名", "案例ID", "案件名称", "综合分", "语义分", "关键词分", "推荐理由"],
            label="检索结果评分表",
            interactive=False,
        )

        txt_snippet = gr.Textbox(label="命中片段", lines=3, interactive=False)

        with gr.Row():
            with gr.Column(scale=6):
                with gr.Tabs():
                    with gr.TabItem("📄 案情正文"):
                        txt_content = gr.TextArea(label="正文内容", lines=18, interactive=False)
                    with gr.TabItem("⚖️ 判决结果"):
                        txt_judgement = gr.TextArea(label="判决结果", lines=18, interactive=False)
            with gr.Column(scale=4):
                btn_analyze = gr.Button("启动 AI 智能分析", variant="secondary")
                txt_summary = gr.Textbox(label="AI/规则案情摘要", lines=8)
                txt_extraction = gr.Code(label="关键要素提取", language="json", lines=10)

        with gr.Accordion("📊 检索评测演示", open=False):
            gr.Markdown("评测模块用于论文实验。当前无真实标注文件时，会使用内置演示查询；正式论文应替换为真实数据人工标注查询。")
            btn_eval = gr.Button("运行演示评测")
            eval_table = gr.Dataframe(
                headers=["检索模式", "查询", "相关ID", "返回ID", "Precision", "Recall", "MRR", "nDCG"],
                label="三种检索模式评测结果",
                interactive=False,
            )

        btn_search.click(
            fn=on_click_search,
            inputs=[txt_query, radio_mode, slider_top_k],
            outputs=[
                dropdown_cases,
                result_table,
                lbl_status,
                state_results,
                txt_content,
                txt_judgement,
                txt_reason,
                txt_snippet,
            ],
        )

        dropdown_cases.change(
            fn=on_select_case,
            inputs=[dropdown_cases, state_results],
            outputs=[txt_content, txt_judgement, txt_reason, txt_snippet, txt_summary, txt_extraction],
        )

        btn_analyze.click(
            fn=on_click_analyze,
            inputs=[txt_content, txt_judgement],
            outputs=[txt_summary, txt_extraction],
        )

        btn_eval.click(fn=run_demo_evaluation, inputs=[slider_top_k], outputs=[eval_table])

    return demo


initialize_runtime()
demo = build_interface()


if __name__ == "__main__":
    print("正在启动 Web 服务...")
    print(RUNTIME.status)
    demo.launch(server_name=APP_CONFIG.server_name, server_port=APP_CONFIG.server_port, show_error=True)
