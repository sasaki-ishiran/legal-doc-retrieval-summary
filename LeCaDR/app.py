import gradio as gr
import pymysql
import faiss
import numpy as np
import json
import os
from sentence_transformers import SentenceTransformer
from modelscope import snapshot_download

# ================= 1. 配置区域 =================

# 数据库配置
DB_CONFIG = {
    'host': 'localhost',
    'user': 'root',
    'password': 'your_password',
    'db': 'legal_ir',
    'charset': 'utf8mb4'
}

# 路径配置
INDEX_PATH = "legal_vector.index"
ID_MAP_PATH = "db_ids.npy"
MODEL_ID = 'Ceceliachenen/bge-large-zh-v1.5'

#大模型配置
ENABLE_LLM = True
LLM_API_KEY = "ollama"
LLM_BASE_URL = "http://localhost:11434/v1"
LLM_MODEL_NAME = "qwen2.5:7b"


# ================= 2. 系统初始化 =================
print("系统启动中...")

# 2.1 加载语义模型
try:
    print(f"正在定位本地模型: {MODEL_ID} ...")

    model_dir = snapshot_download(MODEL_ID)
    print(f"找到模型路径: {model_dir}")

    embed_model = SentenceTransformer(model_dir)
    print("语义模型加载成功！")

except Exception as e:
    print(f"模型加载失败: {e}")
    # 如果失败，抛出异常停止运行
    raise e

# 2.2 加载 FAISS 索引
if os.path.exists(INDEX_PATH) and os.path.exists(ID_MAP_PATH):
    print("正在加载向量索引...")
    search_index = faiss.read_index(INDEX_PATH)
    db_ids_map = np.load(ID_MAP_PATH)
    print(f"索引加载成功，包含 {search_index.ntotal} 条向量")
else:
    raise FileNotFoundError("未找到索引文件！请先运行 build_index.py")


# ================= 3. 核心功能函数 =================

def search_engine(query, top_k=6):
    """
    输入：自然语言查询
    输出：检索结果列表 (ID, 标题, 分数, 摘要片段)
    """
    if not query.strip():
        return [], "请输入查询内容"

    # 1. 向量化
    query_vec = embed_model.encode([query])
    faiss.normalize_L2(query_vec)

    # 2. 检索
    distances, ann_indices = search_index.search(query_vec, top_k)

    results = []
    conn = pymysql.connect(**DB_CONFIG)
    cursor = conn.cursor()

    try:
        for i, idx in enumerate(ann_indices[0]):
            if idx == -1: continue

            # 从 map 中找回真实的 MySQL ID
            # 注意：如果索引和数据库不一致，这里可能会越界，加个 try 保护
            try:
                real_db_id = db_ids_map[idx]
            except IndexError:
                continue

            score = distances[0][i]

            # 查库
            sql = "SELECT id, case_name, content, judgement FROM cases WHERE id = %s"
            cursor.execute(sql, (int(real_db_id),))
            row = cursor.fetchone()

            if row:
                results.append({
                    "id": row[0],
                    "title": row[1],
                    "content": row[2],
                    "judgement": row[3],
                    "score": float(score),
                    "display_label": f"【相似度 {score:.2f}】{row[1]}"
                })
    except Exception as e:
        print(f"检索出错: {e}")
    finally:
        cursor.close()
        conn.close()

    return results, "检索完成"


def llm_process(content):
    """ 调用本地 Ollama 进行分析 """
    if not content:
        return "请先选择一篇文书", "{}"

    if not ENABLE_LLM:
        return "LLM 未启用", "{}"

    from openai import OpenAI

    # 连接本地 Ollama
    client = OpenAI(api_key=LLM_API_KEY, base_url=LLM_BASE_URL)

    # 提示词
    prompt = f"""
    你是一个法律助手。请分析下面的法律文书（已截断）。

    任务：
    1. 生成200字以内的案情摘要。
    2. 提取关键实体：案由、法院、当事人、法律条款、判决结果。

    请务必只输出一段合法的 JSON 文本，不要包含 Markdown 标记（如 ```json），格式如下：
    {{
        "summary": "这里写摘要内容",
        "entities": {{
            "案由": "...", "法院": "...", "当事人": "...", "法律条款": "...", "判决结果": "..."
        }}
    }}

    文书内容：
    {content[:2500]} 
    """

    try:
        response = client.chat.completions.create(
            model=LLM_MODEL_NAME,
            messages=[{"role": "user", "content": prompt}],
            temperature=0.1,
            response_format={"type": "json_object"}
        )
        res_text = response.choices[0].message.content

        # 尝试解析 JSON
        try:
            data = json.loads(res_text)
            return data.get("summary", "无摘要"), json.dumps(data.get("entities", {}), indent=2, ensure_ascii=False)
        except json.JSONDecodeError:
            # 如果模型没输出完美的 JSON，直接把全文显示出来也不错
            return "JSON 解析失败，显示原始输出：", res_text

    except Exception as e:
        print(f"Ollama 调用报错: {e}")
        return f"本地模型调用出错: {e}\n请确认 Ollama 是否已启动。", "{}"


# ================= 4. Gradio 界面构建 =================


def on_click_search(query):
    results, msg = search_engine(query)
    # 返回：下拉框选项，状态信息，暂存的结果数据
    choices = [(r['display_label'], r['id']) for r in results]
    if not choices:
        return gr.update(choices=[], value=None), "未找到相关结果", []
    # 默认选中第一个
    return gr.update(choices=choices, value=choices[0][1]), "检索成功！请在下方查看详情", results


def on_select_case(case_id, results_list):
    # 根据 ID 从暂存列表中找到对应的数据
    for res in results_list:
        if str(res['id']) == str(case_id):
            return res['content'], res['judgement'], "", ""
    return "", "", "", ""


def on_click_analyze(content):
    return llm_process(content)


# 定义界面
with gr.Blocks(title="法律文书智能检索系统") as demo:
    gr.Markdown("# ⚖️ 法律文书智能检索与信息抽取系统")
    gr.Markdown("毕业设计")

    # 状态变量
    state_results = gr.State([])

    with gr.Row():
        with gr.Column(scale=4):
            txt_query = gr.Textbox(label="请输入案情描述或关键词", placeholder="例如：组织传销活动 发展下线层级",
                                   lines=1)
        with gr.Column(scale=1):
            btn_search = gr.Button("智能检索", variant="primary")

    lbl_status = gr.Markdown("等待检索...")

    # 检索结果选择
    dropdown_cases = gr.Dropdown(label="检索结果 (按语义相似度排序)", interactive=True)

    with gr.Row():
        # 左侧：原文展示
        with gr.Column(scale=6):
            with gr.Tabs():
                with gr.TabItem("📄 案情正文"):
                    txt_content = gr.TextArea(label="正文内容", lines=20, interactive=False)
                with gr.TabItem("⚖️ 判决结果"):
                    txt_judgement = gr.TextArea(label="判决书尾部", lines=20, interactive=False)

        # 右侧：智能分析
        with gr.Column(scale=4):
            btn_analyze = gr.Button("启动 AI 智能分析", variant="secondary")
            txt_summary = gr.Textbox(label="AI 案情摘要", lines=8)
            txt_extraction = gr.Code(label="关键要素提取", language="json", lines=10)

    # --- 事件绑定 ---
    btn_search.click(
        fn=on_click_search,
        inputs=[txt_query],
        outputs=[dropdown_cases, lbl_status, state_results]
    )

    dropdown_cases.change(
        fn=on_select_case,
        inputs=[dropdown_cases, state_results],
        outputs=[txt_content, txt_judgement, txt_summary, txt_extraction]
    )

    btn_analyze.click(
        fn=on_click_analyze,
        inputs=[txt_content],
        outputs=[txt_summary, txt_extraction]
    )

if __name__ == "__main__":
    # 启动服务
    print("正在启动 Web 服务...")
    demo.launch(server_name="localhost", server_port=7860, show_error=True)
