"""FAISS 语义向量索引构建脚本。"""

from __future__ import annotations

import os

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

from tqdm import tqdm

from config import DB_CONFIG, MODEL_CONFIG, PATH_CONFIG, RETRIEVAL_CONFIG
from text_utils import build_rag_context, clean_text


INDEX_RETRIEVAL_QUERY = "案由 案件事实 争议焦点 本院认为 法院认为 法律条款 判决如下 裁判结果"


def connect_db():
    """连接 MySQL 数据库。"""

    if pymysql is None:
        raise RuntimeError("未安装 pymysql，无法连接 MySQL")
    return pymysql.connect(**DB_CONFIG.to_pymysql_kwargs())


def remove_old_index_files() -> None:
    """删除旧索引文件，避免新旧数据混用。"""

    for file_path in [PATH_CONFIG.index_path, PATH_CONFIG.id_map_path]:
        if os.path.exists(file_path):
            os.remove(file_path)


def load_embedding_model() -> SentenceTransformer:
    """从 ModelScope 下载并加载语义模型。"""

    if snapshot_download is None or SentenceTransformer is None:
        raise RuntimeError("未安装或无法加载 modelscope/sentence_transformers，无法构建语义索引")

    print("📥 正在从 ModelScope 定位模型：{}".format(MODEL_CONFIG.embedding_model_id))
    model_dir = snapshot_download(MODEL_CONFIG.embedding_model_id)
    print("✅ 模型路径：{}".format(model_dir))
    return SentenceTransformer(model_dir)


def build_index_text(title: str, content: str) -> str:
    """构建向量化输入文本，避免只取长文书开头导致关键信息丢失。"""

    title = clean_text(title)
    content = clean_text(content)
    if len(content) <= RETRIEVAL_CONFIG.index_text_length:
        body = content
    else:
        _, selected_chunks = build_rag_context(
            content,
            "{} {}".format(title, INDEX_RETRIEVAL_QUERY),
            chunk_size=max(RETRIEVAL_CONFIG.rag_chunk_size, RETRIEVAL_CONFIG.index_text_length),
            overlap=RETRIEVAL_CONFIG.rag_chunk_overlap,
            top_k=3,
            max_chars=max(RETRIEVAL_CONFIG.rag_chunk_size, RETRIEVAL_CONFIG.index_text_length * 2),
        )
        body = " ".join(chunk.get("text", "") for chunk in selected_chunks) or content

    max_index_chars = max(RETRIEVAL_CONFIG.index_text_length, RETRIEVAL_CONFIG.rag_chunk_size)
    text = "{} {}".format(title, body[:max_index_chars]).strip()
    return text or "无内容"


def build_vector_index() -> None:
    """从 MySQL cases 表构建 FAISS 向量索引。"""

    missing_dependencies = []
    if faiss is None:
        missing_dependencies.append("faiss")
    if np is None:
        missing_dependencies.append("numpy")
    if pymysql is None:
        missing_dependencies.append("pymysql")
    if snapshot_download is None or SentenceTransformer is None:
        missing_dependencies.append("modelscope/sentence_transformers")
    if missing_dependencies:
        print("❌ 依赖库不可用，无法重建索引：{}".format("、".join(missing_dependencies)))
        return

    remove_old_index_files()
    print("🚀 开始重建索引...")

    try:
        conn = connect_db()
        cursor = conn.cursor()
    except Exception as exc:
        print("❌ 数据库连接失败：{}".format(exc))
        return

    try:
        cursor.execute("SELECT COUNT(*) FROM cases")
        total = int(cursor.fetchone()[0])
        if total == 0:
            print("❌ 数据库 cases 表为空，请先运行 data_import.py")
            return

        model = load_embedding_model()
        dimension = model.get_sentence_embedding_dimension()
        print("📏 模型向量维度：{}".format(dimension))

        index = faiss.IndexFlatIP(dimension)
        batch_size = 500
        db_ids_buffer = []

        cursor.execute("SELECT id, case_name, content FROM cases")
        pbar = tqdm(total=total, unit="doc")

        while True:
            rows = cursor.fetchmany(batch_size)
            if not rows:
                break

            texts_to_encode = []
            current_ids = []
            for row in rows:
                case_id, title, content = row[0], row[1], row[2]
                texts_to_encode.append(build_index_text(title, content))
                current_ids.append(case_id)

            embeddings = model.encode(texts_to_encode)
            embeddings = np.asarray(embeddings, dtype="float32")
            faiss.normalize_L2(embeddings)
            index.add(embeddings)
            db_ids_buffer.extend(current_ids)
            pbar.update(len(rows))

        pbar.close()

        faiss.write_index(index, str(PATH_CONFIG.index_path))
        np.save(str(PATH_CONFIG.id_map_path), np.array(db_ids_buffer))
        print("\n✅ 索引重建完成，包含 {} 条数据。".format(index.ntotal))
        print("索引文件：{}".format(PATH_CONFIG.index_path))
        print("ID 映射：{}".format(PATH_CONFIG.id_map_path))
    finally:
        cursor.close()
        conn.close()


if __name__ == "__main__":
    build_vector_index()
