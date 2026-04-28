"""FAISS 语义向量索引构建脚本。"""

from __future__ import annotations

import os

import faiss
import numpy as np
import pymysql
from modelscope import snapshot_download
from sentence_transformers import SentenceTransformer
from tqdm import tqdm

from config import DB_CONFIG, MODEL_CONFIG, PATH_CONFIG, RETRIEVAL_CONFIG
from text_utils import clean_text


def connect_db():
    """连接 MySQL 数据库。"""

    return pymysql.connect(**DB_CONFIG.to_pymysql_kwargs())


def remove_old_index_files() -> None:
    """删除旧索引文件，避免新旧数据混用。"""

    for file_path in [PATH_CONFIG.index_path, PATH_CONFIG.id_map_path]:
        if os.path.exists(file_path):
            os.remove(file_path)


def load_embedding_model() -> SentenceTransformer:
    """从 ModelScope 下载并加载语义模型。"""

    print("📥 正在从 ModelScope 定位模型：{}".format(MODEL_CONFIG.embedding_model_id))
    model_dir = snapshot_download(MODEL_CONFIG.embedding_model_id)
    print("✅ 模型路径：{}".format(model_dir))
    return SentenceTransformer(model_dir)


def build_index_text(title: str, content: str) -> str:
    """构建向量化输入文本。"""

    title = clean_text(title)
    content = clean_text(content)
    text = "{} {}".format(title, content[: RETRIEVAL_CONFIG.index_text_length]).strip()
    return text or "无内容"


def build_vector_index() -> None:
    """从 MySQL cases 表构建 FAISS 向量索引。"""

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
