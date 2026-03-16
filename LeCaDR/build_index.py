import os
import pymysql
import faiss
import numpy as np
from sentence_transformers import SentenceTransformer
from tqdm import tqdm
from modelscope import snapshot_download


# 配置区域
DB_CONFIG = {
    'host': 'localhost', 'user': 'root', 'password': 'your_password',
    'db': 'legal_ir', 'charset': 'utf8mb4'
}

# 这里只写模型名字，用来给 modelscope 下载
MODEL_ID = 'Ceceliachenen/bge-large-zh-v1.5'
INDEX_FILE = "legal_vector.index"
ID_MAP_FILE = "db_ids.npy"


def build_vector_index():
    # ... (清理旧文件的代码不变) ...
    if os.path.exists(INDEX_FILE): os.remove(INDEX_FILE)
    if os.path.exists(ID_MAP_FILE): os.remove(ID_MAP_FILE)

    print("🚀 开始重建索引...")
    conn = pymysql.connect(**DB_CONFIG)
    cursor = conn.cursor()

    # ======= 修改：使用 ModelScope 下载并加载 =======
    print(f"📥 正在从魔搭社区(ModelScope)下载模型: {MODEL_ID} ...")
    try:
        # cache_dir 可以指定下载路径，不指定则默认在 C盘用户目录
        model_dir = snapshot_download(MODEL_ID)
        print(f"✅ 模型已下载至: {model_dir}")

        # 加载本地下载好的模型
        model = SentenceTransformer(model_dir)
    except Exception as e:
        print(f"❌ 模型下载/加载失败: {e}")
        return
    # ==============================================

    # 2. 检查数据源
    cursor.execute("SELECT count(*) FROM cases")
    total = cursor.fetchone()[0]
    if total == 0:
        print("❌ 严重错误：数据库是空的！请先运行入库脚本！")
        return

    # 3. 初始化索引
    dimension = model.get_sentence_embedding_dimension()
    print(f"📏 模型向量维度自动识别为: {dimension}")
    index = faiss.IndexFlatIP(dimension)

    batch_size = 500
    cursor.execute("SELECT id, case_name, content FROM cases")
    # 【测试模式】只读取前 30 条用于快速验证
    #print("⚠️ 正在运行测试模式：只处理前 30 条数据...")
    #cursor.execute("SELECT id, case_name, content FROM cases LIMIT 30")

    db_ids_buffer = []
    processed_count = 0

    pbar = tqdm(total=total, unit="doc")

    while True:
        results = cursor.fetchmany(batch_size)
        if not results: break

        texts_to_encode = []
        current_ids = []

        for row in results:
            case_id, title, content = row[0], row[1], row[2]

            # 确保内容不是 None
            if content is None: content = ""
            if title is None: title = ""

            # 组合文本
            # 这里的 [:400] 很关键，确保取到了开头
            input_text = f"{title} {content[:400]}".strip().replace('\n', '')

            if not input_text:
                input_text = "无内容"  # 防止空字符串导致报错

            texts_to_encode.append(input_text)
            current_ids.append(case_id)

        # 批量向量化
        embeddings = model.encode(texts_to_encode)
        faiss.normalize_L2(embeddings)  # 归一化
        index.add(embeddings)
        db_ids_buffer.extend(current_ids)

        processed_count += len(results)
        pbar.update(len(results))

    pbar.close()

    # 保存
    faiss.write_index(index, INDEX_FILE)
    np.save(ID_MAP_FILE, np.array(db_ids_buffer))

    print(f"\n✅ 索引重建完成！包含 {index.ntotal} 条数据。")


if __name__ == "__main__":
    build_vector_index()