import os
import json
import pymysql
import re
from tqdm import tqdm

# ================= 1. 配置区域 =================

# 数据库连接配置
DB_CONFIG = {
    'host': 'localhost',
    'user': 'root',
    'password': 'your_password',
    'db': 'legal_ir',
    'charset': 'utf8mb4'
}

# 数据集根目录
# 建议指向 candidates 或 candidates1 这一层，脚本会自动遍历下面所有的子文件夹
DATA_ROOT_PATH = r"D:\PythonStudio\Projects\LS\LeCaDR\LeCaRD-main\data\candidates"


# ================= 2. 数据清洗函数 =================

def clean_text(text):
    """
    清洗文本：去除HTML标签、全角空格、多余换行
    """
    if not text:
        return ""

    # 1. 转换为字符串（防止 None）
    text = str(text)

    # 2. 去除 HTML 标签 (你的数据可能比较干净，但防一手)
    text = re.sub(r'<[^>]+>', '', text)

    # 3. 替换全角空格 (\u3000) 和 &nbsp;
    text = text.replace('\u3000', ' ').replace('&nbsp;', ' ')

    # 4. 去除多余的空白字符（将连续空格/换行合并为一个）
    # 这一步能显著压缩文本体积，提高检索效率
    text = re.sub(r'\s+', ' ', text)

    return text.strip()


# ================= 3. 主程序 =================

def main():
    print("正在连接数据库...")
    try:
        conn = pymysql.connect(**DB_CONFIG)
        cursor = conn.cursor()
    except Exception as e:
        print(f"❌ 数据库连接失败: {e}")
        return

    # 1. 获取已有 ID，用于去重
    print("正在加载已有数据 ID 以便去重...")
    cursor.execute("SELECT aj_id FROM cases")
    existing_ids = set(row[0] for row in cursor.fetchall())
    print(f"✅ 数据库中已有 {len(existing_ids)} 条记录。")

    # 2. 扫描文件列表
    print(f"正在扫描目录: {DATA_ROOT_PATH}")
    json_files = []
    # os.walk 会递归遍历所有子文件夹
    for root, dirs, files in os.walk(DATA_ROOT_PATH):
        for file in files:
            if file.endswith(".json"):
                json_files.append(os.path.join(root, file))

    print(f"📂 共发现 {len(json_files)} 个 JSON 文件。")

    # 3. 开始处理
    success_count = 0
    skip_count = 0
    error_count = 0

    # 使用 tqdm 显示进度条
    for file_path in tqdm(json_files, desc="入库进度", unit="file"):
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                data = json.load(f)

            # --- 提取字段 ---
            aj_id = data.get('ajId', '').strip()

            # 如果没有 ID，无法入库，跳过
            if not aj_id:
                error_count += 1
                continue

            # 去重检查
            if aj_id in existing_ids:
                skip_count += 1
                continue

            writ_id = data.get('writId', '').strip()

            # 案件名称处理：优先 ajName，没有则 writName
            case_name = data.get('ajName')
            if not case_name:
                case_name = data.get('writName', '未知案件名称')

            # 提取正文和判决
            # 注意：LeCaRD 数据集的 qw 字段通常包含了 ajjbqk, cpfxgc 等内容
            # 所以我们主要存 qw 即可，省空间
            content = data.get('qw', '')
            judgement = data.get('pjjg', '')

            # --- 清洗数据 ---
            clean_name = clean_text(case_name)
            clean_content = clean_text(content)
            clean_judgement = clean_text(judgement)

            # --- 入库 ---
            sql = """
                INSERT INTO cases (aj_id, writ_id, case_name, content, judgement)
                VALUES (%s, %s, %s, %s, %s)
            """
            cursor.execute(sql, (aj_id, writ_id, clean_name, clean_content, clean_judgement))

            existing_ids.add(aj_id)  # 更新内存去重表
            success_count += 1

            # 每 100 条提交一次，防止内存溢出并提高速度
            if success_count % 100 == 0:
                conn.commit()

        except Exception as e:
            # print(f"文件出错 {file_path}: {e}") # 调试时可取消注释
            error_count += 1
            continue

    # 提交剩余的数据
    conn.commit()
    cursor.close()
    conn.close()

    print("\n" + "=" * 40)
    print("🎉 处理完成！")
    print(f"✅ 成功入库: {success_count} 条")
    print(f"⏭️ 重复跳过: {skip_count} 条")
    print(f"❌ 错误/忽略: {error_count} 条")
    print("=" * 40)


if __name__ == "__main__":
    main()