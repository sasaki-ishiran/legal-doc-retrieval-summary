"""LeCaRD 数据导入脚本。

回到有真实数据集的电脑后，先在 LeCaDR/config.py 或环境变量 LEGAL_IR_DATA_ROOT 中配置数据集路径，再运行本脚本。
"""

from __future__ import annotations

import json
import os
from typing import Set

import pymysql
from tqdm import tqdm

from config import DB_CONFIG, PATH_CONFIG
from text_utils import clean_text


CREATE_TABLE_SQL = """
CREATE TABLE IF NOT EXISTS cases (
    id INT PRIMARY KEY AUTO_INCREMENT,
    aj_id VARCHAR(100),
    writ_id VARCHAR(100),
    case_name TEXT,
    content LONGTEXT,
    judgement LONGTEXT,
    INDEX idx_aj_id (aj_id)
) DEFAULT CHARSET=utf8mb4;
"""


def connect_db():
    """连接 MySQL 数据库。"""

    return pymysql.connect(**DB_CONFIG.to_pymysql_kwargs())


def ensure_table(cursor) -> None:
    """确保 cases 表存在。"""

    cursor.execute(CREATE_TABLE_SQL)


def load_existing_ids(cursor) -> Set[str]:
    """读取已有 aj_id，用于去重。"""

    cursor.execute("SELECT aj_id FROM cases")
    return {str(row[0]) for row in cursor.fetchall() if row[0]}


def collect_json_files(data_root_path: str) -> list:
    """递归扫描数据集目录下的 JSON 文件。"""

    json_files = []
    for root, _, files in os.walk(data_root_path):
        for file_name in files:
            if file_name.lower().endswith(".json"):
                json_files.append(os.path.join(root, file_name))
    return json_files


def import_cases() -> None:
    """导入 LeCaRD JSON 数据到 MySQL。"""

    data_root_path = PATH_CONFIG.data_root_path
    if not os.path.exists(data_root_path):
        print("❌ 数据集路径不存在：{}".format(data_root_path))
        print("请在 LeCaDR/config.py 中修改 data_root_path，或设置环境变量 LEGAL_IR_DATA_ROOT。")
        return

    print("正在连接数据库...")
    try:
        conn = connect_db()
        cursor = conn.cursor()
        ensure_table(cursor)
        conn.commit()
    except Exception as exc:
        print("❌ 数据库连接或建表失败：{}".format(exc))
        return

    print("正在加载已有数据 ID 以便去重...")
    existing_ids = load_existing_ids(cursor)
    print("✅ 数据库中已有 {} 条记录。".format(len(existing_ids)))

    print("正在扫描目录：{}".format(data_root_path))
    json_files = collect_json_files(data_root_path)
    print("📂 共发现 {} 个 JSON 文件。".format(len(json_files)))

    success_count = 0
    skip_count = 0
    error_count = 0

    insert_sql = """
        INSERT INTO cases (aj_id, writ_id, case_name, content, judgement)
        VALUES (%s, %s, %s, %s, %s)
    """

    for file_path in tqdm(json_files, desc="入库进度", unit="file"):
        try:
            with open(file_path, "r", encoding="utf-8") as file:
                data = json.load(file)

            aj_id = clean_text(data.get("ajId", ""))
            if not aj_id:
                error_count += 1
                continue

            if aj_id in existing_ids:
                skip_count += 1
                continue

            writ_id = clean_text(data.get("writId", ""))
            case_name = clean_text(data.get("ajName") or data.get("writName") or "未知案件名称")
            content = clean_text(data.get("qw", ""))
            judgement = clean_text(data.get("pjjg", ""))

            cursor.execute(insert_sql, (aj_id, writ_id, case_name, content, judgement))
            existing_ids.add(aj_id)
            success_count += 1

            if success_count % 100 == 0:
                conn.commit()
        except Exception:
            error_count += 1
            continue

    conn.commit()
    cursor.close()
    conn.close()

    print("\n" + "=" * 40)
    print("🎉 处理完成！")
    print("✅ 成功入库：{} 条".format(success_count))
    print("⏭️ 重复跳过：{} 条".format(skip_count))
    print("❌ 错误/忽略：{} 条".format(error_count))
    print("=" * 40)


if __name__ == "__main__":
    import_cases()
