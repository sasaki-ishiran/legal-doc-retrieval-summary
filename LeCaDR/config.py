"""项目统一配置模块。

该文件集中管理数据库、模型、索引、演示模式和检索参数，避免配置散落在多个脚本中。
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path


BASE_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = BASE_DIR.parent


@dataclass(frozen=True)
class DatabaseConfig:
    """MySQL 数据库连接配置。"""

    host: str = os.getenv("LEGAL_IR_DB_HOST", "localhost")
    port: int = int(os.getenv("LEGAL_IR_DB_PORT", "3306"))
    user: str = os.getenv("LEGAL_IR_DB_USER", "root")
    password: str = os.getenv("LEGAL_IR_DB_PASSWORD", "your_password")
    db: str = os.getenv("LEGAL_IR_DB_NAME", "legal_ir")
    charset: str = os.getenv("LEGAL_IR_DB_CHARSET", "utf8mb4")

    def to_pymysql_kwargs(self) -> dict:
        """转换为 pymysql.connect 可直接使用的参数。"""

        return {
            "host": self.host,
            "port": self.port,
            "user": self.user,
            "password": self.password,
            "db": self.db,
            "charset": self.charset,
        }


@dataclass(frozen=True)
class ModelConfig:
    """语义向量模型与本地大模型配置。"""

    embedding_model_id: str = os.getenv("LEGAL_IR_EMBEDDING_MODEL", "Ceceliachenen/bge-large-zh-v1.5")
    enable_llm: bool = os.getenv("LEGAL_IR_ENABLE_LLM", "true").lower() in {"1", "true", "yes", "on"}
    llm_api_key: str = os.getenv("LEGAL_IR_LLM_API_KEY", "ollama")
    llm_base_url: str = os.getenv("LEGAL_IR_LLM_BASE_URL", "http://localhost:11434/v1")
    llm_model_name: str = os.getenv("LEGAL_IR_LLM_MODEL", "qwen2.5:7b")


@dataclass(frozen=True)
class PathConfig:
    """数据集、索引和评测文件路径配置。"""

    data_root_path: str = os.getenv(
        "LEGAL_IR_DATA_ROOT",
        r"D:\PythonStudio\Projects\LS\LeCaDR\LeCaRD-main\data\candidates",
    )
    index_path: Path = Path(os.getenv("LEGAL_IR_INDEX_PATH", str(BASE_DIR / "legal_vector.index")))
    id_map_path: Path = Path(os.getenv("LEGAL_IR_ID_MAP_PATH", str(BASE_DIR / "db_ids.npy")))
    evaluation_cases_path: Path = Path(
        os.getenv("LEGAL_IR_EVAL_CASES", str(BASE_DIR / "evaluation_cases.json"))
    )


@dataclass(frozen=True)
class RetrievalConfig:
    """检索参数配置。"""

    default_top_k: int = int(os.getenv("LEGAL_IR_TOP_K", "6"))
    max_top_k: int = int(os.getenv("LEGAL_IR_MAX_TOP_K", "20"))
    semantic_weight: float = float(os.getenv("LEGAL_IR_SEMANTIC_WEIGHT", "0.7"))
    keyword_weight: float = float(os.getenv("LEGAL_IR_KEYWORD_WEIGHT", "0.3"))
    snippet_length: int = int(os.getenv("LEGAL_IR_SNIPPET_LENGTH", "220"))
    index_text_length: int = int(os.getenv("LEGAL_IR_INDEX_TEXT_LENGTH", "500"))


@dataclass(frozen=True)
class AppConfig:
    """Web 应用运行配置。"""

    server_name: str = os.getenv("LEGAL_IR_SERVER_NAME", "localhost")
    server_port: int = int(os.getenv("LEGAL_IR_SERVER_PORT", "7860"))
    auto_demo_mode: bool = os.getenv("LEGAL_IR_AUTO_DEMO_MODE", "true").lower() in {
        "1",
        "true",
        "yes",
        "on",
    }


DB_CONFIG = DatabaseConfig()
MODEL_CONFIG = ModelConfig()
PATH_CONFIG = PathConfig()
RETRIEVAL_CONFIG = RetrievalConfig()
APP_CONFIG = AppConfig()


def get_db_config_dict() -> dict:
    """获取兼容旧脚本的数据库配置字典。"""

    return DB_CONFIG.to_pymysql_kwargs()
