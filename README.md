# ⚖️ 法律文书智能检索与信息抽取系统

## 项目简介

本项目是一个基于自然语言处理（NLP）与深度学习技术实现的法律文书智能检索系统。  
系统能够根据用户输入的案情描述或关键词，利用语义向量检索技术在法律文书数据库中快速找到最相关的案件，并通过大语言模型对文书进行分析，实现案件摘要生成与关键信息抽取。

本系统为本科毕业设计原型系统，旨在探索法律文书语义检索与智能分析技术在司法信息化中的应用。

---

# 系统功能

## 1 语义检索

用户输入案件关键词或案情描述后，系统将：

1. 使用 Sentence-BERT 模型生成语义向量
2. 通过 FAISS 向量索引进行相似度搜索
3. 返回语义最相关的法律文书

相比传统关键词检索，该方法能够更好理解案件语义，提高检索相关性。

---

## 2 法律文书展示

系统从 MySQL 数据库读取案件信息，并展示：

- 案件标题
- 案情正文
- 判决结果

---

## 3 AI 智能分析

系统调用本地大语言模型（Ollama）对案件内容进行分析，实现：

- 案情摘要生成（200 字以内）
- 案件关键要素提取

抽取信息包括：

- 案由
- 法院
- 当事人
- 法律条款
- 判决结果

结果以 JSON 结构形式展示。

---

# 技术架构

系统主要使用以下技术：

| 技术                | 作用               |
| ------------------- | ------------------ |
| Python              | 后端开发           |
| SentenceTransformer | 语义向量编码       |
| FAISS               | 向量相似度检索     |
| MySQL               | 案件数据存储       |
| Gradio              | Web 交互界面       |
| Ollama              | 本地大语言模型调用 |
| ModelScope          | 模型下载           |

系统流程：
用户输入查询
│
Sentence-BERT 生成语义向量
│
FAISS 向量检索
│
MySQL 获取案件数据
│
Gradio Web 界面展示
│
LLM 进行摘要与信息抽取

```
project/
├── LeCaDR/
│   ├── app.py
│   ├── build_index.py
│   └── data_import.py
├── README.md
└── requirements.txt
```


# 数据来源

系统使用 **LeCaRD（Legal Case Retrieval Dataset）** 作为法律文书数据集。

该数据集包含大量真实裁判文书，可用于法律检索与语义匹配研究。

---

# 运行环境

推荐环境：

Python >= 3.9
MySQL >= 5.7

---

# 依赖安装

安装项目依赖：

pip install gradio pymysql faiss-cpu numpy sentence-transformers modelscope openai tqdm

或使用：

pip install -r requirements.txt

---

# 数据导入

首先创建数据库：

legal_ir

并创建数据表 `cases`：

```sql
CREATE TABLE cases (
    id INT PRIMARY KEY AUTO_INCREMENT,
    aj_id VARCHAR(100),
    writ_id VARCHAR(100),
    case_name TEXT,
    content LONGTEXT,
    judgement LONGTEXT
);

然后运行：

python data_import.py

脚本将：

读取 LeCaRD JSON 数据

清洗文本

导入 MySQL 数据库

构建向量索引

数据导入完成后运行：

python build_index.py

该脚本会：

下载语义模型 bge-large-zh-v1.5

将法律文书转换为向量

构建 FAISS 向量索引

生成文件：

legal_vector.index
db_ids.npy
启动系统

运行：

python app.py

启动成功后访问：

http://localhost:7860

即可进入系统界面。

使用流程

输入案件关键词或案情描述

点击 智能检索

系统返回语义最相关案件

选择案件查看详细内容

点击 AI 智能分析

系统生成案件摘要与关键信息

项目说明

本系统为毕业设计实现的研究原型，主要用于验证法律文书语义检索与智能分析方法。

部分功能依赖本地环境（如 Ollama 大语言模型服务）。
```
