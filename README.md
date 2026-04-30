# 法律文书智能检索与信息抽取系统

本项目是一个面向本科毕业设计的法律文书智能检索与辅助分析系统。系统围绕法律文书检索场景，集成 BM25 关键词检索、语义向量检索、混合检索、推荐理由展示、摘要生成、关键要素抽取与检索评测功能。

项目支持两种运行方式：

1. **演示模式**：当前电脑没有 LeCaRD 数据集、MySQL 数据库或 FAISS 索引时，系统会自动使用内置样例案例启动，方便调试界面和答辩演示。
2. **真实数据模式**：导入 LeCaRD 数据并构建 FAISS 索引后，系统使用 MySQL + SentenceTransformer + FAISS 完成真实数据检索。

> 注意：演示模式只用于功能展示和截图兜底，不建议把演示数据结果作为论文正式实验结论。

## 1. 项目定位

论文题目可围绕：

> 基于语义向量与混合检索的法律文书智能检索系统设计与实现

系统目标不是训练全新的大模型，而是在已有法律文书数据集基础上完成一个具有工程完整性和实验分析支撑的检索应用原型。

主要解决的问题：

- 传统关键词检索难以理解案情语义。
- 单一向量检索缺少关键词命中解释。
- 法律文书内容较长，用户需要摘要和关键要素辅助理解。
- 毕业论文需要可复现实验指标支撑系统有效性。

## 2. 功能模块

| 模块 | 文件 | 说明 |
|---|---|---|
| 统一配置 | `LeCaDR/config.py` | 管理数据库、模型、索引路径、检索参数和演示模式 |
| 数据导入 | `LeCaDR/data_import.py` | 递归读取 LeCaRD JSON 文书并写入 MySQL |
| 索引构建 | `LeCaDR/build_index.py` | 使用 SentenceTransformer 生成向量并构建 FAISS 索引 |
| 检索逻辑 | `LeCaDR/retrieval.py` | 实现 BM25 关键词检索、语义结果包装和混合检索融合 |
| 文本处理 | `LeCaDR/text_utils.py` | 文本清洗、分词、命中片段、RAG 分片和推荐理由生成 |
| 智能分析 | `LeCaDR/llm_utils.py` | 调用 Ollama 兼容接口进行摘要和要素抽取，失败时规则兜底 |
| 评测指标 | `LeCaDR/evaluation.py` | Precision@K、Recall@K、MRR、nDCG@K |
| 演示数据 | `LeCaDR/sample_data.py` | 无数据集环境下的内置案例 |
| Web 界面 | `LeCaDR/app.py` | Gradio 交互界面 |
| 评测模板 | `LeCaDR/evaluation_cases.json` | 示例查询和相关案例编号 |

## 3. 系统架构

```text
用户查询
  |
  +-- BM25 关键词检索：计算查询词与文书词项的相关性
  |
  +-- 语义检索：SentenceTransformer 向量编码 + FAISS 相似度搜索
  |
  +-- 混合检索：语义分数与关键词分数归一化后加权融合
        |
        +-- 推荐理由与命中片段展示
        +-- 法律文书正文与判决结果展示
        +-- RAG 分片 + LLM / 规则方法生成摘要与要素抽取
        +-- Precision@K、Recall@K、MRR、nDCG@K 实验评价
```

混合检索默认融合方式：

```text
最终分数 = 语义分数 * 0.7 + 关键词分数 * 0.3
```

权重可在 `LeCaDR/config.py` 中通过环境变量或配置项调整。

## 4. 运行环境

推荐环境：

- Python 3.9+
- MySQL 5.7+
- Windows / Linux
- 可选：Ollama 本地大模型服务

安装依赖：

```bash
pip install -r requirements.txt
```

## 5. 演示模式运行

当前电脑没有真实数据集时，直接运行：

```bash
python LeCaDR/app.py
```

系统会自动检查依赖库、MySQL、FAISS 索引和数据库案例数量。如果检查失败，会自动进入演示模式，并使用 `LeCaDR/sample_data.py` 中的案例启动界面。

访问地址：

```text
http://localhost:7860
```

演示模式可用于界面截图、检索流程展示、推荐理由展示、摘要和要素抽取功能演示、评测模块流程演示。

## 6. 真实数据模式运行

### 6.1 创建数据库

```sql
CREATE DATABASE IF NOT EXISTS legal_ir DEFAULT CHARSET utf8mb4;
USE legal_ir;
```

`LeCaDR/data_import.py` 会自动创建 `cases` 表。

### 6.2 配置数据集和数据库

可以在 `LeCaDR/config.py` 中修改默认值，也可以通过环境变量覆盖：

```bash
set LEGAL_IR_DATA_ROOT=D:\path\to\LeCaRD-main\data\candidates
set LEGAL_IR_DB_HOST=localhost
set LEGAL_IR_DB_PORT=3306
set LEGAL_IR_DB_USER=root
set LEGAL_IR_DB_PASSWORD=你的数据库密码
set LEGAL_IR_DB_NAME=legal_ir
```

### 6.3 导入数据

```bash
python LeCaDR/data_import.py
```

### 6.4 构建向量索引

```bash
python LeCaDR/build_index.py
```

生成文件：

- `LeCaDR/legal_vector.index`
- `LeCaDR/db_ids.npy`

### 6.5 启动系统

```bash
python LeCaDR/app.py
```

如果数据库和索引加载成功，界面会显示“真实数据模式”。

## 7. 检索模式说明

### 7.1 BM25 关键词检索

关键词检索使用 BM25 排分方法，对查询和文书标题、正文、判决结果进行匹配。该方法可解释性强，适合作为论文实验中的基线方法。

### 7.2 语义检索

语义检索使用 SentenceTransformer 模型将查询和文书转化为向量，然后使用 FAISS 进行相似度检索。该方法对自然语言案情描述更友好，但关键词命中解释性较弱。

### 7.3 混合检索

混合检索融合 BM25 分数和语义分数，兼顾语义理解与关键词可解释性。论文中可将其作为系统改进方法，与关键词检索、语义检索做对比实验。

## 8. 智能分析说明

系统在 `LeCaDR/llm_utils.py` 中实现了 RAG 分片与证据约束提示：

- 先按长文书分片并筛选关键证据片段，避免直接截断遗漏“本院认为”“判决如下”等后置内容。
- 提示词要求模型只依据证据片段输出摘要和要素，证据不足时写“未识别”。
- 默认使用隐式 CoT / 证据核验提示，但最终只输出 JSON，不展示推理链。
- 当 OpenAI SDK、Ollama 服务或模型不可用时，系统自动使用规则方法兜底。

可通过环境变量切换提示词策略：

```bash
set LEGAL_IR_LLM_PROMPT_STRATEGY=rag_cot
set LEGAL_IR_LLM_PROMPT_STRATEGY=basic
```

## 9. 检索评测

项目提供以下评价指标：

| 指标 | 含义 |
|---|---|
| Precision@K | 前 K 个检索结果中相关文书比例 |
| Recall@K | 相关文书中被前 K 个结果召回的比例 |
| MRR | 第一个相关结果出现位置的倒数 |
| nDCG@K | 考虑排序位置的归一化折损累计增益 |

评测样例文件：

```text
LeCaDR/evaluation_cases.json
```

当前文件提供 12 条演示查询，覆盖民间借贷、危险驾驶、合同诈骗、劳动合同、交通事故、传销等类型。真实论文实验建议准备 10 到 30 条真实数据查询，每条人工标注 1 到若干相关案例编号，然后在界面评测模块中分别比较：

1. BM25 关键词检索。
2. 语义检索。
3. 混合检索。

论文实验表格建议：

| 方法 | Precision@5 | Recall@5 | MRR | nDCG@5 |
|---|---:|---:|---:|---:|
| BM25 关键词检索 | 待实验 | 待实验 | 待实验 | 待实验 |
| 语义检索 | 待实验 | 待实验 | 待实验 | 待实验 |
| 混合检索 | 待实验 | 待实验 | 待实验 | 待实验 |

## 10. 论文与截图材料

已提供论文初稿提纲与可直接扩写材料：

- `docs/thesis_initial_draft.md`
- `docs/screenshot_checklist.md`

建议截图至少包括：

1. 系统首页与运行状态。
2. 混合检索结果列表。
3. 案情正文、判决结果、推荐理由和命中片段。
4. AI / 规则摘要与要素抽取结果。
5. 三种检索模式评测结果表。

## 11. 后续可扩展方向

- 增加案由、法院、年份等筛选条件。
- 使用交叉编码器进行重排序。
- 将评测结果导出为 CSV。
- 增加数据统计页面，如案由分布、文书长度分布。
- 增加用户查询日志，用于后续系统分析。

本系统为毕业设计原型，重点展示法律文书智能检索系统的完整流程：数据导入、向量索引、检索融合、智能分析、实验评测和 Web 展示。
