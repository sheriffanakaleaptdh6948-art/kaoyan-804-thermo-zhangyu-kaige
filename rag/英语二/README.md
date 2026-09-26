# 英语二 · 单词 RAG + 真题 RAG（向量库）

> 从本地的「英语精读系统」导出的独立 ChromaDB 向量库。
> **只含单词与真题两类**，不含完形 / 新题型 / 写作 / 贴吧四个库。
> 已建好，clone 下来即可检索，不需要重新向量化。

---

## 〇、先看这一条：必须放在纯 ASCII 路径下

**本库在含中文的路径下会读不出来**，报错：

```
chromadb.errors.InternalError:
Error executing plan: ... Error creating hnsw segment reader: Error loading hnsw index
```

这是 chromadb 的 Rust 后端打不开**非 ASCII 路径**下的 HNSW 索引文件所致，**与库本身无关**。
实测（2026-09-26，同一份库、同样的操作）：

| 库所在路径 | 结果 |
|---|---|
| 纯 ASCII 路径（如 `D:\rag_db`） | 正常，6910 条、检索可用 |
| 含中文的路径 | 失败，`Error loading hnsw index` |

（测试时分别放在纯英文目录、一个最短的中文目录、以及本项目的中文目录下，
结果一致：只有纯 ASCII 路径能读。）

所以：**clone 到中文目录后，请把 `rag_db/` 整个拷到一个纯英文路径再用**（例如 `D:\rag_db\`）。

---

## 〇·二、目录里有什么

```
rag/英语二/
├── README.md              本文件
├── rag_db/                向量库本体（ChromaDB，可直接检索）
├── source/                原始数据（JSONL，构建向量库的输入）
│   ├── 真题/              reading_questions / reading_passages / exam_explanations
│   └── 单词/              vocab_merged / redbook_vocab / oxfordCN
├── build/                 构建脚本
│   ├── build_exam_rag.py          真题 -> 向量库
│   ├── build_rag_from_merged.py   单词 -> 向量库
│   ├── extract_explanations_v3.py 逐题解析提取
│   ├── import_ocr_exams.py        真题 OCR 导入
│   └── fix_exam_rag.py            真题库质量修复
└── 纯文本/                **不装任何依赖就能读**
    ├── 真题.md            文章 + 题目 + 选项 + 答案 + 解析（可折叠）
    └── 单词表/A.md…Z.md   6550 词，按首字母分 26 册
```

**三条使用路线，任选：**

| 想干什么 | 用什么 |
|---|---|
| 做 RAG 检索 | `rag_db/`（注意路径必须是纯 ASCII，见上） |
| 直接读 / grep / 喂给别的工具 | `纯文本/`（Markdown，零依赖） |
| 自己重建、改 chunk 策略 | `build/` + `source/` |

---

## 一、内容

| `type` | 条数 | 说明 |
|---|---:|---|
| `vocab` | **6550** | 红宝书 × 牛津 双源词汇 |
| `exam_question` | **300** | 考研英语一 2010–2024 阅读题目（题干 + 选项 + 答案） |
| `exam_passage` | **60** | 阅读文章段落 |
| | **6910** | |

- **collection 名**：`kaoyan_knowledge`
- **向量维度**：384
- **embedding 模型**：`paraphrase-multilingual-MiniLM-L12-v2`
  —— 查询时要用**同一个模型**编码 query，否则向量空间对不上

**真题年份**：2010–2024 **连续 15 年**，每年 24 条，**标注为考研英语一**（不是英语二）。

**题型分布**（`question_type`）：`reading` 200 条（未细分）+ 已标注的 100 条，
其中细节题 56 / 词义题 13 / 推断题 12 / 结构题 8 / 态度题 6 / 主旨题 5。

**词汇词性**（`pos`）前三：`n.` 3335、`v.` 1341、`adj.` 1310。

---

## 二、怎么用

```python
import chromadb
from chromadb.utils import embedding_functions

# 注意路径：必须是纯 ASCII
cli = chromadb.PersistentClient(path=r"D:\rag_db")
col = cli.get_collection("kaoyan_knowledge")

ef = embedding_functions.SentenceTransformerEmbeddingFunction(
    model_name="paraphrase-multilingual-MiniLM-L12-v2")

# 例：找相似的历史真题
vec = ef(["作者对某现象持什么态度"])
res = col.query(query_embeddings=vec, n_results=5,
                where={"type": "exam_question"})
for d, m in zip(res["documents"][0], res["metadatas"][0]):
    print(m.get("source"), m.get("q_num"), d[:80])

# 例：查单词
res = col.query(query_embeddings=ef(["辐射的"]), n_results=3,
                where={"type": "vocab"})
print(res["metadatas"][0])
```

---

## 三、metadata 字段

| 适用 | 字段 |
|---|---|
| 全部 | `type`（`vocab` / `exam_question` / `exam_passage`）、`type_label`、`source` |
| 仅 `vocab` | `word`、`pos`、`phonetic`、`redbook_cn`、`oxford_cn`、`word_family`、`examples`、`word_id` |
| 仅 `exam_question` | `year`、`text_id`、`q_num`、`question_type`、`answer` |
| 仅 `exam_passage` | `year`、`text_id` |

`source` 的取值：`红宝书×牛津`（词汇）与 `2010考研英语一` ～ `2024考研英语一`（真题）。

---

## 四、数据质量提醒（重要）

**逐题解析（`exam_explanations.jsonl`、`纯文本/真题.md` 里可折叠的那部分）来自 OCR，实测约 37%（112/300）含识别错误：**

| 类型 | 例子 |
|---|---|
| 选项标记 | `[B]` 被识别成 `[BJ` |
| 英文错字 | `criticism` -> `cntic1sm`、`His` -> `H,s`、`journalistic` -> `JOurnal1stic` |
| 中文错字 | `新闻` -> `新间`、`工作` -> `丁作`、`已` -> `巳` |

另外**部分解析条目的 `answer` 字段是空的**。

> 这也是**为什么向量库里没有 `exam_strategy` 这一类 chunk** ——
> 解析文本质量不足以直接进库。`build_exam_rag.py` 的代码里设计了这一类，
> 但实际构建出的库里 `type` 只有 `vocab` / `exam_question` / `exam_passage` 三种。

**题目、选项、答案（`reading_questions.jsonl`）质量正常**，可放心用。

---

## 五、没包含什么

1. **向量库里没有逐题解析**（原因见上）。
2. **不含**完形填空 / 新题型 / 写作 / 贴吧 四个库（导出时已剔除，正文与向量均已确认清零）。
3. **不含**任何用户数据（背单词进度、错题、关注词、阅读记录等）。
4. **年份口径不一样**：向量库 `rag_db/` 只含 **2010–2024**
   （`build_exam_rag.py` 里 `EXCLUDE_YEARS = {2025}`）；
   而 `source/` 与 `纯文本/` 保留原始数据的全部 **2010–2025**。

---

## 六、出处与声明

- 词汇释义来自**红宝书**与**牛津高阶英汉双解词典**；真题来自**考研英语一历年真题**（2010–2024）。
- 所有原始资料的**著作权归原作者与出版方所有**。本库不主张对原文内容的任何权利。
- 个人备考用途，**请勿商用**。建议支持正版：购买原书、原课程。
- 若权利方认为内容不妥，请联系删除。

---

*导出日期：2026-09-26*
