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

## 四、没包含什么（请务必知悉）

1. **真题的逐题解析不在库里。** 原项目的构建脚本设计了第三类 chunk
   `exam_strategy`（来自 `exam_explanations.jsonl`，注释里称其为「最核心」），
   但导出的库里 `type` 只有三种取值，**没有 strategy**。
   也就是说本库能检索「文章段落」和「题目+答案」，**不能检索逐题解析**。
2. **不含**完形填空 / 新题型 / 写作 / 贴吧 四个库（导出时已剔除，正文与向量均已确认清零）。
3. **不含**构建脚本与源 JSONL —— 本次只发布建好的库。
   因此本库**只能检索、不能重建、不能调参**。
4. **不含**任何用户数据（背单词进度、错题、关注词、阅读记录等）。

---

## 五、出处与声明

- 词汇释义来自**红宝书**与**牛津高阶英汉双解词典**；真题来自**考研英语一历年真题**（2010–2024）。
- 所有原始资料的**著作权归原作者与出版方所有**。本库不主张对原文内容的任何权利。
- 个人备考用途，**请勿商用**。建议支持正版：购买原书、原课程。
- 若权利方认为内容不妥，请联系删除。

---

*导出日期：2026-09-26*
