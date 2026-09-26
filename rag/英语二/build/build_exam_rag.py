"""
build_exam_rag.py — 把考研阅读真题写入 ChromaDB 向量库（三类 chunk）

  exam_passage   — 文章段落（用于相似话题检索）
  exam_question  — 题目+选项+答案（用于相似历史题检索）
  exam_strategy  — 逐题解析（来自 exam_explanations.jsonl，最核心）

覆盖年份：2010-2024（排除 2025）
全程本地向量化，不消耗任何 API。

用法：
  python build_exam_rag.py          # 追加入库
  python build_exam_rag.py --reset  # 先删真题条目再重建（不影响词汇）
"""

import json
import argparse
import re
from pathlib import Path


QUESTIONS_FILE     = Path("data/exams/reading_questions.jsonl")
PASSAGES_FILE      = Path("data/exams/reading_passages.jsonl")
EXPLANATIONS_FILE  = Path("data/exams/exam_explanations.jsonl")
DB_PATH            = "rag_db"
COLLECTION         = "kaoyan_knowledge"
EMBED_MODEL        = "paraphrase-multilingual-MiniLM-L12-v2"
EXCLUDE_YEARS      = {2025}
BATCH_SIZE         = 50


# ══════════════════════════════════════════════════════
#  构建 rag_text（被向量化的文本）
# ══════════════════════════════════════════════════════

def passage_rag_text(p: dict) -> str:
    """文章chunk：年份+篇号标签 + 正文前600字（语义代表性最强的部分）"""
    text = p['passage'][:600].replace('\n', ' ')
    return f"[{p['year']}考研英语一 Text{p['text_id']}] {text}"


def question_rag_text(q: dict) -> str:
    """题目chunk：题型标签 + 题干 + ABCD选项 + 正确答案"""
    opts = ' '.join(f"[{k}]{v[:40]}" for k, v in sorted(q['options'].items()))
    return (
        f"[{q['question_type']}] {q['stem']}\n"
        f"{opts}\n"
        f"Answer:{q['answer']}"
    )


def strategy_rag_text(q: dict, explanation: str) -> str:
    """解析chunk：题干 + 官方解析文字"""
    expl = (explanation or '')[:800].replace('\n', ' ')
    qtype = q.get('question_type', 'reading')
    return f"[解题策略 {qtype}] {q['stem'][:80]}... {expl}"


# ══════════════════════════════════════════════════════
#  构建 metadata（ChromaDB 过滤用，仅支持 str/int/float/bool）
# ══════════════════════════════════════════════════════

def passage_meta(p: dict) -> dict:
    return {
        "type":       "exam_passage",
        "type_label": "考研真题文章",
        "source":     f"{p['year']}考研英语一",
        "year":       p['year'],
        "text_id":    p['text_id'],
        "q_nums":     ','.join(str(n) for n in p.get('q_nums', [])),
    }


def question_meta(q: dict) -> dict:
    return {
        "type":          "exam_question",
        "type_label":    "考研真题题目",
        "source":        f"{q['year']}考研英语一",
        "year":          q['year'],
        "text_id":       q['text_id'],
        "q_num":         q['q_num'],
        "question_type": q['question_type'],
        "answer":        q['answer'] or '',
    }


def strategy_meta(q: dict) -> dict:
    return {
        "type":          "exam_strategy",
        "type_label":    "解题策略",
        "source":        f"{q['year']}考研英语一解析",
        "year":          q['year'],
        "text_id":       q['text_id'],
        "q_num":         q['q_num'],
        "question_type": q['question_type'],
        "answer":        q['answer'] or '',
    }


# ══════════════════════════════════════════════════════
#  删除旧的真题条目（不影响词汇 vocab 条目）
# ══════════════════════════════════════════════════════

def delete_exam_entries(collection):
    exam_types = ["exam_passage", "exam_question", "exam_strategy"]
    total_deleted = 0
    for t in exam_types:
        try:
            res = collection.get(where={"type": {"$eq": t}})
            ids = res['ids']
            if ids:
                collection.delete(ids=ids)
                total_deleted += len(ids)
                print(f"    已删除 {t}: {len(ids)} 条")
        except Exception as e:
            print(f"    删除 {t} 时出错: {e}")
    print(f"    共删除旧真题条目: {total_deleted}")


# ══════════════════════════════════════════════════════
#  批量写入
# ══════════════════════════════════════════════════════

def batch_add(collection, ids, docs, metas):
    """分批写入，跳过已存在的 id"""
    written = 0
    for i in range(0, len(ids), BATCH_SIZE):
        b_ids   = ids[i:i+BATCH_SIZE]
        b_docs  = docs[i:i+BATCH_SIZE]
        b_metas = metas[i:i+BATCH_SIZE]
        try:
            existing     = collection.get(ids=b_ids)
            existing_set = set(existing['ids'])
        except Exception:
            existing_set = set()

        new = [(a, b, c) for a, b, c in zip(b_ids, b_docs, b_metas)
               if a not in existing_set and b.strip()]
        if new:
            collection.add(
                ids       = [x[0] for x in new],
                documents = [x[1] for x in new],
                metadatas = [x[2] for x in new],
            )
            written += len(new)
    return written


# ══════════════════════════════════════════════════════
#  主函数
# ══════════════════════════════════════════════════════

def main(reset: bool = False, skip_strategies: bool = False):
    try:
        import chromadb
        from chromadb.utils import embedding_functions
    except ImportError:
        print("请先安装: pip install chromadb sentence-transformers")
        return

    print("=" * 55)
    print("  考研真题 RAG 入库工具  (2010-2024)")
    print("=" * 55)

    # 加载数据
    print(f"\n[1] 加载真题数据...")
    with open(QUESTIONS_FILE, encoding='utf-8') as f:
        all_questions = [json.loads(l) for l in f if l.strip()]
    with open(PASSAGES_FILE, encoding='utf-8') as f:
        all_passages = [json.loads(l) for l in f if l.strip()]

    # 加载解析，建立 (year, q_num) → explanation 索引
    expl_index = {}
    if EXPLANATIONS_FILE.exists():
        with open(EXPLANATIONS_FILE, encoding='utf-8') as f:
            for l in f:
                if not l.strip():
                    continue
                e = json.loads(l)
                key = (str(e.get('year')), int(e.get('q_num', 0)))
                expl_index[key] = e.get('explanation', '')

    questions = [q for q in all_questions if q['year'] not in EXCLUDE_YEARS]
    passages  = [p for p in all_passages  if p['year'] not in EXCLUDE_YEARS]

    # 有解析的题目（来自 exam_explanations.jsonl）
    strategies = []
    for q in questions:
        key = (str(q['year']), int(q['q_num']))
        expl = expl_index.get(key, '')
        if expl.strip():
            strategies.append((q, expl))

    print(f"    文章:   {len(passages)} 篇  (排除年份: {EXCLUDE_YEARS})")
    print(f"    题目:   {len(questions)} 道")
    print(f"    解析:   {len(strategies)} 题  (来自 exam_explanations.jsonl)")

    # 连接向量库
    print(f"\n[2] 连接向量库: {DB_PATH}")
    Path(DB_PATH).mkdir(parents=True, exist_ok=True)
    client = chromadb.PersistentClient(path=DB_PATH)
    ef = embedding_functions.SentenceTransformerEmbeddingFunction(
        model_name=EMBED_MODEL
    )
    collection = client.get_or_create_collection(
        name=COLLECTION,
        embedding_function=ef,
        metadata={"hnsw:space": "cosine"}
    )
    print(f"    库中现有条目: {collection.count()}")

    # 可选：删除旧真题条目
    if reset:
        print(f"\n[--reset] 删除旧真题条目...")
        delete_exam_entries(collection)

    # 写入
    print(f"\n[3] 写入三类 chunk...")

    # ── passage chunks ──
    p_ids   = [f"exam_passage_{p['year']}_T{p['text_id']}" for p in passages]
    p_docs  = [passage_rag_text(p)  for p in passages]
    p_metas = [passage_meta(p)      for p in passages]
    w1 = batch_add(collection, p_ids, p_docs, p_metas)
    print(f"    exam_passage  写入: {w1}/{len(passages)}")

    # ── question chunks ──
    q_ids   = [f"exam_q_{q['id']}" for q in questions]
    q_docs  = [question_rag_text(q)  for q in questions]
    q_metas = [question_meta(q)      for q in questions]
    w2 = batch_add(collection, q_ids, q_docs, q_metas)
    print(f"    exam_question 写入: {w2}/{len(questions)}")

    # ── strategy chunks（可选跳过——数据质量差时）──
    if skip_strategies:
        w3 = 0
        print(f"    exam_strategy 写入: 跳过（--skip-strategies）")
    else:
        s_ids   = [f"exam_strategy_{q['id']}" for q, _ in strategies]
        s_docs  = [strategy_rag_text(q, expl) for q, expl in strategies]
        s_metas = [strategy_meta(q)           for q, _ in strategies]
        w3 = batch_add(collection, s_ids, s_docs, s_metas)
        print(f"    exam_strategy 写入: {w3}/{len(strategies)}")


    total_written = w1 + w2 + w3
    print(f"\n    库中总量: {collection.count()}  (本次新增: {total_written})")

    # 验证检索
    print(f"\n[4] 验证检索质量...")
    tests = [
        ("细节题 According to paragraph", "exam_question"),
        ("recommend suggest propose",     "exam_strategy"),
        ("scientific publishing profit",  "exam_passage"),
    ]
    for query, type_filter in tests:
        res = collection.query(
            query_texts=[query],
            n_results=2,
            where={"type": {"$eq": type_filter}},
        )
        hits = []
        for doc, meta in zip(res['documents'][0], res['metadatas'][0]):
            hits.append(f"{meta.get('year','')} {meta.get('question_type', meta.get('text_id',''))} | {doc[:50]}...")
        print(f"\n  查询: [{query[:30]}]")
        for h in hits:
            print(f"    -> {h}")

    print(f"\n[DONE] 真题 RAG 入库完成！")
    print(f"  下一步: 更新 core/rag.py 增加真题检索函数")


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--reset', action='store_true', help='先删除旧真题条目再重建')
    parser.add_argument('--skip-strategies', action='store_true', help='跳过 exam_strategy 入厍（数据质量差时使用）')
    args = parser.parse_args()
    main(reset=args.reset, skip_strategies=args.skip_strategies)
