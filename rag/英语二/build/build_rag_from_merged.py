"""
build_rag_from_merged.py — 将 vocab_merged.jsonl 写入 ChromaDB 向量库

替换之前仅有中文释义的简版，升级为红宝书+牛津双源完整版。

用法：
    python build_rag_from_merged.py
    python build_rag_from_merged.py --input vocab_merged.jsonl --db rag_db --replace
"""

import json
import argparse
from pathlib import Path


def build_rag(input_path: str, db_path: str, replace: bool = True):
    try:
        import chromadb
        from chromadb.utils import embedding_functions
    except ImportError:
        print("❌ 请先安装：pip install chromadb sentence-transformers")
        return

    print("=" * 52)
    print("  词汇知识库写入工具（红宝书 × 牛津双源版）")
    print("=" * 52)

    # 读取数据
    print(f"\n[1] 读取 {input_path} ...")
    with open(input_path, 'r', encoding='utf-8') as f:
        vocab_list = [json.loads(l) for l in f if l.strip()]
    print(f"    共 {len(vocab_list)} 个词条")

    # 初始化向量库
    print(f"\n[2] 连接向量库: {db_path}")
    Path(db_path).mkdir(parents=True, exist_ok=True)
    client = chromadb.PersistentClient(path=db_path)
    ef = embedding_functions.SentenceTransformerEmbeddingFunction(
        model_name="paraphrase-multilingual-MiniLM-L12-v2"
    )

    # 删除旧集合（升级替换）
    if replace:
        try:
            client.delete_collection("kaoyan_knowledge")
            print("    已删除旧集合（升级替换）")
        except Exception:
            print("    旧集合不存在，直接新建")

    collection = client.get_or_create_collection(
        name="kaoyan_knowledge",
        embedding_function=ef,
        metadata={"hnsw:space": "cosine"}
    )

    # 分批写入
    print(f"\n[3] 向量化写入（共 {len(vocab_list)} 条，每批200条）...")
    batch_size = 200
    written    = 0
    skipped    = 0

    for i in range(0, len(vocab_list), batch_size):
        batch = vocab_list[i:i + batch_size]

        ids   = [f"redbook_{v['id']}" for v in batch]
        texts = [v['rag_text'] for v in batch]
        metas = [{
            "source":     "红宝书×牛津",
            "type":       "vocab",
            "type_label": "红宝书+牛津词汇",
            "word":       v.get('word', ''),
            "word_id":    v['id'],
            "phonetic":   v.get('phonetic', ''),
            "pos":        v.get('pos', ''),
            # 把列表字段序列化为字符串（ChromaDB metadata 不支持 list）
            "redbook_cn": ' / '.join(v.get('redbook_cn', []))[:200],
            "oxford_cn":  v.get('oxford_cn', '')[:100],
            "examples":   ' | '.join(v.get('examples', [])[:2])[:200],
            "word_family": v.get('word_family', '')[:100],
        } for v in batch]

        # 过滤空 rag_text
        valid = [(id_, t, m) for id_, t, m in zip(ids, texts, metas) if t.strip()]
        if not valid:
            continue

        try:
            # 检查已存在的（增量模式）
            existing     = collection.get(ids=[x[0] for x in valid])
            existing_ids = set(existing['ids'])
        except Exception:
            existing_ids = set()

        new_items = [(a, b, c) for a, b, c in valid if a not in existing_ids]
        skip_cnt  = len(valid) - len(new_items)
        skipped  += skip_cnt

        if new_items:
            collection.add(
                ids       = [x[0] for x in new_items],
                documents = [x[1] for x in new_items],
                metadatas = [x[2] for x in new_items],
            )
            written += len(new_items)

        done = min(i + batch_size, len(vocab_list))
        bar  = "█" * (done * 30 // len(vocab_list))
        print(f"\r    [{bar:<30}] {done}/{len(vocab_list)}  写入:{written}", end="", flush=True)

    print(f"\n\n[4] 完成！")
    print(f"    新写入: {written}")
    print(f"    已跳过: {skipped}（已存在）")
    print(f"    库中总量: {collection.count()}")

    # 验证：测试检索几个词
    print("\n[5] 验证检索...")
    test_queries = ["推测 猜想 投机", "decline 下降 拒绝", "学术 理论 学者"]
    for q in test_queries:
        results = collection.query(query_texts=[q], n_results=3)
        hits = [(m['word'], m['oxford_cn'][:20]) for m in results['metadatas'][0]]
        print(f"    查询「{q[:10]}」→ {hits}")

    print("\n✅ 词汇知识库升级完成！")
    print(f"   路径: {Path(db_path).absolute()}")


def main():
    parser = argparse.ArgumentParser(description="vocab_merged.jsonl → ChromaDB")
    parser.add_argument('--input',   default='vocab_merged.jsonl', help='输入文件')
    parser.add_argument('--db',      default='rag_db',             help='向量库目录')
    parser.add_argument('--replace', action='store_true', default=True,
                        help='删除旧集合并重建（默认True，确保升级生效）')
    parser.add_argument('--append',  action='store_true',
                        help='追加模式（不删除旧集合，仅写入新词条）')
    args = parser.parse_args()

    replace = not args.append  # append 时不替换
    build_rag(args.input, args.db, replace=replace)


if __name__ == '__main__':
    main()
