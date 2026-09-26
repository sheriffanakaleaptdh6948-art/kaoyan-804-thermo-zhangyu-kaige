#!/usr/bin/env python3
"""
真题 RAG 质量修复脚本（agent04 — Round 3）
检查 exam_explanations.jsonl 的质量问题：长度异常、OCR 伪影、答案缺失、年份错乱。

用法：
  python scripts/fix_exam_rag.py                     # 生成质量报告
  python scripts/fix_exam_rag.py --fix                # 尝试自动修复（生成修复版 JSONL）
  python scripts/fix_exam_rag.py --year 2023          # 仅检查指定年份
  python scripts/fix_exam_rag.py --json               # 输出 JSON 格式（供脚本消费）
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parent.parent
EXAM_JSONL = ROOT / "data" / "exams" / "exam_explanations.jsonl"
REPORT_PATH = ROOT / "scripts" / "rag_quality_report.txt"
FIXED_JSONL = ROOT / "data" / "exams" / "exam_explanations_fixed.jsonl"

# ═══════════════════════════════════════
# OCR 伪影检测规则
# ═══════════════════════════════════════

OCR_PATTERNS: dict[str, re.Pattern] = {
    "alnum_mix": re.compile(
        r"[a-z]{3,}\d+[a-z]{2,}|[A-Z]{2,}\d{2,}[A-Z]{2,}", re.IGNORECASE
    ),
    "odd_case": re.compile(r"\b[a-z][A-Z][a-z]\b"),
    "garbled_run": re.compile(r"[0-9A-Za-z]{25,}"),
    "broken_punct": re.compile(r"[\.\。]{3,}"),  # 异常连续标点
    "repeated_char": re.compile(r"(.)\1{8,}"),  # 单字符重复 8 次以上
    "null_bytes": re.compile(r"\x00"),
}

# 已知 OCR 伪影修复映射（从 v2/v3 提取脚本积累）
OCR_FIXES: dict[str, str] = {
    "fl": "fi", "FI": "FI", "ﬁ": "fi", "ﬂ": "fl",
    "�": "", "\x00": "",
    "  ": " ", "   ": " ",
    " ,": ",", " .": ".", " :": ":", " ;": ";",
    "。.": "。", ".。": "。",
}

# 常见 OCR 词粘连修复
WORD_MERGE_FIXES: dict[str, str] = {
    "developingcountries": "developing countries",
    "developingcountry": "developing country",
    "industrializedcountries": "industrialized countries",
    "climatechange": "climate change",
    "globalwarming": "global warming",
    "economicgrowth": "economic growth",
    "socialmedia": "social media",
    "artificialintelligence": "artificial intelligence",
    "highspeed": "high-speed",
    "wideranging": "wide-ranging",
    "farranging": "far-ranging",
    "longterm": "long-term",
    "shortterm": "short-term",
    "wellknown": "well-known",
    "everincreasing": "ever-increasing",
    "mankind": "mankind",
    "moreover": "moreover",
    "furthermore": "furthermore",
    "nevertheless": "nevertheless",
}


def load_records(path: Path) -> list[dict]:
    records = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                records.append(json.loads(line))
            except json.JSONDecodeError as e:
                print(f"[WARN] JSON 解析失败: {line[:80]}... ({e})")
    return records


def detect_ocr_artifacts(text: str) -> list[dict]:
    """检测 OCR 伪影，返回问题列表。"""
    issues = []
    for name, pat in OCR_PATTERNS.items():
        for m in pat.finditer(text):
            snippet = text[max(0, m.start() - 20):m.end() + 20]
            issues.append({
                "type": f"ocr_{name}",
                "match": m.group(),
                "position": m.start(),
                "snippet": snippet.strip(),
            })
    return issues


def try_fix_ocr(text: str) -> tuple[str, int]:
    """尝试修复常见 OCR 伪影。返回 (fixed_text, fix_count)。"""
    count = 0
    for bad, good in OCR_FIXES.items():
        n = text.count(bad)
        if n:
            text = text.replace(bad, good)
            count += n
    for bad, good in WORD_MERGE_FIXES.items():
        if bad in text.lower():
            # 大小写不敏感替换
            pattern = re.compile(re.escape(bad), re.IGNORECASE)
            n = len(pattern.findall(text))
            if n:
                text = pattern.sub(good, text)
                count += n
    return text, count


def check_quality(records: list[dict]) -> dict:
    """全面质量检查，返回详细报告数据。"""
    stats: dict[str, Any] = {
        "total": len(records),
        "year_dist": defaultdict(int),
        "issues": defaultdict(list),  # record_index → [issue descriptions]
        "summary": defaultdict(int),
    }

    years_seen: set[str] = set()
    for i, rec in enumerate(records):
        year = str(rec.get("year", "?"))
        q_num = rec.get("q_num", "?")
        text_id = rec.get("text_id", "?")
        answer = str(rec.get("answer", "")).strip()
        explanation = rec.get("explanation", "")
        source = rec.get("source", "?")
        rec_id = f"{year}/Q{q_num}/Text{text_id}"

        stats["year_dist"][year] += 1

        # 1) 答案缺失
        if not answer or answer in ("?", "None", ""):
            stats["issues"][i].append(f"[NO_ANSWER] {rec_id}")
            stats["summary"]["no_answer"] += 1

        # 2) 长度异常
        expl_len = len(explanation)
        if expl_len < 50:
            stats["issues"][i].append(
                f"[SHORT] {rec_id} — {expl_len} chars"
            )
            stats["summary"]["short"] += 1
        elif expl_len > 5000:
            stats["issues"][i].append(
                f"[LONG] {rec_id} — {expl_len} chars"
            )
            stats["summary"]["long"] += 1

        # 3) OCR 伪影
        ocr_issues = detect_ocr_artifacts(explanation)
        if ocr_issues:
            issue_types = Counter(o["type"] for o in ocr_issues)
            desc = ", ".join(f"{t}:{c}" for t, c in issue_types.items())
            stats["issues"][i].append(f"[OCR] {rec_id} — {desc}")
            stats["summary"]["ocr_affected"] += 1

        # 4) 年份一致性（检测同 text_id 年份是否一致）
        years_seen.add(f"{year}/{text_id}")

        # 5) 来源标记
        stats["summary"][f"source_{source}"] += 1

    # 年份错乱检测：检查每条记录的 id 前缀是否与 year 一致
    # （id 格式: exam_expl_{year}_q{q_num}_t{text_id}）
    for i, rec in enumerate(records):
        rec_id = rec.get("id", "")
        rec_year = str(rec.get("year", ""))
        # 检测 id 中嵌的年份是否与 year 字段一致
        id_year_match = re.search(r"(\d{4})", rec_id)
        if id_year_match and id_year_match.group(1) != rec_year:
            stats["summary"]["year_conflict"] += 1
            stats["issues"][i].append(
                f"[YEAR_MISMATCH] id={rec_id} vs year={rec_year}"
            )

    return stats


def generate_report(stats: dict, out_path: Path) -> str:
    """生成人类可读的质量报告。"""
    lines = []
    lines.append("=" * 65)
    lines.append("  真题 RAG 质量检查报告")
    lines.append(f"  数据源: {EXAM_JSONL}")
    lines.append(f"  记录数: {stats['total']}")
    lines.append("=" * 65)

    # 年份分布
    lines.append(f"\n## 年份分布（{len(stats['year_dist'])} 年）")
    for y in sorted(stats["year_dist"]):
        lines.append(f"  {y}: {stats['year_dist'][y]} 条")

    # 问题摘要
    lines.append("\n## 问题摘要")
    no_ans = stats["summary"].get("no_answer", 0)
    short = stats["summary"].get("short", 0)
    long = stats["summary"].get("long", 0)
    ocr = stats["summary"].get("ocr_affected", 0)
    yr_conf = stats["summary"].get("year_conflict", 0)

    lines.append(f"  无答案:           {no_ans}/{stats['total']} ({no_ans/stats['total']*100:.1f}%)")
    lines.append(f"  过短 (<50 char):  {short}")
    lines.append(f"  过长 (>5000 char): {long}")
    lines.append(f"  OCR 伪影:         {ocr}/{stats['total']} ({ocr/stats['total']*100:.1f}%)")
    lines.append(f"  年份冲突:         {yr_conf}")

    # 来源分布
    lines.append("\n## 来源分布")
    for k, v in sorted(stats["summary"].items()):
        if k.startswith("source_"):
            lines.append(f"  {k[7:]}: {v}")

    # 各年问题详表
    lines.append("\n## 按年份问题统计")
    lines.append(f"  {'年份':<8} {'无答案':<8} {'过短':<6} {'过长':<6} {'OCR':<6}")
    lines.append(f"  {'-'*40}")
    year_issues: dict[str, dict] = defaultdict(lambda: {"no_answer": 0, "short": 0, "long": 0, "ocr": 0})
    for i, issues in stats["issues"].items():
        if i < 0:
            continue
        for issue in issues:
            rec_year = issue.split("/")[0] if "/" in issue else "?"
            rec_year = rec_year.replace("[NO_ANSWER] ", "").replace("[SHORT] ", "").replace("[LONG] ", "").replace("[OCR] ", "").strip().split("/")[0]
            if issue.startswith("[NO_ANSWER]"):
                year_issues[rec_year]["no_answer"] += 1
            elif issue.startswith("[SHORT]"):
                year_issues[rec_year]["short"] += 1
            elif issue.startswith("[LONG]"):
                year_issues[rec_year]["long"] += 1
            elif issue.startswith("[OCR]"):
                year_issues[rec_year]["ocr"] += 1

    for y in sorted(year_issues):
        yi = year_issues[y]
        lines.append(f"  {y:<8} {yi['no_answer']:<8} {yi['short']:<6} {yi['long']:<6} {yi['ocr']:<6}")

    # 详细问题列表（最多显示 50 条）
    lines.append("\n## 详细问题清单（前 50 条）")
    shown = 0
    for i in sorted(stats["issues"]):
        for issue in stats["issues"][i]:
            if shown >= 50:
                break
            lines.append(f"  {issue}")
            shown += 1
    remaining = sum(len(v) for v in stats["issues"].values()) - shown
    if remaining > 0:
        lines.append(f"  ... 还有 {remaining} 条")

    # 建议
    lines.append("\n## 修复建议")
    if no_ans > 0:
        lines.append(f"  1. {no_ans} 条记录缺少答案 → 启用 answer_inference 从解析文本推断")
    if ocr > 0:
        lines.append(f"  2. {ocr} 条含 OCR 伪影 → 运行 --fix 自动修复已知模式")
    if short > 0:
        lines.append(f"  3. {short} 条过短 → 标记为低质量，出题时跳过这些 RAG 结果")
    if yr_conf > 0:
        lines.append(f"  4. {yr_conf} 处年份冲突 → 手动核查 text_id 映射")

    report = "\n".join(lines)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(report, encoding="utf-8")
    return report


def fix_records(records: list[dict]) -> tuple[list[dict], dict]:
    """尝试自动修复。返回 (fixed_records, fix_stats)。"""
    fix_stats = {"ocr_fixed": 0, "ocr_fix_count": 0, "answer_inferred": 0}

    for rec in records:
        explanation = rec.get("explanation", "")

        # OCR 修复
        fixed_text, n = try_fix_ocr(explanation)
        if n > 0:
            rec["explanation"] = fixed_text
            fix_stats["ocr_fixed"] += 1
            fix_stats["ocr_fix_count"] += n

        # 答案推断：从解析文本正则提取
        if not rec.get("answer") or str(rec["answer"]).strip() in ("?", "None", ""):
            inferred = _infer_answer_from_explanation(explanation)
            if inferred:
                rec["answer"] = inferred
                fix_stats["answer_inferred"] += 1

    return records, fix_stats


_ANSWER_HINTS = [
    re.compile(r"正确选项(?:是|为)\s*([A-D])"),
    re.compile(r"(?:应选|当选|选择)\s*([A-D])"),
    re.compile(r"\[([A-D])\]\s*正确"),
    re.compile(r"([A-D])\s*(?:选项\s*)?(?:是\s*)?正确(?:答案|选项)"),
    re.compile(r"正确答案(?:是|为|：)\s*([A-D])"),
    re.compile(r"答案为\s*([A-D])"),
    re.compile(r"当选\s*([A-D])"),
    re.compile(r"应选\s*([A-D])"),
]


def _infer_answer_from_explanation(text: str) -> str | None:
    """从解析文本推断正确答案。"""
    for pat in _ANSWER_HINTS:
        m = pat.search(text)
        if m:
            return m.group(1).upper()
    return None


def export_json(records: list[dict], out_path: Path) -> int:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    count = 0
    with open(out_path, "w", encoding="utf-8") as f:
        for rec in records:
            json.dump(rec, f, ensure_ascii=False)
            f.write("\n")
            count += 1
    return count


def main():
    parser = argparse.ArgumentParser(description="真题 RAG 质量检查与修复")
    parser.add_argument("--fix", action="store_true", help="尝试自动修复")
    parser.add_argument("--year", type=int, help="仅检查指定年份")
    parser.add_argument("--json", action="store_true", help="JSON 格式输出")
    args = parser.parse_args()

    if not EXAM_JSONL.is_file():
        print(f"[ER] 文件不存在: {EXAM_JSONL}")
        sys.exit(1)

    print(f"[读] {EXAM_JSONL}")
    records = load_records(EXAM_JSONL)
    print(f"    共 {len(records)} 条")

    if args.year:
        records = [r for r in records if str(r.get("year")) == str(args.year)]
        if not records:
            print(f"[ER] 未找到 {args.year} 年数据")
            sys.exit(1)
        print(f"    过滤后 {len(records)} 条")

    # 质量检查
    print("[检] 质量检查中…")
    stats = check_quality(records)

    if args.json:
        # 机器可读输出
        output = {
            "total": stats["total"],
            "summary": dict(stats["summary"]),
            "issues_by_index": {
                str(k): v for k, v in stats["issues"].items()
            },
        }
        print(json.dumps(output, ensure_ascii=False, indent=2))
    else:
        report = generate_report(stats, REPORT_PATH)
        print(report)
        print(f"\n[OK] 报告已保存: {REPORT_PATH}")

    # 自动修复
    if args.fix:
        print("\n[修] 尝试自动修复…")
        records, fix_stats = fix_records(records)
        print(f"    OCR 修复: {fix_stats['ocr_fixed']} 条 ({fix_stats['ocr_fix_count']} 处)")
        print(f"    答案推断: {fix_stats['answer_inferred']} 条")

        count = export_json(records, FIXED_JSONL)
        print(f"[OK] 修复版已保存: {FIXED_JSONL} ({count} 条)")

        # 修复后复检
        stats2 = check_quality(records)
        print(f"\n    修复后: 无答案 {stats2['summary'].get('no_answer', 0)} → 过短 {stats2['summary'].get('short', 0)}")


if __name__ == "__main__":
    main()
