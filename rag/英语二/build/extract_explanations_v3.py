"""
extract_explanations_v3.py  —  考研英语一阅读解析提取（第三版）

v3 核心改进：
  1. 两轮提取 — Pass 1 全量扫描，Pass 2 针对缺失题目定向回扫（含已知遗漏点）
  2. 跨页合并优化 — 基于内容重叠检测 + 放宽至 3 页容差
  3. 答案推断 — VL 模型漏答时，从解析文本中正则提取正确答案
  4. 交叉验证 — 提取后与 reading_questions.jsonl 比对，标记答案不一致
  5. OCR 清理 — 修复常见 PDF 转文字伪影
  6. 遗漏解析感知 — 预加载 遗漏解析.txt 中的已知缺题位置

用法：
  python extract_explanations_v3.py --year 2011 --force
  python extract_explanations_v3.py --year 2023 --force --model qwen-vl-max
  python extract_explanations_v3.py --force              # 全部重提取
  python extract_explanations_v3.py --year 2011 --test   # 调试模式（前 6 页）
  python extract_explanations_v3.py --year 2011 --dry-run  # 不调 API，仅报告计划
"""

import fitz
import base64
import json
import re
import io
import sys
import time
import argparse
import os
import shutil
from pathlib import Path
from collections import defaultdict
from dotenv import load_dotenv
import requests

load_dotenv()
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')

# ── 路径配置 ──────────────────────────────────────────────────────────
PDF_DIR = Path(
    r"英语真题卷\考研英语真题（1980-2024）"
    r"\考研英语一\3. 2010-2024年考研英语真题及解析\2010-2024年考研英语一解析"
)
PROJ_ROOT = Path(__file__).resolve().parent
OUT_FILE = PROJ_ROOT / "data/exams/exam_explanations.jsonl"
QT_FILE = PROJ_ROOT / "data/exams/reading_questions.jsonl"
OVERLOOK_FILE = PDF_DIR / "遗漏解析.txt"

API_KEY = os.getenv("DASHSCOPE_API_KEY")
API_URL = "https://dashscope.aliyuncs.com/compatible-mode/v1/chat/completions"
DPI = 150
TIMEOUT = 90

# ── 页码范围（v3：比 v2 放宽 ±2 页容差）──────────────────────────
PAGE_RANGES = {
    2010: (9,  47),
    2011: (7,  45),
    2012: (7,  47),
    2013: (6,  47),
    2014: (7,  46),
    2015: (10, 52),
    2016: (8,  54),
    2017: (9,  52),
    2018: (9,  51),
    2019: (9,  49),
    2020: (8,  44),
    2021: (8,  45),
    2022: (14, 54),
    2023: (8,  27),
    2024: (8,  32),
}

# ── 已知遗漏题号及页码（来自 遗漏解析.txt）─────────────────────────
KNOWN_GAPS: dict[int, dict[int, str]] = {
    2015: {25: "p20"},
    2016: {23: "p16-17", 24: "p17-18", 39: "p50-51"},
    2017: {24: "p16-17", 39: "p47-49"},
    2018: {22: "p13-15"},
    2019: {22: "p12-14", 23: "p14-15", 28: "p23-24", 31: "p27-28"},
    2021: {29: "p24-25", 34: "p33-34", 36: "p36-37"},
    2023: {35: "p20-21"},
}

# ── OCR 伪影修复映射 ──────────────────────────────────────────────────
OCR_FIXES = {
    "c1t1c1sm": "criticism", "cn t1c1sm": "criticism", "cnt1c1sm": "criticism",
    "山ty": "ility", "wntmgs": "writings", "en t1c1sm": "criticism",
    "po幻tmoclern": "postmodern", "幻": "od", "沁": "al",
    "po山t1cal": "political", "soc1al": "social", "econom1c": "economic",
}

PROMPT_BROAD = """这是考研英语一真题【解析册】的一页图片，属于 Section II Reading Comprehension Part A 的解析部分。

请提取本页中可见的阅读理解题目解析（题号范围：21-40题）。

输出 JSON 数组：
[
  {
    "q_num": 21,
    "answer": "C",
    "explanation": "解析内容（尽量完整，可超过 200 字）",
    "partial": false
  }
]

严格规则：
1. 题号 1-20 的完形填空解析一律忽略
2. 只提取本页可见内容，不要推断未显示的内容
3. 如果某题解析在本页被截断，设 "partial": true，仍提取可见部分
4. answer 必须是 A/B/C/D 单个字母
5. 页面无 21-40 题内容时返回 []
6. 只输出 JSON，不要其他文字"""

PROMPT_TARGETED = """这是考研英语一真题【解析册】的一页图片。

请在本页中查找第 {q_num} 题的解析。这道题应该在附近。

特别注意：
- Q{q_num} 可能跨页，可能在页边、页脚或小字部分
- 如果 Q{q_num} 的解析可见（哪怕是片段），请提取它
- 如果 Q{q_num} 不在本页，返回 []

输出 JSON 数组（即使只有一题）：
[{{"q_num": {q_num}, "answer": "A/B/C/D", "explanation": "...", "partial": false}}]

规则同前：answer 只填单个字母，只输出 JSON。"""


# ── 工具函数 ──────────────────────────────────────────────────────────

def page_to_base64(page) -> str:
    mat = fitz.Matrix(DPI / 72, DPI / 72)
    pix = page.get_pixmap(matrix=mat)
    return base64.b64encode(pix.tobytes("png")).decode()


def _clean_ocr(text: str) -> str:
    """修复常见 OCR 伪影，同时保留中文。"""
    for bad, good in OCR_FIXES.items():
        text = text.replace(bad, good)
    return text


def _infer_answer_from_text(explanation: str) -> str | None:
    """从解析文本中推断正确答案。模式：
    - "[A]正确" / "正确项[A]" / "[B]为正确项" / "应选[C]"
    - "故[D]正确" / "[B]正确" / "正确项是[A]"
    """
    patterns = [
        r'正确项\s*[\[（(]\s*([A-D])\s*[\]）)]',
        r'[\[（(]\s*([A-D])\s*[\]）)]\s*正确',
        r'[\[（(]\s*([A-D])\s*[\]）)]\s*为正确',
        r'应选\s*[\[（(]?\s*([A-D])\s*[\]）)]?',
        r'故\s*[\[（(]?\s*([A-D])\s*[\]）)]?\s*正确',
        r'[\[（(]([A-D])[\]）)]\s*正确',
        r'答案为\s*[\[（(]?\s*([A-D])\s*[\]）)]?',
        r'正确答案\s*[\[（(]?\s*([A-D])\s*[\]）)]?',
    ]
    for pat in patterns:
        m = re.search(pat, explanation)
        if m:
            return m.group(1).upper()
    return None


# ── API 调用 ───────────────────────────────────────────────────────────

def ask_vl(img_b64: str, model: str, prompt: str, retries: int = 3) -> str:
    headers = {"Authorization": f"Bearer {API_KEY}", "Content-Type": "application/json"}
    payload = {
        "model": model,
        "messages": [{"role": "user", "content": [
            {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{img_b64}"}},
            {"type": "text", "text": prompt}
        ]}],
        "max_tokens": 3000,
        "temperature": 0
    }
    for attempt in range(retries):
        try:
            resp = requests.post(API_URL, headers=headers, json=payload, timeout=TIMEOUT)
            resp.raise_for_status()
            return resp.json()["choices"][0]["message"]["content"]
        except Exception as e:
            if attempt < retries - 1:
                wait = 8 * (attempt + 1)
                print(f" [重试{attempt+1} 等{wait}s: {type(e).__name__}]", end="", flush=True)
                time.sleep(wait)
            else:
                print(f" [ERROR: {e}]")
                return "[]"


def parse_response(text: str, enforce_q_range: tuple = (21, 40)) -> list:
    """解析 VL 模型返回的 JSON，自动推断缺失的 answer。"""
    m = re.search(r'\[.*?\]', text, re.DOTALL)
    if not m:
        m = re.search(r'\{.*?\}', text, re.DOTALL)
        if m:
            try:
                return _validate_items([json.loads(m.group())], enforce_q_range)
            except Exception:
                return []
        return []
    try:
        items = json.loads(m.group())
        return _validate_items(items, enforce_q_range)
    except Exception:
        return []


def _validate_items(items: list, q_range: tuple) -> list:
    lo, hi = q_range
    result = []
    for it in items:
        if not isinstance(it, dict):
            continue
        try:
            qn = int(it.get("q_num", 0))
        except (ValueError, TypeError):
            continue
        if not (lo <= qn <= hi):
            continue

        ans = str(it.get("answer", "")).strip().upper()
        expl = _clean_ocr(str(it.get("explanation", "")).strip())

        # 如果 VL 没返回答案，尝试从解析文本推断
        if ans not in "ABCD" or len(ans) != 1:
            inferred = _infer_answer_from_text(expl)
            if inferred:
                ans = inferred

        result.append({
            "q_num": qn,
            "answer": ans if ans in "ABCD" and len(ans) == 1 else "",
            "explanation": expl,
            "partial": bool(it.get("partial", False))
        })
    return result


# ── 跨页合并（v3 改进）────────────────────────────────────────────────

def _content_overlap_ratio(a: str, b: str) -> float:
    """计算两段文本的字符级重叠率（判断是否是同一段内容的不同片段）。"""
    if not a or not b:
        return 0.0
    # 使用 trigram Jaccard
    def _trigrams(s):
        return {s[i:i+3] for i in range(len(s)-2)}
    ta = _trigrams(a)
    tb = _trigrams(b)
    if not ta or not tb:
        return 0.0
    return len(ta & tb) / len(ta | tb)


def merge_fragments(fragments: dict) -> dict:
    """
    v3 改进：
    - 容差从 2 页放宽到 3 页
    - 新增内容重叠检测：高重叠的片段去重（取最长）
    - 合并后用正则二次推断 answer
    """
    merged = {}
    for qn, frags in fragments.items():
        if not frags:
            continue
        frags_sorted = sorted(frags, key=lambda x: x[0])

        if len(frags_sorted) == 1:
            _, ans, expl, _ = frags_sorted[0]
            merged[qn] = {"answer": ans, "explanation": expl}
            continue

        # 去重：检测内容高度重叠的片段（同一页重复识别），保留最长
        deduped = []
        for f in frags_sorted:
            is_dup = False
            for existing in deduped:
                if _content_overlap_ratio(f[2], existing[2]) > 0.6:
                    # 保留更长的
                    if len(f[2]) > len(existing[2]):
                        deduped.remove(existing)
                        deduped.append(f)
                    is_dup = True
                    break
            if not is_dup:
                deduped.append(f)
        deduped.sort(key=lambda x: x[0])

        if len(deduped) == 1:
            merged[qn] = {"answer": deduped[0][1], "explanation": deduped[0][2]}
            continue

        # 合并逻辑：第一个 partial 为 True 时拼接连同页片段（容差 3）
        first_page, first_ans, first_expl, first_partial = deduped[0]
        if first_partial:
            combined = first_expl
            prev_page = first_page
            for page_no, ans, expl, partial in deduped[1:]:
                if page_no <= prev_page + 3:  # v3: 放宽到 3 页
                    combined += " " + expl
                    prev_page = page_no
                else:
                    break
            merged[qn] = {"answer": first_ans, "explanation": combined.strip()}
        else:
            best = max(deduped, key=lambda x: len(x[2]))
            merged[qn] = {"answer": best[1], "explanation": best[2]}

    # 二次推断：合并后重新扫描完整文本以推断答案
    for qn, data in merged.items():
        if not data["answer"] or data["answer"] not in "ABCD":
            inferred = _infer_answer_from_text(data["explanation"])
            if inferred:
                data["answer"] = inferred

    return merged


# ── 页面预检 ──────────────────────────────────────────────────────────

def is_explanation_page(page) -> bool:
    text = page.get_text("text")
    if not text.strip():
        return True
    has_q = bool(re.search(r'\b(2[1-9]|3[0-9]|40)[\.、．。]', text))
    has_cjk = bool(re.search(r'[一-鿿]{3,}', text))
    if not has_cjk:
        return has_q
    return has_q


# ── 加载 ground truth ─────────────────────────────────────────────────

def load_ground_truth() -> dict:
    """从 reading_questions.jsonl 加载正确答案，key 为 id。"""
    if not QT_FILE.exists():
        print(f"⚠️  未找到 {QT_FILE}，跳过答案交叉验证")
        return {}
    gt = {}
    for line in QT_FILE.read_text(encoding='utf-8').splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            rec = json.loads(line)
        except Exception:
            continue
        qid = rec.get("id", "")
        ans = rec.get("answer", "")
        if qid and ans:
            gt[qid] = ans.upper()
    return gt


# ── 单年处理（v3 两轮模式）────────────────────────────────────────────

def process_year(pdf_path: Path, year: int, model: str,
                 existing_ids: set, force: bool,
                 test_mode: bool, dry_run: bool,
                 ground_truth: dict) -> dict:
    """
    两轮提取：
    Pass 1：全量扫描，识别所有可能的解析
    Pass 2：针对缺失题号 + 已知遗漏点定向回扫
    """
    page_start, page_end = PAGE_RANGES.get(year, (1, 9999))
    doc = fitz.open(str(pdf_path))
    total = len(doc)

    if dry_run:
        print(f"\n{'='*60}")
        print(f"  [DRY-RUN] {year}年  PDF共{total}页  目标范围p{page_start}-p{page_end}  模型:{model}")
        gap_info = KNOWN_GAPS.get(year, {})
        print(f"  已知遗漏: {len(gap_info)} 题" if gap_info else "  已知遗漏: 无")
        return {}

    print(f"\n{'='*60}")
    print(f"  {year}年  PDF共{total}页  解析范围p{page_start}-p{page_end}  模型:{model}")

    # 加载已知遗漏
    known_gaps = KNOWN_GAPS.get(year, {})

    fragments = defaultdict(list)

    # ═══════════════════════════════════════
    # Pass 1：全量扫描
    # ═══════════════════════════════════════
    print(f"\n  ── Pass 1：全量扫描 ──")
    for pno in range(total):
        page_1indexed = pno + 1

        if test_mode and page_1indexed > page_start + 5:
            break
        if page_1indexed < page_start or page_1indexed > page_end:
            continue
        if not is_explanation_page(doc[pno]):
            print(f"  p{page_1indexed:03d} [跳过-无解析特征]")
            continue

        print(f"  p{page_1indexed:03d} → ", end="", flush=True)
        t0 = time.time()
        raw = ask_vl(page_to_base64(doc[pno]), model, PROMPT_BROAD)
        items = parse_response(raw)

        found_on_page = []
        for it in items:
            qn = it["q_num"]
            if not force:
                qid = f"{year}-T{(qn-21)//5+1}-Q{qn}"
                if qid in existing_ids:
                    continue
            fragments[qn].append((page_1indexed, it["answer"],
                                   it["explanation"], it["partial"]))
            found_on_page.append(qn)

        elapsed = time.time() - t0
        print(f"{len(found_on_page)}题 ({elapsed:.1f}s)  找到:{sorted(found_on_page)}")
        time.sleep(0.5)

    # ═══════════════════════════════════════
    # Pass 2：缺失题号定向回扫
    # ═══════════════════════════════════════
    found_qnums = set(fragments.keys())
    missing = set(range(21, 41)) - found_qnums
    # 合并已知遗漏题号
    priority_gaps = missing | set(known_gaps.keys())

    if priority_gaps:
        print(f"\n  ── Pass 2：定向回扫缺失/已知遗漏 {sorted(priority_gaps)} ──")
        # 推断缺失题号的可能页码范围
        for qn in sorted(priority_gaps):
            # 从已找到题目推断大致位置
            neighbors = sorted([q for q in found_qnums if abs(q - qn) <= 3])
            if neighbors:
                # 从最近邻居的页码推断搜索窗口
                neighbor_pages = []
                for nq in neighbors:
                    for frag in fragments.get(nq, []):
                        neighbor_pages.append(frag[0])
                if neighbor_pages:
                    avg_page = int(sum(neighbor_pages) / len(neighbor_pages))
                    search_start = max(page_start, avg_page - 2)
                    search_end = min(page_end, avg_page + 2)
                else:
                    search_start, search_end = page_start, page_end
            else:
                search_start, search_end = page_start, page_end

            # 已知遗漏有明确页码提示，优先使用
            if qn in known_gaps:
                gap_info = known_gaps[qn]
                nums = re.findall(r'\d+', gap_info)
                if nums:
                    known_page = int(nums[0])
                    search_start = max(page_start, known_page - 1)
                    search_end = min(page_end, known_page + 2)

            found_in_pass2 = False
            for pno in range(total):
                p1 = pno + 1
                if p1 < search_start or p1 > search_end:
                    continue

                # Pass 2 放宽预检（已知遗漏页面即使无明显特征也要查）
                if not is_explanation_page(doc[pno]) and qn not in known_gaps:
                    continue

                print(f"  p{p1:03d} 🔍 Q{qn} → ", end="", flush=True)
                t0 = time.time()
                prompt = PROMPT_TARGETED.format(q_num=qn)
                raw = ask_vl(page_to_base64(doc[pno]), model, prompt)
                items = parse_response(raw, q_range=(qn - 1, qn + 1))

                for it in items:
                    if it["q_num"] == qn:
                        fragments[qn].append((p1, it["answer"],
                                               it["explanation"], it["partial"]))
                        found_in_pass2 = True

                elapsed = time.time() - t0
                tag = "✓" if found_in_pass2 else "-"
                print(f"{tag} ({elapsed:.1f}s)")
                if found_in_pass2:
                    break
                time.sleep(0.3)

            if not found_in_pass2:
                print(f"  Q{qn} ⚠️  未找到（范围p{search_start}-p{search_end}）")

    # ═══════════════════════════════════════
    # 合并 + 元数据
    # ═══════════════════════════════════════
    merged = merge_fragments(fragments)

    result = {}
    for qn, data in merged.items():
        tid = f"T{(qn - 21) // 5 + 1}"
        qid = f"{year}-{tid}-Q{qn}"
        result[qn] = {
            "id":          qid,
            "year":        year,
            "q_num":       qn,
            "text_id":     tid,
            "answer":      data["answer"],
            "explanation": data["explanation"],
            "source":      "v3"
        }

    # ═══════════════════════════════════════
    # 交叉验证
    # ═══════════════════════════════════════
    if ground_truth:
        mismatches = []
        for qn, rec in result.items():
            gt_ans = ground_truth.get(rec["id"], "")
            if gt_ans and rec["answer"] and rec["answer"] != gt_ans:
                mismatches.append((rec["id"], rec["answer"], gt_ans))
        if mismatches:
            print(f"\n  ⚠️  答案不一致 ({len(mismatches)} 题)：")
            for qid, ext, gt in mismatches[:5]:
                print(f"      {qid}: 提取={ext}  题库={gt}")

    # 报告
    found_final = set(result.keys())
    still_missing = set(range(21, 41)) - found_final
    print(f"\n  {year}年结果：找到 {len(found_final)}/20  缺失：{sorted(still_missing) if still_missing else '无'}")
    if still_missing:
        print(f"  缺失题号: {sorted(still_missing)}")
        for qn in sorted(still_missing):
            neighbors = sorted([q for q in found_final if abs(q - qn) <= 3])
            print(f"      Q{qn} — 相邻已找到: {neighbors}")

    return result


# ── 数据 I/O ────────────────────────────────────────────────────────────

def load_existing(out_file: Path) -> dict:
    data = {}
    if not out_file.exists():
        return data
    for line in out_file.read_text(encoding='utf-8').splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            rec = json.loads(line)
            data[rec['id']] = rec
        except Exception:
            pass
    return data


def save_all(out_file: Path, records: dict):
    sorted_recs = sorted(
        records.values(),
        key=lambda r: (int(r.get('year', 0) if isinstance(r.get('year'), int) else 0), int(r.get('q_num', 0)))
    )
    out_file.parent.mkdir(parents=True, exist_ok=True)
    with open(out_file, 'w', encoding='utf-8') as f:
        for rec in sorted_recs:
            f.write(json.dumps(rec, ensure_ascii=False) + '\n')
    print(f"\n已写入 {len(sorted_recs)} 条记录 → {out_file}")


# ── 主入口 ──────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="考研英语解析提取 v3（两轮提取 + 答案推断 + 交叉验证）")
    parser.add_argument('--year',  type=int, help='指定年份，不填则处理所有年份')
    parser.add_argument('--force', action='store_true',
                        help='强制重新提取（覆盖该年已有数据）')
    parser.add_argument('--model', default='qwen-vl-plus',
                        choices=['qwen-vl-plus', 'qwen-vl-max'],
                        help='视觉模型（默认 qwen-vl-plus）')
    parser.add_argument('--test', action='store_true',
                        help='只处理页码范围的前 6 页（调试用）')
    parser.add_argument('--dry-run', action='store_true',
                        help='不调用 API，仅报告将处理哪些年份/页面')
    args = parser.parse_args()

    if not args.dry_run and not API_KEY:
        print("错误：未找到 DASHSCOPE_API_KEY，请检查 .env 文件")
        return

    pdfs = sorted(PDF_DIR.glob("*.pdf"))
    if args.year:
        pdfs = [p for p in pdfs if str(args.year) in p.name]
    if not pdfs:
        print(f"❌ 未找到 PDF 文件（目录：{PDF_DIR}）")
        return

    print(f"找到 {len(pdfs)} 个 PDF，模型：{args.model}，force：{args.force}")

    ground_truth = load_ground_truth()
    print(f"题库正确答案：{len(ground_truth)} 条")

    existing = load_existing(OUT_FILE)
    existing_ids = set(existing.keys())
    print(f"现有解析数据：{len(existing)} 条记录")

    if args.force and OUT_FILE.exists():
        backup = OUT_FILE.with_suffix('.jsonl.bak')
        shutil.copy2(OUT_FILE, backup)
        print(f"已备份 → {backup}")

    for pdf_path in pdfs:
        m = re.search(r'(20\d{2})', pdf_path.name)
        if not m:
            continue
        year = int(m.group(1))

        if year not in PAGE_RANGES:
            print(f"\n⚠️  {year}年无页码配置，跳过")
            continue

        if not args.force and not args.dry_run:
            year_recs = [k for k in existing_ids if k.startswith(str(year))]
            if len(year_recs) >= 20:
                print(f"\n⏭️  {year}年 已有 {len(year_recs)}/20 题，跳过（用 --force 强制重提取）")
                continue

        new_recs = process_year(
            pdf_path, year, args.model,
            existing_ids, args.force, args.test, args.dry_run,
            ground_truth
        )

        if args.dry_run:
            continue

        if args.force:
            existing = {k: v for k, v in existing.items()
                        if not k.startswith(str(year))}

        for qn, rec in new_recs.items():
            existing[rec['id']] = rec

        save_all(OUT_FILE, existing)
        existing_ids = set(existing.keys())

    if args.dry_run:
        print(f"\n[Dry-run 完成，未调用 API]")
        return

    # 最终统计
    print(f"\n{'='*60}")
    print("全部完成！各年题数统计：")
    year_counts = {}
    for rec in existing.values():
        y = int(rec['year']) if isinstance(rec['year'], int) else rec['year']
        year_counts[y] = year_counts.get(y, 0) + 1
    for y in sorted(year_counts, key=lambda x: int(x) if isinstance(x, int) else 0):
        cnt = year_counts[y]
        flag = "✅" if cnt >= 20 else f"⚠️ 缺{20-cnt}题"
        print(f"  {y}年：{cnt}/20  {flag}")


if __name__ == "__main__":
    main()
