"""
import_ocr_exams.py — 批量解析 OCR 真题 txt + 答案文件，写入 JSONL

用法：
  python scripts/import_ocr_exams.py --ocr_dir "ocr测试" --answers "ocr测试/答案.txt"
  python scripts/import_ocr_exams.py --ocr_dir "ocr测试" --answers "ocr测试/答案.txt" --dry_run
"""

import re, json, argparse, sys, io
from pathlib import Path

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")

ROOT             = Path(__file__).parent.parent
PASSAGES_FILE    = ROOT / "data/exams/reading_passages.jsonl"
QUESTIONS_FILE   = ROOT / "data/exams/reading_questions.jsonl"

# ── 答案解析 ────────────────────────────────────────────────────────────────
def parse_answers(path: Path) -> dict:
    """返回 {year_str: {21:'B', 22:'A', ...}}"""
    result = {}
    current_year = None
    rows = []

    for raw in path.read_text(encoding="utf-8").splitlines():
        line = raw.strip()
        if not line:
            continue
        # 年份行
        if re.fullmatch(r"\d{4}", line):
            if current_year and rows:
                result[current_year] = _rows_to_answers(rows)
            current_year = line
            rows = []
            continue
        # 字母行（只取连续大写字母，去掉 "21-25" 这种前缀）
        letters = re.sub(r"^\d+-\d+", "", line).strip()
        if re.fullmatch(r"[A-D]{5}", letters):
            rows.append(letters)

    if current_year and rows:
        result[current_year] = _rows_to_answers(rows)
    return result

def _rows_to_answers(rows: list) -> dict:
    flat = "".join(rows)          # 20 letters
    return {21 + i: flat[i] for i in range(min(20, len(flat)))}

# ── OCR 文本解析 ─────────────────────────────────────────────────────────────
NOISE = re.compile(r"≦\s*\d+\s*≧|^\d{4}-\d{2}-\d{2}.*|^-\s*\d+\s*-?$|^[-\d]+$")
QNUM  = re.compile(r"^(\d{2})[\.．。]")
OPT   = re.compile(r"^\[([ABCD])\]\s*(.*)")

def fix_ocr_digits(line: str) -> str:
    """修复 OCR 数字/字母混淆"""
    # 行首题号：2l. 4o. 3l. → 21. 40. 31.
    line = re.sub(
        r"^([2-4][0lo])[\.．。]",
        lambda m: str(int(m.group(1).replace("l","1").replace("o","0").replace("O","0"))) + ".",
        line
    )
    # 选项括号内：[Dl → [D]  [Cl → [C]  /D/ → [D]
    line = re.sub(r"\[([ABCD])l\]?", r"[\1]", line)
    line = re.sub(r"/([ABCD])/",     r"[\1]", line)
    return line

def clean_line(line: str) -> str:
    line = line.strip()
    # 去掉页码噪声
    if NOISE.match(line):
        return ""
    # 修复数字OCR错误（行首题号）
    line = fix_ocr_digits(line)
    # 选项标记补空格：[A]xxx → [A] xxx
    line = re.sub(r"\[([ABCD])\](?!\s)", r"[\1] ", line)
    return line

def parse_ocr_file(path: Path) -> dict:
    """返回 {text_id: {"passage": str, "questions": [{q_num, stem, options}]}}"""
    lines = [clean_line(l) for l in path.read_text(encoding="utf-8").splitlines()]
    lines = [l for l in lines if l]

    # 按 Text N 分段
    TEXT_HDR = re.compile(r"^Text\s*(\d)", re.IGNORECASE)
    sections = {}      # text_id → list of lines
    cur_id = None

    for line in lines:
        m = TEXT_HDR.match(line)
        if m:
            cur_id = int(m.group(1))
            sections[cur_id] = []
            continue
        if cur_id is not None:
            sections[cur_id].append(line)

    result = {}
    for tid, sec_lines in sections.items():
        passage_lines = []
        questions = []
        cur_q = None

        for line in sec_lines:
            qm = QNUM.match(line)
            if qm:
                if cur_q:
                    questions.append(cur_q)
                q_num = int(qm.group(1))
                stem = line[qm.end():].strip()
                cur_q = {"q_num": q_num, "stem": stem, "options": {}}
                continue

            if cur_q is not None:
                om = OPT.match(line)
                if om:
                    letter  = om.group(1)
                    content = om.group(2).strip()
                    # 检查是否内嵌了下一个选项：[A]xxx[B]yyy
                    inline = re.search(r'\[([ABCD])\]', content)
                    if inline:
                        cur_q["options"][letter] = content[:inline.start()].strip()
                        remainder = content[inline.start():]
                        om2 = OPT.match(remainder)
                        if om2:
                            cur_q["options"][om2.group(1)] = om2.group(2).strip()
                    else:
                        cur_q["options"][letter] = content
                else:
                    # 续行：接在题干或上一个选项后
                    # 但先检查行中是否含内嵌选项标记
                    inline = re.search(r'\[([ABCD])\]', line)
                    if inline and cur_q["options"]:
                        last_opt = list(cur_q["options"])[-1]
                        cur_q["options"][last_opt] += " " + line[:inline.start()].strip()
                        om2 = OPT.match(line[inline.start():])
                        if om2:
                            cur_q["options"][om2.group(1)] = om2.group(2).strip()
                    elif cur_q["options"]:
                        last_opt = list(cur_q["options"])[-1]
                        cur_q["options"][last_opt] += " " + line
                    else:
                        cur_q["stem"] += " " + line
            else:
                passage_lines.append(line)

        if cur_q:
            questions.append(cur_q)

        # 段落：按空行合并
        passage = "\n".join(passage_lines).strip()
        result[tid] = {"passage": passage, "questions": questions}

    return result

# ── 写入 JSONL ───────────────────────────────────────────────────────────────
def load_existing_ids(path: Path) -> set:
    ids = set()
    if path.exists():
        with open(path, encoding="utf-8") as f:
            for line in f:
                try:
                    obj = json.loads(line)
                    ids.add(obj.get("id", ""))
                except Exception:
                    pass
    return ids

def append_jsonl(path: Path, obj: dict):
    with open(path, "a", encoding="utf-8") as f:
        f.write(json.dumps(obj, ensure_ascii=False) + "\n")

def dedup_jsonl(path: Path):
    """按 id 去重，保留最后一条（最新写入的覆盖旧的）"""
    with open(path, encoding="utf-8") as f:
        lines = [l for l in f if l.strip()]
    seen = {}
    for line in lines:
        obj = json.loads(line)
        seen[obj.get("id", line)] = line
    with open(path, "w", encoding="utf-8") as f:
        for line in seen.values():
            f.write(line if line.endswith("\n") else line + "\n")
    return len(seen)

# ── 主流程 ───────────────────────────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--ocr_dir",  required=True)
    parser.add_argument("--answers",  required=True)
    parser.add_argument("--dry_run",  action="store_true", help="只预览不写入")
    parser.add_argument("--force",    action="store_true", help="强制覆盖已存在记录")
    parser.add_argument("--years",    default="", help="只处理指定年份，逗号分隔，如 2010,2019")
    args = parser.parse_args()

    ocr_dir      = Path(args.ocr_dir)
    answers      = parse_answers(Path(args.answers))
    only_years   = set(args.years.split(",")) if args.years else set()
    print(f"答案文件读取完成，共 {len(answers)} 年：{sorted(answers)}\n")

    if args.force:
        existing_passages  = set()
        existing_questions = set()
        print("⚠ --force 模式：覆盖已存在记录（追加写入，JSONL会有重复，需后处理去重）\n")
    else:
        existing_passages  = load_existing_ids(PASSAGES_FILE)
        existing_questions = load_existing_ids(QUESTIONS_FILE)

    ocr_files = sorted([f for f in ocr_dir.glob("*.txt")
                        if "考研英语一真题" in f.name and "答案" not in f.name])
    if only_years:
        ocr_files = [f for f in ocr_files if any(y in f.name for y in only_years)]
        print(f"只处理年份: {only_years}")
    print(f"找到 OCR 文件 {len(ocr_files)} 个\n")

    total_passages = total_questions = 0

    for fpath in ocr_files:
        m = re.search(r"(\d{4})年", fpath.name)
        if not m:
            continue
        year = m.group(1)
        year_answers = answers.get(year, {})

        print(f"── {year}年  答案:{len(year_answers)}题  ", end="")
        texts = parse_ocr_file(fpath)
        print(f"解析到 {len(texts)} 篇")

        all_q_nums = []
        for tid, data in sorted(texts.items()):
            qs     = data["questions"]
            q_nums = [q["q_num"] for q in qs]
            all_q_nums.extend(q_nums)
        expected = set(range(21, 41))
        got      = set(all_q_nums)
        if expected - got:
            print(f"   ⚠ 缺失题号: {sorted(expected - got)}")

        for tid, data in sorted(texts.items()):
            qs     = data["questions"]
            q_nums = [q["q_num"] for q in qs]
            pid    = f"{year}_text{tid}"

            # 写 passage
            if pid not in existing_passages:
                passage_obj = {
                    "id":      pid,
                    "year":    int(year),
                    "text_id": tid,
                    "passage": data["passage"],
                    "q_nums":  q_nums,
                }
                if not args.dry_run:
                    append_jsonl(PASSAGES_FILE, passage_obj)
                total_passages += 1
            else:
                print(f"   [跳过] passage {pid} 已存在")

            # 写 questions
            for q in qs:
                qid    = f"{year}_q{q['q_num']}"
                answer = year_answers.get(q["q_num"], "")
                if qid not in existing_questions:
                    q_obj = {
                        "id":           qid,
                        "year":         int(year),
                        "text_id":      tid,
                        "q_num":        q["q_num"],
                        "stem":         q["stem"],
                        "options":      q["options"],
                        "answer":       answer,
                        "question_type": "reading",
                        "explanation":  "",
                    }
                    if not args.dry_run:
                        append_jsonl(QUESTIONS_FILE, q_obj)
                    total_questions += 1
                else:
                    print(f"   [跳过] question {qid} 已存在")

    flag = "（dry_run，未实际写入）" if args.dry_run else ""
    print(f"\n完成{flag}：新增 {total_passages} 篇原文 / {total_questions} 道题目")

    if args.force and not args.dry_run:
        n1 = dedup_jsonl(PASSAGES_FILE)
        n2 = dedup_jsonl(QUESTIONS_FILE)
        print(f"去重完成：passages={n1}条  questions={n2}条")

if __name__ == "__main__":
    main()
