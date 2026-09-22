#!/usr/bin/env python3
"""检查 synth-intents 的 jsonl 是否合规. 用法: python3 datasets/synth-intents/check.py [file.jsonl ...]
不给参数就检查目录下全部 jsonl, 并做跨文件的 id 唯一性检查. 全过退出码 0, 否则打印每条问题、退出码 1."""

import json
import pathlib
import re
import sys

HERE = pathlib.Path(__file__).parent
DOMAINS = {
    "ecommerce", "logistics", "telecom", "utilities", "healthcare", "petcare", "it_helpdesk", "education",
    "hr", "legal", "government", "rental", "hotel", "home_services", "automotive", "fitness",
}
FORBIDDEN = re.compile(
    r"\b(bank|debit|credit card|transfer|exchange rate|atm|crypto|bitcoin|loan|top[- ]?up|direct debit|"
    r"alarm|timer|volume|calendar|recipe|podcast|playlist|spotify|smart light|thermostat|"
    r"weather|forecast|taxi|uber|takeaway|food delivery|joke)\b",
    re.I,
)
PII = re.compile(r"(@|\+?\d[\d -]{7,}\d|\b(order|ticket|account)\s*#?\s*\d{3,})", re.I)


def check_file(path: pathlib.Path) -> tuple[list[str], list[dict]]:
    probs, rows = [], []
    domain = path.stem
    if domain not in DOMAINS:
        probs.append(f"{path.name}: 文件名 {domain!r} 不在领域表里")
    for ln, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        try:
            r = json.loads(line)
        except json.JSONDecodeError as e:
            probs.append(f"{path.name}:{ln}: 不是合法 JSON: {e}")
            continue
        rows.append(r)
        where = f"{path.name}:{ln} {r.get('id', '?')}"
        if set(r) != {"id", "domain", "description", "utterances"}:
            probs.append(f"{where}: 字段应为 id/domain/description/utterances, 实际 {sorted(r)}")
            continue
        if r["domain"] != domain:
            probs.append(f"{where}: domain {r['domain']!r} 与文件名不符")
        if not re.fullmatch(rf"{domain}_[a-z0-9]+(_[a-z0-9]+)*", r["id"]):
            probs.append(f"{where}: id 应为 {domain}_<snake_case>")
        d = r["description"]
        if d != d.lower():
            probs.append(f"{where}: description 要全小写")
        if not 3 <= len(d.split()) <= 12:
            probs.append(f"{where}: description {len(d.split())} 词, 要 3–12")
        u = r["utterances"]
        if not (isinstance(u, list) and len(u) == 2):
            probs.append(f"{where}: utterances 要恰好 2 条")
            continue
        for j, s in enumerate(u):
            n = len(s.split())
            if not 6 <= n <= 30:
                probs.append(f"{where}: utterance[{j}] {n} 词, 要 6–30")
            if PII.search(s):
                probs.append(f"{where}: utterance[{j}] 像是有邮箱 / 电话 / 编号: {s!r}")
        if u[0].strip().lower() == u[1].strip().lower():
            probs.append(f"{where}: 两条 utterance 相同")
        for field, s in (("description", d), ("utterance", u[0]), ("utterance", u[1])):
            m = FORBIDDEN.search(s)
            if m:
                probs.append(f"{where}: {field} 撞禁区词 {m.group(0)!r}: {s!r}")
    if len(rows) != 32:
        probs.append(f"{path.name}: {len(rows)} 条, 要 32")
    descs = [r.get("description") for r in rows]
    for d in {x for x in descs if descs.count(x) > 1}:
        probs.append(f"{path.name}: description 重复: {d!r}")
    return probs, rows


def main(argv: list[str]) -> int:
    paths = [pathlib.Path(a) for a in argv] or sorted(HERE.glob("*.jsonl"))
    probs, all_rows = [], []
    for p in paths:
        ps, rows = check_file(p)
        probs += ps
        all_rows += rows
    ids = [r.get("id") for r in all_rows]
    for i in {x for x in ids if ids.count(x) > 1}:
        probs.append(f"id 跨文件重复: {i!r}")
    for p in probs:
        print(p)
    print(f"{len(paths)} 个文件, {len(all_rows)} 条, {len(probs)} 个问题")
    return 1 if probs else 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
