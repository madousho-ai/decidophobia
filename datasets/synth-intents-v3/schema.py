"""synth-intents v3 的数据格式: 一个领域的问题清单与文本, 以及把它们检查后写成 JSON 的函数. 纯标准库.

每个领域一个 <domain>.py (从 template.py 复制), 里面只填数据, 运行它就调 main() 生成两个文件:

  <domain>.questions.json   {"domain", "label", "questions": [问题, ...]}, 一行一道题
  <domain>.src.jsonl        一行一份文本

这两个是生成物, 不进仓库 (见本目录 .gitignore); 仓库里只放 <domain>.py 和参考模型的答案 <domain>.answer.json.
答案由 answer.py 让一个参考模型把题全部答一遍得到, 格式见 answer.py.

问题 (Question) 的键:
  id       snake_case, 领域内唯一. 只用来引用, 模型看不到.
  ask      同一道题的几种问法, 至少 2 种 (只问指定文本的题至少 1 种). 训练时随机挑一种; 判题统一用 ask[0].
  options  选项说明, 2..256 条 (D 槽只有 256 个). 第一版每个选项都要有说明; 答案用下标指选项.
  needs    可选. 这道题指着 JSON state 里的哪些字段; 每种问法都要用反引号写出字段名, 如 `follow_up`.
           只问 state 是 JSON 且含这些字段的文本. 不写就是问整份文本, 每份文本都问.
  texts    可选. 只问这几份文本 (文本 id). 用于一份文本自带的题, 如 v2 每个意图那道细节是非题.
  judged   可选, 默认 true. false = 参考模型不答这道题 (如 256 个选项, 循环移位要 256 次调用),
           标签只来自 stated, 所以每份被问到的文本都要在 stated 里写它.

文本 (Text) 的键:
  id       "<domain>_<后缀>", 领域内唯一.
  style    STYLES 里的一种. json 与 text 是 JSON 对象互为充要.
  text     一段文本; style 为 json 时是 JSON 对象 (值可以是字符串、数字、列表、嵌套对象、null),
           渲染时缩进 2 格展开, 与 TypeSafe 请求里的 state 同形.
  stated   {题 id: 选项下标}: 写文本前预定、并且确实写进了文本的答案. 没列出的题文本里不提,
           判题给出的软标签会分散. 只能列这份文本会被问到的题 (needs 满足).
           训练标签: 写明的题一律用 one-hot 硬标签; 参考模型的分布 (answer.json) 只用在没写明的题上.
           写明题在 answer.json 里的分布留作难度记录 (关思考的参考模型会读错 state, 如 sec_ops 9/63).
  unknown  可选, [题 id]: 文本对这道题一点线索都没给, 细心的读者也没有理由偏向任何一项
           (如文本没提时间, 问持续多久). 这些题的标签是均匀分布, 参考模型不答.
           线索互相矛盾的不算 (如 JSON 字段一个平静一个生气, 问整份的语气), 那种交给参考模型.
           不能与 stated 重叠, 只能列这份文本会被问到、参考模型会答的题.

派生是非题 (不存盘, 训练时现造): 参考模型答过的选择题, 每个选项都能改写成一道 no / yes 题,
问句是 yes_no(问法, 选项), yes 的概率就是那道选择题里这个选项的概率, 不另外判题.

写作规则 (检查不了, 靠人):
  - 题目问文本里的世界: 有哪些人和东西、数量、先后、因果、几件事之后的状态、接下来会怎样.
    多数题只问少数几份文本 (texts 指定), 选项是文本里的东西本身 (房号、人名、时间、金额), 可以是短词.
    world 的写法 (browser_agent / sec_ops / coding_ci 是参照): 一个领域 5-7 种 state, 每种 3 份左右,
    每种配 3-5 道题问它的全部几份; 几份之间答案互不相同, 是非题 yes 与 no 都出现; 全领域至少 8 种题型.
    转哪个组、语气、谁在写、多久、在哪这类描述文本本身的题最多一道.
    (hotel.py 是早先的一份文本配 6-10 道专属题的写法, 正确选项常是最长的、常排在前面, 别照抄.)
  - 数量写成英文单词 (three), 金额带货币符号, 时间带冒号: 参考模型按编号答题, 选项是小整数时会把值当成编号.
  - 答案由文本定死的写进 stated; 要判断的 (先做哪件事、最可能怎样) 不写, 标签交给参考模型.
  - (telecom.py 是早先的写法: 同一套问整份工单的题问每份文本, 纯文本写明 4-5 道, 其余不提.)
  - JSON state 里约一半让字段互相不一致 (报告正文客气、追问生气; 正文要修复、追问只问进度),
    指着某个字段的题只能从那个字段答. 字段不一致时, 问整份文本的同类题不要列进 stated.
"""

from __future__ import annotations

import argparse
import json
import pathlib
import re
import sys
from collections import Counter
from dataclasses import dataclass, field

MAX_OPTIONS = 256
MIN_ASK = 2
STYLES = {
    "short": "一两句随手写的, 可以不讲究标点和大小写",
    "structured": "分段的规整文本, 一行一个小标题: 如 bug 报告的标题 / 复现步骤 / 环境, 或客服表单的主题 / 账户 / 诉求",
    "rant": "长篇抱怨, 情绪多于信息",
    "long": "长而讲清楚的一段, 来龙去脉都交代了 (v2 每个意图的第 2 条消息)",
    "indirect": "只讲处境, 不直说要什么 (v2 每个意图的第 3 条消息)",
    "table": "表格: 一行一条记录, 列用 | 隔开 (名单、账单、排班表)",
    "log": "按时间或日期一行一条的记录 (工作日志、登记簿)",
    "dialogue": "两人或多人的对话, 一行一句, 行首写说话人",
    "story": "一段叙事, 按时间讲发生了什么",
    "notice": "规则、须知或说明, 一条一条列出",
    "json": "state 是 JSON 对象 {字段名: 文本}; 带 needs 的题只问这种",
}
YES_NO = ["no", "yes"]  # 是非题的两个选项, 与 v2 二元题、BoolQ 同形
SNAKE = re.compile(r"[a-z][a-z0-9_]*$")


@dataclass
class Question:
    id: str
    ask: list[str]
    options: list[str]
    needs: list[str] = field(default_factory=list)
    texts: list[str] = field(default_factory=list)
    judged: bool = True

    def to_json(self) -> dict:
        d = {"id": self.id, "ask": self.ask, "options": self.options}
        if self.needs:
            d["needs"] = self.needs
        if self.texts:
            d["texts"] = self.texts
        if not self.judged:
            d["judged"] = False
        return d


@dataclass
class Text:
    id: str
    style: str
    text: str | dict[str, str]
    stated: dict[str, int] = field(default_factory=dict)
    unknown: list[str] = field(default_factory=list)

    def to_json(self) -> dict:
        d = {"id": self.id, "style": self.style, "text": self.text, "stated": self.stated}
        if self.unknown:
            d["unknown"] = self.unknown
        return d


def applies(q: Question, t: Text) -> bool:
    """这份文本会不会被问这道题."""
    if q.texts and t.id not in q.texts:
        return False
    return not q.needs or (isinstance(t.text, dict) and all(f in t.text for f in q.needs))


def yes_no(ask: str, option: str) -> str:
    """把选择题的一个选项改写成是非题的问句. 选项是 YES_NO, yes 的概率取选择题里这个选项的概率."""
    return f'{ask} Is it "{option}"?'


def _nonempty_str(x) -> bool:
    return isinstance(x, str) and x.strip() != ""


def problems(domain: str, label: str, questions: list[Question], texts: list[Text]) -> list[str]:
    """违反格式的地方, 一条一句. 空表 = 可以写文件."""
    out = []
    if not isinstance(domain, str) or not SNAKE.match(domain):
        out.append(f"DOMAIN {domain!r} 要是 snake_case")
    if not _nonempty_str(label):
        out.append("LABEL 不能为空")
    if not questions:
        out.append("QUESTIONS 是空的")
    if not texts:
        out.append("TEXTS 是空的")

    qids = Counter(q.id for q in questions)
    known_tids = {t.id for t in texts}
    for q in questions:
        where = f"题 {q.id!r}"
        if not isinstance(q.id, str) or not SNAKE.match(q.id):
            out.append(f"{where}: id 要是 snake_case")
        if qids[q.id] > 1:
            out.append(f"{where}: id 重复")
        min_ask = 1 if q.texts else MIN_ASK
        if len(q.ask) < min_ask or not all(_nonempty_str(a) for a in q.ask):
            out.append(f"{where}: ask 要有至少 {min_ask} 种非空问法")
        elif len(set(q.ask)) != len(q.ask):
            out.append(f"{where}: ask 有重复的问法")
        if not 2 <= len(q.options) <= MAX_OPTIONS or not all(_nonempty_str(o) for o in q.options):
            out.append(f"{where}: options 要有 2..{MAX_OPTIONS} 条非空说明")
        elif len(set(q.options)) != len(q.options):
            out.append(f"{where}: options 有重复的说明")
        for f in q.needs:
            if not _nonempty_str(f):
                out.append(f"{where}: needs 里有空字段名")
            elif not all(f"`{f}`" in a for a in q.ask if isinstance(a, str)):
                out.append(f"{where}: 每种问法都要写出 `{f}`")
        for tid in q.texts:
            if tid not in known_tids:
                out.append(f"{where}: texts 里的 {tid!r} 不是任何一份文本")
        if not q.judged:
            for t in texts:
                if applies(q, t) and q.id not in t.stated:
                    out.append(f"{where}: judged 是 false, 被问到的文本 {t.id!r} 要在 stated 里写它")

    by_id = {q.id: q for q in questions}
    tids = Counter(t.id for t in texts)
    for t in texts:
        where = f"文本 {t.id!r}"
        if not isinstance(t.id, str) or not t.id.startswith(f"{domain}_") or not SNAKE.match(t.id):
            out.append(f"{where}: id 要写成 {domain}_<snake_case 后缀>")
        if tids[t.id] > 1:
            out.append(f"{where}: id 重复")
        if t.style not in STYLES:
            out.append(f"{where}: style {t.style!r} 不在 {sorted(STYLES)} 里")
        if t.style == "json":
            if not isinstance(t.text, dict) or not t.text:
                out.append(f"{where}: style 是 json, text 要是非空的 {{字段名: 值}}")
            elif not all(_nonempty_str(k) for k in t.text):
                out.append(f"{where}: JSON 字段名要是非空字符串")
            else:
                try:
                    json.dumps(t.text)
                except (TypeError, ValueError) as e:
                    out.append(f"{where}: JSON state 写不成 JSON: {e}")
        elif not _nonempty_str(t.text):
            out.append(f"{where}: style 不是 json, text 要是非空字符串")
        for qid, idx in t.stated.items():
            q = by_id.get(qid)
            if q is None:
                out.append(f"{where}: stated 里的 {qid!r} 不是任何一道题")
                continue
            if isinstance(idx, bool) or not isinstance(idx, int) or not 0 <= idx < len(q.options):
                out.append(f"{where}: stated[{qid!r}] = {idx!r}, 要是 0..{len(q.options) - 1} 的下标")
            if not applies(q, t):
                out.append(f"{where}: 不会被问 {qid!r} (缺字段 {q.needs} 或不在它的 texts 里), 不能列进 stated")
        if len(set(t.unknown)) != len(t.unknown):
            out.append(f"{where}: unknown 有重复的题")
        for qid in t.unknown:
            q = by_id.get(qid)
            if q is None:
                out.append(f"{where}: unknown 里的 {qid!r} 不是任何一道题")
            elif qid in t.stated:
                out.append(f"{where}: {qid!r} 同时在 stated 和 unknown 里")
            elif not applies(q, t):
                out.append(f"{where}: 不会被问 {qid!r}, 不能列进 unknown")
            elif not q.judged:
                out.append(f"{where}: {qid!r} 的 judged 是 false, 标签只来自 stated, 不能列进 unknown")
    return out


def summary(questions: list[Question], texts: list[Text]) -> str:
    """给写的人看: 各风格几份, 每道题问到几份、每个选项被预定成答案几次.
    只问指定文本的题合成一行; 选项多于 12 个的只报被写明次数的最少 / 最多."""
    lines = [f"{len(texts)} 份文本: " + ", ".join(f"{s} {n}" for s, n in sorted(Counter(t.style for t in texts).items()))]
    for q in (q for q in questions if not q.texts):
        asked = sum(applies(q, t) for t in texts)
        c = Counter(t.stated[q.id] for t in texts if q.id in t.stated)
        counts = [c.get(i, 0) for i in range(len(q.options))]
        spread = (" ".join(map(str, counts)) if len(counts) <= 12
                  else f"{len(counts)} 个, 每个被写明 {min(counts)}..{max(counts)} 次")
        unknown = sum(q.id in t.unknown for t in texts)
        lines.append(f"  {q.id:24s} 问到 {asked:3d} 份, 写明答案 {sum(c.values()):3d} 份, "
                     + (f"没提 {unknown:3d} 份, " if unknown else "") + f"各选项 {spread}"
                     + ("  (参考模型不答)" if not q.judged else ""))
    own = [q for q in questions if q.texts]
    if own:
        asked = sum(applies(q, t) for q in own for t in texts)
        stated = sum(q.id in t.stated for q in own for t in texts)
        sizes = sorted(len(q.options) for q in own)
        per_text = Counter(t.id for q in own for t in texts if applies(q, t))
        lines.append(f"  只问指定文本的题 {len(own)} 道, 共问 {asked} 次, 写明答案 {stated} 次; "
                     f"选项数 {sizes[0]}..{sizes[-1]}, 中位 {sizes[len(sizes) // 2]}; "
                     f"每份文本 {min(per_text.values())}..{max(per_text.values())} 道")
    return "\n".join(lines)


def write(domain: str, label: str, questions: list[Question], texts: list[Text], out_dir) -> list[pathlib.Path]:
    """写 <domain>.questions.json (一行一道题) 和 <domain>.src.jsonl (一行一份文本)."""
    out_dir = pathlib.Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    qpath, tpath = out_dir / f"{domain}.questions.json", out_dir / f"{domain}.src.jsonl"
    head = json.dumps({"domain": domain, "label": label}, ensure_ascii=False)[:-1]
    body = ",\n".join(json.dumps(q.to_json(), ensure_ascii=False) for q in questions)
    qpath.write_text(f'{head}, "questions": [\n{body}\n]}}\n', encoding="utf-8")
    tpath.write_text("".join(json.dumps(t.to_json(), ensure_ascii=False) + "\n" for t in texts), encoding="utf-8")
    return [qpath, tpath]


def main(domain: str, label: str, questions: list[Question], texts: list[Text]) -> None:
    """<domain>.py 末尾调它: 检查, 通过就写文件并打印统计, 不通过就列出问题、退出码 1."""
    script = pathlib.Path(sys.argv[0]).resolve()
    ap = argparse.ArgumentParser(description=f"生成 {domain} 的 synth-intents v3 数据")
    ap.add_argument("--out", default=str(script.parent), help="输出目录, 默认是这个脚本所在的目录")
    args = ap.parse_args()
    errs = problems(domain, label, questions, texts)
    if script.stem not in (domain, "template"):
        errs.insert(0, f"文件名 {script.name} 与 DOMAIN {domain!r} 不一致")
    if errs:
        print(f"{len(errs)} 个问题:", *errs, sep="\n  ", file=sys.stderr)
        sys.exit(1)
    paths = write(domain, label, questions, texts, args.out)
    print(summary(questions, texts))
    for p in paths:
        print(f"→ {p}")
