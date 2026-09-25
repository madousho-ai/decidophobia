#!/usr/bin/env python
"""synth-intents v3 一个领域的模板. 复制成 <domain>.py, 改 DOMAIN 和 LABEL, 把 QUESTIONS、TEXTS 换成这个领域的, 然后

  python datasets/synth-intents-v3/<domain>.py              # 写到本目录
  python datasets/synth-intents-v3/<domain>.py --out DIR    # 写到别处

生成 <domain>.questions.json 与 <domain>.src.jsonl. 每个键的意思、检查哪些规则、写作规则, 见 schema.py.
文件名要与 DOMAIN 一致, 否则拒绝生成. 下面的三道题、两份文本是示例, 只示意形状.
"""

from schema import Question, Text, main

DOMAIN = "example"  # 领域 slug, snake_case, 与文件名相同
LABEL = "Bug report"  # 提示里 state 前面的标签, 如 "Customer message"

QUESTIONS = [
    # 问整份文本的题: 不写 needs, 每份文本都问
    Question(
        id="platform",
        ask=[
            "Where does the reporter run into the problem?",
            "On what platform does the bug occur?",
            "Which version of the product are they using when it happens?",
        ],
        options=[
            "In a web browser",
            "In the installed desktop app",
            "In the phone or tablet app",
            "Through the API or a script",
        ],
    ),
    Question(
        id="tone",
        ask=["How does the reporter come across?", "What is the tone of the report?", "How upset is the reporter?"],
        options=["Calm and matter-of-fact", "Frustrated but still polite", "Angry, blaming, or threatening to leave"],
    ),
    # 指着 JSON 字段的题: needs 列出字段, 每种问法都用反引号写出字段名
    Question(
        id="follow_up_tone",
        needs=["follow_up"],
        ask=[
            "How does the reporter come across in `follow_up`?",
            "What is the tone of `follow_up`?",
            "How upset is the reporter in `follow_up`?",
        ],
        options=["Calm and matter-of-fact", "Frustrated but still polite", "Angry, blaming, or threatening to leave"],
    ),
]

TEXTS = [
    # 纯文本 state
    Text(
        id="example_s1",
        style="short",
        text="since the update the save button on the profile screen does nothing on my phone. tapped it like 20 times",
        stated={"platform": 2},
    ),
    # JSON state, 字段互相不一致: 正文客气、追问生气. 于是 follow_up_tone 写明, 整份文本的 tone 不列
    Text(
        id="example_j1",
        style="json",
        text={
            "title": "Password reset email never arrives",
            "report": "Hi, I requested a password reset twice this morning and no email has arrived yet. Could you check?",
            "environment": "Web, Chrome 129",
            "follow_up": "Three days and still nothing. I already told you it's not in spam. Fix this or I'm cancelling.",
        },
        stated={"platform": 0, "follow_up_tone": 2},
    ),
]

if __name__ == "__main__":
    main(DOMAIN, LABEL, QUESTIONS, TEXTS)
