#!/usr/bin/env python
"""synth-intents v3: sec_ops world. 安全运维值班人员眼前的 state, 题目问值班人员要判断的事.

  python datasets/synth-intents-v3/sec_ops.py [--out DIR]

6 种 state, 每种 3 份: 告警队列、登录日志、防火墙规则表、漏洞扫描结果 (JSON)、主机连接表、处置手册加当前案件.
每种配 3-4 道题, 只问这一种的 3 份: 选项在 3 份之间通用的写成一道题 (texts 列 3 份, 3 份答案互不相同);
选项取自某一份 state 的, 每份各一道, 问法一字不差. 答案都由 state 定死, 写进 stated.
登录日志与连接表各有一道 31 项以上的大菜单题 (选项是日志里的全部时刻 / 表里的全部远端地址), 参考模型不答.
外部地址只用文档保留段 (192.0.2.x, 198.51.100.x, 203.0.113.x), 内网用 10.x. 键的意思与写作规则见 schema.py.
"""

import random

from schema import YES_NO, Question, Text, main

DOMAIN = "sec_ops"
LABEL = "Security on-call screen"
QUESTIONS: list[Question] = []
TEXTS: list[Text] = []


def tid(suffix: str) -> str:
    return f"{DOMAIN}_{suffix}"


def question(qid: str, ask, options, suffixes: list[str], *, natural: bool = False, judged: bool = True) -> None:
    """加一道只问 suffixes 这几份 state 的题. 没有自然顺序的选项按题 id 固定打乱, 让答案落在各个位置."""
    options = list(options)
    if not natural and options != YES_NO:
        random.Random(qid).shuffle(options)
    QUESTIONS.append(Question(id=qid, ask=[ask] if isinstance(ask, str) else list(ask), options=options,
                              texts=[tid(s) for s in suffixes], judged=judged))


def state(suffix: str, style: str, text, answers: dict[str, str]) -> None:
    """一份 state. answers 是 {题 id: 答案的选项原文} (写错会在 .index 处报错)."""
    by_id = {q.id: q for q in QUESTIONS}
    stated = {qid: by_id[qid].options.index(a) for qid, a in answers.items()}
    TEXTS.append(Text(id=tid(suffix), style=style, text=text, stated=stated))


def table(cols: list[str], rows: list[tuple]) -> str:
    """列用 | 隔开, 每列按最宽的一格对齐."""
    w = [max(len(str(r[i])) for r in [cols, *rows]) for i in range(len(cols))]
    return "\n".join(" | ".join(str(c).ljust(w[i]) for i, c in enumerate(r)).rstrip() for r in [cols, *rows])


# ---------------------------------------------------------------- 1. 告警队列 (table)
# 9 行, 按严重度排、同级内时间乱序. 8 种 detection 各一次, 外加一条同 detection 同主机的重复.
# 每份都有一条比答案更早、无人认领的 high, 专门接住「只看最早、不看严重度」的读法.

DETECTIONS = [
    "Credential dump tool", "Impossible travel sign-in", "New local admin account",
    "Outbound traffic to a rare domain", "Port scan from inside", "Security logging turned off",
    "Mass file renaming", "Office app spawned a shell",
]
QUEUE_COLS = ["Alert", "Raised", "Severity", "Detection", "Host", "Owner", "State"]
QUEUES = {
    "queue_a": ("23 Sep", "14:30", [
        ("LOG-0719", "12:47", "critical", "Security logging turned off", "db-04", "-", "New"),
        ("NET-0457", "13:58", "critical", "Outbound traffic to a rare domain", "web-02", "-", "New"),
        ("EDR-2291", "12:05", "critical", "Credential dump tool", "dc-01", "Ines", "Working"),
        ("EDR-2310", "14:21", "high", "Mass file renaming", "file-02", "-", "New"),
        ("IDP-1180", "11:52", "high", "Impossible travel sign-in", "vpn-gw", "-", "New"),
        ("LOG-0722", "13:10", "high", "New local admin account", "ws-09", "Lena", "Working"),
        ("NET-0451", "13:36", "medium", "Port scan from inside", "ws-14", "-", "New"),
        ("EDR-2302", "13:44", "medium", "Office app spawned a shell", "ws-22", "Lena", "Working"),
        ("NET-0442", "12:20", "medium", "Port scan from inside", "ws-14", "Omar", "New"),
    ]),
    "queue_b": ("24 Sep", "09:15", [
        ("EDR-4406", "08:03", "critical", "Office app spawned a shell", "ws-31", "-", "New"),
        ("EDR-4415", "08:58", "critical", "Mass file renaming", "file-01", "Raj", "Working"),
        ("NET-1124", "07:35", "critical", "Outbound traffic to a rare domain", "ws-31", "-", "New"),
        ("EDR-4412", "08:39", "high", "Credential dump tool", "ws-18", "-", "New"),
        ("IDP-2046", "07:30", "high", "Impossible travel sign-in", "sso-01", "-", "New"),
        ("LOG-0812", "07:52", "high", "New local admin account", "dc-02", "Mei", "Working"),
        ("EDR-4401", "07:44", "high", "Credential dump tool", "ws-18", "Mei", "Working"),
        ("NET-1130", "09:11", "medium", "Port scan from inside", "srv-05", "-", "New"),
        ("LOG-0815", "08:47", "medium", "Security logging turned off", "ws-12", "-", "New"),
    ]),
    "queue_c": ("25 Sep", "22:40", [
        ("LOG-0921", "20:40", "critical", "Security logging turned off", "ws-44", "Ines", "Working"),
        ("EDR-5520", "21:14", "critical", "Credential dump tool", "srv-08", "-", "New"),
        ("IDP-3307", "20:26", "critical", "Impossible travel sign-in", "mail-01", "Raj", "Working"),
        ("EDR-5531", "22:36", "high", "Office app spawned a shell", "ws-40", "-", "New"),
        ("NET-2281", "20:12", "high", "Outbound traffic to a rare domain", "ws-40", "-", "New"),
        ("EDR-5512", "20:55", "high", "Mass file renaming", "file-03", "-", "New"),
        ("LOG-0934", "22:05", "medium", "New local admin account", "srv-08", "-", "New"),
        ("NET-2288", "21:02", "medium", "Port scan from inside", "ws-17", "-", "New"),
        ("LOG-0926", "21:05", "medium", "New local admin account", "srv-08", "Lena", "New"),
    ]),
}
# (pickup, repeat, latest, new_owned)
QUEUE_ANSWERS = {
    "queue_a": ("LOG-0719", "NET-0451", "Mass file renaming", "yes"),
    "queue_b": ("NET-1124", "EDR-4412", "Port scan from inside", "no"),
    "queue_c": ("EDR-5520", "LOG-0934", "Office app spawned a shell", "yes"),
}

question("queue_latest", ["Which detection raised the most recent alert in the queue?",
                          "What was detected in the newest alert in the queue?"], DETECTIONS, list(QUEUES))
question("queue_new_owned", ["Is any alert in state New already assigned to an owner?",
                             "Does the queue have an alert that someone owns but that is still marked New?"],
         YES_NO, list(QUEUES))
for s, (day, refreshed, rows) in QUEUES.items():
    ids = [r[0] for r in rows]
    question(f"{s}_pickup", ["Under the pick-up rule at the top of the queue, which alert should be taken next?",
                             "Which alert does the pick-up rule say to work on next?"], ids, [s])
    question(f"{s}_repeat", ["Which alert repeats an earlier alert, with the same detection on the same host?",
                             "Which alert is a later copy of another alert: same detection, same host?"], ids, [s])
    pick, rep, latest, owned = QUEUE_ANSWERS[s]
    state(s, "table",
          f"Alert queue, {day}, refreshed {refreshed}\n"
          "Pick-up rule: take the oldest critical alert that has no owner.\n\n" + table(QUEUE_COLS, rows),
          {f"{s}_pickup": pick, f"{s}_repeat": rep, "queue_latest": latest, "queue_new_owned": owned})


# ---------------------------------------------------------------- 2. 登录日志 (log, 大)
# 一次尝试一行. 攻击地址的全部尝试手写; 另有一个外部地址只正常登录一次 (陷阱), 两条内网噪声
# (打错一次密码 / 误点拒绝一次, 随后登上), 其余是内网地址的正常登录, 14 个账号每份都出现.

STAFF = ["ines", "omar", "lena", "raj", "mei", "tomas", "sofia", "kofi", "yuki", "arjun", "nadia", "piet"]
ACCOUNTS = STAFF + ["svc_backup", "svc_build"]
SIGNIN_HEAD = ("Sign-in log, staff portal, {day}. One line per attempt.\n"
               "result=ok: signed in. result=bad_password: the password was wrong. "
               "result=push_denied: the password was right, but the approval prompt on the user's phone "
               "was rejected.\n\n")


def hms(t: int) -> str:
    return f"{t // 3600:02d}:{t // 60 % 60:02d}:{t % 60:02d}"


def burst(start: str, users: list[str], src: str, results: list[str], seed: str, gap: tuple[int, int]) -> list:
    """攻击地址的一串尝试: 从 start 起, 每次间隔 gap 秒内随机."""
    h, m, x = map(int, start.split(":"))
    t, rng, out = h * 3600 + m * 60 + x, random.Random(seed), []
    for u, r in zip(users, results, strict=True):
        out.append((t, u, src, r))
        t += rng.randint(*gap)
    return out


def signin_log(day: str, hour: int, attack: list, benign: tuple, noise: list, fill: int, seed: str):
    """拼一份登录日志. benign 是 (时刻, 账号, 外部地址) 的一次正常登录; noise 是 [(账号, 结果)], 该账号随后
    20-40 秒从同一内网地址登上; fill 行内网正常登录. 返回 (文本, 全部时刻按顺序)."""
    rng = random.Random(seed)
    home = {a: f"10.20.{rng.randint(1, 9)}.{rng.randint(11, 249)}" for a in ACCOUNTS}
    h, m, x = map(int, benign[0].split(":"))
    lines = [*attack, (h * 3600 + m * 60 + x, benign[1], benign[2], "ok")]
    taken = [t for t, *_ in lines]
    lo, hi = hour * 3600 + 20, hour * 3600 + 3540

    def free(t=None):
        while t is None or any(abs(t - u) < 4 for u in taken):
            t = rng.randint(lo, hi)
        taken.append(t)
        return t

    for user, result in noise:
        t = free()
        lines += [(t, user, home[user], result), (free(t + rng.randint(20, 40)), user, home[user], "ok")]
    for u in ACCOUNTS + [rng.choice(STAFF) for _ in range(fill - len(ACCOUNTS))]:
        lines.append((free(), u, home[u], "ok"))
    lines.sort()
    body = "\n".join(f"{hms(t)}  user={u:<10s}  src={src:<14s}  result={r}" for t, u, src, r in lines)
    return SIGNIN_HEAD.format(day=day) + body, [hms(t) for t, *_ in lines]


SIGNINS = {
    # 暴力破解: 同一外部地址对 raj 连试 10 次错密码, 第 11 次登上
    "signin_a": signin_log("22 Sep", 9,
                           burst("09:31:05", ["raj"] * 11, "198.51.100.23", ["bad_password"] * 10 + ["ok"],
                                 "signin_a", (11, 19)),
                           ("09:18:40", "kofi", "192.0.2.30"), [("sofia", "bad_password"), ("omar", "push_denied")],
                           26, "signin_a"),
    # 密码喷洒: 同一外部地址对 12 个账号各试一次, 第 8 个 mei 登上, 之后又试了 4 个
    "signin_b": signin_log("23 Sep", 13,
                           burst("13:08:12", ["arjun", "kofi", "ines", "yuki", "piet", "omar", "nadia", "mei",
                                              "sofia", "lena", "tomas", "svc_build"], "203.0.113.61",
                                 ["bad_password"] * 7 + ["ok"] + ["bad_password"] * 4, "signin_b", (8, 13)),
                           ("13:41:09", "yuki", "198.51.100.140"), [("raj", "bad_password"), ("lena", "push_denied")],
                           23, "signin_b"),
    # 推送轰炸: 密码是对的, 对 tomas 连发 7 次推送被拒, 第 8 次通过
    "signin_c": signin_log("25 Sep", 22,
                           burst("22:47:20", ["tomas"] * 8, "192.0.2.77", ["push_denied"] * 7 + ["ok"],
                                 "signin_c", (25, 45)),
                           ("22:12:33", "nadia", "203.0.113.5"), [("mei", "bad_password"), ("arjun", "push_denied")],
                           23, "signin_c"),
}
SIGNIN_PATTERNS = [
    "Many wrong passwords against one account, all from one address",
    "One address trying one password on each of many accounts",
    "One account's approval prompts rejected again and again until one was accepted",
]
# (pattern, breached, 攻击地址第一次登上的时刻, repeated)
SIGNIN_ANSWERS = {
    "signin_a": (SIGNIN_PATTERNS[0], "raj", "198.51.100.23", "yes"),
    "signin_b": (SIGNIN_PATTERNS[1], "mei", "203.0.113.61", "no"),
    "signin_c": (SIGNIN_PATTERNS[2], "tomas", "192.0.2.77", "yes"),
}

question("signin_pattern", ["Which description fits the attack in this log?",
                            "What kind of attack does this sign-in log show?"], SIGNIN_PATTERNS, list(SIGNINS))
question("signin_breached", ["Which account did the attacker get into?",
                             "Which account did the attacking address manage to sign in to?"], ACCOUNTS, list(SIGNINS))
question("signin_repeated", ["Did the attacking address try any one account more than once?",
                             "Did the attacker go back to the same account for a second try?"], YES_NO, list(SIGNINS))
for s, (text, times) in SIGNINS.items():
    pattern, breached, src, repeated = SIGNIN_ANSWERS[s]
    first_ok = next(ln[:8] for ln in text.splitlines() if f"src={src} " in ln and ln.endswith("result=ok"))
    question(f"{s}_first_ok", "At what time did the attacking address first sign in successfully?", times, [s],
             natural=True, judged=False)
    state(s, "log", text, {"signin_pattern": pattern, "signin_breached": breached, "signin_repeated": repeated,
                           f"{s}_first_ok": first_ok})


# ---------------------------------------------------------------- 3. 防火墙规则表 (table)
# 12 条规则, 从上往下第一条命中的决定. 规则名都是三段, 免得按词数挑. 每份恰有一条被上面的规则完全盖住.
# 待做的改动: A 删掉当前决定的那条 (换成下面的一条), B 挪一条无关的到顶 (不变), C 把一条会命中的挪到顶.

FW_HEAD = (
    "Firewall {name}: traffic coming into the server network, {day}\n"
    "Rules are checked from the top down. The first rule that matches decides; traffic that matches no rule is dropped.\n"
    "A rule matches when protocol, source, destination and port all match. \"any\" matches everything. "
    "An address ending in .0.0/16 covers every address that starts with the same first two numbers; "
    "one ending in .0/24 covers every address that starts with the same first three numbers.\n"
    "Actions: allow = let through. drop = discard silently. reject = discard and send a reset back.\n\n"
)
FW_COLS = ["Rule", "Action", "Proto", "Source", "Destination", "Port"]
FIREWALLS = {
    "fw_a": ("edge-fw-1", "23 Sep", [
        ("allow-web-https", "allow", "tcp", "any", "10.50.1.0/24", "443"),
        ("reject-guest-wifi", "reject", "any", "10.30.0.0/16", "any", "any"),
        ("allow-admin-ssh", "allow", "tcp", "10.8.1.0/24", "10.50.0.0/16", "22"),
        ("allow-app-db", "allow", "tcp", "10.50.1.0/24", "10.50.2.20", "5432"),
        ("allow-backup-db", "allow", "tcp", "10.8.9.0/24", "10.50.2.20", "5432"),
        ("reject-staff-db", "reject", "tcp", "10.8.0.0/16", "10.50.2.0/24", "any"),
        ("allow-web-http", "allow", "tcp", "any", "10.50.1.0/24", "80"),
        ("drop-ssh-rest", "drop", "tcp", "any", "10.50.0.0/16", "22"),
        ("allow-admin-web", "allow", "tcp", "10.8.1.0/24", "10.50.1.0/24", "443"),
        ("allow-snmp-poll", "allow", "udp", "10.8.5.0/24", "10.50.0.0/16", "161"),
        ("drop-db-rest", "drop", "any", "any", "10.50.2.0/24", "any"),
        ("allow-dns-udp", "allow", "udp", "any", "10.50.0.53", "53"),
    ], "tcp from 10.8.3.44 to 10.50.2.20, port 5432", "delete rule reject-staff-db."),
    "fw_b": ("core-fw-2", "24 Sep", [
        ("reject-guest-wifi", "reject", "any", "10.30.0.0/16", "any", "any"),
        ("allow-vpn-web", "allow", "tcp", "10.60.0.0/16", "10.50.1.0/24", "443"),
        ("drop-telnet-all", "drop", "tcp", "any", "any", "23"),
        ("allow-jump-ssh", "allow", "tcp", "10.8.1.0/24", "10.50.9.0/24", "22"),
        ("allow-web-https", "allow", "tcp", "any", "10.50.1.0/24", "443"),
        ("allow-staff-files", "allow", "tcp", "10.8.0.0/16", "10.50.4.0/24", "445"),
        ("drop-ssh-rest", "drop", "tcp", "any", "10.50.0.0/16", "22"),
        ("allow-mail-smtp", "allow", "tcp", "any", "10.50.3.25", "25"),
        ("allow-ops-ssh", "allow", "tcp", "10.8.1.0/24", "10.50.9.4", "22"),
        ("reject-smb-rest", "reject", "tcp", "any", "10.50.0.0/16", "445"),
        ("allow-ntp-sync", "allow", "udp", "any", "10.50.0.123", "123"),
        ("drop-mgmt-rest", "drop", "any", "any", "10.50.9.0/24", "any"),
    ], "tcp from 10.8.1.15 to 10.50.9.4, port 22", "move rule allow-ntp-sync to the top of the list."),
    "fw_c": ("dmz-fw-3", "25 Sep", [
        ("allow-web-https", "allow", "tcp", "any", "10.50.1.0/24", "443"),
        ("reject-guest-wifi", "reject", "any", "10.30.0.0/16", "any", "any"),
        ("allow-admin-ssh", "allow", "tcp", "10.8.1.0/24", "10.50.0.0/16", "22"),
        ("drop-flagged-host", "drop", "any", "10.70.2.8", "any", "any"),
        ("allow-partner-api", "allow", "tcp", "10.70.0.0/16", "10.50.5.0/24", "8443"),
        ("allow-mail-smtp", "allow", "tcp", "any", "10.50.3.25", "25"),
        ("allow-partner-sftp", "allow", "tcp", "10.70.2.0/24", "10.50.5.30", "22"),
        ("drop-ssh-rest", "drop", "tcp", "any", "10.50.0.0/16", "22"),
        ("allow-admin-mail", "allow", "tcp", "10.8.1.0/24", "10.50.3.25", "25"),
        ("reject-smb-rest", "reject", "tcp", "any", "10.50.0.0/16", "445"),
        ("allow-dns-udp", "allow", "udp", "any", "10.50.0.53", "53"),
        ("allow-snmp-poll", "allow", "udp", "10.8.5.0/24", "10.50.0.0/16", "161"),
    ], "tcp from 10.70.2.8 to 10.50.5.30, port 22", "move rule allow-partner-sftp to the top of the list."),
}
FW_OUTCOMES = ["Let through", "Discarded silently", "Discarded with a reset sent back"]
# (现在决定的规则, 改动之后决定的规则, 被完全盖住的规则, 连接的下场)
FW_ANSWERS = {
    "fw_a": ("reject-staff-db", "drop-db-rest", "allow-admin-web", FW_OUTCOMES[2]),
    "fw_b": ("allow-jump-ssh", "allow-jump-ssh", "allow-ops-ssh", FW_OUTCOMES[0]),
    "fw_c": ("drop-flagged-host", "allow-partner-sftp", "allow-admin-mail", FW_OUTCOMES[1]),
}

question("fw_outcome", ["As the rules stand now, before the pending change, what happens to the connection being "
                        "checked?", "Before the pending change is made, how does the firewall treat the connection "
                        "being checked?"], FW_OUTCOMES, list(FIREWALLS))
for s, (name, day, rules, conn, change) in FIREWALLS.items():
    names = [r[0] for r in rules]
    question(f"{s}_rule", ["As the rules stand now, before the pending change, which rule decides what happens to "
                           "the connection being checked?", "Before the pending change is made, which rule is the "
                           "first to match the connection being checked?"], names, [s])
    question(f"{s}_after", "Once the pending change is made, which rule will decide the connection being checked?",
             names, [s])
    question(f"{s}_shadowed", "As the rules stand now, which rule can never match any traffic, because a rule "
             "above it already matches all of that traffic?", names, [s])
    rule, after, shadowed, outcome = FW_ANSWERS[s]
    state(s, "table", FW_HEAD.format(name=name, day=day) + table(FW_COLS, rules)
          + f"\n\nConnection being checked: {conn}\nPending change (not made yet): {change}",
          {f"{s}_rule": rule, f"{s}_after": after, f"{s}_shadowed": shadowed, "fw_outcome": outcome})


# ---------------------------------------------------------------- 4. 漏洞扫描结果 (json)
# 8 条发现. 陷阱: 内网主机上的 critical、面向互联网的 high. 标错级别的那条都在内网或远离 critical 的边界,
# 不影响「两天内修」那道题. id 与发现时间无关, 免得按编号挑最早的.

SCAN_SCALE = "score 9.0 to 10.0 is critical; 7.0 to 8.9 is high; 4.0 to 6.9 is medium; below 4.0 is low"
SCAN_POLICY = ("Critical findings on internet-facing hosts: fix within two days. Other critical findings, and all "
               "high findings: fix within fourteen days. Medium and low findings: fix in the next quarterly cycle.")


def finding(fid, host, facing, issue, score, severity, fix, seen) -> dict:
    return {"id": fid, "host": host, "internet_facing": facing, "issue": issue, "score": score,
            "severity": severity, "fix_available": fix, "first_seen": seen}


SCANS = {
    "scan_a": ("25 Sep 03:10", [
        finding("VS-3817", "web-01", True, "Directory listing enabled on the web server", 5.3, "medium", True, "18 Sep"),
        finding("VS-1094", "db-04", False, "Database server version with a known privilege escalation", 9.1,
                "critical", True, "11 Sep"),
        finding("VS-4418", "build-03", False, "SSH server allows password login for root", 7.2, "medium", True,
                "28 Aug"),
        finding("VS-7731", "mail-01", True, "Weak cipher suites offered", 7.5, "high", True, "4 Sep"),
        finding("VS-6142", "file-02", False, "SMB signing not required", 5.9, "medium", True, "16 Sep"),
        finding("VS-5260", "vpn-gw", True, "Remote code execution in the VPN portal", 9.8, "critical", True, "22 Sep"),
        finding("VS-8309", "wiki-01", False, "Expired TLS certificate", 3.1, "low", True, "9 Sep"),
        finding("VS-2675", "print-01", False, "Printer admin page has no login", 6.5, "medium", False, "21 Aug"),
    ]),
    "scan_b": ("22 Sep 03:10", [
        finding("VS-6604", "vpn-gw", True, "TLS 1.0 still enabled", 5.0, "medium", True, "30 Aug"),
        finding("VS-1733", "jump-01", False, "SSH server allows password login for root", 7.8, "high", True, "12 Aug"),
        finding("VS-9012", "hr-app", False, "Default admin password on the management page", 9.8, "critical", True,
                "16 Sep"),
        finding("VS-4870", "mail-01", True, "Mail server version with a known memory corruption bug", 6.8, "high",
                True, "23 Aug"),
        finding("VS-2291", "web-02", True, "Remote code execution in the web framework", 9.6, "critical", True,
                "19 Sep"),
        finding("VS-5521", "file-02", False, "SMB signing not required", 5.9, "medium", True, "26 Aug"),
        finding("VS-3158", "dc-01", False, "Kernel privilege escalation", 8.8, "high", True, "2 Sep"),
        finding("VS-7046", "print-01", False, "Expired TLS certificate", 2.6, "low", True, "9 Sep"),
    ]),
    "scan_c": ("25 Sep 03:10", [
        finding("VS-4702", "build-03", False, "Kernel privilege escalation", 7.8, "high", True, "7 Aug"),
        finding("VS-3371", "wiki-01", False, "Default admin password on the management page", 9.8, "critical", True,
                "24 Sep"),
        finding("VS-9264", "hr-app", False, "TLS 1.0 still enabled", 5.0, "medium", True, "20 Aug"),
        finding("VS-2047", "web-01", True, "Weak cipher suites offered", 7.4, "high", True, "3 Sep"),
        finding("VS-5903", "file-02", False, "SMB signing not required", 5.9, "medium", True, "29 Aug"),
        finding("VS-8125", "mail-01", True, "Mail server version with a known remote code execution bug", 9.3,
                "critical", True, "23 Sep"),
        finding("VS-1586", "vpn-gw", True, "Directory listing enabled on the portal", 4.3, "low", True, "21 Aug"),
        finding("VS-6390", "db-04", False, "Database server version with a known privilege escalation", 9.0,
                "critical", True, "15 Sep"),
    ]),
}
# (两天内修, 级别标错, 最早发现, 有没有还没出补丁的)
SCAN_ANSWERS = {
    "scan_a": ("VS-5260", "VS-4418", "VS-2675", "yes"),
    "scan_b": ("VS-2291", "VS-4870", "VS-1733", "no"),
    "scan_c": ("VS-8125", "VS-1586", "VS-4702", "no"),
}

question("scan_nofix", ["Does any finding have no fix available yet?",
                        "Is any finding still waiting for a fix to be released?"], YES_NO, list(SCANS))
for s, (finished, found) in SCANS.items():
    ids = [f["id"] for f in found]
    question(f"{s}_urgent", ["Which finding has to be fixed within two days under the `fix_policy`?",
                             "Which finding falls under the two-day deadline in `fix_policy`?"], ids, [s])
    question(f"{s}_mislabeled", ["Which finding has a `severity` that does not fit its `score` on the "
                                 "`severity_scale`?", "Which finding's `severity` label is wrong for its `score`?"],
             ids, [s])
    question(f"{s}_oldest", ["Which finding has been open the longest, going by `first_seen`?",
                             "Which finding was first seen earliest?"], ids, [s])
    urgent, mislabeled, oldest, nofix = SCAN_ANSWERS[s]
    state(s, "json", {"scan": f"Weekly internal scan, finished {finished}", "severity_scale": SCAN_SCALE,
                      "fix_policy": SCAN_POLICY, "findings": found},
          {f"{s}_urgent": urgent, f"{s}_mislabeled": mislabeled, f"{s}_oldest": oldest, "scan_nofix": nofix})


# ---------------------------------------------------------------- 5. 主机连接表 (table, 大)
# 一台主机的连接, 监听行在前、其余乱序. 手写两行: 连到允许清单外的那条, 以及监听在预期清单外端口的进程.
# 其余由代码按主机拼出, 远端全在 10. 开头或正好是镜像 198.51.100.7. 内网地址写成三位数的段, 免得外部地址
# 总是最长的选项. Started by 一列三种值 (httpd / crond / sshd) 每份都有正常行, 监听进程故意取另外的值.

CONN_HEAD = ("Open connections on {host} ({ip}), captured {when}\n"
             "Allowed remote addresses for this host: any address that starts with 10., and the update mirror "
             "198.51.100.7.\n"
             "Expected listening ports: {expected}.\n\n")
CONN_COLS = ["Proto", "Local", "Remote", "State", "Process", "Started by", "User"]


def conn_table(ip: str, service: tuple, clients: list[str], n_clients: int, backends: list[str],
               special: list[tuple], seed: str) -> tuple[list[tuple], list[str]]:
    """service = (进程名, 端口, 用户). 返回 (全部行, 非监听行的远端地址)."""
    rng = random.Random(seed)
    ports: set[int] = set()

    def eph() -> int:
        p = rng.randint(32768, 60999)
        while p in ports:
            p = rng.randint(32768, 60999)
        ports.add(p)
        return p

    proc, port, user = service
    listen = [("tcp", "0.0.0.0:22", "0.0.0.0:*", "LISTEN", "sshd", "systemd", "root"),
              ("tcp", f"0.0.0.0:{port}", "0.0.0.0:*", "LISTEN", proc, "systemd", user),
              ("tcp", "0.0.0.0:9100", "0.0.0.0:*", "LISTEN", "metrics-agent", "systemd", "nobody")]
    rest = [("tcp", f"{ip}:{port}", f"{rng.choice(clients)}:{eph()}", "ESTABLISHED", proc, "systemd", user)
            for _ in range(n_clients)]
    rest += [("tcp", f"{ip}:{eph()}", b, "ESTABLISHED", proc, "systemd", user) for b in backends]
    rest += [
        ("tcp", f"{ip}:22", f"10.108.1.15:{eph()}", "ESTABLISHED", "sshd", "systemd", "root"),
        ("tcp", f"{ip}:9100", f"10.160.12.201:{eph()}", "ESTABLISHED", "metrics-agent", "systemd", "nobody"),
        ("tcp", f"{ip}:{eph()}", "10.160.33.120:5044", "ESTABLISHED", "log-shipper", "systemd", "root"),
        ("udp", f"{ip}:{eph()}", "10.100.0.2:53", "-", "dns-cache", "systemd", "nobody"),
        ("udp", f"{ip}:{eph()}", "10.100.0.3:53", "-", "dns-cache", "systemd", "nobody"),
        ("tcp", f"{ip}:{eph()}", "10.150.8.14:6379", "ESTABLISHED", "render-worker", "httpd", "www-data"),
        ("tcp", f"{ip}:{eph()}", "10.150.8.15:6379", "TIME_WAIT", "render-worker", "httpd", "www-data"),
        ("tcp", f"{ip}:{eph()}", "10.170.44.108:873", "ESTABLISHED", "backup-agent", "crond", "root"),
        ("tcp", f"{ip}:{eph()}", "198.51.100.7:443", "ESTABLISHED", "update-agent", "crond", "root"),
        ("tcp", f"{ip}:{eph()}", "10.50.2.22:5432", "ESTABLISHED", "db-client", "sshd", "deploy"),
    ]
    listen += [r for r in special if r[3] == "LISTEN"]
    rest += [r for r in special if r[3] != "LISTEN"]
    rng.shuffle(rest)
    return listen + rest, [r[2] for r in rest]


CONNS = {
    "conn_a": ("app-03", "10.52.3.7", "23 Sep 15:42", "22 (sshd), 443 (httpd), 9100 (metrics-agent)",
               conn_table("10.52.3.7", ("httpd", 443, "www-data"), ["10.140.100.21", "10.140.100.22"], 22,
                          ["10.150.8.30:8080"],
                          [("tcp", "10.52.3.7:41822", "203.0.113.88:8443", "ESTABLISHED", "kupdate", "httpd",
                            "www-data"),
                           ("tcp", "0.0.0.0:5005", "0.0.0.0:*", "LISTEN", "debug-console", "sshd", "root")],
                          "conn_a")),
    "conn_b": ("db-gw-01", "10.52.8.14", "24 Sep 03:17",
               "22 (sshd), 6432 (db-proxy), 8081 (httpd), 9100 (metrics-agent)",
               conn_table("10.52.8.14", ("db-proxy", 6432, "dbproxy"),
                          ["10.152.20.11", "10.152.20.12", "10.152.20.13"], 22,
                          ["10.50.2.20:5432", "10.50.2.21:5432"],
                          [("tcp", "10.52.8.14:39410", "198.51.100.71:443", "ESTABLISHED", "sync-helper", "crond",
                            "root"),
                           ("tcp", "0.0.0.0:8081", "0.0.0.0:*", "LISTEN", "httpd", "systemd", "www-data"),
                           ("tcp", "0.0.0.0:8000", "0.0.0.0:*", "LISTEN", "tmp-share", "sshd", "deploy")],
                          "conn_b")),
    "conn_c": ("build-03", "10.52.6.30", "25 Sep 11:05", "22 (sshd), 8080 (httpd), 9100 (metrics-agent)",
               conn_table("10.52.6.30", ("httpd", 8080, "www-data"), ["10.140.100.21", "10.140.100.22"], 21,
                          ["10.150.8.31:8080"],
                          [("tcp", "10.52.6.30:50112", "192.0.2.45:53", "ESTABLISHED", "netcheck", "sshd", "deploy"),
                           ("tcp", "0.0.0.0:1080", "0.0.0.0:*", "LISTEN", "relay-svc", "crond", "nobody")],
                          "conn_c")),
}
CONN_STARTERS = ["httpd", "crond", "sshd"]
# (清单外的远端, 多出来的监听进程, 连清单外的进程是谁起的, 监听进程是不是 root)
CONN_ANSWERS = {
    "conn_a": ("203.0.113.88:8443", "debug-console", "httpd", "yes"),
    "conn_b": ("198.51.100.71:443", "tmp-share", "crond", "no"),
    "conn_c": ("192.0.2.45:53", "relay-svc", "sshd", "no"),
}

question("conn_started_by", ["What started the process that talks to the address outside the allowed list?",
                             "Which program launched the process connected to the address that is not allowed?"],
         CONN_STARTERS, list(CONNS))
question("conn_root", ["Does the process listening on the unexpected port run as root?",
                       "Is the process on the port missing from the expected list running as root?"],
         YES_NO, list(CONNS))
for s, (host, ip, when, expected, (rows, remotes)) in CONNS.items():
    question(f"{s}_outside", "Which remote address is outside the allowed list at the top?", remotes, [s],
             judged=False)
    question(f"{s}_listener", "Which process listens on a port that is not in the expected list at the top?",
             sorted({r[4] for r in rows}), [s])
    outside, listener, starter, root = CONN_ANSWERS[s]
    state(s, "table", CONN_HEAD.format(host=host, ip=ip, when=when, expected=expected) + table(CONN_COLS, rows),
          {f"{s}_outside": outside, f"{s}_listener": listener, "conn_started_by": starter, "conn_root": root})


# ---------------------------------------------------------------- 6. 处置手册 + 当前案件 (structured)
# 同一份手册, 三个案件. A: 杀软已删、另一条告警在两天前 (不算) → 关案, 谁都不用告诉.
# B: 杀软已删、但 8 小时前另有告警 → 断网; 用户有服务器管理权限 → 告诉事件负责人.
# C: 杀软没删掉、已断网并存了内存镜像 → 重装; 机器上有客户记录 → 告诉隐私官.

RUNBOOK = (
    "Runbook: malware alert on a staff laptop or desktop\n"
    "(Servers are handled by the server runbook, not this one.)\n"
    "Step 1. If the antivirus removed the file and no other alert fired on the machine in the 24 hours before this "
    "alert, close the case.\n"
    "Step 2. Otherwise, cut the machine off from the network.\n"
    "Step 3. Once the machine is cut off, save a memory image.\n"
    "Step 4. Once the memory image is saved, wipe and reinstall the machine.\n"
    "Who to tell:\n"
    "- The incident lead, if the user has admin rights on any server.\n"
    "- The privacy officer, if the machine holds customer records and the antivirus did not remove the file.\n"
    "- Nobody, in every other case.\n\n"
    "Current case\n"
)
INCIDENTS = {
    "incident_a": ("Machine: ws-22, desktop\n"
                   "Alert: 25 Sep 10:14, trojan found in the Downloads folder\n"
                   "Antivirus: file removed\n"
                   "Other alerts on this machine: 23 Sep 09:50, suspicious script blocked\n"
                   "User: lena. Admin rights on servers: none\n"
                   "Holds customer records: yes\n"
                   "Done so far: nothing yet"),
    "incident_b": ("Machine: lt-107, laptop\n"
                   "Alert: 18 Sep 16:40, password stealer found in a browser add-on folder\n"
                   "Antivirus: file removed\n"
                   "Other alerts on this machine: 18 Sep 08:05, connection to a known bad domain blocked\n"
                   "User: raj. Admin rights on servers: yes, on the build servers\n"
                   "Holds customer records: no\n"
                   "Done so far: nothing yet"),
    "incident_c": ("Machine: ws-61, desktop\n"
                   "Alert: 21 Sep 13:52, ransomware dropper found in a mail attachment folder\n"
                   "Antivirus: could not remove the file\n"
                   "Other alerts on this machine: none\n"
                   "User: sofia. Admin rights on servers: none\n"
                   "Holds customer records: yes\n"
                   "Done so far: cut off from the network at 14:05; memory image saved at 14:40"),
}
NEXT_STEPS = ["Close the case", "Cut the machine off from the network", "Save a memory image",
              "Wipe and reinstall the machine"]
NOTIFY = ["The incident lead", "The privacy officer", "Both the incident lead and the privacy officer", "Nobody"]
# (下一步, 告诉谁, 24 小时内有没有别的告警)
INCIDENT_ANSWERS = {
    "incident_a": (NEXT_STEPS[0], NOTIFY[3], "no"),
    "incident_b": (NEXT_STEPS[1], NOTIFY[0], "yes"),
    "incident_c": (NEXT_STEPS[3], NOTIFY[1], "no"),
}

question("incident_next", ["Following the runbook, what is the next action to take?",
                           "What does the runbook say to do next?"], NEXT_STEPS, list(INCIDENTS))
question("incident_notify", ["According to the runbook, who has to be told about this case?",
                             "Who must be informed, going by the runbook?"], NOTIFY, list(INCIDENTS))
question("incident_recent", ["Did another alert fire on the same machine in the 24 hours before the current alert?",
                             "Was there a different alert on this machine within a day before the current one?"],
         YES_NO, list(INCIDENTS))
for s, case in INCIDENTS.items():
    nxt, notify, recent = INCIDENT_ANSWERS[s]
    state(s, "structured", RUNBOOK + case,
          {"incident_next": nxt, "incident_notify": notify, "incident_recent": recent})


if __name__ == "__main__":
    main(DOMAIN, LABEL, QUESTIONS, TEXTS)
