#!/usr/bin/env python
"""synth-intents v3: browser_agent world. 一个替用户操作网页的浏览器 agent 眼前的 state, 题目问它要判断的事.

  python datasets/synth-intents-v3/browser_agent.py [--out DIR]

6 种 state, 每种 3 份:
  page_*     页面的可交互元素列表: 任务 + 带编号的元素 (structured)
  form_*     带校验报错的表单: 任务 + 用户的原话 + 各栏的值 (json)
  results_*  搜索结果表, 32-40 行 (table)
  overlay_*  被浮层挡住的页面: 任务 + agent 的规则 + 上一步 + 页面元素 + 浮层 (json)
  history_*  任务加操作历史 (log)
  diff_*     一次操作前后的两次快照 (structured)
每种配 3 道题. 选项跨份通用的 (是非题、几类固定说法) 写成一道题问三份, 三份答案互不相同;
选项取自某一份 state 的 (元素、栏、结果行、步骤、页面部位) 每份各写一道, 问法一字不差.
答案全部写明 (stated). 键的意思与写作规则见 schema.py.
"""

import random

from schema import YES_NO, Question, Text, main

DOMAIN = "browser_agent"
LABEL = "Browser agent state"

QUESTIONS: list[Question] = []
TEXTS: list[Text] = []
SHARED: dict[str, Question] = {}


def shuffled(qid: str, options: list[str]) -> list[str]:
    """没有自然顺序的选项按题 id 固定打乱, 让答案在列表里的位置均匀."""
    out = list(options)
    random.Random(qid).shuffle(out)
    return out


def shared(qid: str, ask: list[str], options: list[str]) -> None:
    """三份共用的一道题 (选项跨份通用). 是非题不打乱; 其余按题 id 打乱. texts 由 state() 补上."""
    opts = list(options) if options == YES_NO else shuffled(qid, options)
    SHARED[qid] = Question(id=qid, ask=ask, options=opts)
    QUESTIONS.append(SHARED[qid])


def state(suffix: str, style: str, text, answers: dict[str, str], own: list[tuple] = ()) -> None:
    """一份 state. answers = {共用题 id: 答案原文}; own 每项是 (题名, 问法列表, 选项, 答案原文[, judged]),
    生成只问这份的题 <suffix>_<题名>. 答案写成选项原文, 写错会在 .index 处报错."""
    tid = f"{DOMAIN}_{suffix}"
    stated = {}
    for qid, answer in answers.items():
        SHARED[qid].texts.append(tid)
        stated[qid] = SHARED[qid].options.index(answer)
    for slug, ask, options, answer, *judged in own:
        qid = f"{suffix}_{slug}"
        QUESTIONS.append(Question(id=qid, ask=list(ask), options=list(options), texts=[tid],
                                  judged=judged[0] if judged else True))
        stated[qid] = options.index(answer)
    TEXTS.append(Text(id=tid, style=style, text=text, stated=stated))


# ---------------------------------------------------------------- A. 页面元素快照 (structured)
# 元素行写成 [12] button "Save" (disabled); 选项只写 [12] button "Save", 不带状态注记, 要回 state 里查.

def page(task: str, title: str, url: str, items: list[tuple]) -> tuple[str, list[str]]:
    """items 每项是 (role, name[, 状态]), 编号从 1 起. 返回 (state 文本, 每个元素的选项串)."""
    lines, opts = [], []
    for i, (role, name, *note) in enumerate(items, 1):
        opts.append(f'[{i}] {role} "{name}"')
        lines.append(opts[-1] + (f" ({note[0]})" if note else ""))
    text = f"Task: {task}\nPage title: {title}\nURL: {url}\nElements on the page:\n" + "\n".join(lines)
    return text, opts


def pick(opts: list[str], name: str, nth: int = 0) -> str:
    """名字恰为 name 的第 nth 个 (0 起) 元素的选项串; 同名元素 (如三段里各有一个的复选框) 用 nth 区分."""
    return [o for o in opts if o.endswith(f' "{name}"')][nth]


def slots(day: str, full: set[str]) -> list[tuple]:
    """一天 8 个一小时的送货时段; full 里的开始时刻已满."""
    price = {"08": "£4.50", "09": "£4.50", "10": "£3.50", "11": "£3.50", "12": "£2.50", "14": "£2.50",
             "16": "£3.50", "18": "£5.00"}
    out = []
    for h, p in price.items():
        name = f"{day} {h}:00–{int(h) + 1:02d}:00, {p}"
        out.append(("button", name, "full, disabled") if h in full else ("button", name))
    return out


PAGE_NEXT = ["Which element should the agent click next?", "What should the agent's next click be?"]
PAGE_UNUSABLE = ["Which of these elements can the agent not use right now?",
                 "Which of these elements is out of action at the moment?"]
shared("page_one_click", ["Can the agent finish the task with a single click from here?",
                          "Is one more click enough to complete the task?"], YES_NO)

SLOTS_TEXT, SLOTS = page(
    "Book the earliest delivery slot that is still free on Thursday, then go on to payment.",
    "Choose a delivery slot", "freshcart.example/checkout/delivery",
    [
        ("link", "Skip to main content"), ("link", "FreshCart home"), ("searchbox", "Search products"),
        ("button", "Search"), ("link", "Offers"), ("link", "Favourites"), ("link", "My account"),
        ("link", "Basket, 23 items"), ("link", "Back to basket"),
        ("heading", "Choose a delivery slot"), ("text", "Each slot lasts one hour. Prices include packing."),
        ("radio", "Home delivery", "selected"), ("radio", "Click and collect"),
        ("text", "Delivering to 14 Alder Row"), ("link", "Change address"),
        ("checkbox", "Show only slots under £3"),
        ("heading", "Wednesday 1 October"), *slots("Wed", {"09", "11", "16"}),
        ("heading", "Thursday 2 October"), *slots("Thu", {"08", "09", "10", "14"}),
        ("heading", "Friday 3 October"), *slots("Fri", {"12", "18"}),
        ("button", "Show later days"),
        ("heading", "Delivery pass early slots"), ("text", "For delivery pass members only"),
        ("button", "Thu 07:00–08:00, £0.00", "members only, disabled"), ("link", "Join delivery pass"),
        ("heading", "Order summary"), ("text", "Subtotal £61.20"), ("text", "Delivery: no slot chosen yet"),
        ("text", "Minimum spend of £40 reached"), ("button", "Continue to payment", "disabled"),
        ("link", "Edit basket"),
        *[("link", s) for s in ["Help centre", "Delivery information", "Contact us", "Store finder", "Gift cards",
                                "Product recalls", "Our suppliers", "Sustainability", "Careers", "Press office",
                                "Terms and conditions", "Privacy notice", "Cookie settings", "Accessibility",
                                "Modern slavery statement", "Site map", "Get the app", "Price promise",
                                "Student discount", "Store opening hours"]],
        ("button", "Back to top"),
    ])
state("page_slots", "structured", SLOTS_TEXT, {"page_one_click": "no"}, [
    ("next", PAGE_NEXT, SLOTS, pick(SLOTS, "Thu 11:00–12:00, £3.50"), False),
    ("unusable", PAGE_UNUSABLE,
     [pick(SLOTS, "Search products"), pick(SLOTS, "Click and collect"), pick(SLOTS, "Wed 12:00–13:00, £2.50"),
      pick(SLOTS, "Thu 16:00–17:00, £3.50"), pick(SLOTS, "Continue to payment"), pick(SLOTS, "Delivery information")],
     pick(SLOTS, "Continue to payment")),
])

PREFS_TEXT, PREFS = page(
    "Stop this site from sending promotional text messages. Leave every other notification setting as it is.",
    "Communication preferences", "roamly.example/account/communication",
    [
        ("link", "Skip to main content"), ("link", "Roamly home"), ("link", "Stays"), ("link", "Flights"),
        ("link", "Car hire"), ("link", "Things to do"), ("link", "Deals"), ("searchbox", "Where are you going?"),
        ("button", "Search"), ("button", "Currency: GBP"), ("button", "Language: English"), ("link", "Trips"),
        ("link", "Help"), ("button", "Account: Lena"),
        ("text", "Your stay in Kelstow starts in twelve days"), ("link", "View trip"),
        ("link", "Add a car to your trip"),
        ("heading", "Your account"), ("link", "Profile"), ("link", "Sign-in and security"),
        ("link", "Saved travellers"), ("link", "Payment methods"), ("link", "Addresses"), ("link", "Wish lists"),
        ("link", "Travel documents"), ("link", "Communication preferences", "current page"),
        ("link", "Privacy and data"), ("link", "Linked apps"), ("link", "Rewards"), ("link", "Reviews you wrote"),
        ("button", "Sign out"),
        ("heading", "Communication preferences"),
        ("text", "Tick a box to get that kind of message. Nothing changes until you press Save preferences."),
        ("heading", "Email"),
        ("checkbox", "Booking confirmations", "ticked, disabled"), ("checkbox", "Trip reminders", "ticked"),
        ("checkbox", "Price alerts for saved searches", "ticked"), ("checkbox", "Offers and promotions", "ticked"),
        ("checkbox", "Travel inspiration", "not ticked"), ("checkbox", "Surveys about your stay", "not ticked"),
        ("heading", "Text messages"), ("text", "Sent to the mobile number on your profile"),
        ("checkbox", "Booking confirmations", "ticked"), ("checkbox", "Check-in reminders", "ticked"),
        ("checkbox", "Flight delay updates", "ticked"), ("checkbox", "Offers and promotions", "ticked"),
        ("checkbox", "Surveys about your stay", "not ticked"),
        ("heading", "App notifications"),
        ("checkbox", "Booking confirmations", "ticked"), ("checkbox", "Trip reminders", "ticked"),
        ("checkbox", "Price alerts for saved searches", "not ticked"),
        ("checkbox", "Offers and promotions", "ticked"), ("checkbox", "Messages from hosts", "ticked"),
        ("heading", "Post"), ("checkbox", "Printed brochures", "not ticked"),
        ("heading", "How often we send offers"), ("radio", "As they come", "selected"),
        ("radio", "Weekly summary"), ("radio", "Monthly summary"),
        ("button", "Save preferences"), ("button", "Cancel"), ("link", "Unsubscribe from all marketing"),
        ("text", "You will still get messages we must send, such as booking confirmations."),
        ("heading", "Need help?"), ("link", "How we use your contact details"), ("button", "Chat with us"),
        *[("link", s) for s in [
            "About Roamly", "How Roamly works", "Careers", "Press centre", "Sustainability",
            "Accessibility statement", "Help centre", "Contact customer service", "Cancellation options",
            "Safety resource centre", "Trust and safety", "Partner help", "List your property",
            "Become an affiliate", "Travel articles", "Seasonal deals", "Last-minute stays", "Countries", "Regions",
            "Cities", "Districts", "Airports", "Hotels", "Places of interest", "Holiday homes", "Apartments",
            "Resorts", "Villas", "Hostels", "Guest houses", "Cabins", "Unique places to stay", "Traveller community",
            "Privacy notice", "Terms of service", "Cookie settings", "Modern slavery statement", "Site map",
            "Content guidelines", "Report a problem"]],
        ("button", "Back to top"),
    ])
state("page_prefs", "structured", PREFS_TEXT, {"page_one_click": "no"}, [
    ("next", PAGE_NEXT, PREFS, pick(PREFS, "Offers and promotions", 1), False),
    ("unusable", PAGE_UNUSABLE,
     [pick(PREFS, "Payment methods"), pick(PREFS, "Booking confirmations", 0), pick(PREFS, "Offers and promotions", 0),
      pick(PREFS, "Offers and promotions", 1), pick(PREFS, "Save preferences"),
      pick(PREFS, "Unsubscribe from all marketing")],
     pick(PREFS, "Booking confirmations", 0)),
])


def order_block(line: str, links: list[tuple]) -> list[tuple]:
    return [("text", line), ("link", "View order details"), *links]


ORDERS_TEXT, ORDERS = page(
    "Download the invoice for the order that arrived on 9 September.",
    "Your orders", "brightlane.example/account/orders",
    [
        ("link", "Skip to main content"), ("link", "Brightlane home"), ("button", "Deliver to Sam, 5 Wren Close"),
        ("searchbox", "Search Brightlane"), ("button", "Search"), ("link", "Account and lists"),
        ("link", "Returns and orders"), ("link", "Basket, 0 items"),
        *[("link", s) for s in ["All departments", "Today's deals", "Books", "Garden", "Home and kitchen", "Toys",
                                "Electronics", "Fashion", "Beauty", "Sports", "Pets", "Grocery", "Gift ideas",
                                "Customer service"]],
        ("link", "Your account"), ("heading", "Your orders"), ("link", "Orders", "current tab"),
        ("link", "Buy again"), ("link", "Not yet dispatched"), ("link", "Cancelled orders"),
        ("searchbox", "Search all orders"), ("button", "Search orders"),
        ("combobox", "Orders placed in", "past three months"), ("text", "Six orders placed in the past three months"),
        *order_block("Placed 18 September · Total £23.40 · Not yet dispatched", [
            ("link", "Download invoice (PDF)", "disabled, ready after dispatch"), ("button", "Track package"),
            ("button", "Cancel items")]),
        *order_block("Placed 12 September · Total £56.10 · Delivered 16 September", [
            ("link", "Download invoice (PDF)"), ("button", "Buy it again"), ("link", "Return or replace items"),
            ("link", "Write a product review")]),
        *order_block("Placed 9 September · Total £12.99 · Delivered 13 September", [
            ("link", "Download invoice (PDF)"), ("button", "Buy it again"), ("link", "Return or replace items"),
            ("link", "Write a product review")]),
        *order_block("Placed 4 September · Total £41.75 · Delivered 9 September", [
            ("link", "Download invoice (PDF)"), ("button", "Buy it again"), ("link", "Return or replace items"),
            ("link", "Write a product review")]),
        *order_block("Placed 29 August · Total £8.50 · Delivered 2 September", [
            ("link", "Download invoice (PDF)"), ("button", "Buy it again"), ("link", "Return or replace items"),
            ("link", "Write a product review")]),
        *order_block("Placed 3 July · Total £64.00 · Delivered 9 July", [
            ("link", "Download invoice (PDF)"), ("button", "Buy it again"),
            ("link", "Return or replace items", "disabled, return window closed"),
            ("link", "Write a product review")]),
        ("heading", "Recommended for you"), ("link", "Bamboo chopping board"), ("link", "Linen tea towels, set of four"),
        ("link", "Enamel pie dish"), ("link", "Beeswax food wraps"),
        *[("link", s) for s in ["Conditions of use", "Privacy notice", "Cookie settings", "Help", "Accessibility"]],
        ("button", "Back to top"),
    ])
state("page_orders", "structured", ORDERS_TEXT, {"page_one_click": "yes"}, [
    ("next", PAGE_NEXT, ORDERS, pick(ORDERS, "Download invoice (PDF)", 3), False),
    ("unusable", PAGE_UNUSABLE,
     [pick(ORDERS, "Track package"), pick(ORDERS, "Return or replace items", 0),
      pick(ORDERS, "Download invoice (PDF)", 2), pick(ORDERS, "Return or replace items", 4),
      pick(ORDERS, "Enamel pie dish"), pick(ORDERS, "Cookie settings")],
     pick(ORDERS, "Return or replace items", 4)),
])

# ---------------------------------------------------------------- B. 带校验报错的表单 (json)
# 错误横幅不点名是哪一栏, 要从内容推; 用户的原话 (user_note) 与各栏的值比对.

def fields(rows: list[tuple]) -> list[dict]:
    """rows 每项是 (栏名, 当前的值, 是否必填)."""
    return [{"label": label, "value": value, "required": req} for label, value, req in rows]


FORM_ERROR = ["Which field is the error banner about?", "Which field does the error banner refer to?"]
FORM_MISMATCH = ["Which field has a value that goes against the user's note?",
                 "Which field did the agent fill in differently from what the user's note says?"]
shared("form_note_complete", ["Does the user's note give a value for every required field?",
                              "Is everything the required fields ask for somewhere in the user's note?"], YES_NO)


def form(suffix: str, doc: dict, answers: dict, error: str, mismatch: str) -> None:
    labels = [f["label"] for f in doc["fields"]]
    state(suffix, "json", doc, answers, [("error_field", FORM_ERROR, labels, error),
                                         ("mismatch", FORM_MISMATCH, labels, mismatch)])


form("form_race", {
    "task": "Enter the user in the Riverside Half Marathon on Sunday 12 October, using the details in their note.",
    "user_note": "Sign me up please! Mara Quint, born 3 June 1990. I run with Hillside Harriers. Medium T-shirt. "
                 "I'm hoping to finish in about 1 hour 55. My emergency contact is my brother, Tomas Quint.",
    "page": "Riverside Half Marathon entry, step 2 of 3",
    "error_banner": "Please enter the time as hours and minutes, for example 2:05.",
    "fields": fields([
        ("First name", "Mara", True), ("Last name", "Quint", True), ("Date of birth", "3 June 1990", True),
        ("Name of your running club", "Hillside Harriers", False), ("T-shirt size", "Large", True),
        ("Expected finish time", "about 1 hour 55", True), ("Best half marathon time so far", "", False),
        ("Emergency contact name", "Tomas Quint", True), ("Emergency contact phone number", "", True),
        ("Emergency contact relationship", "Brother", True), ("Medical conditions we should know about", "", False),
        ("Charity you are running for", "", False)]),
    "submit_button": "Continue to payment",
}, {"form_note_complete": "no"}, error="Expected finish time", mismatch="T-shirt size")

form("form_bike", {
    "task": "Book a bike service using the details in the user's note.",
    "user_note": "Can you book my bike in for a full service at the Mill Lane workshop? It's a green Dalby Tourer "
                 "with a 54 cm frame, last serviced on Saturday 7 June. Saturday 4 October suits me best, and I'd "
                 "drop it off around 09:00. I'm Kofi Mensah, postcode BS3 4QT.",
    "page": "Book a service",
    "error_banner": "Our workshops are closed at weekends. Please choose a day from Monday to Friday.",
    "fields": fields([
        ("Full name", "Kofi Mensah", True), ("Postcode", "BS3 4QT", True), ("Workshop", "Canal Street", True),
        ("Service type", "Full service", True), ("Bike brand", "Dalby", True), ("Bike model", "Tourer", True),
        ("Frame size", "54 cm", False), ("Frame colour", "Green", False),
        ("Date of the bike's last service", "Sat 7 June", False), ("Drop-off date", "Sat 4 October", True),
        ("Drop-off time", "09:00", True), ("Notes for the mechanic", "", False)]),
    "submit_button": "Confirm booking",
}, {"form_note_complete": "yes"}, error="Drop-off date", mismatch="Workshop")

form("form_flat", {
    "task": "Ask for a viewing of the flat at 22 Quarry Lane, using the details in the user's note.",
    "user_note": "Please request a viewing for me. I'm Ines Paredes. Two of us would move in, no children, and we "
                 "have one cat. We'd like to move in on 1 November. I have a permanent job. Any evening after 18:00 "
                 "next week works, ideally Tuesday.",
    "page": "22 Quarry Lane: request a viewing",
    "error_banner": "Please use digits only in this box, for example 2.",
    "fields": fields([
        ("First name", "Ines", True), ("Surname", "Paredes", True), ("Adults moving in", "two", True),
        ("Children moving in", "0", True), ("Pets", "One cat", True), ("Move-in date", "1 November", True),
        ("Employment status", "", True), ("Monthly budget", "", False), ("Preferred viewing day", "Tuesday", True),
        ("Preferred viewing time", "17:30", True), ("Where you heard about the flat", "", False),
        ("Message to the landlord", "", False)]),
    "submit_button": "Send request",
}, {"form_note_complete": "yes"}, error="Adults moving in", mismatch="Preferred viewing time")


# ---------------------------------------------------------------- C. 搜索结果表 (table)
# 行由代码按固定种子拼: 每个填充行至少违反任务的一条要求, 答案行由我放进去, 再断言恰有它一行全满足.

def only_match(rows: list[tuple], ok, answer: str) -> None:
    hits = [r[0] for r in rows if ok(r)]
    if hits != [answer]:
        raise ValueError(f"满足全部要求的应只有 {answer!r}, 实际是 {hits}")


def distinct(rng: random.Random, used: set, lo: int, hi: int) -> int:
    while True:
        x = rng.randint(lo, hi)
        if x not in used:
            used.add(x)
            return x


RESULTS_MATCH = ["Which result meets every requirement in the task?",
                 "Which listed item fits all of the task's conditions?"]
shared("results_order", ["In what order are the results listed?", "How has the site sorted these results?"],
       ["Lowest price first", "Highest rating first", "Newest listing first", "Soonest arrival first",
        "Alphabetical by name"])
shared("results_more", ["Are there more results than the ones shown on this page?",
                        "Does the search have results beyond the ones listed here?"], YES_NO)


def headphones() -> tuple[str, list[str], str]:
    rng, used = random.Random("results_headphones"), {54}
    kinds = ["Wireless Over-Ear Headphones", "Wireless On-Ear Headphones", "Wired Over-Ear Headphones",
             "Wired On-Ear Headphones", "Headphone Case"]
    brands = ["Kestrel", "Solace", "Brio", "Nimbus", "Arden", "Quill", "Tamber", "Vesta", "Orla", "Halden", "Marlo",
              "Pike"]
    answer = "Orla Wireless On-Ear Headphones"
    names = [f"{b} {k}" for b in brands for k in kinds if f"{b} {k}" != answer]
    rng.shuffle(names)
    days = ["Tue 30 Sep", "Wed 1 Oct", "Thu 2 Oct", "Fri 3 Oct", "Sat 4 Oct", "Mon 6 Oct"]
    rows = []
    for name in names[:35]:
        if "Case" in name:
            p, r = distinct(rng, used, 9, 29), rng.randint(36, 49)
        elif "Wireless" in name and rng.random() < 0.5:  # 太贵
            p, r = distinct(rng, used, 60, 189), rng.randint(40, 49)
        elif "Wireless" in name:  # 价格合适, 评分不够
            p, r = distinct(rng, used, 25, 59), rng.randint(38, 44)
        else:  # 有线
            p, r = distinct(rng, used, 15, 120), rng.randint(38, 49)
        rows.append((name, p, r / 10, rng.choice(days)))
    rows.append((answer, 54, 4.6, "Thu 2 Oct"))
    rows.sort(key=lambda r: r[1])
    only_match(rows, lambda r: "Wireless" in r[0] and "Headphones" in r[0] and r[1] < 60 and r[2] >= 4.5, answer)
    text = ("Site: soundhaus.example\n"
            "Task: Find wireless headphones that cost less than £60 and have a rating of 4.5 or higher.\n"
            "Search: wireless headphones\n"
            f"Showing 1–{len(rows)} of {len(rows)} results\n"
            "Product | Price | Rating | Arrives\n"
            + "\n".join(f"{n} | £{p}.99 | {r:.1f} | {d}" for n, p, r, d in rows))
    return text, [r[0] for r in rows], answer


def lamps() -> tuple[str, list[str], str]:
    rng, used = random.Random("results_lamps"), {27}
    kinds = ["LED Desk Lamp", "Clamp Desk Lamp", "Folding Desk Lamp", "Architect Desk Lamp"]
    brands = ["Lumo", "Beacon", "Glowe", "Arc", "Tilde", "Nook", "Halo", "Ember", "Pivot", "Crane", "Orbit", "Fable"]
    sellers = ["LampLand", "Brightco (marketplace seller)", "HomeGlow (marketplace seller)",
               "DeskDirect (marketplace seller)"]
    lights = ["Warm white only", "Warm and cool white", "Cool white only", "Daylight white only"]
    answer = "Tilde Clamp Desk Lamp"
    names = [f"{b} {k}" for b in brands for k in kinds if f"{b} {k}" != answer]
    rng.shuffle(names)
    rows = []
    for i, name in enumerate(names[:39]):
        p, seller, light = distinct(rng, used, 12, 89), rng.choice(sellers), rng.choice(lights)
        fail = i % 3
        if fail == 0 and "Warm" in light:  # 没有暖光
            light = rng.choice(lights[2:])
        elif fail == 1 and p < 35:  # 太贵
            p = distinct(rng, used, 35, 89)
        elif fail == 2 and seller == "LampLand":  # 第三方卖家
            seller = rng.choice(sellers[1:])
        rows.append((name, p, rng.randint(34, 50) / 10, seller, light))
    rows.append((answer, 27, 4.8, "LampLand", "Warm and cool white"))
    rows.sort(key=lambda r: -r[2])
    only_match(rows, lambda r: "Warm" in r[4] and r[1] < 35 and r[3] == "LampLand", answer)
    text = ("Site: lampland.example\n"
            "Task: Find a desk lamp that has a warm light setting, costs less than £35, and is sold by LampLand "
            "itself rather than by a marketplace seller.\n"
            "Search: desk lamp\n"
            "Showing 1–40 of 212 results\n"
            "Product | Price | Rating | Sold by | Light\n"
            + "\n".join(f"{n} | £{p}.99 | {r:.1f} | {s} | {li}" for n, p, r, s, li in rows))
    return text, [r[0] for r in rows], answer


def bikes() -> tuple[str, list[str], str]:
    rng, used = random.Random("results_bikes"), {340}
    models = {"road": ["Aero", "Sprint", "Tempo"], "hybrid": ["Commuter", "Metro"], "mountain": ["Trail", "Summit"],
              "folding": ["Compact"]}
    brands = ["Vantor", "Hollis", "Crestline", "Ridgeway", "Moss", "Kinley", "Talbot", "Fenwick"]
    answer = "Hollis Tempo road bike"
    names = [f"{b} {m} {k} bike" for b in brands for k, ms in models.items() for m in ms]
    names = [n for n in names if n != answer]
    rng.shuffle(names)
    frames = ["50 cm", "52 cm", "54 cm", "56 cm", "58 cm", "60 cm"]
    rows = []
    for i, name in enumerate(names[:31]):
        frame = "One size" if "folding" in name else rng.choice(frames)
        handover = rng.choice(["Collection only", "Can post"])
        if "road" in name and frame == "56 cm" and handover == "Can post":
            if i % 2:
                frame = rng.choice(["54 cm", "58 cm"])
            else:
                handover = "Collection only"
        rows.append([name, frame, distinct(rng, used, 90, 900), handover])
    rows.insert(26, [answer, "56 cm", 340, "Can post"])
    minutes = [4, 12, 25, 41, 58, 75, 130, 190, 260, 340, 430, 530, 650, 790, 960, 1150, 1400, 1700, 2100, 2600,
               3200, 3900, 4700, 5600, 6600, 7800, 9100, 10500, 12000, 14000, 16500, 19000]  # 挂出多久, 从新到旧
    only_match(rows, lambda r: " road " in r[0] and r[1] == "56 cm" and r[3] == "Can post", answer)

    def ago(m: int) -> str:
        if m < 60:
            return f"{m} min ago"
        if m < 1440:
            return "1 hour ago" if m < 120 else f"{m // 60} hours ago"
        return "1 day ago" if m < 2880 else f"{m // 1440} days ago"

    text = ("Site: cyclemart.example\n"
            "Task: Find a used road bike with a 56 cm frame that the seller can post, because the user cannot "
            "collect.\n"
            "Search: used bike\n"
            "Showing 1–32 of 57 results\n"
            "Bike | Frame | Price | Handover | Listed\n"
            + "\n".join(f"{n} | {f} | £{p} | {h} | {ago(m)}" for (n, f, p, h), m in zip(rows, minutes)))
    return text, [r[0] for r in rows], answer


for suffix, build, order, more in [("results_headphones", headphones, "Lowest price first", "no"),
                                   ("results_lamps", lamps, "Highest rating first", "yes"),
                                   ("results_bikes", bikes, "Newest listing first", "yes")]:
    text, names, answer = build()
    state(suffix, "table", text, {"results_order": order, "results_more": more},
          [("match", RESULTS_MATCH, names, answer, False)])

# ---------------------------------------------------------------- D. 被浮层挡住的页面 (json)
# agent 的规则三份一字不差, 每条写明管哪一类浮层. 浮层的种类要从 top_layer 的内容推, state 里不写种类名.

AGENT_RULES = [
    "Cookie banners: pick the choice that keeps only the cookies the site needs to work.",
    "Sign-up or discount pop-ups: close them without typing anything into them.",
    "Help chat windows: make them smaller so they stop covering the page; never send a message.",
    "Sign-in prompts: close them; only sign in if the task says to.",
]
OVERLAY_DISMISS = ["Following the agent's rules, which element should the agent click first?",
                   "Which element does the agent's rulebook say to click before anything else?"]
MISSED = "The click did not reach the button: the layer on top of the page got it instead."
shared("overlay_cover", ["What is covering the page?", "What has opened on top of the page?"],
       ["A cookie choices banner", "A discount sign-up pop-up", "An open help chat window",
        "A prompt asking to sign in", "A country and region picker"])
shared("overlay_took_effect", ["Did the agent's last action take effect on the page?",
                               "Did the last thing the agent did actually go through?"], YES_NO)


def overlay(suffix: str, doc: dict, answers: dict, options: list[str], answer: str) -> None:
    listed = doc["page"] + doc["top_layer"]["elements"]
    for o in options:
        if not any(line.startswith(o) for line in listed):
            raise ValueError(f"{suffix}: 选项 {o!r} 不在 state 里")
    state(suffix, "json", doc, answers, [("dismiss", OVERLAY_DISMISS, options, answer)])


overlay("overlay_cookies", {
    "task": "Add the Fernhill 30 m garden hose to the basket.",
    "agent_rules": AGENT_RULES,
    "url": "greenshed.example/watering/fernhill-hose-30m",
    "last_action": 'click [16] button "Add to basket"',
    "last_action_result": MISSED,
    "basket_badge": "0 items",
    "page": ['[5] link "Greenshed home"', '[6] searchbox "Search the shop"', '[7] link "Sign in"', '[8] link "Help"',
             '[9] link "Basket"', '[10] link "Watering"', '[11] link "Hoses and reels"',
             '[12] heading "Fernhill garden hose, 30 m"', '[13] text "£24.99"', '[14] combobox "Length" (30 m)',
             '[15] spinbutton "Quantity" (1)', '[16] button "Add to basket"', '[17] link "Delivery and returns"',
             '[18] link "Reviews"', '[19] link "Cookie settings"', '[20] link "Privacy notice"'],
    "top_layer": {
        "text": "We value your privacy. We and our partners use cookies to measure visits and to show ads that suit "
                "you. You can change your mind at any time.",
        "elements": ['[1] button "Accept all"', '[2] button "Necessary only"', '[3] button "Manage choices"',
                     '[4] link "Read our cookie policy"'],
    },
}, {"overlay_cover": "A cookie choices banner", "overlay_took_effect": "no"},
    ['[1] button "Accept all"', '[2] button "Necessary only"', '[3] button "Manage choices"',
     '[4] link "Read our cookie policy"', '[7] link "Sign in"', '[16] button "Add to basket"',
     '[19] link "Cookie settings"'],
    '[2] button "Necessary only"')

overlay("overlay_offer", {
    "task": "Add the Oakmoor rechargeable AA batteries, pack of 8, to the basket.",
    "agent_rules": AGENT_RULES,
    "url": "voltbox.example/batteries/oakmoor-aa-8",
    "last_action": 'click [12] button "Add to basket"',
    "last_action_result": "The page changed, then a new layer opened on top of it.",
    "basket_badge": "1 item",
    "page": ['[1] link "Voltbox home"', '[2] searchbox "Search"', '[3] link "Sign in"', '[4] link "Help"',
             '[5] link "Basket"', '[6] link "Batteries"', '[7] heading "Oakmoor rechargeable AA batteries, pack of 8"',
             '[8] text "£12.49"', '[9] combobox "Pack size" (8)', '[10] spinbutton "Quantity" (1)',
             '[11] text "In stock"', '[12] button "Add to basket"', '[13] text "Added to your basket"',
             '[14] link "Go to basket"', '[15] link "Cookie settings"', '[16] link "Privacy notice"'],
    "top_layer": {
        "text": "Wait! Take 15% off your first order. Join our mailing list and we'll send your code straight away.",
        "elements": ['[40] textbox "Your email address"', '[41] button "Send my code"', '[42] link "Maybe later"'],
    },
}, {"overlay_cover": "A discount sign-up pop-up", "overlay_took_effect": "yes"},
    ['[3] link "Sign in"', '[12] button "Add to basket"', '[14] link "Go to basket"', '[15] link "Cookie settings"',
     '[40] textbox "Your email address"', '[41] button "Send my code"', '[42] link "Maybe later"'],
    '[42] link "Maybe later"')

overlay("overlay_chat", {
    "task": "Change the delivery address for the user's parcel that is due tomorrow.",
    "agent_rules": AGENT_RULES,
    "url": "parcelpath.example/track",
    "last_action": 'click [14] button "Change delivery address"',
    "last_action_result": MISSED,
    "page": ['[1] link "ParcelPath home"', '[2] link "Track a parcel"', '[3] link "Send a parcel"', '[4] link "Help"',
             '[5] link "Sign in"', '[10] heading "Your parcel is due tomorrow between 08:00 and 12:00"',
             '[11] text "Going to: 31 Tanner Street"', '[12] button "Change delivery day"',
             '[13] button "Leave with a neighbour"', '[14] button "Change delivery address"',
             '[15] link "Cookie settings"'],
    "top_layer": {
        "text": "Hi, I'm Pip, the ParcelPath help assistant. What can I help you with today?",
        "elements": ['[30] textbox "Type your message"', '[31] button "Minimise"', '[32] button "Send"',
                     '[33] button "Talk to a person"'],
    },
}, {"overlay_cover": "An open help chat window", "overlay_took_effect": "no"},
    ['[5] link "Sign in"', '[12] button "Change delivery day"', '[14] button "Change delivery address"',
     '[30] textbox "Type your message"', '[31] button "Minimise"', '[32] button "Send"',
     '[33] button "Talk to a person"'],
    '[31] button "Minimise"')


# ---------------------------------------------------------------- E. 任务加操作历史 (log)
# 每行: 时刻  动作  →  页面的反应. 选项是 "时刻 动作" (同一个动作可能出现两次, 时刻让选项唯一), 保持先后顺序.

HISTORY_WRONG = ["Which step was the first one that went against the task?", "Where did the agent first go wrong?"]
NO_MISTAKE = "Nothing so far was a mistake"
shared("history_done", ["Is the task finished?", "Has the agent completed what the task asks?"], YES_NO)
shared("history_next", ["What should the agent do next?", "What is the right next move for the agent?"],
       ["Report that the task is done", "Go back and fix an earlier step", "Keep going with the steps left",
        "Ask the user for missing details", "Wait for the page to finish loading"])


def history(suffix: str, task: str, steps: list[tuple], now: str, answers: dict, wrong: str) -> None:
    """steps 每项是 (时刻, 动作[, 页面的反应]); wrong 是第一步出错的动作的时刻, 或 None."""
    lines = [f"Task: {task}", "Steps so far:"]
    options = []
    for when, action, *result in steps:
        options.append(f"{when} {action}")
        lines.append(f"{when}  {action}" + (f"  →  {result[0]}" if result else ""))
    lines.append(f"Now showing: {now}")
    options.append(NO_MISTAKE)
    answer = NO_MISTAKE if wrong is None else next(o for o in options if o.startswith(wrong))
    state(suffix, "log", "\n".join(lines), answers, [("wrong_step", HISTORY_WRONG, options, answer)])


history("history_table",
        "Book a table for four at Nonna Rosa's Mill Road branch on Friday 3 October at 19:30.",
        [("14:02:10", "open nonnarosa.example", "home page"),
         ("14:02:15", 'click "Book a table"', "booking form"),
         ("14:02:18", 'choose branch "Harbour Street"'),
         ("14:02:24", 'choose party size "4 people"'),
         ("14:02:30", 'choose date "Friday 3 October"'),
         ("14:02:33", 'choose time "19:30"'),
         ("14:02:40", 'click "Check availability"', '"19:30 is free at Harbour Street"')],
        'booking summary with a "Confirm booking" button',
        {"history_done": "no", "history_next": "Go back and fix an earlier step"}, wrong="14:02:18")

history("history_ink",
        "Order two black ink cartridges for the user's Inkwell 300 printer and have them sent to the saved home "
        "address.",
        [("09:40:02", "open inkstop.example", "home page"),
         ("09:40:07", 'search "Inkwell 300 black"', "12 results"),
         ("09:40:12", 'open "Inkwell 300 black cartridge, single"', "product page"),
         ("09:40:16", "set quantity to 2"),
         ("09:40:18", 'click "Add to basket"', "basket shows 2 items"),
         ("09:40:25", 'click "Checkout"', "delivery step"),
         ("09:40:31", 'choose saved address "Home, 31 Tanner Street"'),
         ("09:40:34", 'choose delivery "Standard, 3 to 5 days"', "order review")],
        'order review with a "Place order" button',
        {"history_done": "no", "history_next": "Keep going with the steps left"}, wrong=None)

history("history_trial",
        "Cancel the user's free trial of StreamNest before it turns into a paid plan.",
        [("19:15:03", "open streamnest.example/account", "account page"),
         ("19:15:09", 'click "Membership"', '"Plan: free trial, ends 30 September"'),
         ("19:15:14", 'click "Cancel free trial"', '"Before you go, why are you leaving?"'),
         ("19:15:20", 'click "Skip"', '"Stay for half price for three months?"'),
         ("19:15:24", 'click "Accept offer"', '"Offer applied: paid plan at half price from 30 September"'),
         ("19:15:35", 'click "Membership"', '"Plan: paid, half price from 30 September"'),
         ("19:15:40", 'click "Cancel plan"', '"Before you go, why are you leaving?"'),
         ("19:15:44", 'click "Skip"', '"Stay for half price for three months?"'),
         ("19:15:49", 'click "No thanks, cancel"', '"Cancelled. You will not be charged, and nothing renews on '
                                                   '30 September."')],
        '"Cancelled. You will not be charged, and nothing renews on 30 September."',
        {"history_done": "yes", "history_next": "Report that the task is done"}, wrong="19:15:24")

# ---------------------------------------------------------------- F. 一次操作前后的两次快照 (structured)
# changed 的选项是两次快照里都有的部位, 每份只挑恰好一项变了的那一组 (变了的其余部位不进选项).

DIFF_CHANGED = ["Which of these shows a different value in the second snapshot?",
                "Which of these changed between the two snapshots?"]
shared("diff_effect", ["What did the action do?", "What was the effect of the click?"],
       ["It updated values on the same page", "It took the browser to another page",
        "It showed an error and saved nothing", "It opened the link in a new browser tab",
        "It made no visible change at all"])
shared("diff_success", ["Did the action do what the agent intended?", "Did the agent get the result it was after?"],
       YES_NO)


def diff(suffix: str, intent: str, action: str, before: tuple[str, list], after: tuple[str, list], answers: dict,
         options: list[str], answer: str) -> None:
    """before / after 是 (时刻, [(部位, 值)]); 两次快照的部位一一对应."""
    def snap(title: str, when: str, rows: list) -> str:
        return f"Snapshot {title} ({when})\n" + "\n".join(f"{k}: {v}" for k, v in rows)

    if [k for k, _ in before[1]] != [k for k, _ in after[1]]:
        raise ValueError(f"{suffix}: 两次快照的部位要一一对应")
    text = f"Intent: {intent}\nAction: {action}\n\n{snap('before', *before)}\n\n{snap('after', *after)}"
    state(suffix, "structured", text, answers,
          [("changed", DIFF_CHANGED, shuffled(f"{suffix}_changed", options), answer)])


diff("diff_promo", "apply the promo code AUTUMN15 to the basket", 'click [27] button "Apply"',
     ("10:41:07", [("URL", "potterly.example/basket"), ("Title", "Your basket"), ("Signed in as", "Rae"),
                   ("Items in basket", "3"), ("Promo code box", '"AUTUMN15"'),
                   ("Message under the promo code box", "none"), ("Subtotal", "£80.00"), ("Discount", "none"),
                   ("Delivery charge", "£4.95"), ("Total to pay", "£84.95")]),
     ("10:41:09", [("URL", "potterly.example/basket"), ("Title", "Your basket"), ("Signed in as", "Rae"),
                   ("Items in basket", "3"), ("Promo code box", "empty"),
                   ("Message under the promo code box", '"AUTUMN15 applied: 15% off"'), ("Subtotal", "£80.00"),
                   ("Discount", "-£12.00"), ("Delivery charge", "£4.95"), ("Total to pay", "£72.95")]),
     {"diff_effect": "It updated values on the same page", "diff_success": "yes"},
     ["The URL", "The signed-in name", "The number of items in the basket", "The subtotal", "The delivery charge",
      "The total to pay"],
     "The total to pay")

diff("diff_sizes", "open the size guide for the Harlow rain jacket", 'click [33] link "Size guide"',
     ("16:05:12", [("URL", "wearwell.example/jackets/harlow-rain-jacket"), ("Title", "Harlow rain jacket"),
                   ("Signed in", "no"), ("Basket", "0 items"), ("Currency", "GBP"), ("Language", "English"),
                   ("Search box", '"rain jacket"'), ("Main heading", '"Harlow rain jacket"'), ("Size chosen", "M")]),
     ("16:05:13", [("URL", "wearwell.example/help/size-guide/jackets"), ("Title", "Jacket size guide"),
                   ("Signed in", "no"), ("Basket", "0 items"), ("Currency", "GBP"), ("Language", "English"),
                   ("Search box", '"rain jacket"'), ("Main heading", '"Jacket sizes: chest and sleeve"'),
                   ("Size chosen", "not shown")]),
     {"diff_effect": "It took the browser to another page", "diff_success": "yes"},
     ["The URL", "The signed-in state", "The number of items in the basket", "The currency", "The language",
      "The search box text"],
     "The URL")

diff("diff_address", "save the new delivery address", 'click [19] button "Save address"',
     ("11:20:40", [("URL", "tidyhome.example/account/addresses/new"), ("Title", "Add a new address"),
                   ("Name box", '"Jo Arden"'), ("Street box", '"8 Kiln Yard"'), ("Message under the street box", "none"),
                   ("Town box", '"Millbridge"'), ("Postcode box", '"MB9 9ZZ"'),
                   ("Message under the postcode box", "none"), ("Saved addresses", "1"),
                   ("Save address button", "enabled")]),
     ("11:20:41", [("URL", "tidyhome.example/account/addresses/new"), ("Title", "Add a new address"),
                   ("Name box", '"Jo Arden"'), ("Street box", '"8 Kiln Yard"'), ("Message under the street box", "none"),
                   ("Town box", '"Millbridge"'), ("Postcode box", '"MB9 9ZZ"'),
                   ("Message under the postcode box", '"We can\'t find this postcode. Check it and try again."'),
                   ("Saved addresses", "1"), ("Save address button", "enabled")]),
     {"diff_effect": "It showed an error and saved nothing", "diff_success": "no"},
     ["The URL", "The message under the street box", "The town box", "The postcode box",
      "The message under the postcode box", "The number of saved addresses"],
     "The message under the postcode box")

if __name__ == "__main__":
    main(DOMAIN, LABEL, QUESTIONS, TEXTS)
