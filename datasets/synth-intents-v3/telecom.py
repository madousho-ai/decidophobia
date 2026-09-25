#!/usr/bin/env python
"""synth-intents v3: telecom 领域 (手机 / 宽带 / 电视 / 座机运营商的客服工单). 从 template.py 复制.

  python datasets/synth-intents-v3/telecom.py [--out DIR]

11 道问整份工单的题 (其中 issue 20 选 1、next_step 17 选 1、feeling 10 选 1 是选项多的细分题) + 3 道指着 JSON 字段的题.
36 份文本: 手写 16 份 (short / structured / rant / json 各 4), 另有 20 份 short 取自 v2 (见文件末尾 V2_SHORT).
json 的 4 份里 j2、j4 字段互相不一致 (j2 正文平静、追问生气; j4 正文生气、追问平静),
这两份问整份工单的 tone 不列进 stated. 键的意思与写作规则见 schema.py.
文本对某道题一点线索都没给的 (多是账单 / 咨询类问 duration、location), 列进 unknown, 标签是均匀分布;
V2_SHORT 里这类条目带第 4 个元素 (题 id 列表).
"""

from schema import Question, Text, main

DOMAIN = "telecom"
LABEL = "Support ticket"

TONE = ["Calm and matter-of-fact", "Frustrated but still polite", "Angry, blaming, or threatening to leave"]

QUESTIONS = [
    Question(
        id="service",
        ask=["Which service is this about?", "Which of the customer's services does the ticket concern?",
             "What service is the customer writing about?"],
        options=[
            "Their mobile phone line: calls, texts or mobile data",
            "Their home broadband or wifi",
            "Their TV package or set-top box",
            "Their home landline phone",
        ],
    ),
    Question(
        id="team",
        ask=["Which team at the provider should pick this up?", "Which department is this ticket for?",
             "Who at the provider should deal with this?"],
        options=[
            "Billing: bills and charges on the account",
            "Network: signal, coverage, speed or outages",
            "Plans: signing up for, changing or ending a plan or contract",
            "Equipment: phones, routers, TV boxes and SIM cards",
            "Installations: setting up new services and engineer appointments",
        ],
    ),
    Question(
        id="problem",
        ask=["What kind of problem does the customer describe?", "What is going wrong, if anything?",
             "Which option best describes the customer's situation?"],
        options=[
            "The service is not working at all",
            "The service works, but badly: slow, patchy or dropping out",
            "They were charged more than they expected",
            "Something they ordered has not arrived or been set up yet",
            "One feature does not work, such as voicemail or roaming",
            "Nothing is broken; they want to change something or ask a question",
        ],
    ),
    Question(
        id="ask",
        ask=["What is the customer asking the provider to do?", "What outcome is the customer hoping for?",
             "What would the customer like done?"],
        options=[
            "Get the fault fixed",
            "Money back or a reduction on their bill",
            "Change, upgrade or cancel their plan",
            "An explanation or information, with no action needed",
            "An update on something already in progress",
        ],
    ),
    Question(
        id="tone",
        ask=["How does the customer come across?", "What is the tone of the ticket?", "How upset is the customer?"],
        options=TONE,
    ),
    Question(
        id="customer",
        ask=["Who is writing?", "What kind of customer is this?", "Who most likely sent this ticket?"],
        options=[
            "A customer with a personal account",
            "A business customer writing about work lines or an office connection",
            "Someone writing on behalf of the account holder, such as a relative",
            "Someone who is not a customer yet",
        ],
    ),
    Question(
        id="duration",
        ask=["How long has the problem been going on?", "Since when has this been happening?",
             "How long has the customer had this issue?"],
        options=["It started today", "For a few days", "For weeks or longer"],
    ),
    Question(
        id="location",
        ask=["Where is the customer when the problem happens?", "Where does the issue occur?",
             "Where is the customer using the service when it goes wrong?"],
        options=["At home", "Abroad", "Out and about in their own country", "At their workplace"],
    ),
    # ---- 选项多的细分题: issue 细分 service × problem, next_step 细分 ask, feeling 细分 tone
    Question(
        id="issue",
        ask=["What exactly is the ticket about?", "Which of these is the customer's issue?",
             "Which issue does the customer raise?"],
        options=[
            "No mobile signal at all",
            "Mobile data slow or not connecting",
            "Calls cutting off or sounding bad",
            "Texts not being sent or received",
            "Home broadband down completely",
            "Home broadband slow, patchy or dropping out",
            "TV channels missing, freezing or not working",
            "Home phone line dead or faulty",
            "A bill higher than expected, or a charge they think is wrong",
            "Switching to a different plan or adding an extra",
            "Cancelling or ending a contract",
            "Signing up as a new customer, or checking what is available",
            "An order, SIM or device that has not arrived",
            "Booking, missing or rearranging an engineer visit",
            "A phone that is broken, lost or stolen",
            "Moving house and taking the service along",
            "Voicemail, call forwarding or another calling feature",
            "Anything about using the phone abroad",
            "Keeping their number when switching, or getting a new number",
            "Logging in, passwords, or who can manage the account",
        ],
    ),
    Question(
        id="next_step",
        ask=["What should the provider do next?", "Which action should the agent take?",
             "What is the best next step for the provider?"],
        options=[
            "Check the network or the line for a fault",
            "Book or rebook an engineer visit",
            "Send out a replacement SIM, router, TV box or phone",
            "Remove a charge or give a refund or credit",
            "Explain the bill or a charge",
            "Move them to a different plan or add an extra",
            "Process a cancellation and say what it will cost",
            "Chase a late order or delivery",
            "Switch a feature or setting on or off on their account",
            "Block a lost or stolen SIM or phone",
            "Arrange to move their number over, or give them a new one",
            "Say what is available or what a plan includes",
            "Talk them through steps to try on their own device",
            "Pass it on as a formal complaint",
            "Nothing more: the matter is already settled",
            "Help them get back into their account",
            "Move their service to a new address",
        ],
    ),
    Question(
        id="feeling",
        ask=["Which word best describes how the customer feels?", "How is the customer feeling?",
             "What mood is the customer in?"],
        options=[
            "Calm: just stating the facts",
            "Curious: asking out of interest",
            "Grateful: thanking the provider",
            "Confused: unsure what has happened or why",
            "Worried: anxious about what might happen",
            "Impatient: wants it done quickly",
            "Annoyed: irritated but holding back",
            "Angry: blaming or threatening",
            "Excited: looking forward to something new",
            "Desperate: stuck and needs help urgently",
        ],
    ),
    Question(
        id="agent_reply_action",
        needs=["agent_reply"],
        ask=["What does `agent_reply` do about the customer's issue?", "How does the provider respond in `agent_reply`?",
             "Which option best describes `agent_reply`?"],
        options=[
            "Says the problem is fixed, or fixes it",
            "Gives the customer steps to try themselves",
            "Asks the customer for more information",
            "Books an engineer visit or sends replacement equipment",
            "Says the charge or behaviour is correct and explains why",
            "Says it is being looked into, with nothing concrete yet",
        ],
    ),
    Question(
        id="follow_up_outcome",
        needs=["follow_up"],
        ask=["What does `follow_up` say about the problem now?", "According to `follow_up`, where do things stand?",
             "Going by `follow_up`, what has happened since the reply?"],
        options=[
            "The problem is solved",
            "Nothing has changed",
            "It is worse, or a new problem has appeared",
            "They have not tried the suggested fix yet",
        ],
    ),
    Question(
        id="follow_up_tone",
        needs=["follow_up"],
        ask=["How does the customer come across in `follow_up`?", "What is the tone of `follow_up`?",
             "How upset is the customer in `follow_up`?"],
        options=TONE,
    ),
]

TEXTS = [
    # ---- short
    Text(
        id="telecom_s1", style="short",
        text="no signal on my phone at home since this morning, not even one bar. cant call out at all, can you sort it",
        stated={"service": 0, "problem": 0, "duration": 0, "location": 0, "ask": 0, "issue": 0},
    ),
    Text(
        id="telecom_s2", style="short",
        text="hi, my bill is £38 this month, it's usually £25 and i haven't changed anything. could you explain the difference? thanks",
        stated={"team": 0, "problem": 2, "ask": 3, "tone": 0, "issue": 8, "next_step": 4},
        unknown=["duration", "location"],
    ),
    Text(
        id="telecom_s3", style="short",
        text="hi, is your fibre broadband available on my street yet? thinking of switching over from my current provider",
        stated={"customer": 3, "service": 1, "problem": 5, "ask": 3, "issue": 11, "next_step": 11},
        unknown=["duration"],
    ),
    Text(
        id="telecom_s4", style="short",
        text="since you swapped my tv box on monday the picture freezes for a few seconds every evening, then carries on. "
             "old box never did this",
        stated={"service": 2, "team": 3, "problem": 1, "duration": 1, "issue": 6},
    ),
    # ---- structured
    Text(
        id="telecom_t1", style="structured",
        text="Subject: Engineer didn't turn up\n"
             "Account type: Business\n"
             "Service: Broadband installation at our office\n"
             "Details: The installation was booked for Tuesday, 8am to 12pm. Two of us waited all morning. "
             "Nobody came and nobody called, and the office still has no connection.\n"
             "What we need: a new appointment this week, as early as possible please.",
        stated={"customer": 1, "service": 1, "team": 4, "problem": 3, "location": 3, "issue": 13, "next_step": 1},
    ),
    Text(
        id="telecom_t2", style="structured",
        text="Subject: Roaming not working in Spain\n"
             "Account type: Personal\n"
             "Plan: Unlimited SIM-only, roaming included\n"
             "Issue: I landed in Madrid yesterday. Calls and texts work fine here, but mobile data won't connect at "
             "all. Data roaming is switched on in settings and I have restarted the phone twice.\n"
             "Request: please get this working, I'm here until Friday.",
        stated={"customer": 0, "service": 0, "location": 1, "problem": 4, "ask": 0},
    ),
    Text(
        id="telecom_t3", style="structured",
        text="Subject: Ending my father's contract\n"
             "Account holder: my father. I'm his daughter and I manage his accounts for him.\n"
             "Services: home phone and broadband\n"
             "Request: He is moving into a care home at the end of next month. Please end the contract from the 30th, "
             "and tell me whether there is a fee for leaving early.",
        stated={"customer": 2, "team": 2, "problem": 5, "ask": 2, "issue": 10, "next_step": 6},
        unknown=["duration"],
    ),
    Text(
        id="telecom_t4", style="structured",
        text="Subject: Broadband speed lower than the plan\n"
             "Plan: 500 Mb fibre\n"
             "Measured: 40 to 60 Mb over a cable, several tests a day for the past three weeks\n"
             "Already tried: a new cable, restarting the router, a factory reset\n"
             "Wanted: either fix the speed, or move me to a cheaper plan that matches what I actually get.",
        stated={"service": 1, "team": 1, "problem": 1, "duration": 2, "tone": 0, "issue": 5},
    ),
    # ---- rant
    Text(
        id="telecom_r1", style="rant",
        text="This is the fourth time I've written in. For three weeks my mobile data has slowed to nothing every time "
             "I'm in town. Pages take a full minute to load, and every time I'm told it's 'being looked at'. I pay for "
             "5G and I'm getting less than dial-up. Sort it out or I'm leaving the minute my contract lets me.",
        stated={"service": 0, "problem": 1, "duration": 2, "location": 2, "tone": 2, "issue": 1},
    ),
    Text(
        id="telecom_r2", style="rant",
        text="You have charged me £60 for a phone I sent back in the first week. I have the return receipt. It has been "
             "on my bill every month since, and every month someone promises it'll be taken off. I want that money back "
             "and the charge gone for good, not another promise.",
        stated={"team": 0, "problem": 2, "ask": 1, "duration": 2, "issue": 8, "next_step": 3},
        unknown=["location"],
    ),
    Text(
        id="telecom_r3", style="rant",
        text="I understand things go wrong, I really do, but my mum is 84 and her home phone has been completely dead "
             "since Saturday. It's the only way she can call anyone. I've rung your helpline twice on her behalf and "
             "been cut off both times. Could someone please just tell me when it will be working again?",
        stated={"customer": 2, "service": 3, "problem": 0, "tone": 1, "duration": 1, "issue": 7},
    ),
    Text(
        id="telecom_r4", style="rant",
        text="Signed up online nine days ago, got a confirmation email, and since then absolutely nothing. No SIM, no "
             "tracking number, no reply to two emails. I've already given notice to my old provider, so in four days "
             "I'll have no phone at all. Where is my SIM? How is it this hard to post one?",
        stated={"service": 0, "team": 3, "problem": 3, "ask": 4, "issue": 12, "next_step": 7},
    ),
    # ---- json: j1 字段一致 (都平静), j2 不一致, j3 一致 (都生气), j4 不一致
    Text(
        id="telecom_j1", style="json",
        text={
            "subject": "Voicemail stopped working",
            "message": "Since I moved my mobile to the new plan on Monday, people who call my mobile can't leave "
                       "voicemail. They hear it ring out and then the call just ends. Could you check the setup on "
                       "your side?",
            "agent_reply": "Thanks for letting us know. Voicemail wasn't switched back on when your plan changed. We've "
                           "turned it on again, so callers should be able to leave messages straight away.",
            "follow_up": "Thanks. I'm away from my phone until this evening, so I haven't been able to test it yet. "
                         "I'll let you know if it's still not working.",
        },
        stated={"service": 0, "problem": 4, "ask": 0, "issue": 16,
                "agent_reply_action": 0, "follow_up_outcome": 3, "follow_up_tone": 0},
        unknown=["location"],
    ),
    Text(
        id="telecom_j2", style="json",
        text={
            "subject": "Broadband dropping out in the evenings",
            "message": "Hello, my home broadband has been dropping out for a few minutes most evenings this week. It "
                       "comes back on its own each time. Is there anything I can check on my end?",
            "agent_reply": "Sorry to hear that. Please move the router away from the TV and other electronics, and "
                           "restart it once a day for the next few days. Let us know how it goes.",
            "follow_up": "I did exactly what you said for five days and it's still dropping every single night, same as "
                         "before. I'm not rearranging my living room again. Send someone who can actually fix it.",
        },
        stated={"service": 1, "problem": 1, "location": 0, "issue": 5,
                "agent_reply_action": 1, "follow_up_outcome": 1, "follow_up_tone": 2},
    ),
    Text(
        id="telecom_j3", style="json",
        text={
            "subject": "Charged for calls I never made",
            "message": "This is outrageous. My bill has £14 of calls to a premium number I have never rung. Nobody else "
                       "uses my phone. Take them off now.",
            "agent_reply": "We've checked the call records. The calls were made from your number to a premium quiz line "
                           "on the 3rd and 4th, so the charges are correct. You can block premium numbers in your "
                           "account settings.",
            "follow_up": "Rubbish. I never called that line, and now there's another £9 of the same calls on this "
                         "month's usage. If this isn't off my bill by Friday I'm cancelling everything.",
        },
        stated={"problem": 2, "ask": 1, "tone": 2, "issue": 8,
                "agent_reply_action": 4, "follow_up_outcome": 2, "follow_up_tone": 2},
        unknown=["location"],
    ),
    Text(
        id="telecom_j4", style="json",
        text={
            "subject": "No TV channels at all",
            "message": "Every channel has said 'no signal' since Thursday. I've missed three matches I pay extra for, "
                       "and this is the second time this year. Absolutely useless.",
            "agent_reply": "We're sorry about this. There's a fault with your set-top box, so we're sending a "
                           "replacement by courier. It should arrive tomorrow before 1pm.",
            "follow_up": "The new box arrived this morning and every channel is back. Thank you for sorting it so "
                         "quickly, all good now.",
        },
        stated={"service": 2, "problem": 0, "duration": 1, "issue": 6, "next_step": 14,
                "agent_reply_action": 3, "follow_up_outcome": 0, "follow_up_tone": 0},
    ),
]

# ---- short, 取自 v2: datasets/synth-intents/telecom.jsonl 里意图 <后缀> 的第 1 条消息 (短口语), 原样照抄.
# 这 20 个意图按 issue 挑, 每条都有一个明确的 issue; 上面 16 份没写到的 8 个 issue 在这里各至少一条.
# v2 消息只讲意图, stated 通常只有 2-4 道.
V2_SHORT = [
    ("report_dropped_calls", "calls keep cutting out after a minute or two",
     {"problem": 1, "issue": 2}, ["location"]),
    ("texts_not_arriving", "havent had a single text since friday",
     {"service": 0, "duration": 1, "issue": 3}, ["location"]),
    ("report_broadband_outage", "internet at home is dead, router light is red",
     {"service": 1, "problem": 0, "location": 0, "issue": 4}, ["duration"]),
    ("upgrade_plan_more_data", "keep running out of gigs, what's the next tier up",
     {"problem": 5, "ask": 2, "issue": 9, "next_step": 5}, ["location"]),
    ("blacklist_stolen_phone", "someone stole my phone, brick it so they can't use it",
     {"service": 0, "issue": 14, "next_step": 9}),
    ("move_broadband_to_new_address", "moving next month, can i take my internet with me",
     {"service": 1, "problem": 5, "issue": 15, "next_step": 16}, ["duration"]),
    ("change_my_number", "can i get a different number on my phone",
     {"service": 0, "problem": 5, "issue": 18, "next_step": 10}, ["duration", "location"]),
    ("reset_online_password", "cant log into my account on your website",
     {"issue": 19, "next_step": 15}, ["service", "duration", "location"]),
    ("report_no_signal_at_home", "zero bars inside my house, works fine outside",
     {"service": 0, "location": 0, "issue": 0}),
    ("mobile_data_not_connecting", "no internet at all on my phone unless im on wifi",
     {"service": 0, "issue": 1}, ["duration"]),
    ("broadband_keeps_dropping", "home internet keeps vanishing then coming back, driving me mad",
     {"service": 1, "problem": 1, "location": 0, "issue": 5}),
    ("landline_no_dial_tone", "home phone is completely silent when i pick it up",
     {"service": 3, "problem": 0, "location": 0, "issue": 7}, ["duration"]),
    ("dispute_bill_charge", "there's a 20 dollar fee on here i never agreed to",
     {"team": 0, "problem": 2, "issue": 8}, ["service", "duration", "location"]),
    ("cancel_contract", "i want out, how do i close my line",
     {"problem": 5, "ask": 2, "issue": 10, "next_step": 6}, ["duration", "location"]),
    ("order_home_broadband", "need home internet for a house of 5, what do you offer",
     {"service": 1, "problem": 5, "ask": 3, "issue": 11, "next_step": 11}, ["duration"]),
    ("track_phone_delivery", "ordered a phone monday, where is it",
     {"problem": 3, "ask": 4, "issue": 12, "next_step": 7}),
    ("reschedule_technician_visit", "can't be home thursday, can the engineer come another day",
     {"team": 4, "issue": 13, "next_step": 1}, ["duration"]),
    ("callers_cannot_leave_voicemail", "people calling me get a message saying my mailbox is full",
     {"problem": 4, "issue": 16}, ["location"]),
    ("no_service_abroad", "roaming is on but no signal in spain",
     {"service": 0, "location": 1, "issue": 17}, ["duration"]),
    ("faulty_new_phone", "phone came yesterday and the screen flickers",
     {"service": 0, "team": 3, "issue": 14}, ["location"]),
]
TEXTS += [Text(id=f"telecom_v2_{suffix}", style="short", text=text, stated=stated, unknown=rest[0] if rest else [])
          for suffix, text, stated, *rest in V2_SHORT]

if __name__ == "__main__":
    main(DOMAIN, LABEL, QUESTIONS, TEXTS)
