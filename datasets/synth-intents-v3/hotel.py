#!/usr/bin/env python
"""synth-intents v3: hotel 领域. 一家 48 间房 (4 层, 每层 12 间) 的酒店的日常文档, 题目问文档里发生的事.

  python datasets/synth-intents-v3/hotel.py [--out DIR]

12 份文本, 各是一种文档: 交班记录、到店名单、入住对话、客房日志、住客点评、房态 JSON、住客须知、
住客短信、失物登记、会议单、账单、排班表. 每份配 6-8 道只问它的题 (Question.texts), 问里面的人、东西、
数量、先后、因果、状态和下一步; 选项是文档里的东西本身. 另有一道所有文本共用的题: 这是哪种文档.
答案由文档定死的写进 stated; 要判断的 (先做哪件事) 不写, 标签交给参考模型. 键的意思与写作规则见 schema.py.
"""

from schema import YES_NO, Question, Text, main

DOMAIN = "hotel"
LABEL = "Hotel document"

ROOMS = [f"Room {floor}{n:02d}" for floor in range(1, 5) for n in range(1, 13)]  # 每层 12 间: Room 101 .. Room 412
FLOOR2, FLOOR3 = ROOMS[12:24], ROOMS[24:36]
NUMBER_WORDS = ["none", "one", "two", "three", "four", "five", "six", "seven", "eight", "nine", "ten"]
DAYS = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]


def count(lo: int, hi: int) -> list[str]:
    """数量题的选项写成英文单词: 参考模型按编号答题, 选项是小整数时会把值当成编号."""
    return NUMBER_WORDS[lo:hi + 1]


def clock(start: str, end: str, step: int) -> list[str]:
    """从 start 到 end (含) 每 step 分钟一个时刻, 如 clock("18:00", "23:45", 15)."""
    h, m = map(int, start.split(":"))
    t, stop = h * 60 + m, int(end[:2]) * 60 + int(end[3:])
    out = []
    while t <= stop:
        out.append(f"{t // 60:02d}:{t % 60:02d}")
        t += step
    return out


def dates(month: str, first: int, last: int) -> list[str]:
    return [f"{d} {month}" for d in range(first, last + 1)]


DOC_KINDS = [
    "A night handover note",
    "A list of arriving bookings",
    "A conversation at check-in",
    "A housekeeping log",
    "A guest's review of their stay",
    "A room status report",
    "The hotel's rules for guests",
    "A message from a guest",
    "A lost property book",
    "A sheet for a meeting or event",
    "A guest's bill",
    "A staff rota",
]

QUESTIONS = [
    Question(
        id="document",
        ask=["What kind of document is this?", "Which of these best describes the text?", "What is this text?"],
        options=DOC_KINDS,
    ),
]
TEXTS = []


def doc(suffix: str, style: str, kind: str, text, questions: list[tuple]) -> None:
    """一份文本和只问它的题. questions 每项是 (题名, 问法或 [问法...], 选项, 答案或 None[, needs]).
    答案写成选项原文 (写错会在 .index 处报错); None 表示文本没把答案定死, 交给参考模型."""
    tid = f"{DOMAIN}_{suffix}"
    stated = {"document": DOC_KINDS.index(kind)}
    for slug, ask, options, answer, *needs in questions:
        qid = f"{suffix}_{slug}"
        QUESTIONS.append(Question(id=qid, ask=[ask] if isinstance(ask, str) else ask, options=options,
                                  needs=needs[0] if needs else [], texts=[tid]))
        if answer is not None:
            stated[qid] = options.index(answer)
    TEXTS.append(Text(id=tid, style=style, text=text, stated=stated))


doc("handover", "structured", "A night handover note",
    "Night handover, Friday 22:00\n"
    "- 204 Okafor: arrived today, staying 3 nights. Asked for extra pillows at 21:30, not sent yet.\n"
    "- 207: was empty until 20:15, when the Bauers moved in from 312.\n"
    "- 312: radiator dead since 19:40. Engineer booked for Saturday 09:00. Keep empty.\n"
    "- 311 Lindqvist and her daughter: leaving Saturday 07:00, breakfast box ordered.\n"
    "- Lost property: a black umbrella found in the lift.",
    [
        ("out_of_use", ["Which room is out of use tonight?", "Which room must not be given to anyone tonight?"],
         ROOMS, "Room 312"),
        ("first", "Which of these happened first on Friday evening?",
         ["The radiator in 312 stopped working", "The Bauers moved into 207", "Mr Okafor asked for extra pillows"],
         "The radiator in 312 stopped working"),
        ("why_moved", ["Why are the Bauers in room 207?", "What made the Bauers change rooms?"],
         ["The radiator in their first room stopped working", "They asked for a bigger room",
          "Room 207 has a better view", "Their booking was for 207 all along"],
         "The radiator in their first room stopped working"),
        ("waiting", "What is room 204 still waiting for?",
         ["Extra pillows", "A breakfast box", "An engineer", "A black umbrella"], "Extra pillows"),
        ("leaving", "Who does the note say is leaving on Saturday morning?",
         ["Mr Okafor", "The Bauers", "Ms Lindqvist and her daughter", "Nobody"], "Ms Lindqvist and her daughter"),
        ("engineer", ["When is the engineer due?", "At what time should someone come to fix the radiator?"],
         ["Friday 22:00", "Saturday 07:00", "Saturday 09:00", "Monday morning"], "Saturday 09:00"),
        ("moved_at", ["At what time did the Bauers move into 207?", "When did room 207 stop being empty?"],
         clock("18:00", "23:45", 15), "20:15"),
        ("first_task", "What should the night staff do first?",
         ["Send the extra pillows up to 204", "Prepare the breakfast box for 311",
          "Take the umbrella to lost property", "Try to fix the radiator in 312", "Move the Bauers back to 312"],
         None),
    ])

doc("arrivals", "table", "A list of arriving bookings",
    "Arrivals, week of 6 October\n"
    "Guest     | Arrives | Nights | Room type | Rate per night\n"
    "Adebayo   | Mon 6   | 2      | Double    | £95\n"
    "Carvalho  | Mon 6   | 5      | Suite     | £210\n"
    "Dimitrov  | Tue 7   | 1      | Single    | £70\n"
    "Eriksen   | Wed 8   | 3      | Double    | £95\n"
    "Fontaine  | Wed 8   | 4      | Twin      | £90\n"
    "Gallagher | Fri 10  | 1      | Suite     | £210",
    [
        ("most_nights", ["Which guest is staying the most nights?", "Who has the longest stay?"],
         ["Adebayo", "Carvalho", "Dimitrov", "Eriksen", "Fontaine", "Gallagher"], "Carvalho"),
        ("wednesday", "Who arrives on Wednesday?",
         ["Adebayo only", "Eriksen and Fontaine", "Dimitrov and Eriksen", "Fontaine only", "Nobody"],
         "Eriksen and Fontaine"),
        ("cheapest", "Which room type has the lowest rate per night?", ["Single", "Twin", "Double", "Suite"], "Single"),
    ])

doc("checkin", "dialogue", "A conversation at check-in",
    "Guest: Hi, I'm checking in, the name's Moreau. I booked a double for two nights.\n"
    "Receptionist: Welcome, Ms Moreau. I have you in room 305, a double on the third floor.\n"
    "Guest: Is there anything lower down? My knee's not great with stairs if the lift is busy.\n"
    "Receptionist: I can offer 108 on the ground floor. It's a twin rather than a double, same price.\n"
    "Guest: A twin is fine, it's just me. I'll take 108.\n"
    "Receptionist: Done. Breakfast is 07:00 to 10:00 in the garden room. Will you need parking?\n"
    "Guest: No, I don't have a car. Oh, and could I check out at 14:00 on the last day instead of 11:00?\n"
    "Receptionist: Late check-out is £15, or free for members. Are you a member?\n"
    "Guest: I'm not. That's all right, I'll pay the £15.",
    [
        ("room", ["Which room does Ms Moreau end up in?", "Where will Ms Moreau sleep tonight?"], ROOMS, "Room 108"),
        ("why", "Why did she ask for a different room?",
         ["She has a bad knee", "She wanted a bigger bed", "Room 305 was not clean", "She wanted a cheaper room"],
         "She has a bad knee"),
        ("room_type", "What kind of room is she staying in?", ["Single", "Twin", "Double", "Suite"], "Twin"),
        ("checkout", ["At what time will she check out on her last day?", "When does she plan to leave her room?"],
         ["07:00", "10:00", "11:00", "14:00"], "14:00"),
        ("extra", "How much will she pay on top of the room price?", ["Nothing", "£15", "£30", "£45"], "£15"),
        ("member", "Is Ms Moreau a member?", YES_NO, "no"),
        ("breakfast", "Where is breakfast served?", ["The garden room", "Room 108", "The third floor", "The lobby"],
         "The garden room"),
        ("nights", "How many nights is she staying?", count(1, 6), "two"),
    ])

doc("housekeeping", "log", "A housekeeping log",
    "Housekeeping log, 2nd floor, Tuesday\n"
    "08:40  Priya starts on 201 (checkout).\n"
    "09:25  201 finished. 205 has a do-not-disturb sign, so she skips it and starts 203 (checkout).\n"
    "10:15  203 finished. 204 only wants fresh towels; towels left at the door.\n"
    "10:20  Tomasz joins and takes 206 and 208.\n"
    "11:05  Tomasz finishes 206.\n"
    "11:50  Tomasz finishes 208 except the shower rail, which is broken; reported to maintenance.\n"
    "11:55  The sign on 205 is gone. Priya starts 205.\n"
    "12:30  205 finished.",
    [
        ("last", ["Which room was finished last?", "Which room was the last one done?"], FLOOR2, "Room 205"),
        ("why_skipped", "Why did Priya skip 205 at first?",
         ["It had a do-not-disturb sign", "Its shower rail was broken", "Its guests were checking out",
          "She had run out of towels"], "It had a do-not-disturb sign"),
        ("tomasz", "Which rooms did Tomasz work on?", ["206 and 208", "201 and 203", "204 and 205", "203 and 206"],
         "206 and 208"),
        ("broken", ["Which room still has something that needs fixing?", "Where does maintenance need to go?"],
         FLOOR2, "Room 208"),
        ("towels", "What did room 204 get?",
         ["Fresh towels only", "A full clean", "Nothing at all", "A new shower rail"], "Fresh towels only"),
        ("tomasz_join", ["At what time did Tomasz start work on the floor?", "When did Tomasz join Priya?"],
         clock("08:00", "12:55", 5), "10:20"),
    ])

doc("review", "story", "A guest's review of their stay",
    "We arrived on Thursday evening to find our room smelled strongly of paint, so the duty manager moved us to a "
    "room at the back. It was quieter than the first one, but the window wouldn't close all the way and the cold "
    "came in all night. On Friday morning we mentioned it at breakfast and a man came up within the hour and fixed "
    "it. The breakfast itself was the highlight: fresh bread, good coffee, and the staff remembered our names by "
    "the second day. When we checked out on Sunday they took the first night off the bill without us asking. We'd "
    "happily come back, though we'll ask for a room away from any decorating next time.",
    [
        ("first_problem", "What was the first problem the guests had?",
         ["A strong smell of paint", "A window that would not close", "Cold coffee", "A noisy room"],
         "A strong smell of paint"),
        ("window_fixed", ["When was the window fixed?", "On which morning did someone repair the window?"],
         ["Thursday evening", "Friday morning", "Saturday morning", "Sunday at check-out"], "Friday morning"),
        ("at_checkout", "What did the hotel do when they checked out?",
         ["Took the first night off the bill", "Gave them a free breakfast", "Moved them to a suite",
          "Asked them to write a review"], "Took the first night off the bill"),
        ("liked_most", ["What did they like most?", "What was the best part of the stay for them?"],
         ["The breakfast", "The room at the back", "The view", "The check-out"], "The breakfast"),
        ("come_back", "Would they stay again?",
         ["Yes, happily", "Only if the price drops", "No", "They do not say"], "Yes, happily"),
        ("next_time", "What will they ask for next time?",
         ["A room away from decorating", "A room at the front", "A later breakfast", "A room with a bath"],
         "A room away from decorating"),
        ("overall", "How do the guests feel about the stay overall?",
         ["Pleased", "Angry", "Disappointed", "Indifferent"], None),
    ])

doc("status", "json", "A room status report",
    {
        "floor": "3",
        "checked": "Saturday 13:00",
        "301": "occupied, due out Sunday",
        "302": "vacant, clean",
        "303": "vacant, needs cleaning",
        "304": "occupied, due out today",
        "305": "vacant, clean",
        "306": "out of order: leak in the ceiling",
        "307": "occupied, due out Monday",
        "308": "vacant, clean, twin beds",
        "309": "vacant, needs cleaning",
        "310": "occupied, due out today",
        "311": "occupied, due out Sunday",
        "312": "occupied, due out Sunday",
    },
    [
        ("twin", "Which room would suit two friends who want separate beds and are arriving now?",
         FLOOR3, "Room 308"),
        ("unusable", "Which room cannot be used at all?", FLOOR3, "Room 306"),
        ("last_out", "Which room's guests are due to leave last?", FLOOR3, "Room 307"),
        ("field_306", ["What does `306` say is wrong?", "According to `306`, what is the problem with that room?"],
         ["A leak in the ceiling", "It needs cleaning", "Its guests are leaving today", "Nothing is wrong"],
         "A leak in the ceiling", ["306"]),
        ("clean_first", "Which room should housekeeping clean first?",
         ["Room 303", "Room 309", "Room 304", "Room 310", "Room 306"], None),
    ])

doc("rules", "notice", "The hotel's rules for guests",
    "Guest information\n"
    "- Check-in from 15:00. Check-out by 11:00. Late check-out until 14:00 costs £15, free for members.\n"
    "- Dogs are welcome in ground-floor rooms only (rooms 101 to 112), £20 per dog per night, up to two dogs per room. "
    "No dogs in the restaurant.\n"
    "- Pool open 07:00 to 21:00. Under-16s must be with an adult. Lengths only before 09:00.\n"
    "- Quiet hours 22:00 to 07:00.\n"
    "- Parking £10 per night, free for guests staying 3 nights or more.",
    [
        ("dog_room", "Which of these rooms could be given to a guest with a dog?",
         ["Room 108", "Room 204", "Room 312", "Room 405"], "Room 108"),
        ("teen_pool", "Can a 14-year-old swim on their own at 17:00?",
         ["Yes", "No, they need an adult with them", "No, the pool is closed then", "Only if they swim lengths"],
         "No, they need an adult with them"),
        ("parking_four", "What does parking cost for a guest staying four nights?",
         ["Nothing", "£10", "£30", "£40"], "Nothing"),
        ("breaks_rule", ["Which of these would break the hotel's rules?", "Which of these is not allowed?"],
         ["A party in a room at 23:00", "Swimming lengths at 08:00", "Checking out at 10:30",
          "A dog in room 105"], "A party in a room at 23:00"),
        ("latest_free", "What is the latest a non-member can check out without paying extra?",
         ["10:00", "11:00", "14:00", "15:00"], "11:00"),
        ("member_late", "How much does a member pay to check out at 14:00?", ["Nothing", "£10", "£15", "£20"],
         "Nothing"),
        ("dogs_per_room", "How many dogs may share one room?",
         ["Only one", "Up to two", "Up to three", "As many as the guest likes"], "Up to two"),
        ("pool_closes", ["At what time does the pool close?", "What is the last time the pool is open?"],
         clock("00:00", "23:00", 60), "21:00"),
    ])

doc("message", "short", "A message from a guest",
    "hi its room 407, the kettle isn't working and we've only got one cup between us. "
    "also can we keep the room till 1pm tomorrow? thanks",
    [
        ("room", ["Which room is the message from?", "Where are these guests staying?"], ROOMS, "Room 407"),
        ("broken", "What is broken?", ["The kettle", "The cup", "The TV", "The shower"], "The kettle"),
        ("short_of", "What do they not have enough of?", ["Cups", "Towels", "Pillows", "Tea bags"], "Cups"),
        ("keep_until", "Until what time do they want to keep the room tomorrow?",
         ["11:00", "12:00", "13:00", "14:00"], "13:00"),
        ("people", "How many people are staying in the room?", ["One person", "More than one person"],
         "More than one person"),
        ("first", "What should reception do first?",
         ["Send up a working kettle and a second cup", "Tell them check-out is at 11:00",
          "Ask them to come down to reception", "Move them to another room"], None),
    ])

doc("lost", "log", "A lost property book",
    "Lost property book\n"
    "3 Mar   reading glasses, restaurant. Claimed 4 Mar.\n"
    "9 Mar   child's blue scarf, pool. Not claimed.\n"
    "15 Mar  phone charger, room 212. Claimed 15 Mar.\n"
    "21 Mar  black umbrella, lift. Not claimed.\n"
    "28 Mar  wedding ring, pool changing room. Claimed 28 Mar.\n"
    "2 Apr   paperback novel, restaurant. Not claimed.",
    [
        ("oldest_waiting", ["Which item has been waiting longest for its owner?",
                            "Of the things nobody has collected, which was found first?"],
         ["The reading glasses", "The blue scarf", "The phone charger", "The black umbrella", "The wedding ring",
          "The paperback novel"], "The blue scarf"),
        ("restaurant", "Which items were found in the restaurant?",
         ["Reading glasses and a paperback novel", "A scarf and a wedding ring", "A charger and an umbrella",
          "Only the reading glasses"], "Reading glasses and a paperback novel"),
        ("day_after", "Which item was claimed the day after it was found?",
         ["The reading glasses", "The phone charger", "The wedding ring", "The black umbrella"],
         "The reading glasses"),
        ("umbrella_date", ["On which date was the umbrella found?", "When did the umbrella turn up?"],
         dates("March", 1, 31) + dates("April", 1, 5), "21 March"),
        ("false", ["Which of these statements is false?", "Which statement does the book contradict?"],
         ["The umbrella was found in the lift", "The charger was found in room 212", "The scarf has been claimed",
          "The novel was found in the restaurant"], "The scarf has been claimed"),
    ])

doc("function", "structured", "A sheet for a meeting or event",
    "Function sheet: Sato & Partners, Thursday\n"
    "Room: Linden Room (seats 40 theatre style, 24 boardroom)\n"
    "Guests: 30\n"
    "Layout: boardroom\n"
    "09:00 room open, coffee and pastries\n"
    "10:30 coffee break\n"
    "12:30 lunch in the restaurant, 30 covers, 2 vegetarian, 1 gluten-free\n"
    "15:00 afternoon tea\n"
    "16:30 finish\n"
    "Contact: Emi Sato, arriving 08:15",
    [
        ("problem", ["What is wrong with this booking as written?", "Which part of the sheet cannot work?"],
         ["30 guests will not fit the boardroom layout, which seats 24", "Lunch is booked for too few people",
          "There is no coffee break", "The contact arrives after the room opens"],
         "30 guests will not fit the boardroom layout, which seats 24"),
        ("layout_fit", "Which layout of the Linden Room can seat everyone?",
         ["Theatre style", "Boardroom", "Neither", "Both"], "Theatre style"),
        ("lunch_time", ["At what time is lunch?", "When does the group go to the restaurant?"],
         clock("08:00", "17:45", 15), "12:30"),
    ])

doc("bill", "table", "A guest's bill",
    "Room 402, Mr Haddad, 3 nights (Monday to Thursday)\n"
    "Mon | Room                | £120\n"
    "Mon | Dinner, restaurant  | £46\n"
    "Tue | Room                | £120\n"
    "Tue | Minibar             | £9\n"
    "Tue | Minibar             | £9\n"
    "Wed | Room                | £120\n"
    "Wed | Room                | £120\n"
    "Thu | Parking, 3 nights   | £30",
    [
        ("certain_error", ["Which line is certainly a mistake?", "Which charge should definitely not be there?"],
         ["The second room charge on Wednesday", "The second minibar charge on Tuesday", "The dinner on Monday",
          "The parking charge"], "The second room charge on Wednesday"),
        ("room", ["Which room is Mr Haddad staying in?", "Which room is this bill for?"], ROOMS, "Room 402"),
        ("ask_first", "What should reception ask Mr Haddad before changing anything else on the bill?",
         ["Whether he took two things from the minibar on Tuesday", "Whether he ate dinner on Monday",
          "Whether he stayed on Wednesday night", "Whether he used the parking"], None),
    ])

doc("rota", "table", "A staff rota",
    "Front desk rota, week of 13 October\n"
    "Early 07:00-15:00, Late 15:00-23:00, Night 23:00-07:00\n"
    "\n"
    "      | Mon   | Tue   | Wed   | Thu   | Fri   | Sat   | Sun\n"
    "Early | Ana   | Ana   | Ben   | Ben   | Ana   | Chloe | Chloe\n"
    "Late  | Ben   | Chloe | Chloe | Dev   | Dev   | Ana   | Ben\n"
    "Night | Dev   | Dev   | Dev   | Chloe | Chloe | Ben   | Ana",
    [
        ("thu_night", "Who is on the night shift on Thursday?", ["Ana", "Ben", "Chloe", "Dev"], "Chloe"),
        ("no_break", ["Whose shifts run straight into each other with no rest in between?",
                      "Who has to start a shift the moment their previous one ends?"],
         ["Ana", "Ben", "Chloe", "Dev", "Nobody"], "Chloe"),
        ("cover", "Chloe needs her Saturday early shift covered. Who is not working at all on Saturday?",
         ["Ana", "Ben", "Dev", "Nobody"], "Dev"),
        ("ana_late", "On which day does Ana work a late shift?", DAYS, "Saturday"),
        ("ben_off", "Which two days does Ben have off?",
         ["Tuesday and Friday", "Monday and Thursday", "Wednesday and Saturday", "Tuesday and Sunday"],
         "Tuesday and Friday"),
    ])

if __name__ == "__main__":
    main(DOMAIN, LABEL, QUESTIONS, TEXTS)
