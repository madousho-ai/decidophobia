#!/usr/bin/env python
"""生成 datasets/synth-simple-eval/synth-simple-eval.jsonl. 纯标准库, 固定种子, 重跑得到同一个文件.

用途: sanity check. 客户消息直接说出答案 (I want apple.), 选项是常见名词, 题目只剩「照着找到那一项」,
看得出菜单拉长时模型还选不选得对, 不掺题目本身的语义难度. 只做评估.

- 选择题: 菜单 5 / 10 / 20 / 40 / 60 / 100 / 255 项, 每档 10 题. 问句 "Which item does the customer want?".
  10 题的正确答案等距铺在菜单上, 第 i 题在第 round(i * (k - 1) / 9) 行: 第一行、最后一行都有.
  干扰项从名词表里随机抽, 每题重抽.
- 二元题: 10 题, 问 "does the customer want <名词>?", 选项 no / yes. 5 题问的就是消息里那样 (yes),
  5 题问另一样 (no); no 排第一的 5 题. 10 题做不到答案、选项顺序、正确答案位置三样同时各半,
  这里保前两样, 正确答案在第一行的是 4 题.

名词两两不互为子串, 也不出现在消息模板和问句的固定文字里 —— 消息里除了答案, 找不到任何一个选项.

每行一题: {"id", "qtype": "choice" | "bool", "context", "question", "options": [...], "answer": 正确项的下标}.
"""

from __future__ import annotations

import json
import pathlib
import random

SIZES = (5, 10, 20, 40, 60, 100, 255)
PER_SIZE = 10
N_BOOL = 10
SEED = 20260924
CHOICE_QUESTION = "Which item does the customer want?"
TEMPLATES = ("I want {x}.", "i want {x}", "I'd like {x}, please.", "can I get {x}?", "{x} for me, please.", "give me {x}")

NOUNS = """
apple banana cherry grape lemon mango orange peach plum kiwi apricot coconut papaya lime fig guava lychee
raspberry strawberry blueberry pear
carrot potato tomato onion garlic cabbage lettuce spinach celery cucumber pumpkin broccoli radish turnip
zucchini mushroom asparagus artichoke
bread cheese butter cookie muffin bagel pretzel noodle rice pasta pizza burger sandwich taco burrito sushi
dumpling waffle croissant donut soup salad omelette yogurt honey jam popcorn cereal porridge sausage bacon
chocolate candy biscuit
coffee juice milk soda cocoa
dog horse rabbit tiger lion zebra giraffe elephant kangaroo koala panda penguin dolphin whale shark octopus
turtle frog snake lizard parrot eagle duck goose swan sheep goat camel squirrel hedgehog fox wolf deer moose
otter beaver hamster spider crab lobster flamingo peacock ostrich raccoon gorilla cheetah leopard buffalo
hippo rhino llama alpaca
hammer screwdriver wrench drill shovel ladder axe chisel pliers glue scissors stapler ruler pencil eraser
crayon marker envelope stamp calendar clipboard magnet compass battery candle lantern broom bucket sponge mop
towel soap shampoo comb mirror pillow blanket mattress curtain carpet lamp sofa bench shelf drawer cabinet
wardrobe clock vase kettle spoon fork knife plate bowl jar bottle tray
shirt jacket sweater scarf glove sock boot sandal sneaker helmet belt dress skirt vest apron umbrella wallet
backpack purse necklace bracelet watch
guitar piano violin trumpet flute drum harp cello banjo accordion saxophone tuba harmonica xylophone tambourine
bicycle scooter tractor truck boat canoe kayak yacht helicopter rocket sled wagon ambulance taxi van jeep
motorcycle
football baseball tennis golf skateboard surfboard frisbee kite puzzle chess dice domino yoyo balloon trophy
medal
camera laptop printer speaker microphone television radio telescope microscope calculator robot
leaf pebble feather acorn pinecone cactus tulip daisy sunflower lily orchid bamboo seaweed
anchor crown sword shield wand map globe flag whistle ticket coin diamond emerald ribbon button zipper needle
thread yarn quilt puppet doll marble sticker postcard poster statue fountain bridge tent hammock igloo castle
windmill chimney garage rope chain wheel engine heater fridge oven toaster blender spatula ladle whisk
""".split()


def check_nouns(nouns: list[str]) -> None:
    if len(set(nouns)) != len(nouns):
        raise SystemExit(f"repeated nouns: {sorted({n for n in nouns if nouns.count(n) > 1})}")
    fixed = " ".join(t.replace("{x}", " ") for t in TEMPLATES).lower() + " " + CHOICE_QUESTION.lower() \
        + " does the customer want"
    clash = [(a, b) for a in nouns for b in nouns if a != b and a in b] + [(a, "<fixed text>") for a in nouns if a in fixed]
    if clash:
        raise SystemExit(f"nouns inside other words: {clash}")
    if len(nouns) < max(SIZES):
        raise SystemExit(f"{len(nouns)} nouns, need {max(SIZES)}")


def build(rng: random.Random) -> list[dict]:
    rows = []
    for k in SIZES:
        golds = rng.sample(NOUNS, PER_SIZE)
        for i, gold in enumerate(golds):
            others = rng.sample([n for n in NOUNS if n != gold], k - 1)
            pos = round(i * (k - 1) / (PER_SIZE - 1))
            options = others[:pos] + [gold] + others[pos:]
            rows.append({"id": f"k{k}-{i:02d}", "qtype": "choice", "context": rng.choice(TEMPLATES).format(x=gold),
                         "question": CHOICE_QUESTION, "options": options, "answer": pos})
    # 二元题: (答案 yes?, no 排第一?) 各 5 题, 组合 3/2/2/3
    plan = [(True, True)] * 2 + [(True, False)] * 3 + [(False, True)] * 3 + [(False, False)] * 2
    rng.shuffle(plan)
    for i, (yes, no_first) in enumerate(plan):
        said, other = rng.sample(NOUNS, 2)
        asked = said if yes else other
        options = ["no", "yes"] if no_first else ["yes", "no"]
        rows.append({"id": f"bool-{i:02d}", "qtype": "bool", "context": rng.choice(TEMPLATES).format(x=said),
                     "question": f"does the customer want {asked}?", "options": options,
                     "answer": options.index("yes" if yes else "no")})
    return rows


def main() -> None:
    check_nouns(NOUNS)
    rows = build(random.Random(SEED))
    out = pathlib.Path(__file__).resolve().parent / "synth-simple-eval.jsonl"
    out.write_text("".join(json.dumps(r) + "\n" for r in rows), encoding="utf-8")
    print(f"{len(NOUNS)} nouns, {len(rows)} questions -> {out}")


if __name__ == "__main__":
    main()
