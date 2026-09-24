# synth-intents: 合成意图数据集

决策头的训练集. 16 个领域 × 256 个虚构意图 = 4096 个, 每个一条自然语言描述 (菜单里显示的就是它)、
3 条用户消息 (作上下文) 和一道 yes/no 问题. 共 12288 条消息, 每条消息出两道题: 一道菜单题, 一道二元题.

**全部用于训练. 评估用 Banking77、MASSIVE 和 BoolQ.** BoolQ 退出训练后, 二元题由这里的二元题顶上.

## 菜单题: 一题一个领域

训练题的菜单是正确答案所在领域的全部 256 个意图, 随机排序, 恰好占满 D0..D255.
不同领域的意图不进同一份菜单. 这与部署一致 (一个应用的请求带的是它自己的全部意图),
每个 D 码当正确答案的机会也相同.

## 二元题: 问消息里的一个细节

形状与 BoolQ 相同: 上下文 + 问句 + 两项菜单 `no` / `yes` (qtype `bool`), 上下文标签是 `Customer message`.
每个意图一道问题, 3 条消息各有一个答案, 共 12288 道.

问题问消息里的一个细节, 同一意图的 3 条消息在这个细节上**有 yes 也有 no**. 这样模型认出意图也答不了,
得读消息. 答案由意图决定 (3 条全 yes) 或拿别的意图的消息凑 no, 模型都能不读消息就答对.

- **只凭消息文字能答.** 不要需要专业知识的问题: 紧急与否、严重程度、是否符合条件、价格与政策.
- **问的事在消息里要么写了、要么明确没写.** 「宠物是狗吗」碰上只写 my pet 的消息就只能猜, 不能用.
- **不复述意图描述.** 「顾客是不是想改约」是把菜单题换成二选一.
- 英文, 全小写, 以 `?` 结尾, 主语用 the customer (与 BoolQ 的小写问句一致).

**答案配比.** 每个意图预先分到一种答案模式, 六种轮流分: YYN、YNY、NYY、YNN、NYN、NNY.
三种风格的消息各有一半答 yes, 全部二元题也是一半 yes —— 恒答 yes 只拿一半分, 看风格答题也拿不到分.
旧意图的两条消息已写好, 它的模式由所选问题决定, 第 3 条补齐; 整体配比的偏差由新意图的模式分配抵掉.

例, `move my pet's vet visit to another time` —— `does the customer give a reason for moving the visit?`

| 消息 | 答案 |
| --- | --- |
| cant make tuesday at the vets for the dog, got anything on thursday instead | no |
| My rabbit is booked in for a checkup on Saturday morning but I have to work that day now, could we shift it to a weekday evening? | yes |
| my boss just put me in a meeting on thursday morning, right on top of the dog's check up, and I still want him seen | yes |

## 文件

一个领域一个文件: `datasets/synth-intents/<domain>.jsonl`, 每行一个意图:

```json
{"id": "telecom_report_dropped_calls", "domain": "telecom", "description": "report that calls keep dropping", "utterances": ["calls keep cutting out after a minute or two", "For the past week every voice call I make ends abruptly after a couple of minutes, even when I have full bars, and I have to redial each time.", "I was halfway through explaining something to my mum and suddenly I was talking to nobody, third time today"], "question": "does the customer mention their signal strength?", "answers": [false, true, false]}
```

16 个领域 × 256 个意图 = 4096:

| domain | 内容 |
| --- | --- |
| ecommerce | 网购: 订单、退换货、优惠券、账户 |
| logistics | 快递 / 货运: 寄件、追踪、丢件、清关 |
| telecom | 手机 / 宽带运营商: 套餐、信号、账单、换卡 |
| utilities | 水电燃气: 抄表、账单、停复供、报修 |
| healthcare | 诊所 / 医院患者服务: 预约、处方、化验、转诊 |
| petcare | 宠物医院 / 宠物服务: 疫苗、寄养、美容 |
| it_helpdesk | 企业 IT 帮助台: 账号、设备、VPN、软件 |
| education | 高校学生事务: 选课、成绩、学籍、奖学金 |
| hr | 人事: 请假、薪资单、报销、入离职 |
| legal | 律所客户接待: 咨询、文件、案件进度 |
| government | 政务: 证件、登记、税务申报、投诉 |
| rental | 租房 / 物业: 租约、维修、押金、停车 |
| hotel | 酒店客务: 房型、入住、设施、投诉 |
| home_services | 家政 / 上门服务: 保洁、水管、电工预约 |
| automotive | 4S 店 / 修车: 保养、召回、配件、保修 |
| fitness | 健身房: 会籍、私教、课程、冻结 |

窄领域凑不满 256 条分得清的意图时, 放宽它的业务范围 (例如 petcare 扩到宠物用品、训练), 领域数保持 16.

## 禁区 (评估集要留干净)

**银行** (Banking77 的领域): 银行账户、借记卡 / 信用卡、转账、汇率、ATM、加密货币、贷款、充值、直接扣款、卡支付退款.
给某项服务交账单是可以的 (telecom / utilities 的账单意图正常写), 但不要写「卡被拒」「退款没到账」这种.

**语音助手** (MASSIVE 的 18 个场景): 闹钟 / 计时器、音量、日历、菜谱、日期时间、邮件、笑话 / 寒暄 / 闲聊、
智能家居 (灯 / 插座 / 咖啡机 / 扫地机)、购物清单、音乐、新闻、播放媒体 (播客 / 电台 / 有声书 / 游戏)、
百科问答 (定义 / 事实 / 算术 / 股票 / 汇率)、推荐 (活动 / 地点 / 电影)、社交媒体、外卖 / 点餐、
交通 (打车 / 车票 / 机票 / 路况)、天气.
hotel 里不要写「帮我叫车」「推荐附近餐厅」, 那是 MASSIVE 的.

## 字段约束

- `id`: `<domain>_<snake_case>`, 全局唯一
- `domain`: 上表里的 slug
- `description`: 3–12 个英文单词, 全小写, 说用户想要什么, 例如 `change the delivery address of an order`.
  **同一领域的 256 条两两之间都要分得清** —— 给一条用户消息, 人能在整个领域的菜单里认出它归哪条.
  可以相近 (`cancel an order` / `cancel one item in an order`), 不能重复或可互换.
  不同领域之间允许相近, 它们不会出现在同一份菜单上.
- `utterances`: 恰好 3 条, 第一人称的用户消息, 6–30 个单词, 英文. 三种风格各一条, 按这个顺序:
  1. 短、口语 (可以没有标点、小写)
  2. 长, 把情况讲清楚
  3. 间接: 只描述处境, 不直说想要什么, 读者要自己推断意图.
     例如 `move an existing appointment to a different time` →
     `my boss just put a big meeting on thursday morning, right when I was supposed to be at the clinic`

  不要照抄描述里的关键短语. 不要人名、邮箱、电话、地址、订单号 (写 "my order" 就行).
- `question`: 二元题的问句, 规则见「二元题」一节.
- `answers`: 3 个布尔值, 与 `utterances` 逐条对应, 至少一个 `true` 一个 `false`.

每个领域里放几组彼此相近但可区分的意图 (像 Banking77 的 `card lost` / `card stolen` / `card swallowed`),
那是教模型仔细读菜单的东西.

## 生成方式

分两步, 描述定稿之前不写消息:

1. **描述.** 每个领域把业务分成 8 个子业务, 各列 32 条描述, 凑满 256.
   现有的 32 个意图算在里面, 新写 224 个, 要避开和它们撞. 过近义检测 (见下), 改掉太近的.
2. **消息与二元题.** 先定问题, 再按分到的答案模式写消息. 新意图写 3 条.
   现有意图保留原来 2 条 (已是短口语在前、长的在后), 选一个在这两条上答案已确定的问题, 补 1 条间接型凑出 yes 与 no.

每个领域的描述写在 `<domain>.py` (`SUBS` + `INTENTS`), 问题与消息写在 `<domain>_msgs.py`,
装配脚本合成 jsonl 后跑 `python3 datasets/synth-intents/check.py <domain>.jsonl` 过检查.
工作目录、文件格式、每次追加多少 (描述一个子业务 32 条, 消息最多 16 个意图) 和各个脚本,
见生成时的执行指南 `/tmp/madousho-speckit/decidophobia/synth-v2/GUIDE.md`.
生成脚本、质检脚本与结果、各领域的生成笔记都只在那个临时目录里, 没有进仓库.

## 质检

- **近义检测**: 同一领域的描述两两比较, 太近的一对改写或合并. 用远程 embedding 余弦 + reranker 复核,
  只算初筛 —— 用词不同而场景重叠的一对 (例如「通知诊所宠物去世」与「别再发提醒」) 它排不到前面.
- **做题**: 让一个强模型对着该领域完整的 256 项菜单逐条做题 (12288 条消息, 约 3300 万 token 输入).
  做错的消息有歧义, 改消息或改描述, 改完再做一遍.
- **二元题复核**: 强模型只看消息和问句答 yes/no (12288 道, 输入很短). 与 `answers` 不一致的,
  改问句或改消息; 答不出、说要看专业知识或说消息没交代的, 换问题.
- **乱序终验**: 上面的做题按文件顺序编号菜单, 相近意图常排在相邻位置, 训练时菜单却是打乱的.
  所以最后再用两种固定的随机顺序各做一遍菜单题. 两种顺序都选错的消息算真歧义, 要修;
  只错一种的多是位置偏向或判题噪声 (温度 0 下仍有约 0.5–1% 翻转), 不追.

判题模型是 claude-sonnet-5. 最终结果 (2026-09-24):

| | 准确率 |
| --- | --- |
| 菜单题, 文件顺序 | 12194/12288 = 99.2% |
| 菜单题, 随机顺序 1 / 2 | 96.96% / 96.97% |
| 菜单题, 两种随机顺序都错 | 12 条 (0.1%), 原因记在各领域的生成笔记里 |
| 二元题 | 12277/12288 = 99.9% |

判题模型有位置偏向: 错题里多数选的是排在正确项之前、大致沾边的那一项.
这些分数只说明「强模型分得清」; 对小模型来说仍可能有相当一部分题是难题.
