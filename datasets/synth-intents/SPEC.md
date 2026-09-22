# synth-intents: 合成意图数据集

给决策头训练用的第二个意图数据集. 512 个虚构意图, 每个一条自然语言描述 (菜单里显示的就是它)
和 2 条用户消息 (作上下文). 用途: 训练时菜单干扰项从这 512 个里抽, k 可以拉到 256 而每一项都像一条真意图.

**只做训练. 评估仍用真实数据 (Banking77 留出类、BoolQ、MASSIVE).**

## 文件

一个领域一个文件: `datasets/synth-intents/<domain>.jsonl`, 每行一个意图:

```json
{"id": "telecom_report_dropped_calls", "domain": "telecom", "description": "report that calls keep dropping", "utterances": ["My calls have been cutting out mid-conversation all week, what's going on?", "every time I'm on the phone for more than a minute the line just dies"]}
```

16 个领域 × 32 个意图 = 512:

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
  同一领域内两条描述不能可互换 —— 给一条用户消息, 人能分清该归哪条. 可以相近 (`cancel an order` / `cancel one item in an order`), 不能重复.
- `utterances`: 恰好 2 条, 第一人称的用户消息, 6–30 个单词, 英文.
  两条风格要不同: 一条短、口语 (可以没有标点、小写), 一条长、把情况讲清楚.
  不要照抄描述里的关键短语. 不要人名、邮箱、电话、地址、订单号 (写 "my order" 就行).

每个领域里放几组彼此相近但可区分的意图 (像 Banking77 的 `card lost` / `card stolen` / `card swallowed`),
那是教模型仔细读菜单的东西.

## 生成方式

数据写在 Python 文件里 (`/tmp/madousho-speckit/decidophobia/synth/<domain>.py`), 一个 `INTENTS` 列表,
**每次 tool call 追加 8 条**, 写满 32 条后运行它, 用 `json.dumps(..., ensure_ascii=False)` 逐行写到 jsonl.
最后跑 `python3 datasets/synth-intents/check.py <domain>.jsonl` 过检查.
