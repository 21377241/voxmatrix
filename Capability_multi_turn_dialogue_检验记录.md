# Capability：multi_turn_dialogue（多轮对话）

负责人：丙（T4）  
日期：2026-09-03  
模型：Qwen3-Omni-30B-A3B-Instruct（`qwen3-omni-local`）  
Job：12550–12557（v2，170 条全量）；12567（MTalk 修正重跑，10 条）  
产物：`tools/capability_t4_smoke.py`、`tools/run_capability_t4_infer.sh`；`/mnt/afs/users/shizs/capability_t4_run_20260903/{samples.jsonl,preflight.json,audited.jsonl,summary.json,predictions-v2-*.jsonl,predictions-v3-mtalk.jsonl}`

本记录覆盖 AudioAgentBench Suite、MTalk-Bench、VocalBench、VocalBench-zh、VoiceBench 共 50 条。每集固定抽 10 条；多轮历史按原生顺序构造，50/50 prediction，inference error=0。正式 contextual judge/ELO 只在有 assistant reference/rubric 与评测服务时成立。

## 统一 template

- **任务形态**：至少两轮存在跨轮依赖的语音对话；模型须维护实体、目标、约束与当前状态，只回答最新一轮。
- **Prompt 骨架**：`system: Use all preceding dialogue turns and answer the latest spoken turn; return only the response.`；历史用户轮使用 audio，历史 assistant 轮使用原生 assistant response（若数据没有 assistant reference 则不伪造）。
- **选择题是否改为问答**：否。所有本轮格均按开放式多轮回答处理；没有将 VoiceBench/MTalk 的 rubric 改成选择题。
- **多音频 / 多轮约定**：AudioAgentBench 保留目标轮前最多 16 个 user audio + assistant history；MTalk 按 `(number,type)` 分组，仅送 user audio，避免拼接 ambient/paralinguistic/semantic 变体；VocalBench 用文本 context + 目标音频；VoiceBench MTBench 的 `audio1`、`audio2` 两段均实际进入 prompt，native transcripts 只留 metadata。

## 主指标

- **推荐主指标**：首选逐轮 contextual LLM judge（实体/指代/目标完成/连贯性/安全性）及有真实任务后端的 task_success；可复算时报告 token F1/ROUGE-L 作为诊断，不把 lexical overlap 当正式能力分。
- **数据流**：有序 audio/history → Qwen3-Omni → latest response；AudioAgentBench 与 golden_text 做 offline token F1，VocalBench 两集做 ROUGE-L + token F1，MTalk/VoiceBench 交给原生 rubric/judge。
- **算分差异**：AudioAgentBench 是 native-golden text proxy；VocalBench/VocalBench-zh 是 reference-based lexical diagnostic；MTalk 没有 assistant reference，VoiceBench 多数 reference 为空，因此两者正式 judge/ELO 无法从单模型 batch 独立计算。中文 tokenization 还会把同义中文答案打成 0。

| benchmark | 当前 proxy | 10 条汇总 | 可否作为本能力主分 |
|---|---|---:|---|
| audioagentbench_suite | golden_text token F1（offline） | 均值 0.4232；10/10 有值 | 否（无真实运行时） |
| mtalk_bench | 原生 rubric 保留；无 assistant ref | judge/ELO 不可计算；10/10 推理成功 | 本轮不评正式分 |
| vocalbench | ROUGE-L + token F1 | ROUGE-L 0.2851；token F1 0.4106 | 否（diagnostic） |
| vocalbench_zh | ROUGE-L + token F1 | ROUGE-L 0；token F1 近 0 | 否（语言/tokenization 风险） |
| voicebench | 原生 MTBench judge | reference 多数为空；10/10 双音频已送入 | 本轮不评正式分 |

## 各 benchmark 核验

### audioagentbench_suite

- 抽样：total_turns.json 等距抽样 10 个目标轮。
- 当前用的指标及版本：`text_f1`（golden_text offline proxy）
- 当前指标（数值）：offline golden_text token F1 均值 0.4232。
- 人核结论：**方案待定**
- 结果是否可信：修复 scorer 后 10/10 可离线计算；但没有工具执行/状态后端，text F1 只代表词面重叠，不能代替 task_success 或 judge。
- 问题编号：丙-MD-01

**10 条抽样核验（audio 列显示实际送入的音频文件名/数量）**

| # | sample_id | native_id / audio | ref / expected（截断） | pred（截断） | score |
|---:|---|---|---|---|---|
| 0 | t4_multi_turn_dialogue_audioagentbench_suite_cfaf6b312931 | conversation-bench:0 / turn_000.wav | Workshop day is on Tuesday, June 3rd. There will be exclusive, hands-on workshops across five different tracks, with to… | Workshops are scheduled for Saturday, October 14th, from 9:00 AM to 4:00 PM. | textF1=0.1026 |
| 1 | t4_multi_turn_dialogue_audioagentbench_suite_e80dc01a1195 | conversation-bench:24 / 17× turn_008.wav; turn_009.wav; turn_010.wav; turn_011.wa… | There are two speakers named Dr. Liu in the conference program. Dr. Thomas Liu (VP Infrastructure, Anthropic) speaks on… | Dr. Liu's talk on June 4th is titled **"Scaling LLM Inference to Millions of Users"** and it starts at **12:00 PM** in the *… | textF1=0.3421 |
| 2 | t4_multi_turn_dialogue_audioagentbench_suite_9b869ea74a3f | conversation-bench:49 / 17× turn_033.wav; turn_034.wav; turn_035.wav; turn_036.wa… | You reported that you couldn't access the location maps on the mobile app. I submitted a tech support ticket for that i… | You reported that the registration system was returning an unexpected error when trying to register for the 'Fine-tuning Fou… | textF1=0.2553 |
| 3 | t4_multi_turn_dialogue_audioagentbench_suite_ae1db952dbea | conversation-bench:73 / 17× turn_057.wav; turn_058.wav; turn_059.wav; turn_060.wa… | Yes — Harald Kirschner from the VS Code team at Microsoft. He's the only speaker who appears on all three days: he lead… | Yes, there is one speaker presenting on all three days: **Dr. Elena Rodriguez** from Google DeepMind is scheduled for: - Jun… | textF1=0.3802 |
| 4 | t4_multi_turn_dialogue_audioagentbench_suite_4050131d0b89 | assistant-bench:23 / 17× turn_007.wav; turn_008.wav; turn_009.wav; turn_010.wa… | No — your hotel is in New York City, not San Francisco. The Manhattan House in Midtown Manhattan, February 3rd to 5th. | No, the hotel was in New York — The Manhattan House. You're traveling to New York, not San Francisco. | textF1=0.6 |
| 5 | t4_multi_turn_dialogue_audioagentbench_suite_b5903be7deda | product-bench:16 / 17× turn_000.wav; turn_001.wav; turn_002.wav; turn_003.wa… | The S15 Pro is $1,199 new, so with the 10 percent student discount, it would be $1,079.10. That's the most affordable o… | With the student discount, the S15 Pro is $1,199 minus 10 percent — $1,079.10. | textF1=0.6 |
| 6 | t4_multi_turn_dialogue_audioagentbench_suite_602d182b7658 | grocery-bench:10 / 11× turn_000.wav; turn_001.wav; turn_002.wav; turn_003.wa… | Updated to two bottles of Maple Syrup — nineteen ninety-eight for the syrup. | Updated to two bottles of Maple Syrup — nineteen ninety-eight total. | textF1=0.8333 |
| 7 | t4_multi_turn_dialogue_audioagentbench_suite_0bb38e228e6d | appointment-bench:4 / 5× turn_000.wav; turn_001.wav; turn_002.wav; turn_003.wav… | Got it — 9:15 AM with Dr. Perry for your cleaning, and 3:30 PM with Dr. Barry for Danielle's orthodontic consultation. … | Perfect! I've booked your cleaning for Monday, January 13th at 9:15 AM with Dr. Perry, and Danielle's orthodontic consultati… | textF1=0.619 |
| 8 | t4_multi_turn_dialogue_audioagentbench_suite_d0c6d7899101 | event-bench:4 / 5× turn_000.wav; turn_001.wav; turn_002.wav; turn_003.wav… | Got it — Priya Mehta, M-E-H-T-A. And your phone number is five-one-two, eight-four-seven, three-one-six-zero. Is that c… | Thank you, Priya. I've noted your details. I'll go ahead and reserve the Garden Pavilion for your company gala on Saturday, … | textF1=0.1282 |
| 9 | t4_multi_turn_dialogue_audioagentbench_suite_b723ebba6477 | event-bench:28 / 17× turn_012.wav; turn_013.wav; turn_014.wav; turn_015.wa… | You're welcome, Priya! I'll send a confirmation to Raj at 512-849-3016. Your event at the Garden Pavilion on March 15th… | You're welcome! I'm glad we got everything sorted. Your deposit of $6,375 is due by March 8th to secure the booking. If you … | textF1=0.3714 |

### mtalk_bench

- 抽样：parquet 按 type 分层抽样 10 条；native_id 含 number、源行号和 turn。
- 当前用的指标及版本：原生 `llm_judge+elo`（本轮不可计算）
- 当前指标（数值）：正式 LLM judge/ELO：不可计算（10/10 judge_unavailable）。
- 人核结论：**本轮不评**
- 结果是否可信：输入历史音频已送入且无 placeholder 回显；发布数据没有 assistant reference，单模型无法做正式响应质量比较。
- 问题编号：丙-MD-02

**10 条抽样核验（audio 列显示实际送入的音频文件名/数量）**

| # | sample_id | native_id / audio | ref / expected（截断） | pred（截断） | score |
|---:|---|---|---|---|---|
| 0 | t4_multi_turn_dialogue_mtalk_bench_6c2733c5a834 | 1-1:row0:turnturn1 / 0.wav | 无 assistant reference；转写={"transcription": "I want to invite my family to a weekend barbecue. How can I phrase it … | Hey everyone! I was thinking of hosting a weekend barbecue—just a casual get-together with some good food and great company.… | judge unavailable |
| 1 | t4_multi_turn_dialogue_mtalk_bench_9c20d5397d7e | 1-1:row1:turnturn2 / 2× 0.wav; 1.wav | 无 assistant reference；转写={"transcription": "Some are concerned about the weather. How can I address this and reass… | - "Hey everyone, I'm planning a weekend BBQ and would love for you all to come! It's going to be a fun, low-key way to catch… | judge unavailable |
| 2 | t4_multi_turn_dialogue_mtalk_bench_3428b95808d3 | 1-2:row2:turnturn1 / 2.wav | 无 assistant reference；转写={"transcription": "I'm planning to invite my Muslim boyfriend to a family dinner. For the… | Since your Muslim boyfriend follows halal dietary guidelines, you'll need to ensure all ingredients and preparation methods … | judge unavailable |
| 3 | t4_multi_turn_dialogue_mtalk_bench_e1608e143223 | 1-1:row187:turnturn1 / 187.wav | 无 assistant reference；转写={"transcription": "I want to invite my family to a weekend barbecue. How can I phrase it … | Hey everyone! I'm thinking of hosting a casual weekend BBQ—just some good food, drinks, and easy vibes. Would love for you a… | judge unavailable |
| 4 | t4_multi_turn_dialogue_mtalk_bench_ce2d9586abae | 1-1:row188:turnturn2 / 2× 187.wav; 188.wav | 无 assistant reference；转写={"transcription": "*Sigh* Some are concerned about the weather. How can I address this an… | - "Hey everyone, I'm planning a weekend barbecue and would love for you all to come! It's going to be a fun, low-key way to … | judge unavailable |
| 5 | t4_multi_turn_dialogue_mtalk_bench_4d999c02c467 | 1-2:row189:turnturn1 / 189.wav | 无 assistant reference；转写={"transcription": "I'm planning to invite my Muslim boyfriend to a family dinner. For the… | Since your Muslim boyfriend follows halal dietary guidelines, you'll need to ensure all ingredients and preparation methods … | judge unavailable |
| 6 | t4_multi_turn_dialogue_mtalk_bench_9af31e3bf932 | 1-1:row377:turnturn1 / 377.wav | 无 assistant reference；转写={"transcription": "I want to invite my family to a weekend barbecue. How can I phrase it … | Hey everyone! I'm planning a casual weekend barbecue and would love for you all to come. It'll be a great chance to catch up… | judge unavailable |
| 7 | t4_multi_turn_dialogue_mtalk_bench_fe11452a7760 | 1-1:row378:turnturn2 / 2× 377.wav; 378.wav | 无 assistant reference；转写={"transcription": "Some are concerned about the weather. How can I address this and reass… | - "I know the forecast is a bit uncertain, but I've got a backup plan—maybe we can move it to the covered patio or even indo… | judge unavailable |
| 8 | t4_multi_turn_dialogue_mtalk_bench_ad5d23be221c | 1-2:row379:turnturn1 / 379.wav | 无 assistant reference；转写={"transcription": "I'm planning to invite my Muslim boyfriend to a family dinner. For the… | Since your Muslim boyfriend follows halal dietary guidelines, you'll need to ensure all ingredients and preparation methods … | judge unavailable |
| 9 | t4_multi_turn_dialogue_mtalk_bench_55c5d647466d | 1-2:row380:turnturn2 / 2× 379.wav; 380.wav | 无 assistant reference；转写={"transcription": "Please list these dishes according to the ingredients to be prepared, … | Here is a list of the dishes you mentioned, organized by ingredients, seasonings, and estimated baking times. Since your Mus… | judge unavailable |

### vocalbench

- 抽样：multi_round.json 等距抽样 10 条。
- 当前用的指标及版本：`rouge_l+token_f1`（diagnostic）
- 当前指标（数值）：ROUGE-L 均值 0.2851；token F1 均值 0.4106。
- 人核结论：**方案待定**
- 结果是否可信：可作 lexical diagnostic；缺少统一 judge 时，长答案的词面差异会显著影响分数。
- 问题编号：丙-MD-03

**10 条抽样核验（audio 列显示实际送入的音频文件名/数量）**

| # | sample_id | native_id / audio | ref / expected（截断） | pred（截断） | score |
|---:|---|---|---|---|---|
| 0 | t4_multi_turn_dialogue_vocalbench_0fa98b2ebec8 | multi_round-000 / 000.wav | Based on the updated information, if D is higher than A and A is already known to be taller than B and B is taller than… | Now, we know the following: - A > B > C - D > B - E > D - D > A From D > A and A > B > C, we get: D > A > B > C Also, E > D,… | ROUGE-L=0.1942;tokenF1=0.3373 |
| 1 | t4_multi_turn_dialogue_vocalbench_01b19a5640ab | multi_round-044 / 044.wav | Common side effects of Atorvastatin include symptoms like headache, sore throat, runny or stuffy nose, digestive issues… | Common side effects of atorvastatin (Lipitor) may include: - Headache - Muscle pain or weakness - Joint pain - Upset stomach… | ROUGE-L=0.2083;tokenF1=0.4179 |
| 2 | t4_multi_turn_dialogue_vocalbench_293c1b86f1f4 | multi_round-089 / 089.wav | Yes, starting with materials that match your current level of proficiency is important. As you become more comfortable,… | Yes, start with materials that match your current level—like graded readers, children's books, or simple news articles. As y… | ROUGE-L=0.2564;tokenF1=0.3457 |
| 3 | t4_multi_turn_dialogue_vocalbench_8e8ad89179d5 | multi_round-133 / 133.wav | I, Alexander, born in 356 BC, ascended the throne of Macedonia and carved out one of history's grandest empires before … | I am Alexander, son of Philip of Macedon, and I have conquered much of the known world. From a young age, I was trained in w… | ROUGE-L=0.1761;tokenF1=0.325 |
| 4 | t4_multi_turn_dialogue_vocalbench_6ae1954eab94 | multi_round-177 / 177.wav | When a person is described as "hardy," it typically means that the person is robust, healthy, and capable of enduring d… | Yes, when we say a person is "hardy," it often means they are tough, resilient, and able to endure difficult conditions—incl… | ROUGE-L=0.274;tokenF1=0.4027 |
| 5 | t4_multi_turn_dialogue_vocalbench_33dc36bad1d0 | multi_round-222 / 222.wav | Absolutely. Think of synthetic motor oil as the super athlete of engine oils. It keeps your car's engine in top shape, … | Think of synthetic motor oil as a high-performance athlete for your engine. While conventional oil is like a reliable team p… | ROUGE-L=0.2238;tokenF1=0.4156 |
| 6 | t4_multi_turn_dialogue_vocalbench_af991c5b5d6c | multi_round-266 / 266.wav | 5 gallons is about 18.93 liters. | 5 gallons is approximately 18.93 liters. | ROUGE-L=0.8333;tokenF1=0.8571 |
| 7 | t4_multi_turn_dialogue_vocalbench_b481b29edacf | multi_round-310 / 310.wav | Given your high risk tolerance and interest in high returns, you might want to consider investing in equity or stock mu… | If you have a high risk tolerance and are seeking higher returns, you might consider investing in **equity mutual funds**, p… | ROUGE-L=0.2788;tokenF1=0.4368 |
| 8 | t4_multi_turn_dialogue_vocalbench_2e317bbcfe8d | multi_round-355 / 355.wav | Henna designs can be quite intricate. Did this experience make you interested in learning more about other Indian custo… | Yes, the henna ceremony, or Mehndi, is a special pre-wedding ritual. The intricate designs symbolize beauty, joy, and spirit… | ROUGE-L=0.1667;tokenF1=0.2041 |
| 9 | t4_multi_turn_dialogue_vocalbench_fb0199f1088c | multi_round-399 / 399.wav | Certainly, I misspoke earlier. While there are 9 players on the field, a full baseball team roster includes 26 players … | You're right to ask for clarification. On the field during a game, there are 9 players from a baseball team. However, a team… | ROUGE-L=0.2394;tokenF1=0.3636 |

### vocalbench_zh

- 抽样：multi_round.json 等距抽样 10 条。
- 当前用的指标及版本：`rouge_l+token_f1`（diagnostic）
- 当前指标（数值）：ROUGE-L 均值 0；token F1 近 0。
- 人核结论：**有问题**
- 结果是否可信：路径/音频链路正常，但中文 ref 与自然中文表达几乎无 token overlap；0 不能解释为模型不会多轮对话。
- 问题编号：丙-MD-03

**10 条抽样核验（audio 列显示实际送入的音频文件名/数量）**

| # | sample_id | native_id / audio | ref / expected（截断） | pred（截断） | score |
|---:|---|---|---|---|---|
| 0 | t4_multi_turn_dialogue_vocalbench_zh_879160fd679a | multi_round-000 / 000.wav | 基于更新的信息，如果阿阳比小明高，而已知小明比小红高，小红比小亮高，阿紫又比阿阳高，那么阿紫目前是最高的。 | 根据所有信息： - 小明 > 小红 > 小亮 - 阿紫 > 阿阳 > 小红 - 阿阳 > 小明 将这些关系串联起来： 阿紫 > 阿阳 > 小明 > 小红 > 小亮 因此，**阿紫是最高的**。 | ROUGE-L=0.0;tokenF1=0.0 |
| 1 | t4_multi_turn_dialogue_vocalbench_zh_cc7180b3d0e9 | multi_round-044 / 044.wav | 阿托伐他汀的常见副作用包括头痛、喉咙痛、流鼻涕或鼻塞、腹泻、便秘、恶心等消化问题，以及肌肉和关节疼痛。如果您有任何疑虑或出现严重副作用，请咨询您的医疗保健提供者。 | 阿托伐他汀是一种常用的降脂药物，用于降低胆固醇和预防心血管疾病。常见的副作用包括： 1. 肌肉相关症状：如肌肉疼痛、无力或痉挛，严重时可能导致横纹肌溶解（罕见但需警惕）。 2. 肝功能异常：少数患者可能出现肝酶升高，通常在停药后可恢复。 3. 消化… | ROUGE-L=0.0;tokenF1=0.0 |
| 2 | t4_multi_turn_dialogue_vocalbench_zh_884023ed5491 | multi_round-089 / 089.wav | 是的，从与你当前熟练程度相符的材料开始是很重要的。当你变得更加自如时，逐渐增加阅读内容的复杂性。接触各种体裁和主题也会丰富你的词汇量。 | 当然，根据你的语言水平选择合适的材料很重要。初学者可以从简单的儿童读物、分级读物或带有双语对照的文章开始。随着水平提高，可以尝试新闻简报、博客、短篇小说或你感兴趣的领域的文章。关键是选择你感兴趣的内容，这样阅读会更愉快，也更容易坚持。 | ROUGE-L=0.0;tokenF1=0.0 |
| 3 | t4_multi_turn_dialogue_vocalbench_zh_53fe552bf6cb | multi_round-133 / 133.wav | 我，亚历山大，生于公元前356年，登上了马其顿的王位，并在三十岁之前开辟了历史上最伟大的帝国之一。我以在战场上从未尝败绩而闻名，不仅征服了广大的领土，还传播了希腊文化，从而为后来的希腊化世界奠定了基础。 | 当然可以。以下是以亚历山大大帝第一人称视角的叙述： 我是亚历山大，马其顿的国王，宙斯之子，也是希腊世界的征服者。我生于公元前356年，是腓力二世之子。自幼便接受亚里士多德的教导，热爱知识与英雄史诗。我一生追求荣耀与不朽，渴望将希腊文明传播至世界的尽… | ROUGE-L=0.0;tokenF1=0.0 |
| 4 | t4_multi_turn_dialogue_vocalbench_zh_60f46ffefb35 | multi_round-177 / 177.wav | 当形容一个人“生猛”时，并不是指他们像海鲜那样“新鲜度高”。这个词语在形容人时，通常是指这个人性格上比较冲动、做事直接、精力旺盛，甚至有些莽撞或有冲劲。它和新鲜度没有关系，而是用来描述一个人的行为风格或性格特点。 | 不是的，如果说一个人“生猛”，这并不是指他们“新鲜度很高”，而是形容这个人精力充沛、性格刚烈、有冲劲或带点野性。这个词语通常用来描述人的气质或行为特征，而不是像形容海鲜那样强调新鲜程度。 例如： - “他打球生猛，一上来就抢得特别凶。” - “这小… | ROUGE-L=0.0;tokenF1=0.0571 |
| 5 | t4_multi_turn_dialogue_vocalbench_zh_6a71cf950081 | multi_round-222 / 222.wav | 可以把合成机油看作是发动机油中的超级运动员。它让你的汽车发动机保持最佳状态，就像均衡的饮食和定期的锻炼让运动员表现最佳一样。它比传统机油更能长久地抵御发动机的敌人——比如油泥和磨损，而且无论是在寒冷的冬天早晨还是酷热的夏天，它总是准备好跑… | 当然可以。把合成机油比作“发动机的高级营养餐”就再合适不过了。 就像我们吃营养均衡、易吸收的食物能让身体更健康、精力更充沛一样，合成机油为发动机提供了更纯净、更稳定的润滑，能更好地保护内部零件，减少磨损，让发动机在各种环境下都保持最佳状态，延长使用… | ROUGE-L=0.0;tokenF1=0.0 |
| 6 | t4_multi_turn_dialogue_vocalbench_zh_e1384f76ff0f | multi_round-266 / 266.wav | 5加仑约为18.93升。 | 5加仑大约等于18.927升。 | ROUGE-L=0.0;tokenF1=0.0 |
| 7 | t4_multi_turn_dialogue_vocalbench_zh_748c5be6cb99 | multi_round-310 / 310.wav | 鉴于你有较高的风险承受能力和对高回报的兴趣，你可以考虑投资股票或股票型共同基金。这些基金投资于股票，相对于其他类型的共同基金，如固定收益基金，具有更高的风险，但它们也有更高的回报潜力。然而，重要的是要记住，所有投资都带有一定的风险，进行自… | 如果您有较高的风险承受能力并且追求高回报，可以考虑投资股票型共同基金，特别是专注于成长型股票或新兴市场的基金。这些基金主要投资于股票，长期来看可能带来较高的回报，但波动性也更大。 | ROUGE-L=0.0;tokenF1=0.0 |
| 8 | t4_multi_turn_dialogue_vocalbench_zh_9e391acf73e4 | multi_round-355 / 355.wav | 指甲花设计可以非常复杂。这次经历是否让你对了解其他印度习俗产生了兴趣？ | 是的，这种传统被称为“海娜纹身”，在印度婚礼中非常普遍。它不仅是一种装饰，还象征着好运和祝福。你对这种文化细节感兴趣吗？ | ROUGE-L=0.0;tokenF1=0.0 |
| 9 | t4_multi_turn_dialogue_vocalbench_zh_b7a814368a38 | multi_round-399 / 399.wav | 当然，我之前说错了。虽然场上有9名球员，但根据2020赛季的调整，一个完整的棒球队名单包括26名在册球员。这其中包括首发球员和替补球员。 | 在职业棒球比赛中，每支队伍的正式大名单（通常称为“26人名单”或“28人名单”）根据不同的联赛和赛季规则有所不同。例如： - 在美国职业棒球大联盟（MLB），每支球队在赛季期间通常有 **26名球员** 的大名单（2020年起从25人增加到26人，… | ROUGE-L=0.0;tokenF1=0.0 |

### voicebench

- 抽样：MTBench parquet 等距抽样 10 条；每条 audio1+audio2。
- 当前用的指标及版本：原生 MTBench LLM judge（reference 缺失）
- 当前指标（数值）：正式 judge unavailable；reference 多数为空。
- 人核结论：**本轮不评**
- 结果是否可信：已确认双音频都进入 prompt，未把转写文本旁路给模型；没有可用 reference/judge，不能给正式分。
- 问题编号：丙-MD-04

**10 条抽样核验（audio 列显示实际送入的音频文件名/数量）**

| # | sample_id | native_id / audio | ref / expected（截断） | pred（截断） | score |
|---:|---|---|---|---|---|
| 0 | t4_multi_turn_dialogue_voicebench_a67c645493f7 | 81 / 2× 81-audio1.wav; 81-audio2.wav | — | Aloha, adventurers! Awe-inspiring Hawaii awaits all who adore adventure and aloha spirit. Amidst azure oceans and emerald mo… | judge unavailable |
| 1 | t4_multi_turn_dialogue_voicebench_42908ceca791 | 86 / 2× 86-audio1.wav; 86-audio2.wav | — | Bustling crowds weave through narrow alleys, brushing past stalls piled high with colorful wares. Carts creak under the weig… | judge unavailable |
| 2 | t4_multi_turn_dialogue_voicebench_76301639047a | 91 / 2× 91-audio1.wav; 91-audio2.wav | — | Dancing? I'm not really a dancer. I'm more of a... rocket scientist. But hey, if you're into it, go for it! Just don't expec… | judge unavailable |
| 3 | t4_multi_turn_dialogue_voicebench_8bae07ee59dd | 97 / 2× 97-audio1.wav; 97-audio2.wav | — | Great question! Let's break it down in simple terms. **Euclidean Geometry** is the kind of geometry you probably learned in … | judge unavailable |
| 4 | t4_multi_turn_dialogue_voicebench_e90f7de8938d | 102 / 2× 102-audio1.wav; 102-audio2.wav | ["The answer is \"Washington, DC\".", "No."] | No, the original question does not contain any clues to definitively determine the location of the White House. | judge unavailable |
| 5 | t4_multi_turn_dialogue_voicebench_7e1a8e2f55ae | 108 / 2× 108-audio1.wav; 108-audio2.wav | ["Car does not belong because all others are components of a car.", ""] | Sure! The word "car" does not belong with the others because it is the whole vehicle, while "tire," "steering wheel," and "e… | judge unavailable |
| 6 | t4_multi_turn_dialogue_voicebench_705847b2f794 | 144 / 2× 144-audio1.wav; 144-audio2.wav | ["Genetic information flows from DNA to RNA to Protein. Three processes: replication, transcription, and translation. F… | In my previous response, I incorrectly stated that Francis Crick proposed the central dogma in 1958. The correct year is 195… | judge unavailable |
| 7 | t4_multi_turn_dialogue_voicebench_5a021a4e363e | 149 / 2× 149-audio1.wav; 149-audio2.wav | — | Yes, in theory, you could use supervised learning to solve problems typically addressed by reinforcement learning, but it wo… | judge unavailable |
| 8 | t4_multi_turn_dialogue_voicebench_2b68f22b4491 | 154 / 2× 154-audio1.wav; 154-audio2.wav | — | **Day 1: Introduction to the Opium Wars and Character Exploration** **Objective:** Students will gain foundational knowledge… | judge unavailable |
| 9 | t4_multi_turn_dialogue_voicebench_26629bb66799 | 160 / 2× 160-audio1.wav; 160-audio2.wav | — | Title: "Rising from the Ashes" Pitch: "Rising from the Ashes" is a powerful, character-driven documentary that follows the j… | judge unavailable |

## 问题

### 问题 丙-MD-01

- benchmark：audioagentbench_suite
- 发现时间：2026-09-03

样例
- sample_id：t4_multi_turn_dialogue_audioagentbench_suite_cfaf6b312931
- 音频如何定位：/mnt/afs/eval_data/benchmarks/AudioAgentBench/conversation-bench/audio/turn_000.wav
- ref：Workshop day is on Tuesday, June 3rd. There will be exclusive, hands-on workshops across five different tracks, with to…
- pred：Workshops are scheduled for Saturday, October 14th, from 9:00 AM to 4:00 PM.
- 当前指标：textF1=0.1026
- 必要摘录：native_id=conversation-bench:0

原因
AudioAgentBench 目标轮有 golden_text，但 native task_success 还依赖工具/状态执行；词面 F1 只能反映回答与 reference 的重叠，无法检查实体、日期、任务状态是否真正正确。

解决方案
- 建议动作：保留 golden_text 作为离线诊断，同时接沙箱工具和 contextual judge；按跨轮依赖类型分层报告，不用 .4232 作为统一主分。
- 若涉及数据：不修改原始数据；仅提出字段、ontology、rubric 或评测实现方案。
- 方案是否确定：方案待定（需运行时与 judge）

### 问题 丙-MD-02

- benchmark：mtalk_bench
- 发现时间：2026-09-03

样例
- sample_id：t4_multi_turn_dialogue_mtalk_bench_6c2733c5a834
- 音频如何定位：/mnt/afs/users/shizs/capability_t4_run_20260903/audio/mtalk/0.wav
- ref：无 assistant reference；转写={"transcription": "I want to invite my family to a weekend barbecue. How can I phrase it …
- pred：Hey everyone! I was thinking of hosting a weekend barbecue—just a casual get-together with some good food and great company. Would love for you all to come hang out and enjoy the day. Let me know if you can make it!
- 当前指标：judge unavailable
- 必要摘录：native_id=1-1:row0:turnturn1

原因
MTalk-Bench parquet 原生只有 user audio/transcription 与 rubric，没有 assistant response reference；若填占位 assistant 历史会污染模型输入并可能回显，因此本轮明确不填。

解决方案
- 建议动作：补充可核验的 assistant reference 或运行原生 pairwise judge；在数据补齐前只报告输入构造/推理成功，不计算单模型 ELO。
- 若涉及数据：不修改原始数据；仅提出字段、ontology、rubric 或评测实现方案。
- 方案是否确定：确定（本轮不评正式分）

### 问题 丙-MD-03

- benchmark：vocalbench / vocalbench_zh
- 发现时间：2026-09-03

样例
- sample_id：t4_multi_turn_dialogue_vocalbench_af991c5b5d6c
- 音频如何定位：/mnt/afs/eval_data/99_auxiliary_or_out_of_scope/VocalBench/audio/multi_round/266.wav
- ref：5 gallons is about 18.93 liters.
- pred：5 gallons is approximately 18.93 liters.
- 当前指标：ROUGE-L=0.8333;tokenF1=0.8571
- 必要摘录：native_id=multi_round-266

- sample_id：t4_multi_turn_dialogue_vocalbench_zh_879160fd679a
- 音频如何定位：/mnt/afs/eval_data/99_auxiliary_or_out_of_scope/VocalBench-zh/audio/multi_round/000.wav
- ref：基于更新的信息，如果阿阳比小明高，而已知小明比小红高，小红比小亮高，阿紫又比阿阳高，那么阿紫目前是最高的。
- pred：根据所有信息： - 小明 > 小红 > 小亮 - 阿紫 > 阿阳 > 小红 - 阿阳 > 小明 将这些关系串联起来： 阿紫 > 阿阳 > 小明 > 小红 > 小亮 因此，**阿紫是最高的**。
- 当前指标：ROUGE-L=0.0;tokenF1=0.0
- 必要摘录：native_id=multi_round-000

原因
多轮长答案存在大量合理改写；当前 ROUGE/token F1 对中文按空格 token，中文自然答案与 ref 无空格重叠，VocalBench-zh 整批近 0。词面指标把语义正确打成低分。

解决方案
- 建议动作：采用同语言、多 reference 的 judge；中文至少使用字/词规范化与语义评分，ROUGE/token F1 仅作辅诊断。
- 若涉及数据：不修改原始数据；仅提出字段、ontology、rubric 或评测实现方案。
- 方案是否确定：方案待定（需统一 tokenizer 与 judge）

### 问题 丙-MD-04

- benchmark：voicebench / mtbench
- 发现时间：2026-09-03

样例
- sample_id：t4_multi_turn_dialogue_voicebench_a67c645493f7
- 音频如何定位：/mnt/afs/users/shizs/capability_t4_run_20260903/audio/voicebench_mtbench/81-audio1.wav；/mnt/afs/users/shizs/capability_t4_run_20260903/audio/voicebench_mtbench/81-audio2.wav
- ref：—
- pred：Aloha, adventurers! Awe-inspiring Hawaii awaits all who adore adventure and aloha spirit. Amidst azure oceans and emerald mountains, I unearthed authentic cultural gems and awe-inspiring attractions. Arriving on Oahu, I attended a traditional luau, where islanders shared ancient chants and danced to rhythmic drums. Awe-struck, I watched fire knife dancers, their movements as fluid as the Pacific waves. Afterward, I …
- 当前指标：judge unavailable
- 必要摘录：native_id=81

原因
VoiceBench MTBench 每条确有 `audio1`、`audio2`，但 reference 多数为 null/空列表；没有 reference 和官方 judge，不能从当前 batch 推出质量/ELO。

解决方案
- 建议动作：保留双音频顺序和 prompt hash；补齐官方 judge/reference 后再评。不得用模型自答或转写文本替代 reference。
- 若涉及数据：不修改原始数据；仅提出字段、ontology、rubric 或评测实现方案。
- 方案是否确定：确定（本轮不评）

## 复现与边界

- `python3 -m py_compile tools/capability_t4_smoke.py` 已通过；AudioAgentBench 的 `golden_text` fallback 已用于本轮离线 text F1。
- MTalk 修正结果来自 `predictions-v3-mtalk.jsonl`（Job 12567）；其余组合使用 v2 shards。
- 未修改 MTalk/VocalBench/VoiceBench 原始数据或正式 manifest。
