# Capability：clarification（澄清追问）

负责人：丙（T4）  
日期：2026-09-03（初检）；2026-09-10（生产链路复验）
模型：Qwen3-Omni-30B-A3B-Instruct（`qwen3-omni-local`）  
Job：12550–12557（v2，170 条全量）；12567（MTalk 修正重跑，10 条）  
产物：`tools/capability_t4_smoke.py`、`tools/run_capability_t4_infer.sh`；`/mnt/afs/users/shizs/capability_t4_run_20260903/{samples.jsonl,preflight.json,audited.jsonl,summary.json,predictions-v2-*.jsonl,predictions-v3-mtalk.jsonl}`

本记录覆盖 AudioAgentBench Suite 与 EAR WDYL 共 20 条。目标是判断信息缺失/歧义时是否追问，而不是把所有噪声重复请求都算业务澄清。20/20 音频存在、prediction 完整、inference error=0；当前数值均为离线 decision proxy。

## 2026-09-10 生产链路复验

- 真实 adapter smoke：2 个格各加载 10 条，共 20 条；20/20 V2 schema、音频、Prompt 与 evaluator route 通过。AudioAgentBench 不再从问号启发式造标签，而是只读仓库内 `mesh_eval/data/audioagentbench_clarification_review.v1.json` 的 28 条人工复核记录（10 正、18 负）；每个语义轮标签覆盖两位录音说话人，dialogue view 同时排除这些样本。
- 审核标签不写回原始 benchmark；`input_text`/`golden_text` 的 SHA256 与当前 native metadata 绑定。缺标签、schema 非法、正负类不全或 native 内容漂移均明确失败，不再静默产出空 clarification view。复核中发现外部临时 sidecar 的 `grocery-bench:5` 仍引用旧 golden；当前 native golden 为 “How many dozens would you like?”，故按缺数量改为正例并固定 digest。
- 统一输出协议改为显式 JSON：`{"should_clarify": boolean, "response": string}`。自然语言里出现问号、`which` 或“请问”不再自动算作澄清决策。
- 2026-09-03 的 20 条旧预测都是自然语言，修复后回放均 `clarification_parse_valid=0`，因此 AudioAgentBench 与 EAR 的 `clarification_accuracy/P/R/F1` 均为 `0.0000`；正例上的非法输出分别贡献 FN=4 和 FN=5，而不是从分母消失。
- P/R/F1 由 TP/FP/FN 充分统计量聚合；零预测正例时严格返回 0。回放证据与 SHA256 同 instruction_following 记录。

指标合理性结论：旧 `decision=8/10` 与 `5/10` 来自问号词法猜测，已经作废。新的 0 分反映旧模型输出没有遵循显式决策协议，并不声称追问文本质量为 0。丙-CL-01 的人工正负标签、prompt、解析、漂移校验与聚合链已解决；这些标签仅作为 `diagnostic_evidence`，不冒充 AudioAgentBench 官方标签。丙-CL-02 的 EAR `answerable→should_clarify` 可作为决策 proxy，但追问内容质量仍因缺官方 question/gold/rubric 而不可评。

> 下文逐条 `ask=` 与 text-F1 是 2026-09-03 历史启发式证据，仅用于说明当时为什么会虚高。

## 统一 template

- **任务形态**：音频请求 + 可用历史上下文 → 若关键实体/参数缺失或有歧义，提出一个最小、定向的问题；若信息充分，直接回答/执行，不无谓追问。
- **Prompt 骨架**：`system: If a required entity is ambiguous or missing, ask one targeted clarification before acting; otherwise answer directly without guessing.`；`user: audio/history`。
- **选择题是否改为问答**：否。本轮没有 MCQ；追问必须是自然语言问题，不用 A/B/C/D 代替。
- **多音频 / 多轮约定**：AudioAgentBench 保留目标轮前历史音频和 assistant history；EAR 为单段含/不含关键字的噪声音频。模型输入不附加 native transcript。

## 主指标（2026-09-03 初检口径）

- **推荐主指标**：`clarification_decision`（should-ask precision/recall/F1，按真正歧义与可回答分层）+ 追问质量/答案正确性的 contextual judge。
- **数据流**：音频/历史 → pred；用标注的 should_clarify 与 pred 中是否提出定向问题对齐，得到 binary decision；有 golden answer 时另算 text F1。
- **算分差异**：AudioAgentBench 的 should_clarify 是从 native golden_text 中问句启发式抽出的，并混有工具/可直接回答轮；EAR 只有 `answerable`、`masked_word` 等字段，缺少官方问题、gold answer 和 judge。因此两集 0.8/0.5 不能合并成正式能力分。

| benchmark | 当前 proxy | 10 条汇总 | 可否作为本能力主分 |
|---|---|---:|---|
| audioagentbench_suite | native-golden decision + text F1 | decision 8/10；text F1 均值 0.4687 | 否（标签混合/离线） |
| ear_wdyl | answerable→should-ask decision | decision 5/10；无 text F1 | 否（官方字段不完整） |

## 各 benchmark 核验（2026-09-03 样本明细）

### audioagentbench_suite

- 抽样：clarification_turns.json 等距抽样 10 条。
- 当前用的指标及版本：`clarification_decision+text_f1`（offline native-golden proxy）
- 当前指标（数值）：decision 8/10；text F1 均值 0.4687。
- 人核结论：**方案待定**
- 结果是否可信：能检查是否问句，但 native view 混有真正歧义与 answerable/tool turns；需显式标签与 judge。
- 问题编号：丙-CL-01

**10 条抽样核验**

| # | sample_id | native_id / audio | ref / expected（截断） | pred（截断） | score |
|---:|---|---|---|---|---|
| 0 | t4_clarification_audioagentbench_suite_2caf2a6593d3 | conversation-bench:20 / 17× turn_004.wav; turn_005.wav; turn_006.wav; turn_007.wa… | {"gold": "Of course. Harald Kirschner is involved in three sessions. Could you please tell me which one you'd like to v… | Thank you, Jennifer. I'll look up the session by Harold Kirshner. Could you please confirm the date of the talk you'd like t… | ask=1/1;correct=1;F1=0.2718 |
| 1 | t4_clarification_audioagentbench_suite_6bb739b3899a | conversation-bench:32 / 17× turn_016.wav; turn_017.wav; turn_018.wav; turn_019.wa… | {"gold": "There are three speakers named Alex at the conference:\n\n1. Alex Volkov (AI Evangelist) — MCP observability … | There are two speakers named Alex at the conference: 1. Alex Chen from Google, presenting on June 4th at 1:00 PM in the LLM … | ask=0/0;correct=1;F1=0.4124 |
| 2 | t4_clarification_audioagentbench_suite_c153be481c8a | conversation-bench:43 / 17× turn_027.wav; turn_028.wav; turn_029.wav; turn_030.wa… | {"gold": "You're registered for Kevin Zhang's 'Red Teaming AI Systems: Lessons from the Field' session, which is in the… | I've canceled your registration for the Security track session. You'll no longer receive updates or reminders for that track. | ask=1/0;correct=0;F1=0.2759 |
| 3 | t4_clarification_audioagentbench_suite_efabccc0861d | assistant-bench:5 / 6× turn_000.wav; turn_001.wav; turn_002.wav; turn_003.wav… | {"gold": "Right — the email to Alex Reed about the meeting agenda. What would you like the email to say?", "input": "Gr… | Got it. What's the subject of the email to Alex, and what should the agenda include? | ask=1/1;correct=1;F1=0.4571 |
| 4 | t4_clarification_audioagentbench_suite_20b0650e238c | assistant-bench:26 / 17× turn_010.wav; turn_011.wav; turn_012.wav; turn_013.wa… | {"gold": "Priya Kapoor's email is priya.kapoor@meridian.com.", "input": "What was her email address again — I want to w… | Priya Kapoor's email is priya.kapoor@meridian.com. | ask=0/0;correct=1;F1=1.0 |
| 5 | t4_clarification_audioagentbench_suite_70d43fdde441 | grocery-bench:2 / 3× turn_000.wav; turn_001.wav; turn_002.wav | {"gold": "All-Purpose Flour, five-pound bag for six ninety-nine. Added to your order.", "input": "Yes. I need flour for… | Understood. I'll add a 5-pound bag of flour to your order for delivery. Is there anything else you'd like to include? | ask=0/1;correct=0;F1=0.3684 |
| 6 | t4_clarification_audioagentbench_suite_13db527c6944 | grocery-bench:9 / 10× turn_000.wav; turn_001.wav; turn_002.wav; turn_003.wa… | {"gold": "Maple Syrup, Grade A, twelve-ounce bottle for nine ninety-nine. Added.", "input": "OK good. Now I need maple … | Maple Syrup, one pint for five ninety-nine. Adding that to your order. | ask=0/0;correct=1;F1=0.4 |
| 7 | t4_clarification_audioagentbench_suite_d4e0987c93e8 | grocery-bench:14 / 15× turn_000.wav; turn_001.wav; turn_002.wav; turn_003.wa… | {"gold": "Kalamata Olives, ten-ounce jar for eight forty-nine. Added — that's separate from the olive oil.", "input": "… | Kalamata Olives, twelve-ounce jar for seven ninety-nine. Added. | ask=0/0;correct=1;F1=0.5185 |
| 8 | t4_clarification_audioagentbench_suite_2733806b7e7a | grocery-bench:20 / 17× turn_004.wav; turn_005.wav; turn_006.wav; turn_007.wa… | {"gold": "Organic Bananas, ninety-nine cents per pound. Fifteen pounds comes to fourteen eighty-five. Added. Your runni… | Organic Bananas, one pound for one forty-nine. You said fifteen — that's fifteen pounds, so total is twenty-two thirty-five.… | ask=0/0;correct=1;F1=0.459 |
| 9 | t4_clarification_audioagentbench_suite_19764065791b | event-bench:8 / 9× turn_000.wav; turn_001.wav; turn_002.wav; turn_003.wav… | {"gold": "We have two music options — a professional DJ for $800 which includes a full sound system, lighting effects, … | Great — let's add those. For music, we have a DJ package for $1,200, which includes a full sound system, lighting, and a 6-h… | ask=1/1;correct=1;F1=0.5238 |

### ear_wdyl

- 抽样：ear_wdyl manifest 按 condition 分层抽样 10 条（kw 5、nokw 5）。
- 当前用的指标及版本：`clarification_decision`（answerable proxy；方案待定）
- 当前指标（数值）：decision 5/10；官方问题/gold/judge 缺失。
- 人核结论：**方案待定**
- 结果是否可信：当前只验证 masked keyword 是否可用的离线 proxy；不能评价追问内容是否有效。
- 问题编号：丙-CL-02

**10 条抽样核验**

| # | sample_id | native_id / audio | ref / expected（截断） | pred（截断） | score |
|---:|---|---|---|---|---|
| 0 | t4_clarification_ear_wdyl_50aeced37dd9 | ear_wdyl_kw_0 / 0_Chicago.wav | {"answerable": false, "masked_word": "Chicago"} | I like Chicago the most. | ask=1/0;correct=0;F1=— |
| 1 | t4_clarification_ear_wdyl_fc7c9bfdf65f | ear_wdyl_kw_100 / 100_Seattle.wav | {"answerable": false, "masked_word": "Seattle"} | Your favorite city is Miami. My favorite city is Seattle. | ask=1/0;correct=0;F1=— |
| 2 | t4_clarification_ear_wdyl_f11ecc753a7b | ear_wdyl_kw_101 / 101_pasta.wav | {"answerable": false, "masked_word": "pasta"} | Your favorite food is ramen. My favorite food is pizza. | ask=1/0;correct=0;F1=— |
| 3 | t4_clarification_ear_wdyl_4f36fb81ee67 | ear_wdyl_kw_102 / 102_milk.wav | {"answerable": false, "masked_word": "milk"} | Coffee is my favorite beverage. | ask=1/0;correct=0;F1=— |
| 4 | t4_clarification_ear_wdyl_981174173368 | ear_wdyl_kw_103 / 103_Boston.wav | {"answerable": false, "masked_word": "Boston"} | Your favorite city is Miami. | ask=1/0;correct=0;F1=— |
| 5 | t4_clarification_ear_wdyl_39c31149c889 | ear_wdyl_nokw_0 / 0_really.wav | {"answerable": true, "masked_word": "really"} | I like Chicago the most. | ask=0/0;correct=1;F1=— |
| 6 | t4_clarification_ear_wdyl_ebe969235c36 | ear_wdyl_nokw_100 / 100_your.wav | {"answerable": true, "masked_word": "your"} | My favorite city is Seattle. | ask=0/0;correct=1;F1=— |
| 7 | t4_clarification_ear_wdyl_d0f2bff4d717 | ear_wdyl_nokw_101 / 101_your.wav | {"answerable": true, "masked_word": "your"} | My favorite food is pasta. | ask=0/0;correct=1;F1=— |
| 8 | t4_clarification_ear_wdyl_dda4fa102ad9 | ear_wdyl_nokw_102 / 102_is.wav | {"answerable": true, "masked_word": "is"} | Coffee is my favorite beverage. | ask=0/0;correct=1;F1=— |
| 9 | t4_clarification_ear_wdyl_ae2535ae029f | ear_wdyl_nokw_103 / 103_your.wav | {"answerable": true, "masked_word": "your"} | My favorite city is Miami. | ask=0/0;correct=1;F1=— |

## 问题

### 问题 丙-CL-01

- benchmark：audioagentbench_suite
- 发现时间：2026-09-03

样例
- sample_id：t4_clarification_audioagentbench_suite_c153be481c8a
- 音频如何定位：/mnt/afs/eval_data/benchmarks/AudioAgentBench/conversation-bench/audio/turn_027.wav；/mnt/afs/eval_data/benchmarks/AudioAgentBench/conversation-bench/audio/turn_028.wav；/mnt/afs/eval_data/benchmarks/AudioAgentBench/conversation-bench/audio/turn_029.wav；…共 17 个音频（完整列表见 samples.jsonl）
- ref：{"gold": "You're registered for Kevin Zhang's 'Red Teaming AI Systems: Lessons from the Field' session, which is in the…
- pred：I've canceled your registration for the Security track session. You'll no longer receive updates or reminders for that track.
- 当前指标：ask=1/0;correct=0;F1=0.2759
- 必要摘录：native_id=conversation-bench:43

- sample_id：t4_clarification_audioagentbench_suite_70d43fdde441
- 音频如何定位：/mnt/afs/eval_data/benchmarks/AudioAgentBench/grocery-bench/audio/turn_000.wav；/mnt/afs/eval_data/benchmarks/AudioAgentBench/grocery-bench/audio/turn_001.wav；/mnt/afs/eval_data/benchmarks/AudioAgentBench/grocery-bench/audio/turn_002.wav
- ref：{"gold": "All-Purpose Flour, five-pound bag for six ninety-nine. Added to your order.", "input": "Yes. I need flour for…
- pred：Understood. I'll add a 5-pound bag of flour to your order for delivery. Is there anything else you'd like to include?
- 当前指标：ask=0/1;correct=0;F1=0.3684
- 必要摘录：native_id=grocery-bench:2

原因
clarification view 同时包含 ambiguous_entity、取消/工具执行和可直接回答轮。当前从 golden_text 是否含问句推 should_clarify，无法区分“应澄清哪一个实体”和“应直接执行”；样本 2 的取消请求被模型直接处理而 proxy 判错，样本 5 的可回答 flour 请求被模型追加问句而判错。

解决方案
- 建议动作：评测侧已增加显式 `should_clarify` 人工审核资产，按业务澄清与可直接回答的困难负例分轨；二分类决策使用严格 JSON 与 TP/FP/TN/FN。后续如需评“问题是否最小且可行动”，仍须增加有效问题 rubric 和 judge。
- 若涉及数据：未修改原始数据或正式 manifest；审核资产只保存 turn key、诊断标签及 native 文本 digest，且在加载时校验漂移。
- 方案是否确定：**评测链路已解决**；审核标签仍标记为 diagnostic evidence，待 AudioAgentBench 标注方确认后方可升级为官方指标。

### 问题 丙-CL-02

- benchmark：ear_wdyl
- 发现时间：2026-09-03

样例
- sample_id：t4_clarification_ear_wdyl_50aeced37dd9
- 音频如何定位：/mnt/afs/eval_data/benchmarks/EAR/raw/wdyl_kw_whitenoise/0_Chicago.wav
- ref：{"answerable": false, "masked_word": "Chicago"}
- pred：I like Chicago the most.
- 当前指标：ask=1/0;correct=0;F1=—
- 必要摘录：native_id=ear_wdyl_kw_0

- sample_id：t4_clarification_ear_wdyl_39c31149c889
- 音频如何定位：/mnt/afs/eval_data/benchmarks/EAR/raw/wdyl_nokw_whitenoise/0_really.wav
- ref：{"answerable": true, "masked_word": "really"}
- pred：I like Chicago the most.
- 当前指标：ask=0/0;correct=1;F1=—
- 必要摘录：native_id=ear_wdyl_nokw_0

原因
EAR 当前 manifest 只有 condition、masked_word、answerable 等字段，没有官方问题文本、gold answer、追问质量 rubric 或 judge；`kw`/`nokw` 只能说明关键字是否被噪声遮蔽，不能说明模型是否问对了问题。

解决方案
- 建议动作：补充原生 question、gold answer、should_clarify 与有效追问 rubric；若无法补齐，正式报告标方案待定/本轮不评，只保留可复现的 answerable proxy。
- 若涉及数据：不修改原始数据；仅提出字段、ontology、rubric 或评测实现方案。
- 方案是否确定：方案待定（缺官方标注与评测方案）

## 复现与边界

- 本 capability 20/20 prediction、error=0；所有音频路径由 `samples.jsonl` 预检。
- 代理规则只用于发现口径问题，不修改 EAR/AudioAgentBench 原始字段。
- AudioAgentBench 审核资产缺失或与 native metadata 不一致时 fail closed；不会回退到问号启发式。
