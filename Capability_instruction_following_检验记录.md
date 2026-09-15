# Capability：instruction_following（指令遵循）

负责人：丙（T4）  
日期：2026-09-03（初检）；2026-09-10（生产链路复验）
模型：Qwen3-Omni-30B-A3B-Instruct（`qwen3-omni-local`）  
Job：12550–12557（v2，170 条全量）；12567（MTalk 修正重跑，10 条）  
产物：`tools/capability_t4_smoke.py`、`tools/run_capability_t4_infer.sh`；`/mnt/afs/users/shizs/capability_t4_run_20260903/{samples.jsonl,preflight.json,audited.jsonl,summary.json,predictions-v2-*.jsonl,predictions-v3-mtalk.jsonl}`

本记录按任务说明 §4 只覆盖 fluent_speech_commands、SLURP、VocalBench、VocalBench-zh、VoiceBench。每集固定抽 10 条（共 50 条），音频预检 50/50 存在；最终 prediction 50/50，inference error=0。VocalBench/VoiceBench 的本地分数是诊断值，不能与正式 LLM judge 分数混合。

## 2026-09-10 生产链路复验

- 真实 adapter smoke：上述 5 个格各从原生 test 标注只读加载 10 条，共 50 条；50/50 为 `runtime-sample/2.0`，音频存在、schema 校验通过、Prompt 可渲染，且 metrics/evaluators 与 benchmark map 完全一致。
- Qwen3-Omni 预测复用：只回放 2026-09-03 已产生的 50 条预测，不重新生成或修改 benchmark 数据。回放使用生产 evaluator，不再使用 `tools/capability_t4_smoke.py` 的旧启发式 scorer。
- 证据：`/tmp/voxmatrix-t4-production-replay-20260910-final.OKSYj9/{audited.jsonl,summary.json}`；`summary.json` SHA256 为 `b68c92b07e0f201e755e4e84e636b0dca76111473cb1fa797ca8832568026545`。临时目录不是正式数据资产，复现入口为 `tests/test_t4_agent.py` 和 `tools/capability_t4_smoke.py evaluate`。

| benchmark | 修复后生产指标（10 条旧预测回放） | 结论 |
|---|---:|---|
| fluent_speech_commands | `intent_acc=0.0000`；字段级 `slot_f1=0.1700` | 可复算的 SLU diagnostic；不是通用 IF 主分 |
| slurp | `intent_acc=0.0000`；字段级 `slot_f1=0.2000` | 可复算的 SLU diagnostic；不是通用 IF 主分 |
| vocalbench | `constraint_satisfaction=not_evaluated`（4 条缺生成音频；6 条缺 rubric judge） | 不再输出局部启发式伪分 |
| vocalbench_zh | `constraint_satisfaction=not_evaluated`（4 条缺生成音频；6 条缺 rubric judge） | 不再输出局部启发式伪分 |
| voicebench | canonical IFEval strict prompt pass rate `0.6000`（10/10） | 使用完整 `reference_obj` 与官方 checker；旧 `0.7000` 作废 |

指标合理性结论：FSC/SLURP 的低分来自模型没有遵循注入的 native ontology，字段级 F1 不再对序列化 JSON 做 token overlap；VoiceBench 的 0.6 在 `[0,1]` 且逐条 strict checker 可追溯；VocalBench 的语气、语速、语音复述及语义约束在缺音频或 judge 时没有数值。丙-IF-01 的 ontology/prompt/字段级 scorer 已解决；丙-IF-02/03 涉及生成音频、audio-capable judge 或 gold 规范，链路已 fail-closed，外部条件本轮仍未满足；丙-IF-04 的 canonical IFEval 读取与 parquet 数值类型问题已解决。

> 下文逐条表格保留 2026-09-03 初检证据，其中“当前指标”均是历史启发式值，仅用于解释问题样例；不得覆盖上表的修复后结果或作为正式榜分。

## 统一 template

- **任务形态**：单段用户语音指令 → 只输出请求的答案或结构化结果。能力层面要求同时覆盖内容、格式、长度、禁止项、语气及步骤；当任务是工具选择/参数生成时转入 `tool_call`。
- **Prompt 骨架**：`system: Listen to the audio instruction and satisfy every explicit content/format/length constraint; return only the requested answer.`；`user: audio`。FSC/SLURP 为同一外壳下的 JSON SLU diagnostic（`intent`+`slots`），VocalBench/VocalBench-zh/VoiceBench 保持自然语言答案。
- **选择题是否改为问答**：否。本轮 5 个抽样格均为开放回答或结构化 JSON，没有把官方选择题改作主分；若后续接入 MCQ，只保留 diagnostic。
- **多音频 / 多轮约定**：无；每条仅一个用户音频，历史对话不进入本 capability。

## 主指标（2026-09-03 初检口径）

- **推荐主指标**：VocalBench、VocalBench-zh、VoiceBench 应使用逐条 rubric/LLM judge 的 constraint-pass rate（需同时检查语义、格式、语气、语速和多步完整性）；FSC/SLURP 的 intent accuracy + schema-aware slot F1 只能作为 SLU diagnostic。
- **数据流**：音频 + prompt → Qwen3-Omni → pred；结构化轨先解析 JSON，再将 intent 与 gold exact-match、slots 做字段级 F1；开放轨将 pred 送入 checker/judge，与问题约束及 reference/rubric 对齐后聚合。
- **算分差异**：FSC/SLURP 都是本地 `intent_accuracy+slot_f1`，但 native ontology 不同；VocalBench 两集是本地 `constraint_satisfaction`（只对可执行文本约束计分）；VoiceBench 是本地 IFEval-compatible 子集 checker；三类不可横向平均。推荐正式榜使用各集原生 judge/明确 rubric，保留本轮 proxy 并标注 `diagnostic`。

| benchmark | 当前 proxy | 10 条汇总 | 可否作为本能力主分 |
|---|---|---:|---|
| fluent_speech_commands | intent accuracy + slot F1 | intent 0/10；slot F1 均值 0.275；组合均值 0.1375 | 否（SLU ontology diagnostic） |
| slurp | intent accuracy + slot F1 | intent 0/10；slot F1 均值 0.4167；组合均值 0.2083 | 否（SLU ontology diagnostic） |
| vocalbench | local constraint checks | 8 条有数值；均值 0.9375；2 条仅 unsupported | 否（缺官方 judge） |
| vocalbench_zh | local constraint checks | 8 条有数值；均值 0.625；2 条仅 unsupported | 否（缺官方 judge） |
| voicebench | partial IFEval checker | 均值 0.7000（10/10 有数值） | 否（仅 checker 子集） |

## 各 benchmark 核验（2026-09-03 样本明细）

### fluent_speech_commands

- 抽样：test 等距抽样 10 条；`intent_accuracy+slot_f1`（JSON 解析，字段级 F1）。
- 当前用的指标及版本：intent_accuracy+slot_f1（诊断 proxy）
- 当前指标（数值）：intent 0/10；slot F1 均值 0.275；组合均值 0.1375。
- 人核结论：**有问题**
- 结果是否可信：音频和 JSON 链路可复现，但 native gold（如 `activate_lights`）与模型自然 ontology（如 `turn_on`）不一致；组合分数不能代表一般指令遵循。
- 问题编号：丙-IF-01

**10 条抽样核验（音频路径均由 `samples.jsonl` 定位；表中 audio 为文件名）**

| # | sample_id | native_id / audio | ref / expected（截断） | pred（截断） | score |
|---:|---|---|---|---|---|
| 0 | t4_instruction_following_fluent_speech_commands_340f2dbe94ec | wavs/speakers/4BrX8aDqK2cLZRYl/cbdf5700-452c-11e9-b1e4-e5… / cbdf5700-452c-11e9-b1e4-e5985dca719e.wav | Turn on the lights；gold={"intent": "activate_lights", "slots": {"action": "activate", "location": "none", "object… | {"intent": "turn_on", "slots": {"device": "lights"}} | I=0;slotF1=0.25;v=0.125 |
| 1 | t4_instruction_following_fluent_speech_commands_c3265ae60b9b | wavs/speakers/4BrX8aDqK2cLZRYl/8e92af90-4532-11e9-b1e4-e5… / 8e92af90-4532-11e9-b1e4-e5985dca719e.wav | Decrease the temperature in the washroom；gold={"intent": "decrease_heat", "slots": {"action": "decrease", "location": "washr… | {"intent": "adjust_temperature", "slots": {"location": "washroom", "temperature_direction": "decrease"}} | I=0;slotF1=0.6;v=0.3 |
| 2 | t4_instruction_following_fluent_speech_commands_7a4e87242cdd | wavs/speakers/7B4XmNppyrCK977p/27bc1d50-45ce-11e9-9992-df… / 27bc1d50-45ce-11e9-9992-dfba019cb646.wav | That’s too loud；gold={"intent": "decrease_volume", "slots": {"action": "decrease", "location": "none", "object… | {"intent": "adjust_volume", "slots": {"volume_level": "too loud"}} | I=0;slotF1=0.0;v=0.0 |
| 3 | t4_instruction_following_fluent_speech_commands_ab4a41784029 | wavs/speakers/k5bqyxx2lzIbrlg9/12b65b60-4528-11e9-a843-8d… / 12b65b60-4528-11e9-a843-8db76f4b5e29.wav | Resume music；gold={"intent": "activate_music", "slots": {"action": "activate", "location": "none", "object"… | {"intent": "resume_music", "slots": {}} | I=0;slotF1=0.0;v=0.0 |
| 4 | t4_instruction_following_fluent_speech_commands_418c0df5493c | wavs/speakers/NgQEvO2x7Vh3xy2xz/3415a830-45c1-11e9-8ec0-7… / 3415a830-45c1-11e9-8ec0-7bf21d1cfe30.wav | Turn the bedroom heat up；gold={"intent": "increase_heat", "slots": {"action": "increase", "location": "bedroom", "objec… | {"intent": "adjust_temperature", "slots": {"location": "bedroom", "direction": "up"}} | I=0;slotF1=0.4;v=0.2 |
| 5 | t4_instruction_following_fluent_speech_commands_0b94a0443e00 | wavs/speakers/oOK5kxoW7dskMbaK/b5c7b170-44ea-11e9-a1ea-79… / b5c7b170-44ea-11e9-a1ea-79ca03012c0e.wav | Bathroom heat down；gold={"intent": "decrease_heat", "slots": {"action": "decrease", "location": "washroom", "obje… | {"intent": "adjust_temperature", "slots": {"location": "bathroom", "temperature_direction": "down"}} | I=0;slotF1=0.2;v=0.1 |
| 6 | t4_instruction_following_fluent_speech_commands_c722404a3bbb | wavs/speakers/ppymZZDb2Bf4NQnE/d4c8bad0-44f2-11e9-a1ea-79… / d4c8bad0-44f2-11e9-a1ea-79ca03012c0e.wav | Turn the temperature down in the bathroom；gold={"intent": "decrease_heat", "slots": {"action": "decrease", "location": "wash… | {"intent": "adjust_temperature", "slots": {"location": "bathroom", "direction": "down"}} | I=0;slotF1=0.2;v=0.1 |
| 7 | t4_instruction_following_fluent_speech_commands_4a8831e3a4b1 | wavs/speakers/V4ejqNL4xbUKkYrV/ed16b800-459a-11e9-bf4b-21… / ed16b800-459a-11e9-bf4b-216bfa333ac2.wav | Make it hotter；gold={"intent": "increase_heat", "slots": {"action": "increase", "location": "none", "object":… | {"intent": "adjust_temperature", "slots": {"temperature": "hotter"}} | I=0;slotF1=0.0;v=0.0 |
| 8 | t4_instruction_following_fluent_speech_commands_f90246727aeb | wavs/speakers/V4ZbwLm9G5irobWn/060ef1e0-4525-11e9-a843-8d… / 060ef1e0-4525-11e9-a843-8db76f4b5e29.wav | Lights off in the kitchen；gold={"intent": "deactivate_lights", "slots": {"action": "deactivate", "location": "kitchen", … | {"intent": "turn_off_lights", "slots": {"location": "kitchen"}} | I=0;slotF1=0.5;v=0.25 |
| 9 | t4_instruction_following_fluent_speech_commands_543f13b808fd | wavs/speakers/V4ZbwLm9G5irobWn/71b7c510-452b-11e9-a843-8d… / 71b7c510-452b-11e9-a843-8db76f4b5e29.wav | Turn the washroom lights off；gold={"intent": "deactivate_lights", "slots": {"action": "deactivate", "location": "washroom",… | {"intent": "turn_off", "slots": {"location": "washroom", "device": "lights"}} | I=0;slotF1=0.6;v=0.3 |

### slurp

- 抽样：test 等距抽样 10 条；`intent_accuracy+slot_f1`（JSON 解析，字段级 F1）。
- 当前用的指标及版本：intent_accuracy+slot_f1（诊断 proxy）
- 当前指标（数值）：intent 0/10；slot F1 均值 0.4167；组合均值 0.2083。
- 人核结论：**有问题**
- 结果是否可信：音频和解析链路正常；SLURP native intent/scenario/slot 命名与模型输出 schema 不同，exact intent 将语义近似全部判错。
- 问题编号：丙-IF-01

**10 条抽样核验（音频路径均由 `samples.jsonl` 定位；表中 audio 为文件名）**

| # | sample_id | native_id / audio | ref / expected（截断） | pred（截断） | score |
|---:|---|---|---|---|---|
| 0 | t4_instruction_following_slurp_447d31477c32 | 9054 / audio-1497872916-headset.flac | event reminder mona tuesday；gold={"intent": "calendar_set", "slots": {"date": "tuesday", "event_name": "mona"}} | ＇＇＇json { "intent": "set_reminder", "slots": { "event": "mona chisei", "time": "event_reminder" } } ＇＇＇ | I=0;slotF1=0.2222;v=0.1111 |
| 1 | t4_instruction_following_slurp_9edc30d813d7 | 5047 / audio-1490184631.flac | alexa please turn down the lights in the house；gold={"intent": "iot_hue_lightdim", "slots": {"house_place": "house"}} | ＇＇＇json { "intent": "adjust_lighting", "slots": { "action": "turn down", "light": "lights", "location": "house" } } ＇＇＇ | I=0;slotF1=0.2222;v=0.1111 |
| 2 | t4_instruction_following_slurp_acce171dd203 | 4672 / audio-1502308546-headset.flac | this is the best band ever；gold={"intent": "music_likeness", "slots": {}} | ＇＇＇json { "intent": "express_opinion", "slots": { "subject": "band", "opinion": "best" } } ＇＇＇ | I=0;slotF1=0.0;v=0.0 |
| 3 | t4_instruction_following_slurp_d0b84b4527f5 | 14154 / audio-1490287353-headset.flac | africa info；gold={"intent": "qa_factoid", "slots": {"place_name": "africa"}} | ＇＇＇json { "intent": "get_info", "slots": { "topic": "Africa" } } ＇＇＇ | I=0;slotF1=0.5;v=0.25 |
| 4 | t4_instruction_following_slurp_1f393491d8a2 | 9853 / audio-1496265565.flac | alexa start from my last session of joes book；gold={"intent": "play_audiobook", "slots": {"audiobook_name": "joes book", "pl… | ＇＇＇json { "intent": "resume_book", "slots": { "book_title": "Joe's book" } } ＇＇＇ | I=0;slotF1=0.2222;v=0.1111 |
| 5 | t4_instruction_following_slurp_8b803f33633d | 2075 / audio-1501415123.flac | i want more light；gold={"intent": "iot_hue_lightup", "slots": {}} | ＇＇＇json { "intent": "adjust_lighting", "slots": { "lighting_level": "more" } } ＇＇＇ | I=0;slotF1=0.0;v=0.0 |
| 6 | t4_instruction_following_slurp_ef7a849892e2 | 832 / audio-1490288927-headset.flac | hey what will be the predicted weather for tomorrow；gold={"intent": "weather_query", "slots": {"date": "tomorrow"}} | ＇＇＇json { "intent": "get_weather_prediction", "slots": { "date": "tomorrow" } } ＇＇＇ | I=0;slotF1=1.0;v=0.5 |
| 7 | t4_instruction_following_slurp_918e2ab82253 | 10805 / audio-1492784202.flac | please show me the list that i have；gold={"intent": "lists_query", "slots": {}} | ＇＇＇json { "intent": "show_list", "slots": {} } ＇＇＇ | I=0;slotF1=1.0;v=0.5 |
| 8 | t4_instruction_following_slurp_873e535decf6 | 5497 / audio-1497885659.flac | what is the time difference between california and new york；gold={"intent": "datetime_convert", "slots": {"place_name": "new… | ＇＇＇json { "intent": "get_time_difference", "slots": { "location1": "California", "location2": "New York" } } ＇＇＇ | I=0;slotF1=0.5;v=0.25 |
| 9 | t4_instruction_following_slurp_400e979536a6 | 1308 / audio-1502893102-headset.flac | get todays climate forecast here；gold={"intent": "weather_query", "slots": {"date": "todays"}} | ＇＇＇json { "intent": "get_climate_forecast", "slots": { "date": "today" } } ＇＇＇ | I=0;slotF1=0.5;v=0.25 |

### vocalbench

- 抽样：instruction_following 子集 10 条；local constraint checker。
- 当前用的指标及版本：local checker；不等价于官方 judge
- 当前指标（数值）：8 条可执行检查均值 0.9375；instruction-350/400 的多步/条件维度无数值。
- 人核结论：**方案待定**
- 结果是否可信：可执行的重复、关键词、模式检查能运行；emotion、speed、no-explanation、conditional、multi-step 仍需音频输出/时间或 judge，不能把 0.9375 当正式分。
- 问题编号：丙-IF-02、丙-IF-03

**10 条抽样核验（音频路径均由 `samples.jsonl` 定位；表中 audio 为文件名）**

| # | sample_id | native_id / audio | ref / expected（截断） | pred（截断） | score |
|---:|---|---|---|---|---|
| 0 | t4_instruction_following_vocalbench_099d28afe0a3 | instruction-000 / 000.wav | Repeat the following question without any explanation. Why did my sound sensitivity and motion issues suddenly come bac… | Why did my sound sensitivity and motion issues suddenly come back after I thought they were gone, and why hasn't anyone ackn… | checks=1/1;v=1.0 |
| 1 | t4_instruction_following_vocalbench_defa30e5a5af | instruction-050 / 050.wav | Repeat the following question with angry emotion. Why did my sound sensitivity and motion issues suddenly come back aft… | Why did my sound sensitivity and motion issues suddenly come back after I thought they were gone? And why hasn't anyone ackn… | checks=1/1;v=1.0 |
| 2 | t4_instruction_following_vocalbench_b80b39d53623 | instruction-100 / 100.wav | Repeat the following question at half my speaking speed. Why did my sound sensitivity and motion issues suddenly come b… | Why did my sound sensitivity and motion issues suddenly come back after I thought they were gone, and why hasn't anyone ackn… | checks=1/1;v=1.0 |
| 3 | t4_instruction_following_vocalbench_88207fbed751 | instruction-150 / 150.wav | Repeat the following question in angry emotion at half my speaking speed. Why did my sound sensitivity and motion issue… | Why did my sound sensitivity and motion issues suddenly come back after I thought they were gone, and why hasn't anyone ackn… | checks=1/1;v=1.0 |
| 4 | t4_instruction_following_vocalbench_244b65164e66 | instruction-200 / 200.wav | What are different drawers I should have for clothes? Could you please give me five suggestions? | Sure! Here are five suggestions for different drawers you should have for clothes: 1. **Tops Drawer** – Use this for shirts,… | checks=1/1;v=1.0 |
| 5 | t4_instruction_following_vocalbench_ba0394fe8d46 | instruction-250 / 250.wav | Tell me a short blog post about a trip to Japan. In your response, the word "adventure" must appear at least three time… | Japan greeted me with a whirlwind of neon lights, ancient temples, and the promise of an unforgettable adventure. From the m… | checks=2/2;v=1.0 |
| 6 | t4_instruction_following_vocalbench_fe4ecd780cb5 | instruction-300 / 300.wav | Tell me a short blog post about a trip to Japan. Do not forget to exclude the keywords temple, Tokyo, travel, culture, … | **A Journey Through Japan: An Unforgettable Adventure** Last year, I embarked on a journey that remains one of the most enri… | checks=1/2;v=0.5 |
| 7 | t4_instruction_following_vocalbench_db16f9002c23 | instruction-350 / 350.wav | What are some general tips for beginners starting an exercise routine? Then, identify the best exercises for beginners … | **General Tips for Beginners Starting an Exercise Routine:** 1. **Start Slowly**: Avoid overexertion by beginning with low-i… | checks=0/0;v=None |
| 8 | t4_instruction_following_vocalbench_bce61d778f20 | instruction-400 / 400.wav | Did Patrick Swayze have pancreatic cancer? If yes, name two other celebrities who also battled pancreatic cancer. If no… | Yes, Patrick Swayze had pancreatic cancer. Two other celebrities who also battled pancreatic cancer are: 1. Steve Jobs 2. Go… | checks=0/0;v=None |
| 9 | t4_instruction_following_vocalbench_18088a6b9635 | instruction-450 / 450.wav | Generate a thank-you letter. Wait a minute, I misspoke, make it an apology letter, with a more sincere tone. | Dear [Recipient's Name], I am writing this letter to sincerely apologize for my recent actions and the impact they have had … | checks=1/1;v=1.0 |

### vocalbench_zh

- 抽样：instruction_following 子集 10 条；local constraint checker。
- 当前用的指标及版本：local checker；不等价于官方 judge
- 当前指标（数值）：8 条可执行检查均值 0.625；instruction-150/200/250/300 有可见失败或口径风险。
- 人核结论：**方案待定**
- 结果是否可信：路径与推理正常；中文重复/关键词可作诊断，但计数、首句和禁止词需更精确的 rubric，且语气/语速未评。
- 问题编号：丙-IF-02、丙-IF-03

**10 条抽样核验（音频路径均由 `samples.jsonl` 定位；表中 audio 为文件名）**

| # | sample_id | native_id / audio | ref / expected（截断） | pred（截断） | score |
|---:|---|---|---|---|---|
| 0 | t4_instruction_following_vocalbench_zh_4a77eab2b48d | instruction-000 / 000.wav | 请听我领读并跟我朗诵这句诗。滚滚长江东逝水，浪花淘尽英雄。 | 滚滚长江东逝水，浪花淘尽英雄。 | checks=1/1;v=1.0 |
| 1 | t4_instruction_following_vocalbench_zh_da0f790b87db | instruction-050 / 050.wav | 请用伤心的语气重复我下面说的问题，注意不要回答。我因朋友的背叛感到受伤和脆弱，担心无法再信任他人。在这种情况下，我该如何重建对他人的信任并持续关心自己？ | 我因朋友的背叛感到受伤和脆弱，担心无法再信任他人。在这种情况下，我该如何重建对他人的信任，并持续关心自己？ | checks=1/1;v=1.0 |
| 2 | t4_instruction_following_vocalbench_zh_e741f3b8e050 | instruction-100 / 100.wav | 请用两倍于我的语速，跟我重复下面这句话。深秋的黄昏，一片金黄的银杏叶缓缓飘落在寂静无人的公园长椅上。 | 深秋的黄昏，一片金黄的银杏叶缓缓飘落在寂静无人的公园长椅上。 | checks=1/1;v=1.0 |
| 3 | t4_instruction_following_vocalbench_zh_efedc078b0cf | instruction-150 / 150.wav | 请用比我慢一倍的语速和生气的语气，跟我重复下面这句话。我和我的好朋友闹掰了，他怎么能说我是一个自私的人！ | 我……和……我……的……好……朋……友……闹……掰……了……他……怎……么……能……说……我……是……一……个……自……私……的……人…… | checks=0/1;v=0.0 |
| 4 | t4_instruction_following_vocalbench_zh_52c8edf8ca31 | instruction-200 / 200.wav | 我应该怎样规划好自己的橱柜来放不同的衣服？你能给我划分五个类别吗？ | 当然可以。以下是将衣柜划分为五个类别的建议，帮助你更高效地收纳和管理衣物： 1. 上衣区 用于悬挂或叠放衬衫、T恤、毛衣、外套等上身衣物。 - 悬挂区：适合西装、大衣、连衣裙、衬衫等易皱衣物。 - 折叠区：适合T恤、卫衣、针织衫等不易皱的衣物。 2… | checks=0/1;v=0.0 |
| 5 | t4_instruction_following_vocalbench_zh_8b5410ff0b72 | instruction-250 / 250.wav | 请生成一篇关于日本之旅的短博文。在你的博文中，“冒险”一词至少要出现三次，第一句话应该以“日本用什么什么迎接我”开头。 | 日本用樱花与晨钟迎接我，开启了一场充满冒险的旅程。踏上这片土地，每一步都像是在书写新的冒险故事。无论是攀登富士山的挑战，还是在京都小巷中寻找隐藏的茶馆，每一次探索都让我感受到冒险带来的惊喜与成长。在北海道的雪原上，我体验了极限滑雪的刺激，这无疑是一… | checks=1/2;v=0.5 |
| 6 | t4_instruction_following_vocalbench_zh_646442456ebf | instruction-300 / 300.wav | 请给我一篇关于日本之旅的简短博客文章。不要忘记排除以下关键词：寺庙、东京、旅行、文化。包含关键词：冒险、城市、体验、旅程、风景、记忆。 | 标题：在知里的城市探险：一段难忘的旅程记忆 当踏上日本知里的土地，我意识到这将是一次不同寻常的冒险。不同于传统印象中的繁华都市，知里以其宁静的街道、独特的建筑风格和迷人的自然风景，为每一位到访者提供了别具一格的城市体验。 知里坐落在北海道的西南部，… | checks=1/2;v=0.5 |
| 7 | t4_instruction_following_vocalbench_zh_2ddf6b92b709 | instruction-350 / 350.wav | 对于刚开始锻炼的初学者，有哪些建议呢？接下来，根据这些建议，找出最适合初学者的几项运动。然后，详细说明如何正确地完成其中一项运动。最后，探讨初学者如何随着时间的推移逐步提高训练强度。 | 对于刚开始锻炼的初学者，以下是一些建议： 1. **循序渐进**：不要一开始就进行高强度训练，应从低强度、短时间开始，逐步增加运动量。 2. **选择喜欢的运动**：选择自己感兴趣的运动项目，有助于长期坚持。 3. **注重热身与拉伸**：每次锻炼… | checks=0/0;v=None |
| 8 | t4_instruction_following_vocalbench_zh_7e26f5929e12 | instruction-400 / 400.wav | 以武术闻名的佛教寺院是白马寺还是少林寺？如果是少林寺，请简单介绍一下它的地理位置；如果是白马寺，请介绍一下它的建造年代。 | 以武术闻名的佛教寺院是少林寺。 少林寺位于中国河南省郑州市登封市嵩山少室山北麓，地处嵩山风景区内，是嵩山的重要组成部分。 | checks=0/0;v=None |
| 9 | t4_instruction_following_vocalbench_zh_f9194b15d9e1 | instruction-450 / 450.wav | 写一封感谢信。等等，我口误了，改成一封道歉信，语气要更诚恳一些。 | 尊敬的[收信人姓名]： 您好！ 首先，请允许我以最诚恳的态度向您致以深深的歉意。对于我之前在[具体事件或行为]中的不当之处，我深感懊悔与自责。我的行为不仅给您带来了困扰，也辜负了您对我的信任与期待，对此我负有不可推卸的责任。 我深知，无论出于何种原… | checks=1/1;v=1.0 |

### voicebench

- 抽样：IFEval constraint checker 的本地可执行子集。
- 当前用的指标及版本：local checker；不等价于官方 judge
- 当前指标（数值）：0.7000；10/10 有数值，均标 `partial_checker`。
- 人核结论：**方案待定**
- 结果是否可信：支持的格式/关键词/长度规则可复核；不支持的官方指令及 LLM judge 未运行，不能当完整 VoiceBench 分。
- 问题编号：丙-IF-04

**10 条抽样核验（音频路径均由 `samples.jsonl` 定位；表中 audio 为文件名）**

| # | sample_id | native_id / audio | ref / expected（截断） | pred（截断） | score |
|---:|---|---|---|---|---|
| 0 | t4_instruction_following_voicebench_5092284a7d51 | 1001 / 1001.wav | {"ids": ["punctuation:no_comma"], "kwargs": [{"capital_frequency": null, "capital_relation": null, "end_… | Hark thou traveler of far-flung lands prepare thy heart and mind for a sojourn most fair through the realm of Nippon where c… | checks=1/1;v=1.0 |
| 1 | t4_instruction_following_voicebench_92b8ac9ddca3 | 1322 / 1322.wav | {"ids": ["detectable_format:title"], "kwargs": [{"capital_frequency": null, "capital_relation": null, "e… | <<Kotlin vs Java>> **Pros of Kotlin:** 1. **Concise Syntax**: Kotlin reduces boilerplate code significantly, making code mor… | checks=1/1;v=1.0 |
| 2 | t4_instruction_following_voicebench_5fd38c744d02 | 1643 / 1643.wav | {"ids": ["punctuation:no_comma", "length_constraints:number_words"], "kwargs": [{"capital_frequency": nu… | The quote "Coincidence is God's way of remaining anonymous" is a poetic and philosophical reflection on the nature of chance… | checks=1/2;v=0.5 |
| 3 | t4_instruction_following_voicebench_f8fec28baa7f | 1967 / 1967.wav | {"ids": ["change_case:english_lowercase", "length_constraints:number_sentences"], "kwargs": [{"capital_f… | 1. mobile health clinics with telemedicine access 2. solar-powered charging stations for phones and medical devices 3. low-c… | checks=2/2;v=1.0 |
| 4 | t4_instruction_following_voicebench_e29ea72fdca9 | 2314 / 2314.wav | {"ids": ["detectable_format:number_bullet_lists"], "kwargs": [{"capital_frequency": null, "capital_relat… | - Mochi - Shadow - Snowflake | checks=1/1;v=1.0 |
| 5 | t4_instruction_following_voicebench_8e361b6a0f20 | 2590 / 2590.wav | {"ids": ["length_constraints:nth_paragraph_first_word"], "kwargs": [{"capital_frequency": null, "capital… | Send your inquiries to Grafton, Vermont, and you'll find a community that values tranquility, tradition, and natural beauty.… | checks=0/1;v=0.0 |
| 6 | t4_instruction_following_voicebench_81b8143dbb03 | 2853 / 2853.wav | {"ids": ["change_case:capital_word_frequency", "detectable_format:title"], "kwargs": [{"capital_frequenc… | <<Whispers Beneath the Rising Waters>> The Brazos hums a low, insistent tune, As Donnell’s streets grow slick with silver sh… | checks=1/2;v=0.5 |
| 7 | t4_instruction_following_voicebench_c3ef2bfd2118 | 3156 / 3156.wav | {"ids": ["keywords:existence"], "kwargs": [{"capital_frequency": null, "capital_relation": null, "end_ph… | [Your Name] [Your Address] [City, State, ZIP Code] [Email Address] [Phone Number] [Date] Customer Service Department [Compan… | checks=0/1;v=0.0 |
| 8 | t4_instruction_following_voicebench_ce81750c603c | 3442 / 3442.wav | {"ids": ["length_constraints:number_words", "detectable_format:title"], "kwargs": [{"capital_frequency":… | <<The Answer to the Riddle>> A cold (catch) and a ball (throw). | checks=2/2;v=1.0 |
| 9 | t4_instruction_following_voicebench_287d124108e8 | 3757 / 3757.wav | {"ids": ["detectable_format:constrained_response"], "kwargs": [{"capital_frequency": null, "capital_rela… | My answer is yes. | checks=1/1;v=1.0 |

## 问题

### 问题 丙-IF-01

- benchmark：fluent_speech_commands / slurp
- 发现时间：2026-09-03

样例
- sample_id：t4_instruction_following_fluent_speech_commands_340f2dbe94ec
- 音频如何定位：/mnt/afs/eval_data/04_dialogue_slu/fluent_speech_commands_dataset/wavs/speakers/4BrX8aDqK2cLZRYl/cbdf5700-452c-11e9-b1e4-e5985dca719e.wav
- ref：Turn on the lights；gold={"intent": "activate_lights", "slots": {"action": "activate", "location": "none", "object…
- pred：{"intent": "turn_on", "slots": {"device": "lights"}}
- 当前指标：I=0;slotF1=0.25;v=0.125
- 必要摘录：native_id=wavs/speakers/4BrX8aDqK2cLZRYl/cbdf5700-452c-11e9-b1e4-e5985dca719e.wav

- sample_id：t4_instruction_following_slurp_447d31477c32
- 音频如何定位：/mnt/afs/oss_data/datasets/04_dialogue_slu/SLURP_repo/audio/slurp_real/audio-1497872916-headset.flac
- ref：event reminder mona tuesday；gold={"intent": "calendar_set", "slots": {"date": "tuesday", "event_name": "mona"}}
- pred：＇＇＇json { "intent": "set_reminder", "slots": { "event": "mona chisei", "time": "event_reminder" } } ＇＇＇
- 当前指标：I=0;slotF1=0.2222;v=0.1111
- 必要摘录：native_id=9054

原因
两集原生任务是 intent/slot SLU，而不是一般 instruction-following；gold 使用固定 ontology（例如 `activate_lights`、`calendar_set`），模型输出采用自然的 `turn_on`、`set_reminder` 等命名。当前 exact intent 0/10 会把语义可能正确的样本整体判错。

解决方案
- 建议动作：为每集登记 canonical ontology 映射，分别报告 intent exact、schema-aware 参数 F1 和语义 judge；不得把两集的组合均值并入开放式指令遵循主榜。
- 若涉及数据：不修改原始数据；仅提出字段、ontology、rubric 或评测实现方案。
- 方案是否确定：确定

### 问题 丙-IF-02

- benchmark：vocalbench / vocalbench_zh
- 发现时间：2026-09-03

样例
- sample_id：t4_instruction_following_vocalbench_defa30e5a5af
- 音频如何定位：/mnt/afs/eval_data/99_auxiliary_or_out_of_scope/VocalBench/audio/instruction_following/050.wav
- ref：Repeat the following question with angry emotion. Why did my sound sensitivity and motion issues suddenly come back aft…
- pred：Why did my sound sensitivity and motion issues suddenly come back after I thought they were gone? And why hasn't anyone acknowledged how frustrating this is for me?
- 当前指标：checks=1/1;v=1.0
- 必要摘录：native_id=instruction-050

- sample_id：t4_instruction_following_vocalbench_zh_e741f3b8e050
- 音频如何定位：/mnt/afs/eval_data/99_auxiliary_or_out_of_scope/VocalBench-zh/audio/instruction/100.wav
- ref：请用两倍于我的语速，跟我重复下面这句话。深秋的黄昏，一片金黄的银杏叶缓缓飘落在寂静无人的公园长椅上。
- pred：深秋的黄昏，一片金黄的银杏叶缓缓飘落在寂静无人的公园长椅上。
- 当前指标：checks=1/1;v=1.0
- 必要摘录：native_id=instruction-100

原因
这些样本要求 angry/sad emotion 或指定语速；当前模型只生成文本，离线 checker 只能核对重复内容，`unsupported` 维度没有被测量。文本通过不等于语气/语速通过。

解决方案
- 建议动作：增加可回放的生成语音、ASR/时长与 prosody 检查，或接入有明确 rubric 的音频 judge；在此之前将相关样本标为方案待定，不用 heuristic 均值作能力分。
- 若涉及数据：不修改原始数据；仅提出字段、ontology、rubric 或评测实现方案。
- 方案是否确定：方案待定（需确定生成音频与 judge 协议）

### 问题 丙-IF-03

- benchmark：vocalbench_zh
- 发现时间：2026-09-03

样例
- sample_id：t4_instruction_following_vocalbench_zh_52c8edf8ca31
- 音频如何定位：/mnt/afs/eval_data/99_auxiliary_or_out_of_scope/VocalBench-zh/audio/instruction/200.wav
- ref：我应该怎样规划好自己的橱柜来放不同的衣服？你能给我划分五个类别吗？
- pred：当然可以。以下是将衣柜划分为五个类别的建议，帮助你更高效地收纳和管理衣物： 1. 上衣区 用于悬挂或叠放衬衫、T恤、毛衣、外套等上身衣物。 - 悬挂区：适合西装、大衣、连衣裙、衬衫等易皱衣物。 - 折叠区：适合T恤、卫衣、针织衫等不易皱的衣物。 2. 裤子与裙子区 专门存放长裤、短裤、半身裙、连衣裙等下装。 - 悬挂：适合西裤、牛仔裤、长裙等保持版型的衣物。 - 折叠：适合运动裤、休闲裤等日常穿着的衣物。 3. 内衣与袜子区 用于收纳内衣、文胸、内裤、袜子、腰带等贴身小件。 - 使用抽屉分隔盒或收纳盒，按类型分类，方便查找。 4. 鞋帽与配饰区 用于存放鞋子、帽子、围巾、手套、皮带、包包等配饰。
- 当前指标：checks=0/1;v=0.0
- 必要摘录：native_id=instruction-200

- sample_id：t4_instruction_following_vocalbench_zh_8b5410ff0b72
- 音频如何定位：/mnt/afs/eval_data/99_auxiliary_or_out_of_scope/VocalBench-zh/audio/instruction/250.wav
- ref：请生成一篇关于日本之旅的短博文。在你的博文中，“冒险”一词至少要出现三次，第一句话应该以“日本用什么什么迎接我”开头。
- pred：日本用樱花与晨钟迎接我，开启了一场充满冒险的旅程。踏上这片土地，每一步都像是在书写新的冒险故事。无论是攀登富士山的挑战，还是在京都小巷中寻找隐藏的茶馆，每一次探索都让我感受到冒险带来的惊喜与成长。在北海道的雪原上，我体验了极限滑雪的刺激，这无疑是一次难忘的冒险。冒险不仅限于自然景观，更在于与当地人的交流中发现文化的深度。这次日本之旅，让我明白，真正的冒险，是心灵的觉醒与世界的连接。
- 当前指标：checks=1/2;v=0.5
- 必要摘录：native_id=instruction-250

原因
instruction-200 的输出含嵌套编号，简单行数 checker 将五个顶层类别判为失败；instruction-250 的 gold 首句“日本用什么什么迎接我”本身不自然，模型写成“日本用樱花与晨钟迎接我”而被字面 starts_with 判错。两者是 checker/gold 口径风险，不应直接归因模型。

解决方案
- 建议动作：将 exact_items 定义为顶层项目 AST/列表计数；对不自然的 starts_with 提供规范化 gold 或语义 judge，并保留原题版本以便追溯。
- 若涉及数据：不修改原始数据；仅提出字段、ontology、rubric 或评测实现方案。
- 方案是否确定：方案待定（需数据负责人拍板 gold 规范化）

### 问题 丙-IF-04

- benchmark：voicebench / ifeval
- 发现时间：2026-09-03

样例
- sample_id：t4_instruction_following_voicebench_5fd38c744d02
- 音频如何定位：/mnt/afs/users/shizs/capability_t4_run_20260903/audio/voicebench_ifeval/1643.wav
- ref：{"ids": ["punctuation:no_comma", "length_constraints:number_words"], "kwargs": [{"capital_frequency": nu…
- pred：The quote "Coincidence is God's way of remaining anonymous" is a poetic and philosophical reflection on the nature of chance and the possibility of a higher power operating subtly within the fabric of everyday life. At first glance it may seem like a whimsical or even mystical statement but upon deeper contemplation it reveals a profound truth about how humans interpret events that appear to be random yet carry deep…
- 当前指标：checks=1/2;v=0.5
- 必要摘录：native_id=1643

- sample_id：t4_instruction_following_voicebench_c3ef2bfd2118
- 音频如何定位：/mnt/afs/users/shizs/capability_t4_run_20260903/audio/voicebench_ifeval/3156.wav
- ref：{"ids": ["keywords:existence"], "kwargs": [{"capital_frequency": null, "capital_relation": null, "end_ph…
- pred：[Your Name] [Your Address] [City, State, ZIP Code] [Email Address] [Phone Number] [Date] Customer Service Department [Company Name] [Company Address] [City, State, ZIP Code] Subject: Formal Request for Refund – Unusable Product and Breach of Customer Policy Dear Customer Service Team, I am writing to formally request a full refund for a product I purchased from your company on [Purchase Date] via your website, [Webs…
- 当前指标：checks=0/1;v=0.0
- 必要摘录：native_id=3156

原因
当前仅实现 IFEval 的一部分规则（本批 10 条均有数值，但不支持的 instruction id 与官方 judge 未覆盖）；例如长度/关键词失败只能说明局部 checker 结果。

解决方案
- 建议动作：接入官方 IFEval checker 与 VoiceBench LLM judge，记录 instruction_id、kwargs、每条规则及 unsupported 清单；本地 0.7000 只作 diagnostic。
- 若涉及数据：不修改原始数据；仅提出字段、ontology、rubric 或评测实现方案。
- 方案是否确定：确定（本轮结论保持方案待定）

## 复现与边界

- `python3 -m py_compile tools/capability_t4_smoke.py` 已通过。
- `python3 tools/capability_t4_smoke.py evaluate --work-root /mnt/afs/users/shizs/capability_t4_run_20260903` 已重新生成 `audited.jsonl`/`summary.json`；本 capability 50/50 有 prediction、error=0。
- 原始 benchmark、正式 subset manifest 与运行目录均未写入仓库；本记录中的问题只提出方案，不改数据。
