# Capability：tool_call（工具调用）

负责人：丙（T4）  
日期：2026-09-03  
模型：Qwen3-Omni-30B-A3B-Instruct（`qwen3-omni-local`）  
Job：12550–12557（v2，170 条全量）；12567（MTalk 修正重跑，10 条）  
产物：`tools/capability_t4_smoke.py`、`tools/run_capability_t4_infer.sh`；`/mnt/afs/users/shizs/capability_t4_run_20260903/{samples.jsonl,preflight.json,audited.jsonl,summary.json,predictions-v2-*.jsonl,predictions-v3-mtalk.jsonl}`

本记录覆盖 AudioAgentBench Suite、fluent_speech_commands、SLURP、StepEval-Audio-Toolcall 共 40 条。抽样均为 test/native index 的确定性等距选择；40/40 音频可定位，40/40 prediction，inference error=0。AudioAgentBench、FSC/SLURP、StepEval 均为离线 schema/expected-call proxy，不等价于真实工具执行成功。

## 统一 template

- **任务形态**：语音请求（必要时带历史音频）→ 选择工具 → 生成符合 schema 的参数；无关请求必须返回 `tool: null`，不得执行副作用。
- **Prompt 骨架**：`system: Use the complete tool schemas; return JSON only as {"tool": string|null, "arguments": object}.`；`user: audio`；AudioAgentBench/StepEval 先按原生顺序提供历史轮，再提供目标音频。
- **选择题是否改为问答**：否。本轮没有将 MCQ 当工具调用主形态；StepEval 的 positive/negative 是触发/不触发协议，不是选择题。
- **多音频 / 多轮约定**：AudioAgentBench 保留目标轮之前的上下文音频及 assistant 历史文本；StepEval 保留会话历史；FSC/SLURP 为单音频。模型输入不附加 native 转写，避免文本泄露。

## 主指标

- **推荐主指标**：按顺序报告 `call_exact`（有序 tool name + canonical 参数完全一致）、tool selection accuracy、schema-aware parameter F1；StepEval 另报 positive trigger/type/parameter 与 negative false-trigger，不能压成单一 exact。
- **数据流**：音频/历史 → Qwen3-Omni → 解析 JSON 或兼容的 `<tool_call>` → 规范化 call 列表 → 与 expected name/args 对齐。参数 F1 是诊断重叠，不能替代执行结果。
- **算分差异**：AudioAgentBench 用多 call ordered exact + 参数 F1；FSC/SLURP 是把 SLU ontology 投影到人为工具 schema；StepEval 用 trigger correctness（负样本要求不触发）及正样本 type accuracy。没有真实工具后端，当前分数不代表 side-effect/task success。

| benchmark | 当前 proxy | 10 条汇总 | 可否作为本能力主分 |
|---|---|---:|---|
| audioagentbench_suite | ordered call exact + parameter F1 | exact 2/10；parameter F1 均值 0.4716 | 否（无工具执行） |
| fluent_speech_commands | tool selection + parameter F1 | tool 10/10；parameter F1 0.6802；exact 0/10 | 否（ontology projection） |
| slurp | tool selection + parameter F1 | tool 10/10；parameter F1 0.5133；exact 0/10 | 否（ontology projection） |
| stepeval_audio_toolcall | trigger/type diagnostic | trigger 8/10；positive type 4/5；正负均覆盖 | 否（离线触发 proxy） |

## 各 benchmark 核验

### audioagentbench_suite

- 抽样：tool_call_turns.json 等距抽样 10 条（含 1 条多工具期望）。
- 当前用的指标及版本：`tool_use_success` offline ordered-call proxy
- 当前指标（数值）：ordered exact 2/10；parameter F1 均值 0.4716。
- 人核结论：**有问题**
- 结果是否可信：解析/对照可复现，但仅与 native expected call 对齐，没有调用真实工具或核验副作用；多轮目标还混有直接回答。
- 问题编号：丙-TC-01

**10 条抽样核验（audio 为输入文件名；多轮样本列出音频数量）**

| # | sample_id | native_id / audio | ref / expected（截断） | pred（截断） | score |
|---:|---|---|---|---|---|
| 0 | t4_tool_call_audioagentbench_suite_4d1141da7bab | conversation-bench:10 / 11× turn_000.wav; turn_001.wav; turn_002.wav; turn_003.wa… | {"args": {"name": "Jennifer Smith", "suggestion_text": "A session about open telemetry tracing."}, "name": "submit_sess… | {"tool": "submit_session_suggestion", "arguments": {"name": "Jennifer Smith", "suggestion_text": "A session about OpenTeleme… | exact=0;tool=0;pF1=0.8421 |
| 1 | t4_tool_call_audioagentbench_suite_41a1dd3ec8bc | conversation-bench:38 / 17× turn_022.wav; turn_023.wav; turn_024.wav; turn_025.wa… | [{"args": {"name": "Jennifer Smith", "session_id": "923101"}, "name": "register_for_session"}, {"args": {"name": "Jenni… | I've registered you for Dr. Stephanie Liu's 'Robotics and Embodied AI' session on June 5th at 10:15 AM. | exact=0;tool=0;pF1=0.0 |
| 2 | t4_tool_call_audioagentbench_suite_8f1a14f16dbb | assistant-bench:6 / 7× turn_000.wav; turn_001.wav; turn_002.wav; turn_003.wav… | {"args": {"body": "Hi Alex, I'll be in New York on February 3rd for our meeting. Can you put together the agenda and sh… | {"tool": "send_email", "arguments": {"to": "alex.reed@company.com", "subject": "Meeting Agenda Request - Feb 3rd", "body": "… | exact=0;tool=0;pF1=0.8738 |
| 3 | t4_tool_call_audioagentbench_suite_772cb2775649 | assistant-bench:29 / 17× turn_013.wav; turn_014.wav; turn_015.wav; turn_016.wa… | {"args": {"date": "2025-02-02", "message": "Pack bag for NYC trip", "time": "21:00"}, "name": "set_reminder"} | {"tool": "set_reminder", "arguments": {"date": "2025-02-02", "time": "21:00", "message": "Pack bag for NYC trip"}} {"tool": … | exact=0;tool=0;pF1=0.0 |
| 4 | t4_tool_call_audioagentbench_suite_c648e67682d6 | product-bench:24 / 17× turn_008.wav; turn_009.wav; turn_010.wav; turn_011.wa… | {"args": {"cart_id": "CART-5501", "field": "warranty", "new_value": "extended"}, "name": "update_cart"} | Switching to the 3-year extended warranty at $149. Your total is now $1,348.99. | exact=0;tool=0;pF1=0.0 |
| 5 | t4_tool_call_audioagentbench_suite_11a42d6b7a5c | grocery-bench:11 / 12× turn_000.wav; turn_001.wav; turn_002.wav; turn_003.wa… | {"args": {"query": "maple candy"}, "name": "lookup_item"} | Maple Candy, one-pound bag for eleven ninety-nine. Added. | exact=0;tool=0;pF1=0.0 |
| 6 | t4_tool_call_audioagentbench_suite_a205dcbc3a54 | grocery-bench:22 / 17× turn_006.wav; turn_007.wav; turn_008.wav; turn_009.wa… | {"args": {"order_id": "ORD-5521"}, "name": "verify_details"} | {"tool": "verify_details", "arguments": {"order_id": "ORD-5521"}} | exact=1;tool=1;pF1=1.0 |
| 7 | t4_tool_call_audioagentbench_suite_872a0be56552 | appointment-bench:10 / 11× turn_000.wav; turn_001.wav; turn_002.wav; turn_003.wa… | {"args": {"date": "2025-01-13", "doctor": "Perry", "patient_name": "Daniel Nolan", "phone": "415-916-1640", "service_ty… | {"tool": "book_appointment", "arguments": {"patient_name": "Daniel Nolan", "phone": "415-916-1640", "date": "2025-01-13", "t… | exact=1;tool=1;pF1=1.0 |
| 8 | t4_tool_call_audioagentbench_suite_aedfc2c07864 | event-bench:9 / 10× turn_000.wav; turn_001.wav; turn_002.wav; turn_003.wa… | {"args": {"add_ons": ["dj", "photographer"], "catering_package": "gold", "guest_count": 80, "venue_id": "garden_pavilio… | Let me update your quote with the DJ and photographer. | exact=0;tool=0;pF1=0.0 |
| 9 | t4_tool_call_audioagentbench_suite_388092bb3135 | event-bench:28 / 17× turn_012.wav; turn_013.wav; turn_014.wav; turn_015.wa… | {"args": {}, "name": "end_session"} | You're welcome! Your booking is confirmed and secure. If you need anything else, just reach out. Have a wonderful event! | exact=0;tool=0;pF1=1.0 |

### fluent_speech_commands

- 抽样：FSC test 等距抽样 10 条。
- 当前用的指标及版本：`tool_exact_match` + parameter F1（diagnostic）
- 当前指标（数值）：tool_correct 10/10；parameter F1 0.6802；schema exact 0/10。
- 人核结论：**有问题**
- 结果是否可信：工具名是人为统一的 `control_device`，选择本身全对；参数词面与 native action/object/location ontology 不同，exact 被系统性压低。
- 问题编号：丙-TC-02

**10 条抽样核验（audio 为输入文件名；多轮样本列出音频数量）**

| # | sample_id | native_id / audio | ref / expected（截断） | pred（截断） | score |
|---:|---|---|---|---|---|
| 0 | t4_tool_call_fluent_speech_commands_6245936a6509 | wavs/speakers/4BrX8aDqK2cLZRYl/cbdf5700-452c-11e9-b1e4-e5… / cbdf5700-452c-11e9-b1e4-e5985dca719e.wav | {"arguments": {"action": "activate", "location": "none", "object": "lights"}, "tool": "control_device"} | {"tool": "control_device", "arguments": {"action": "turn on", "object": "lights", "location": "living room"}} | tool=1;pF1=0.5714;exact=0 |
| 1 | t4_tool_call_fluent_speech_commands_49f133d96d4a | wavs/speakers/4BrX8aDqK2cLZRYl/8e92af90-4532-11e9-b1e4-e5… / 8e92af90-4532-11e9-b1e4-e5985dca719e.wav | {"arguments": {"action": "decrease", "location": "washroom", "object": "heat"}, "tool": "control_device"} | {"tool": "control_device", "arguments": {"action": "decrease", "object": "temperature", "location": "washroom"}} | tool=1;pF1=0.8333;exact=0 |
| 2 | t4_tool_call_fluent_speech_commands_32065c2cbc34 | wavs/speakers/7B4XmNppyrCK977p/27bc1d50-45ce-11e9-9992-df… / 27bc1d50-45ce-11e9-9992-dfba019cb646.wav | {"arguments": {"action": "decrease", "location": "none", "object": "volume"}, "tool": "control_device"} | {"tool": "control_device", "arguments": {"action": "lower", "object": "volume", "location": "living room"}} | tool=1;pF1=0.6154;exact=0 |
| 3 | t4_tool_call_fluent_speech_commands_c389b3ed04da | wavs/speakers/k5bqyxx2lzIbrlg9/12b65b60-4528-11e9-a843-8d… / 12b65b60-4528-11e9-a843-8db76f4b5e29.wav | {"arguments": {"action": "activate", "location": "none", "object": "music"}, "tool": "control_device"} | {"tool": "control_device", "arguments": {"action": "resume", "object": "music", "location": "living room"}} | tool=1;pF1=0.6154;exact=0 |
| 4 | t4_tool_call_fluent_speech_commands_28b983a4bfcd | wavs/speakers/NgQEvO2x7Vh3xy2xz/3415a830-45c1-11e9-8ec0-7… / 3415a830-45c1-11e9-8ec0-7bf21d1cfe30.wav | {"arguments": {"action": "increase", "location": "bedroom", "object": "heat"}, "tool": "control_device"} | {"tool": "control_device", "arguments": {"action": "turn", "object": "heat", "location": "bedroom"}} | tool=1;pF1=0.8333;exact=0 |
| 5 | t4_tool_call_fluent_speech_commands_230e4d1394cf | wavs/speakers/oOK5kxoW7dskMbaK/b5c7b170-44ea-11e9-a1ea-79… / b5c7b170-44ea-11e9-a1ea-79ca03012c0e.wav | {"arguments": {"action": "decrease", "location": "washroom", "object": "heat"}, "tool": "control_device"} | {"tool": "control_device", "arguments": {"action": "turn", "object": "heat", "location": "bathroom"}} | tool=1;pF1=0.6667;exact=0 |
| 6 | t4_tool_call_fluent_speech_commands_8c9b52cd953f | wavs/speakers/ppymZZDb2Bf4NQnE/d4c8bad0-44f2-11e9-a1ea-79… / d4c8bad0-44f2-11e9-a1ea-79ca03012c0e.wav | {"arguments": {"action": "decrease", "location": "washroom", "object": "heat"}, "tool": "control_device"} | {"tool": "control_device", "arguments": {"action": "turn down", "object": "temperature", "location": "bathroom"}} | tool=1;pF1=0.4615;exact=0 |
| 7 | t4_tool_call_fluent_speech_commands_59edb6a8df83 | wavs/speakers/V4ejqNL4xbUKkYrV/ed16b800-459a-11e9-bf4b-21… / ed16b800-459a-11e9-bf4b-216bfa333ac2.wav | {"arguments": {"action": "increase", "location": "none", "object": "heat"}, "tool": "control_device"} | {"tool": "control_device", "arguments": {"action": "increase", "object": "temperature", "location": "here"}} | tool=1;pF1=0.6667;exact=0 |
| 8 | t4_tool_call_fluent_speech_commands_235f3f3d26e8 | wavs/speakers/V4ZbwLm9G5irobWn/060ef1e0-4525-11e9-a843-8d… / 060ef1e0-4525-11e9-a843-8db76f4b5e29.wav | {"arguments": {"action": "deactivate", "location": "kitchen", "object": "lights"}, "tool": "control_device"} | {"tool": "control_device", "arguments": {"action": "turn off", "object": "lights", "location": "kitchen"}} | tool=1;pF1=0.7692;exact=0 |
| 9 | t4_tool_call_fluent_speech_commands_764666fd0d9f | wavs/speakers/V4ZbwLm9G5irobWn/71b7c510-452b-11e9-a843-8d… / 71b7c510-452b-11e9-a843-8db76f4b5e29.wav | {"arguments": {"action": "deactivate", "location": "washroom", "object": "lights"}, "tool": "control_device"} | {"tool": "control_device", "arguments": {"action": "turn off", "object": "lights", "location": "washroom"}} | tool=1;pF1=0.7692;exact=0 |

### slurp

- 抽样：SLURP test 等距抽样 10 条。
- 当前用的指标及版本：`tool_exact_match` + parameter F1（diagnostic）
- 当前指标（数值）：tool_correct 10/10；parameter F1 0.5133；schema exact 0/10。
- 人核结论：**有问题**
- 结果是否可信：`execute_slurp_intent` 只是投影工具；模型重新命名 intent/scenario/slot，不能视为真实 API 参数正确。
- 问题编号：丙-TC-02

**10 条抽样核验（audio 为输入文件名；多轮样本列出音频数量）**

| # | sample_id | native_id / audio | ref / expected（截断） | pred（截断） | score |
|---:|---|---|---|---|---|
| 0 | t4_tool_call_slurp_2afeefe32540 | 9054 / audio-1497872916-headset.flac | {"arguments": {"intent": "calendar_set", "scenario": "calendar", "slots": {"date": "tuesday", "event_name": "mona"}}, "… | {"tool": "execute_slurp_intent", "arguments": {"intent": "event_reminder", "scenario": "mona_chisai", "slots": {}}} | tool=1;pF1=0.4286;exact=0 |
| 1 | t4_tool_call_slurp_6048bf9ba454 | 5047 / audio-1490184631.flac | {"arguments": {"intent": "iot_hue_lightdim", "scenario": "iot", "slots": {"house_place": "house"}}, "tool": "execute_sl… | {"tool": "execute_slurp_intent", "arguments": {"intent": "turn_down_light", "scenario": "house", "slots": {}}} | tool=1;pF1=0.6667;exact=0 |
| 2 | t4_tool_call_slurp_5702c04f6339 | 4672 / audio-1502308546-headset.flac | {"arguments": {"intent": "music_likeness", "scenario": "music", "slots": {}}, "tool": "execute_slurp_intent"} | {"tool": "execute_slurp_intent", "arguments": {"intent": "express_opinion", "scenario": "listening_to_music", "slots": {"sub… | tool=1;pF1=0.4286;exact=0 |
| 3 | t4_tool_call_slurp_3385d9d79d4c | 14154 / audio-1490287353-headset.flac | {"arguments": {"intent": "qa_factoid", "scenario": "qa", "slots": {"place_name": "africa"}}, "tool": "execute_slurp_int… | {"tool": "execute_slurp_intent", "arguments": {"intent": "get_info", "scenario": "africa", "slots": {}}} | tool=1;pF1=0.6667;exact=0 |
| 4 | t4_tool_call_slurp_e16023ac4f9c | 9853 / audio-1496265565.flac | {"arguments": {"intent": "play_audiobook", "scenario": "play", "slots": {"audiobook_name": "joes book", "player_setting… | {"tool": "execute_slurp_intent", "arguments": {"intent": "resume", "scenario": "reading", "slots": {"book": "Joe's book", "s… | tool=1;pF1=0.381;exact=0 |
| 5 | t4_tool_call_slurp_556534743d96 | 2075 / audio-1501415123.flac | {"arguments": {"intent": "iot_hue_lightup", "scenario": "iot", "slots": {}}, "tool": "execute_slurp_intent"} | {"tool": "execute_slurp_intent", "arguments": {"intent": "adjust_lighting", "scenario": "morning", "slots": {"light_level": … | tool=1;pF1=0.5;exact=0 |
| 6 | t4_tool_call_slurp_44b2a25e7d03 | 832 / audio-1490288927-headset.flac | {"arguments": {"intent": "weather_query", "scenario": "weather", "slots": {"date": "tomorrow"}}, "tool": "execute_slurp… | {"tool": "execute_slurp_intent", "arguments": {"intent": "get_weather_prediction", "scenario": "get_weather_prediction", "sl… | tool=1;pF1=0.5;exact=0 |
| 7 | t4_tool_call_slurp_d3648f5508da | 10805 / audio-1492784202.flac | {"arguments": {"intent": "lists_query", "scenario": "lists", "slots": {}}, "tool": "execute_slurp_intent"} | {"tool": "execute_slurp_intent", "arguments": {"intent": "show", "scenario": "list", "slots": {}}} | tool=1;pF1=0.6;exact=0 |
| 8 | t4_tool_call_slurp_8365ab3fbeec | 5497 / audio-1497885659.flac | {"arguments": {"intent": "datetime_convert", "scenario": "datetime", "slots": {"place_name": "new york"}}, "tool": "exe… | {"tool": "execute_slurp_intent", "arguments": {"intent": "get_time_difference", "scenario": "between_california_and_new_york… | tool=1;pF1=0.4615;exact=0 |
| 9 | t4_tool_call_slurp_1f8e85e56fbd | 1308 / audio-1502893102-headset.flac | {"arguments": {"intent": "weather_query", "scenario": "weather", "slots": {"date": "todays"}}, "tool": "execute_slurp_i… | {"tool": "execute_slurp_intent", "arguments": {"intent": "get_climate_forecast", "scenario": "today", "slots": {}}} | tool=1;pF1=0.5;exact=0 |

### stepeval_audio_toolcall

- 抽样：8 个 tool family 的 positive/negative 混合抽样 10 条。
- 当前用的指标及版本：trigger precision/recall + positive type（10-row diagnostic）
- 当前指标（数值）：trigger 8/10；positive type 4/5；negative 4/5 正确。
- 人核结论：**方案待定**
- 结果是否可信：正负触发维度可核验；参数语义与真实执行、不同 serialization 的兼容性仍需官方 evaluator/后端确认。
- 问题编号：丙-TC-03

**10 条抽样核验（audio 为输入文件名；多轮样本列出音频数量）**

| # | sample_id | native_id / audio | ref / expected（截断） | pred（截断） | score |
|---:|---|---|---|---|---|
| 0 | t4_tool_call_stepeval_audio_toolcall_0c3e6754a5cf | get_date_time_negative.jsonl:0 / 6× get_date_time_negative.jsonl-0-0.wav; get_date_time_ne… | {"gold_call": {"args": {"location": "chengdu"}, "name": "get_weather"}, "polarity": "negative", "target_tool": "get_dat… | {"tool": "get_weather", "arguments": {"location": "成都"}} | trigger=1;type=None;negative |
| 1 | t4_tool_call_stepeval_audio_toolcall_4c24ba17f548 | get_date_time_positive.jsonl:0 / 6× get_date_time_positive.jsonl-0-0.wav; get_date_time_po… | {"gold_call": {"args": {}, "name": "get_date_time"}, "polarity": "positive", "target_tool": "get_date_time"} | {"tool": "get_date_time", "arguments": {}} | trigger=1;type=1;positive |
| 2 | t4_tool_call_stepeval_audio_toolcall_66359f6fc8d0 | get_weather_negative.jsonl:0 / 6× get_weather_negative.jsonl-0-0.wav; get_weather_negati… | {"gold_call": {"args": {"expression": "35 * (60 + 25) + 1200 + 500"}, "name": "calculate"}, "polarity": "negative", "ta… | {"tool": "calculate", "arguments": {"expression": "35 * 60 + 35 * 25 + 1200 + 500"}} | trigger=1;type=None;negative |
| 3 | t4_tool_call_stepeval_audio_toolcall_e1a1d70d7b43 | get_weather_positive.jsonl:0 / 4× get_weather_positive.jsonl-0-0.wav; get_weather_positi… | {"gold_call": {"args": {"location": "chengdu"}, "name": "get_weather"}, "polarity": "positive", "target_tool": "get_wea… | {"tool": "get_weather", "arguments": {"location": "成都"}} | trigger=1;type=1;positive |
| 4 | t4_tool_call_stepeval_audio_toolcall_9cde3f62cab0 | timbre_negative.jsonl:0 / 2× timbre_negative.jsonl-0-0.wav; timbre_negative.jsonl-0… | {"gold_call": null, "polarity": "negative", "target_tool": "timbre"} | {"tool": "timbre_rag", "arguments": {"query": "小月 语速 加快 点 我 赶 时间 呢"}} | trigger=0;type=None;negative |
| 5 | t4_tool_call_stepeval_audio_toolcall_1b09e8a9bd03 | timbre_positive.jsonl:0 / 10× timbre_positive.jsonl-0-0.wav; timbre_positive.jsonl-… | {"gold_call": {"args": {"query": "默认男音色"}, "name": "timbre_rag"}, "polarity": "positive", "target_tool": "timbre"} | <tool_call>function timbre_rag {"query": "恢复原始音色"}</tool_call> | trigger=1;type=1;positive |
| 6 | t4_tool_call_stepeval_audio_toolcall_c4b47954e65b | web_search_negative.jsonl:0 / 2× web_search_negative.jsonl-0-0.wav; web_search_negative… | {"gold_call": null, "polarity": "negative", "target_tool": "web_search"} | {"tool": "timbre_rag", "arguments": {"query": "推荐一款性价比高的安卓手机"}} | trigger=1;type=None;negative |
| 7 | t4_tool_call_stepeval_audio_toolcall_539924090ab6 | web_search_positive.jsonl:0 / 8× web_search_positive.jsonl-0-0.wav; web_search_positive… | {"gold_call": {"args": {"query": "太阳能推广 政策 补贴"}, "name": "web_search"}, "polarity": "positive", "target_tool": "web_sea… | {"tool": "web_search", "arguments": {"query": "最新太阳能推广政策 补贴细节"}} | trigger=1;type=1;positive |
| 8 | t4_tool_call_stepeval_audio_toolcall_275795cded60 | get_date_time_negative.jsonl:1 / 4× get_date_time_negative.jsonl-1-0.wav; get_date_time_ne… | {"gold_call": {"args": {"query": "附近爬山路线推荐"}, "name": "web_search"}, "polarity": "negative", "target_tool": "get_date_t… | {"tool": "web_search", "arguments": {"query": "附近推荐的爬山路线信息"}} | trigger=1;type=None;negative |
| 9 | t4_tool_call_stepeval_audio_toolcall_a98626e4c05b | web_search_positive.jsonl:199 / 4× web_search_positive.jsonl-199-0.wav; web_search_positi… | {"gold_call": {"args": {"query": "苹果公司起源"}, "name": "web_search"}, "polarity": "positive", "target_tool": "web_search"} | {"tool": "timbre_rag", "arguments": {"query": "苹果公司最早是做什么的？它的起源是什么？"}} | trigger=0;type=0;positive |

## 问题

### 问题 丙-TC-01

- benchmark：audioagentbench_suite
- 发现时间：2026-09-03

样例
- sample_id：t4_tool_call_audioagentbench_suite_4d1141da7bab
- 音频如何定位：/mnt/afs/eval_data/benchmarks/AudioAgentBench/conversation-bench/audio/turn_000.wav；/mnt/afs/eval_data/benchmarks/AudioAgentBench/conversation-bench/audio/turn_001.wav；/mnt/afs/eval_data/benchmarks/AudioAgentBench/conversation-bench/audio/turn_002.wav；…共 11 个音频（完整列表见 samples.jsonl）
- ref：{"args": {"name": "Jennifer Smith", "suggestion_text": "A session about open telemetry tracing."}, "name": "submit_sess…
- pred：{"tool": "submit_session_suggestion", "arguments": {"name": "Jennifer Smith", "suggestion_text": "A session about OpenTelemetry tracing."}}
- 当前指标：exact=0;tool=0;pF1=0.8421
- 必要摘录：native_id=conversation-bench:10

- sample_id：t4_tool_call_audioagentbench_suite_41a1dd3ec8bc
- 音频如何定位：/mnt/afs/eval_data/benchmarks/AudioAgentBench/conversation-bench/audio/turn_022.wav；/mnt/afs/eval_data/benchmarks/AudioAgentBench/conversation-bench/audio/turn_023.wav；/mnt/afs/eval_data/benchmarks/AudioAgentBench/conversation-bench/audio/turn_024.wav；…共 17 个音频（完整列表见 samples.jsonl）
- ref：[{"args": {"name": "Jennifer Smith", "session_id": "923101"}, "name": "register_for_session"}, {"args": {"name": "Jenni…
- pred：I've registered you for Dr. Stephanie Liu's 'Robotics and Embodied AI' session on June 5th at 10:15 AM.
- 当前指标：exact=0;tool=0;pF1=0.0
- 必要摘录：native_id=conversation-bench:38

原因
AudioAgentBench 的 expected call/context 是离线 gold；当前没有真实工具注册、执行返回值、权限/副作用或任务完成状态。一个 call exact=0 可能是模型回答了文本，也可能是格式/调用时机问题，不能直接等同业务失败。

解决方案
- 建议动作：接入沙箱工具运行时，记录 tool name、args、返回值、状态变更和 task_success；离线 expected-call 分保留为 diagnostic，并按单 call/多 call 分层。
- 若涉及数据：不修改原始数据；仅提出字段、ontology、rubric 或评测实现方案。
- 方案是否确定：方案待定（需工具沙箱与 task-success rubric）

### 问题 丙-TC-02

- benchmark：fluent_speech_commands / slurp
- 发现时间：2026-09-03

样例
- sample_id：t4_tool_call_fluent_speech_commands_6245936a6509
- 音频如何定位：/mnt/afs/eval_data/04_dialogue_slu/fluent_speech_commands_dataset/wavs/speakers/4BrX8aDqK2cLZRYl/cbdf5700-452c-11e9-b1e4-e5985dca719e.wav
- ref：{"arguments": {"action": "activate", "location": "none", "object": "lights"}, "tool": "control_device"}
- pred：{"tool": "control_device", "arguments": {"action": "turn on", "object": "lights", "location": "living room"}}
- 当前指标：tool=1;pF1=0.5714;exact=0
- 必要摘录：native_id=wavs/speakers/4BrX8aDqK2cLZRYl/cbdf5700-452c-11e9-b1e4-e5985dca719e.wav

- sample_id：t4_tool_call_slurp_2afeefe32540
- 音频如何定位：/mnt/afs/oss_data/datasets/04_dialogue_slu/SLURP_repo/audio/slurp_real/audio-1497872916-headset.flac
- ref：{"arguments": {"intent": "calendar_set", "scenario": "calendar", "slots": {"date": "tuesday", "event_name": "mona"}}, "…
- pred：{"tool": "execute_slurp_intent", "arguments": {"intent": "event_reminder", "scenario": "mona_chisai", "slots": {}}}
- 当前指标：tool=1;pF1=0.4286;exact=0
- 必要摘录：native_id=9054

原因
FSC/SLURP 本来是 SLU ontology；本轮将其投影为统一工具 schema。例：FSC gold `activate_lights/action=activate/location=none`，模型输出 `turn_on/living room`；SLURP 也出现 `calendar_set`→`event_reminder`。tool selection 10/10 不能掩盖参数语义对齐问题。

解决方案
- 建议动作：为每集维护版本化 canonical mapping 与多值别名，先做 schema-aware 参数/实体归一化，再报告 exact；不要把投影集与真实 API benchmark 合并。
- 若涉及数据：不修改原始数据；仅提出字段、ontology、rubric 或评测实现方案。
- 方案是否确定：方案待定（需数据负责人确认 canonical ontology）

### 问题 丙-TC-03

- benchmark：stepeval_audio_toolcall
- 发现时间：2026-09-03

样例
- sample_id：t4_tool_call_stepeval_audio_toolcall_1b09e8a9bd03
- 音频如何定位：/mnt/afs/eval_data/benchmarks/StepEval-Audio-Toolcall/Step-Audio_Toolcall/wav/timbre_positive.jsonl-0-0.wav；/mnt/afs/eval_data/benchmarks/StepEval-Audio-Toolcall/Step-Audio_Toolcall/wav/timbre_positive.jsonl-0-1.wav；/mnt/afs/eval_data/benchmarks/StepEval-Audio-Toolcall/Step-Audio_Toolcall/wav/timbre_positive.jsonl-0-2.wav；…共 10 个音频（完整列表见 samples.jsonl）
- ref：{"gold_call": {"args": {"query": "默认男音色"}, "name": "timbre_rag"}, "polarity": "positive", "target_tool": "timbre"}
- pred：<tool_call>function timbre_rag {"query": "恢复原始音色"}</tool_call>
- 当前指标：trigger=1;type=1;positive
- 必要摘录：native_id=timbre_positive.jsonl:0

- sample_id：t4_tool_call_stepeval_audio_toolcall_9cde3f62cab0
- 音频如何定位：/mnt/afs/eval_data/benchmarks/StepEval-Audio-Toolcall/Step-Audio_Toolcall/wav/timbre_negative.jsonl-0-0.wav；/mnt/afs/eval_data/benchmarks/StepEval-Audio-Toolcall/Step-Audio_Toolcall/wav/timbre_negative.jsonl-0-0.wav
- ref：{"gold_call": null, "polarity": "negative", "target_tool": "timbre"}
- pred：{"tool": "timbre_rag", "arguments": {"query": "小月 语速 加快 点 我 赶 时间 呢"}}
- 当前指标：trigger=0;type=None;negative
- 必要摘录：native_id=timbre_negative.jsonl:0

原因
模型在 timbre positive 样本输出了原生 `<tool_call>function …</tool_call>`，而不是提示要求的 JSON；同一 batch 还包含 negative 不应触发样本。若 scorer 不兼容该 serialization，会把已调用误记为未调用；兼容后仍需按 tool 名和参数单独判定。

解决方案
- 建议动作：保留原始 serialization；解析器已增加 XML-ish 兼容并在本轮重算为 trigger 8/10、positive type 4/5。正式评测应同时记录格式合规率、trigger、type、参数及 negative false-trigger。
- 若涉及数据：不修改原始数据；仅提出字段、ontology、rubric 或评测实现方案。
- 方案是否确定：确定（解析修复已落地；正式口径仍待官方确认）

## 复现与边界

- `python3 -m py_compile tools/capability_t4_smoke.py` 已通过；重新 evaluate 后 StepEval 使用兼容解析结果。
- 本 capability 40/40 有 prediction、error=0；`audited.jsonl` 保留 raw prediction，解析后的结构只用于 score。
- 未修改原始 benchmark、tool schema 或正式 subset manifest；运行目录在仓库外。
