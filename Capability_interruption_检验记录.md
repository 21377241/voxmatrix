# Capability：interruption（打断恢复）

负责人：丙（T4）  
日期：2026-09-03（初检）；2026-09-10（生产链路复验）
模型：Qwen3-Omni-30B-A3B-Instruct（`qwen3-omni-local`）  
Job：12550–12557（v2，170 条全量）；12567（MTalk 修正重跑，10 条）  
产物：`tools/capability_t4_smoke.py`、`tools/run_capability_t4_infer.sh`；`/mnt/afs/users/shizs/capability_t4_run_20260903/{samples.jsonl,preflight.json,audited.jsonl,summary.json,predictions-v2-*.jsonl,predictions-v3-mtalk.jsonl}`

本记录覆盖 IHBench 10 条。抽样包含 filler、topic_switch、correction、normal、pushback 五类；所有历史/打断音频均物化为可读 wav，10/10 prediction，inference error=0。当前 batch 只做离线恢复文本 proxy，不声称完成实时打断评测。

## 2026-09-10 生产链路复验

- 真实 adapter smoke：IHBench 从原生 `baseline.parquet`/`conversations.parquet` 加载 10 条；10/10 V2 schema、完整历史音频、TF/RQ rubrics、Prompt 和 evaluator route 通过。
- 生产 evaluator 只接受官方口径的结构化 judge 结果：`tf_win∈{0,0.5,1}` 且 `rq_pass∈{0,1}`，两字段缺一即报错；不会再计算或回填 token F1。
- 旧 10 条 Qwen 预测回放结果为 `tf_win_score=not_evaluated`、`rq_pass=not_evaluated`（0/10 有数值），原因是当前环境没有 `OPENAI_API_KEY`/`LLMCenterUserToken`。旧 `recovery tokenF1=0.3810` 已降级为历史回归信息，不是 IHBench 指标。
- 单测以 fake judge 验证 TF/RQ 字段传递、取值范围和“不得出现 lexical F1”；真实 adapter smoke 验证评测输入链。回放证据与 SHA256 同 instruction_following 记录。

指标合理性结论：离线 TF/RQ judge 链已经正确且 fail-closed，但本轮仍不能声称评了实时打断能力，因为没有流式 speech onset、停止播放、取消工具、resume latency 等事件。丙-IN-02 的指标错配已解决；丙-IN-01 的 streaming harness 属于仍未满足的运行时依赖，因此正式 capability 结论仍为“本轮不评”。

> 下文 2026-09-03 的 token-F1 表格只保留为问题样例，不能作为修复后数值。

## 统一 template

- **任务形态**：助手正在识别/生成/执行时收到新的用户音频；系统应检测打断、停止或暂停原任务，正确理解新意图，并从正确状态继续/切换。
- **Prompt 骨架**：`system: maintain the current conversation state; process the user interruption and respond to the new turn without replaying cancelled content.`；按时间顺序送入历史 assistant/user 音频，最后送入打断后的 user 音频。
- **选择题是否改为问答**：否。IHBench 是开放式实时交互，不改成选择题。
- **多音频 / 多轮约定**：保留 interruption turn 之前的 assistant transcript 与 user audio；当前离线 batch 只生成最终文本，没有播放流/取消信号。

## 主指标（2026-09-03 初检口径）

- **推荐主指标**：官方 `tf_win_rate`（打断/继续处理是否正确）+ `rq_pass_rate`（恢复质量 rubric），并分 interruption_type 报告；同时记录 detection、stop、cancel、resume latency。
- **数据流**：实时音频流与播放状态 → 检测事件 → 停止/切换 → 新响应；本轮仅把历史与打断音频离线送入模型，再将 pred 与 baseline 做 token F1，故 `recovery tokenF1=0.381` 仅为 proxy。
- **算分差异**：token F1 是 lexical overlap，不检查停止时延、原任务是否继续播放、状态是否切换或 rubric 条件；不能替代官方 tf/rq。样本类型分布为 normal 3、filler 2、topic_switch 2、pushback 2、correction 1。

| benchmark | 当前 proxy | 10 条汇总 | 可否作为本能力主分 |
|---|---|---:|---|
| ihbench | baseline token F1（offline recovery proxy） | 均值 0.3810；10/10 有值 | 本轮不评正式分 |

## 各 benchmark 核验（2026-09-03 样本明细）

### ihbench

- 抽样：interruptions_all.json 等距抽样 10 条，覆盖 5 类 interruption_type。
- 当前用的指标及版本：`tf_win_rate+rq_pass_rate`（官方目标）；本轮实际为 baseline token F1 offline proxy。
- 当前指标（数值）：token F1 均值 0.3810（0.1370–0.6538）；正式 tf/rq 未计算。
- 人核结论：**本轮不评**（正式分）；仅确认历史/打断音频和恢复文本可复现。
- 结果是否可信：可用于检查 batch 输入和 lexical recovery 趋势，不能判断实时打断能力。
- 问题编号：丙-IN-01、丙-IN-02

**10 条抽样核验**

| # | sample_id | native_id / audio | ref / expected（截断） | pred（截断） | score |
|---:|---|---|---|---|---|
| 0 | t4_interruption_ihbench_fc0811daef3d | f16d3f68-3667-4a93-945a-7947bac1061c:1 / f16d3f68-3667-4a93-945a-7947bac1061c-1.wav | {"baseline": "No problem. Just to confirm, would you like to continue now, or would you prefer we schedule a time later… | Hi Jordan, this is BayVista Bank. I’m calling about a pending decision on overdraft coverage for ATM and one-time debit card… | recovery tokenF1=0.137 |
| 1 | t4_interruption_ihbench_ee2bd2649b72 | d5cbea6c-e559-44bc-9099-260ae0cd44e8:19 / 19× d5cbea6c-e559-44bc-9099-260ae0cd44e8-1.wav; d5cbea6c-… | {"baseline": "Thanks for confirming. Let me quickly summarize everything before I send the email.\n\nWe have Nordcrest … | Can you confirm the HR ops email is hr ops at nordcrest dot com? | recovery tokenF1=0.1504 |
| 2 | t4_interruption_ihbench_7431aa7bb4d8 | 8c5bfe54-d23d-44d9-ad40-58c9b83371a7:3 / 3× 8c5bfe54-d23d-44d9-ad40-58c9b83371a7-1.wav; 8c5bfe54-d… | {"baseline": "Property taxes around Lake Merritt, like most of California, typically range around 1% to 1.25% of the pr… | Great question. Property taxes in the Lake Merritt area—like most of Alameda County—can vary quite a bit by neighborhood. On… | recovery tokenF1=0.3605 |
| 3 | t4_interruption_ihbench_a6932fd04dea | 79941712-5598-4666-97da-c72049ed7a2c:7 / 7× 79941712-5598-4666-97da-c72049ed7a2c-1.wav; 79941712-5… | {"baseline": "No problem! I’ll update that to daniela.ruiz+contracts@asterionlabs.com.\n\nLet me read that back: d a n … | Understood, I’ll use daniela.ruiz+contracts@asterionlabs.com. I have Daniela Ruiz, Director of Finance, with email daniela d… | recovery tokenF1=0.3478 |
| 4 | t4_interruption_ihbench_6b3ff5a6b55e | a0e06e0d-2d4f-4174-8a9d-91c8eeb2dadc:10 / 10× a0e06e0d-2d4f-4174-8a9d-91c8eeb2dadc-1.wav; a0e06e0d-… | {"baseline": "I understand you’re curious about the pool hours and whether it’s heated. Unfortunately, I don’t have liv… | I know you’ve asked about the pool, and I want to help. The pool at Lakeside Grand Hotel Chicago is heated and typically ope… | recovery tokenF1=0.2333 |
| 5 | t4_interruption_ihbench_4c8a473caa5b | 3fe4cbd2-14a6-42ef-bf7d-bdac50631f63:1 / 3fe4cbd2-14a6-42ef-bf7d-bdac50631f63-1.wav | {"baseline": "No problem at all. An address verification mismatch means that the system flagged a difference between th… | No, your address isn’t necessarily wrong. An Address Verification Service (AVS) mismatch means that the billing address we h… | recovery tokenF1=0.4493 |
| 6 | t4_interruption_ihbench_8eab9fe1e905 | cd9025f7-9324-4d2c-a4bc-92b5c6c0db90:5 / 5× cd9025f7-9324-4d2c-a4bc-92b5c6c0db90-1.wav; cd9025f7-9… | {"baseline": "No, soreness is common and not considered severe. Thanks for clarifying.\n\nNext, are you currently exper… | No, soreness at the injection site is common and not considered severe. That’s fine. Now, are you currently experiencing a f… | recovery tokenF1=0.6538 |
| 7 | t4_interruption_ihbench_932619109aab | 3b1fb3b1-b092-4dd3-95b9-9a7c90944582:7 / 7× 3b1fb3b1-b092-4dd3-95b9-9a7c90944582-1.wav; 3b1fb3b1-b… | {"baseline": "Absolutely, keeping it simple. That’s correct: $12 a day plus local taxes, UK only, and you only pay on d… | Understood. I’ll activate the Global DayPass for the UK only, from March nineteenth to twenty seventh. It’s twelve dollars a… | recovery tokenF1=0.4259 |
| 8 | t4_interruption_ihbench_efc5f6e6a129 | e7ef65fe-234d-4ff6-9f76-518733225e3a:13 / 13× e7ef65fe-234d-4ff6-9f76-518733225e3a-1.wav; e7ef65fe-… | {"baseline": "Yes, we can definitely add your Australia ETA reference now. That will help keep things on track. Once yo… | Yes, you can provide your Australia ETA reference now, and then send your passport details via the secure SMS form later tod… | recovery tokenF1=0.5273 |
| 9 | t4_interruption_ihbench_740816a0f864 | 18b0c81b-057e-41a7-99dd-fb62c8372acc:19 / 19× 18b0c81b-057e-41a7-99dd-fb62c8372acc-1.wav; 18b0c81b-… | {"baseline": "Good question. The total of fifty-three thousand eight hundred sixty-five dollars is before any applicabl… | The fifty three thousand eight hundred sixty five is the base amount before tax or fees. Sales tax will be added based on yo… | recovery tokenF1=0.5248 |

## 问题

### 问题 丙-IN-01

- benchmark：ihbench
- 发现时间：2026-09-03

样例
- sample_id：t4_interruption_ihbench_fc0811daef3d
- 音频如何定位：/mnt/afs/users/shizs/capability_t4_run_20260903/audio/ihbench/f16d3f68-3667-4a93-945a-7947bac1061c-1.wav
- ref：{"baseline": "No problem. Just to confirm, would you like to continue now, or would you prefer we schedule a time later…
- pred：Hi Jordan, this is BayVista Bank. I’m calling about a pending decision on overdraft coverage for ATM and one-time debit card transactions on your checking account. Before we discuss any account details, I’ll need to verify your identity
- 当前指标：recovery tokenF1=0.137
- 必要摘录：native_id=f16d3f68-3667-4a93-945a-7947bac1061c:1

原因
当前脚本以离线消息列表一次性生成文本；没有检测用户插入的时间点、停止正在播放的 assistant 音频、取消未完成工具、测量 detection/stop/resume latency，也无法观察原任务是否错误继续。

解决方案
- 建议动作：实现 streaming harness，记录事件时间线（user speech onset、detector、cancel ack、new response）和播放/工具状态；接入 IHBench 官方 tf_win_rate、rq_pass_rate judge。
- 若涉及数据：不修改原始数据；仅提出字段、ontology、rubric 或评测实现方案。
- 方案是否确定：方案待定（需实时运行时/设备接口）

### 问题 丙-IN-02

- benchmark：ihbench
- 发现时间：2026-09-03

样例
- sample_id：t4_interruption_ihbench_8eab9fe1e905
- 音频如何定位：/mnt/afs/users/shizs/capability_t4_run_20260903/audio/ihbench/cd9025f7-9324-4d2c-a4bc-92b5c6c0db90-1.wav；/mnt/afs/users/shizs/capability_t4_run_20260903/audio/ihbench/cd9025f7-9324-4d2c-a4bc-92b5c6c0db90-2.wav；/mnt/afs/users/shizs/capability_t4_run_20260903/audio/ihbench/cd9025f7-9324-4d2c-a4bc-92b5c6c0db90-3.wav；…共 5 个音频（完整列表见 samples.jsonl）
- ref：{"baseline": "No, soreness is common and not considered severe. Thanks for clarifying.\n\nNext, are you currently exper…
- pred：No, soreness at the injection site is common and not considered severe. That’s fine. Now, are you currently experiencing a fever or any acute illness, like a cold or flu?
- 当前指标：recovery tokenF1=0.6538
- 必要摘录：native_id=cd9025f7-9324-4d2c-a4bc-92b5c6c0db90:5

- sample_id：t4_interruption_ihbench_7431aa7bb4d8
- 音频如何定位：/mnt/afs/users/shizs/capability_t4_run_20260903/audio/ihbench/8c5bfe54-d23d-44d9-ad40-58c9b83371a7-1.wav；/mnt/afs/users/shizs/capability_t4_run_20260903/audio/ihbench/8c5bfe54-d23d-44d9-ad40-58c9b83371a7-2.wav；/mnt/afs/users/shizs/capability_t4_run_20260903/audio/ihbench/8c5bfe54-d23d-44d9-ad40-58c9b83371a7-3.wav
- ref：{"baseline": "Property taxes around Lake Merritt, like most of California, typically range around 1% to 1.25% of the pr…
- pred：Great question. Property taxes in the Lake Merritt area—like most of Alameda County—can vary quite a bit by neighborhood. On average, homeowners pay about 1% of the home’s assessed value annually, but that’s just a baseline. Some neighborhoods have higher assessments due to proximity to downtown Oakland, better schools, or newer construction, which can push taxes higher. For example, homes near the lakefront or in t…
- 当前指标：recovery tokenF1=0.3605
- 必要摘录：native_id=8c5bfe54-d23d-44d9-ad40-58c9b83371a7:3

原因
baseline token F1（如 0.6538/0.3605）只表示词面相似；它不等价于 rubric 中的 fresh start、continuation coherence、信息保留或安全要求。不同 interruption_type 也不应混合成一个平均数解释。

解决方案
- 建议动作：按 type 分层运行官方 rubric judge，分别报告 tf、rq、latency 和错误恢复类别；token F1 只保留为可复算辅助指标。
- 若涉及数据：不修改原始数据；仅提出字段、ontology、rubric 或评测实现方案。
- 方案是否确定：确定（指标定位）；正式数值待实时/judge）

## 复现与边界

- 本轮 10/10 推理无错误，音频物化在运行目录 `audio/ihbench/`；该目录不提交仓库。
- 由于没有实时播放与打断事件，正式 capability 结论保留为本轮不评。
