# Capability 抽样检验任务说明

> 日期：2026-08-30  
> 对象：三人并行，对每个 capability 下的 benchmark 做实际检验  
> 能力与数据对照：`/mnt/afs/users/wangyl/benchmark_annotation_audit/native_v2_pipeline/outputs/task_inventory_capability.csv`

---

## 1. 目的

评测流程已经跑通，链路能出分。但每个 capability 下挂了多个 benchmark，**现有评测结果并不可靠**：切片、路径类型、音频格式、Prompt/任务形态、归一化、指标口径等细节会把分数做虚或做崩。这些问题只能对着真实样本查。**不要直接改数据**；能判断的提出方案并及时反馈，不能假定「流程通了分数就可信」。

`asr` 与 `long_form_asr` 已按此做过一轮（每集约 10 条），并写进 `/mnt/afs/users/wangyl/VoxMatrix/问题记录.md`，**不必重做听音**。本轮覆盖其余可执行 capability。

**本轮略过：**

- `meeting_summary`
- `todo_extraction`

---

## 2. 需要做的工作

每个负责人对其名下 **每一个 capability**，必须覆盖该 capability 下 **每一个 benchmark**（清单见 §4），不得漏集。

对每个 `(capability, benchmark)`：

1. **抽样 10 条**（尽量跨文件 / 跨会话）。
2. 用 **Qwen3-Omni**（见 §7）跑通推理，拿到 pred 与该集当前评测指标。
3. **人工核验**推理结果、指标、音频和 reference：分数是否在评该能力、有没有把分做虚或做崩。
4. 发现问题：**不要直接改数据**。写成问题说明（含样例、原因、解决方案）并及时反馈；方案不确定的同样及时反馈，不要自行落盘改原始数据或正式 manifest。
5. 该 capability 给出 **统一 template**（该能力下各 benchmark 共用一套任务形态 / Prompt 骨架；选择题是否改成问答等由负责人判断后写入 template）。
6. 该 capability **推荐主指标**（可以不止一个）：说明数据如何从推理结果和 ref 得到指标、为什么选它。核对该能力下各 benchmark **计算指标的方式是否相同**；若存在不同版本（例如 BLEU-1 / BLEU-2、不同 tokenizer 或归一化），把差异写清，并标明推荐用哪一个版本。

`asr` 已做过的集不重复抽听。

---

## 3. 原则（方向约定，写法由各人定稿）

- **覆盖**：每个 capability 下每个 benchmark 都抽 10 条，对照推理、指标、人核。漏集不算完成。
- **统一 template**：每个 capability 交一份该能力共用的 template（任务形态 + Prompt 骨架）。选择题是否改成问答等，由负责人看数据后写入这份 template，本文不预写正文。
- **推荐主指标**：写清数据如何从推理结果和 ref 得到该指标，以及为什么选它。可以不止一个。核对各 benchmark 的计算方式是否相同；有 BLEU-1/2 等不同版本时写明差异并推荐其一。本文不规定指标名。
- **不改数据，只提案并反馈**：原始数据、正式评测 manifest 一律不直接改。需要改的写出方案后及时反馈；方案拿不准的也及时反馈，等对齐后再动。
- **问题必须带样例**：每条问题写清样例、原因、解决方案；解决方案不确定就标明不确定并反馈，不要空着或硬写一个。
- **按集判断**：切片、格式、路径类型等对着样本看。三人之间不必共用同一套 Prompt。

---

## 4. 分工

按 Task 族划分，便于同一人连续处理相近的数据与指标。  
会议类音频在 ASR 轮已听过：甲做 `diarization` / `speaker_attribution` 时 **不必重听整段**，重点看协议、说话人标签与该能力指标是否成立。

MMSU 跨多个 capability：甲先摸清该库各子集长什么样；乙、丙若用到 MMSU 条目，可复用甲已切的听音样例，但 **形态与指标仍由该 cap 负责人自己定**。

VocalBench / VoiceBench / EAR 等一库多 cap：**各抽各的 10 条**（问答实例 ≠ 指令遵循实例）。

### 甲：T1 收尾 + T3 + T5

先核 ASR 轮已发现、尚未收口的问题（长会未切片、MISP pcm、SBCSAE 路径类型等），**只出方案并反馈**，避免同样问题传到说话人任务却无人知晓。其余按自己判断排序。

| Capability | 中文 | Benchmark |
|---|---|---|
| `speech_translation` | 语音翻译 | covost2；mmsu |
| `code_switch` | 语码转换 | mmsu；vocalbench_zh |
| `speaker_count` | 说话人计数 | mmsu |
| `speaker_verification` | 说话人确认 | voxceleb1 |
| `diarization` | 说话人分离 | aishell_4；alimeeting；ami；chime_6；misp |
| `speaker_attribution` | 说话人归属 | aishell_4；aishell_5；alimeeting；ami；cn_celeb |
| `target_speaker` | 目标说话人 | aishell_5 |
| `spoof_detection` | 伪造检测 | mmsu |
| `speech_generation` | 语音生成 | libritts |

`diarization`：10 条仍要抽、要看数据和现有结果。若判断当前模型给不出可用时间戳输出，结论可写「本轮不评」，并说明依据；不能不看就跳过。

### 乙：T2 音频理解

| Capability | 中文 | Benchmark |
|---|---|---|
| `qa` | 语音问答 | ear_wdyl；spoken_squad；uro_bench；vocalbench；vocalbench_zh；voicebench |
| `audio_reasoning` | 音频推理 | mmsu；uro_bench；vocalbench；vocalbench_zh；voicebench |
| `audio_caption` | 音频描述 | audiocaps；clotho |
| `sound_event` | 声音事件 | esc50；mavd_traffic；mmsu；sonyc_ust_v23；tut_sound_events_2017 |
| `acoustic_scene` | 声学场景 | dcase2013_scenes；mmsu；tut_sound_events_2017；tut_urban_acoustic_scenes_2018_mobile |
| `paralinguistic_recognition` | 副语言识别 | mmsu |

### 丙：T4 语音智能体（不含已略过两项）

| Capability | 中文 | Benchmark |
|---|---|---|
| `instruction_following` | 指令遵循 | fluent_speech_commands；slurp；vocalbench；vocalbench_zh；voicebench |
| `tool_call` | 工具调用 | audioagentbench_suite；fluent_speech_commands；slurp；stepeval_audio_toolcall |
| `multi_turn_dialogue` | 多轮对话 | audioagentbench_suite；mtalk_bench；vocalbench；vocalbench_zh；voicebench |
| `clarification` | 澄清追问 | audioagentbench_suite；ear_wdyl |
| `interruption` | 打断恢复 | ihbench |

同一数据集挂到两个 capability（如 SLURP → IF 与 tool_call）时，每个 capability 仍各抽 10 条；template 是否拆成两套由丙判断后写进交付。

---

## 5. 交付物格式

记录可以各自存放，**不规定路径**。**每个 capability 一份记录**即可：template、主指标、该能力下各 benchmark 的核验都写在同一份里，不要每个 benchmark 再单独开文档。

```markdown
# Capability：<id>  （<中文名>）
负责人：
日期：

## 统一 template
- 任务形态：
- Prompt 骨架：
- 选择题是否改为问答（是/否及做法）：
- 多音频 / 多轮约定（无则写无）：

## 主指标
可不止一个。写清下面几件事即可，不必拆很多字段：
- 选了哪些指标、为什么能代表该能力
- 数据怎么流动：推理输出 → （如何解析）→ 与 ref 如何对齐/比较 → 得到指标
- 各 benchmark 算分方式是否相同：当前各集分别用的是哪套实现/哪个版本（如 BLEU-1 与 BLEU-2、不同分词或归一化）。相同则写「各集相同」；不同则列出差异，并写明 **推荐采用的版本**（及原因）

## 各 benchmark 核验
（该 cap 下每个集一小节，抽 10 条后写结论；有问题把问题条附在本份记录末尾）

### <benchmark_id>
- 抽样：10 条
- 当前用的指标及版本：
- 当前指标（数值）：
- 人核结论：pass / 有问题 / 方案待定 / 本轮不评
- 结果是否可信：
- 问题编号（无则写无）：

## 问题（无则写「未发现问题」）
```

### 问题条格式（发现一条写一条，附在该 capability 记录里）

独立编号，例如 `甲-ST-01`。**不要改数据**，只写方案并及时反馈。方案不确定必须标「方案待定」并反馈。

```markdown
### 问题 <编号>
- benchmark：
- 发现时间：

样例
- sample_id：
- 音频如何定位：
- ref：
- pred：
- 当前指标：
- 必要摘录：

原因
（为什么结果不可信，或为什么评错了能力）

解决方案
- 建议动作（数据 / 评测 / template / 指标）：
- 若涉及数据：建议改哪类内容、改成什么（本人不改）
- 方案是否确定：确定 / **方案待定**（待定则写清卡在哪、需要谁拍板）
```

问题或「方案待定」**发现后及时反馈**，不要攒到全部检完再一次性抛出。反馈时带上问题编号，正文仍用上述问题条格式。

---

## 6. 同步与交叉

- 问题发现后及时反馈；方案不确定的立刻提出，不要压到全部检完再一次性抛出。
- **系统性数据病**（路径类型、未切片、打不开的格式、空 ref）发现后立刻告诉另外两人。
- 交叉抽查：各抽对方约 2 个 benchmark、每集 2 条，核对该 capability 的 template 与指标算法是否用到这 10 条上。
- 甲看乙：答案是否在音频里、描述是否被当成转写。
- 乙看丙：工具 JSON / 多轮 / 打断是否按该能力 template 在评。
- 丙看甲：双音频、翻译方向、生成回评是否与自述 template / 指标一致。

---

## 7. 资源索引

### 代码

| 项 | 说明 |
|---|---|
| 仓库 | [https://github.com/yingyingxia666/VoxMatrix](https://github.com/yingyingxia666/VoxMatrix) |
| 分支 | `add capability` |
| 用法 | 本轮检验在该分支上跑推理与评测，不要用别的 fork/分支各跑各的 |

```bash
git clone https://github.com/yingyingxia666/VoxMatrix.git
cd VoxMatrix
git checkout "add capability"
```

### 模型（Qwen3-Omni）

本轮统一用 **Qwen3-Omni**，不要换模型比分。

| 项 | 路径 / 名称 |
|---|---|
| 权重 | `/mnt/afs/models/Qwen3-Omni-30B-A3B-Instruct` |
| Registry 名（smoke 已用） | `qwen3-omni-local` |
| 适配器 | `audio_evals.models.qwen3_omni.Qwen3Omni` |
| 隔离环境 | `/mnt/afs/users/wangyl/VoxMatrix/envs/qwen3-omni` |
| 配置示例 | `/mnt/afs/users/wangyl/VoxMatrix/smoke_asr_all/registry/model/qwen3_omni_local.yaml` |

### 数据集路径

以这份 **subset 级索引** 为准，按 `framework.capability` + `benchmark.id` 找到自己负责的格子，再取路径：

`/mnt/afs/users/wangyl/benchmark_annotation_audit/native_v2_pipeline/outputs/benchmark_subset_manifest.jsonl`

一行是一个 `(benchmark, subset, capability)`，抽 10 条时看这些字段：

| 字段 | 用途 |
|---|---|
| `benchmark.id` | 数据集 id |
| `framework.capability` | 挂到哪个能力 |
| `source.dataset_root` | 数据根目录 |
| `subset.native_index_files` | 原生索引 / 可定位到样本的文件 |
| `subset.split` / `subset.id` | 用哪个 split（优先 test） |
| `protocol.metrics` | 该格子当前登记的指标（核验时对照是否与推荐版本一致） |

不要改原始数据。不必另造一份路径表。

**MMSU 路径纠偏（2026-08-31）：** 索引里旧的 `source.dataset_root`  
`/mnt/afs/oss_data/datasets/07_speech_translation/MMSU` **不存在**。本地可用副本在：

| 项 | 路径 |
|---|---|
| 数据根 | `/mnt/afs/eval_data/benchmarks/MMSU` |
| 原生索引 | `/mnt/afs/eval_data/benchmarks/MMSU/mmsu_audioqa.jsonl`（5000 条，47 子任务） |
| 音频 | 索引里是相对路径 `audio/...wav`，拼到数据根即可；目录里约 5905 个 wav |

按 capability 抽 10 条：打开 jsonl，用 `id` 前缀过滤（与 `subset.selector.prefixes` 一致），例如说话人计数是 `MMSU_total_speaker_counting_`。不要用 `vocal_bench/work/full_eval/mmsu/final/manifest.jsonl` 里的绝对路径，那些仍指向已失效的 `oss_data`。

能力定义（只读）：`/mnt/afs/users/wangyl/benchmark_annotation_audit/交接包_端侧语音Benchmark框架标注统计_0805/01_任务体系/Capability具体定义与任务边界.md`

