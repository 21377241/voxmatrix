# Capability：speaker_attribution（说话人归属）检验记录

负责人：甲（T3）  
日期：2026-09-07；工程修复与文档对齐：2026-09-09  
模型：Qwen3-Omni-30B-A3B-Instruct（`qwen3-omni-local`）  
Smoke Job：  
- **15278**（SUCCEEDED）：主 suite 五集 ×10（aishell_4/5、alimeeting、ami、cn_celeb）  
- **16421**（SUCCEEDED）：CHiME-6 **原轨裁剪窗** ×10（`results_chime6_crop/`）  
产物：  
- 原始推理：`VoxMatrix/smoke_attr_all/results/`、`results_chime6_crop/`  
- **工程抢救后主报**：`VoxMatrix/smoke_attr_all/results_rescored/`（离线重打，未重推理）

> **协议 vs 本轮实际**  
> 正式 speaker_attribution = 多说话人音频 → 带 speaker 的转写 utterances → **cpCER%（主）** / attribution_acc（辅）。  
> 本轮因 Omni 无法稳定处理数十分钟长会，统一切 **30s 多说话人窗**（多数复用 `smoke_diar_all/clips`）；AMI 仍为句级 ASR **utt overlay**；CHiME-6 已改为连续 ref-array **原轨裁剪**。  
> **cn_celeb** 实为 SID，**本轮不评 attribution**。

> **如何听样例音频**  
> 请听已切好的 **16-bit / 16k / 单声道 / 30s** wav（与 diar 共用）：  
> `VoxMatrix/smoke_diar_all/clips/{aishell_4,aishell_5,alimeeting,ami,chime_6}/`  
> 勿直接打开会议原轨（多通道 / 超长）。ASR 短句听音目录 `问题记录_听音样例/` 与本任务窗级协议不同，仅作旁证。

---

## 状态总览（2026-09-09 更新）

| # | 问题 | 状态 | smoke 当前指标 | 下一步 |
| --- | --- | --- | --- | --- |
| 甲-ATTR-01 | 长会不可整段喂 Omni | 📝 **adapted 已落地** | 统一 30s 窗；会议四集可出分 | 正式评测再定窗长/滑窗策略；本轮分数按 30s 协议解读 |
| 甲-ATTR-02 | AMI overlay / CHiME 原轨 | 📝 **部分收口** | AMI=`utt_overlay_sdm`；CHiME=`ref_array_window`（Job 16421） | AMI 若将来有连续 Mix 可改裁剪；CHiME 已可与 diar crop 对齐 |
| 甲-ATTR-03 | cn_celeb = SID 错挂 | ✅ **本轮不评** | SID accuracy **0**（10/10 复读 `idXXXXX`） | 从 attribution 榜剔除或单列 SID |
| 甲-ATTR-04 | pred 非法/截断 JSON 抬高 cpCER | ✅ **已收口（评测侧）** | 见下表 rescored；invalid 会议集 →0 | 可选：guided decoding / 加长 max_tokens（**框架未做、无 A/B**） |
| 甲-ATTR-05 | CHiME 英文归一化后空 ref 崩评测 | ✅ **已收口** | 原 Job 16421 failure **20%**；评测侧丢弃归一化后无词 gold 后 10/10 可评，cpCER≈**73.4%** | 无（与 ASR filler 口径对齐；见下） |

**主 suite 可直接参考的 smoke 集（rescored cpCER）**：aishell_4、alimeeting、ami。  

**可评但偏弱 / 难例**：aishell_5（车载口语）、chime_6 crop（多人远场）。  

**仅 diagnostic / 不评**：cn_celeb（ATTR-03）。

**口径**：排行看 **cpCER%↓**；`attribution_acc` 仅辅分（pred 常无 utterance id、句数 ≠ gold 时按下标对齐，易虚低）。  
**主报数字**：会议四集用 `results_rescored/`；CHiME 用离线抢救均值（见汇总表）。Job 15278/16421 目录内 overall 为**修复前**原始分。

---

## Smoke 汇总

### 会议四集（Job **15278** 原始 → `results_rescored`）

| Benchmark | failure_rate | 原 cpCER% | **rescored cpCER%** | 原 attr_acc | rescored attr_acc | invalid 行 | 结论 |
|-----------|-------------:|----------:|--------------------:|------------:|------------------:|-----------:|------|
| aishell_4 | 0% | 61.6 | **59.8** | 0.55 | 0.58 | 1→0 | ✅ 可评 |
| aishell_5 | 0% | 77.6 | **66.1** | 0.32 | 0.44 | 3→0 | ✅ 可评但弱；含截断胡言 |
| alimeeting | 0% | 56.8 | **45.9** | 0.26 | 0.37 | 3→0 | ✅ 可评；工程修复收益最大 |
| ami | 0% | 61.0 | **61.0** | 0.48 | 0.48 | 0→0 | ✅ 可评（顶层数组原本就可解析） |
| cn_celeb | 0% | — | — | SID acc **0** | — | — | ❌ **不评 attribution** |

### CHiME-6 原轨裁剪（Job **16421**）

| Benchmark | failure_rate | 原 cpCER%（8 条有效） | **rescored cpCER%**（同 8 条） | invalid / salvage | 结论 |
|-----------|-------------:|---------------------:|------------------------------:|-------------------|------|
| chime_6 | 原 **20%**（ATTR-05×2）→ 评测修后 **0** | 74.0（仅 8 条） | **~73.4**（10/10，含 ATTR-04/05 修） | 非法 JSON + 事件-only speaker 已可评 | 可作难例 |

人核：**链路通；Omni 能做带 speaker 的转写，但 cpCER 偏高（约 46–66% 主集）。**  
主分建议看 **alimeeting / aishell_4 / ami（rescored）**；aishell_5、chime 作难例；**cn_celeb 不进总评**。

---

## 问题列表

### 甲-ATTR-01. 长会不可整段喂 Omni → 统一 30s 多说话人窗


| 字段 | 内容 |
| --- | --- |
| 发现时间 | 2026-09-07 |
| 场景 | `smoke_attr_all`（与 `smoke_diar_all` 同窗策略） |
| 相关产物 | `smoke_attr_all/*/manifest.jsonl`；音频多在 `smoke_diar_all/clips/` |
| 严重程度 | 高（协议 adapted；分数不能直接当「整场会议 attribution」） |
| 状态 | 📝 **adapted 已落地**：30s 窗可复现出分；正式整场协议未做 |


#### 现象

与 diarization 相同：AISHELL-4 / AliMeeting 等原是长会 + 时间线标注。整段喂 Omni 不稳定，smoke 改为：

1. 选取含多人的 **30.0 s** 窗；
2. `WavPath` 指向预切 16k 单声道 wav；
3. gold 为窗内 utterances：`{"utterances":[{"id","speaker","text"},...]}`（相对窗起点的时序文本归属）。

| 数据集 | protocol / subset | 音频来源 | 窗长 |
| --- | --- | --- | ---: |
| aishell_4 | `textgrid_window` / test | diar clips | 30 s |
| aishell_5 | `msswift_window` / eval1 | attr 自建 clips | 30 s |
| alimeeting | `textgrid_window` / Eval\|Test_Ali_far | diar clips | 30 s |
| ami | 见 ATTR-02 | diar clips | 30 s |
| chime_6 | 见 ATTR-02 | diar crop clips | 30 s |

#### 样例（听这个文件）

| 文件 | 说明 |
| --- | --- |
| `smoke_diar_all/clips/aishell_4/aishell4_diar_L_R003S01C02_w0000.wav` | 30s 窗，protocol=`textgrid_window` |
| `smoke_diar_all/clips/alimeeting/` 下 `alimeeting_diar_*_w*.wav` | 远场会议窗 |

#### 影响

1. 指标反映的是 **短窗归属+转写**，不是整场会议端到端。
2. 与 ASR 的「句级 utt_clips」不同：本任务窗内常有多句多人，更接近 diar 窗。
3. 未切片的长会路径若误用，会出现与 ASR #1 同类的虚高错误率——本轮已避免。

#### 建议方向

1. 正式评测明确写清 measurement protocol：`30s_window_attribution`。
2. 可选滑窗聚合 / 更长窗，需单独验证 Omni 长度与截断行为。
3. 报告中标注 **adapted**，勿与「整场」协议混比。

#### 收口结果

- 构建：`smoke_attr_all/build_manifests.py`（`WIN=30`）  
- Job 15278 会议四集 **failure_rate=0**（链路通）  
- **未**改为整场评测；状态保持 adapted 说明，而非「协议已正式化」。

---

### 甲-ATTR-02. AMI 句级 overlay；CHiME 改为原轨裁剪


| 字段 | 内容 |
| --- | --- |
| 发现时间 | 2026-09-07（AMI/CHiME 同源问题）；CHiME 改裁剪 2026-09-09 |
| 场景 | `smoke_attr_all` → ami / chime_6 |
| 相关产物 | ami：`protocol=utt_overlay_sdm`；chime：`protocol=ref_array_window`，Job **16421** → `results_chime6_crop/` |
| 严重程度 | 中（协议真实性）；AMI 仍 adapted |
| 状态 | 📝 **部分收口**：CHiME ✅ 原轨裁剪；AMI 仍 overlay |


#### 现象

1. **AMI**：磁盘上主要是句级 wav（HF 导出），无连续 Mix-Headset 可供裁剪 → smoke 用 utt 时间线 **叠加重建** 30s 窗（数字静音填空隙、多 utt 混叠），与真实会议室连续拾音有差。  
2. **CHiME-6（旧）**：打包 ref_array 短片曾与「utt overlay」混用，和连续会场拾音不完全一致。  
3. **CHiME-6（现）**：连续 session 在  
   `/mnt/afs/oss_data/datasets/03_multi_speaker/CHiME/.../audio/`  
   可按窗对 ref-array CH1–4 **等权平均** 后裁 30s；gold 仍来自 utt 时间线文本。attr 复用 diar 的 crop clips。

| 集 | 当前 protocol | 说明 |
| --- | --- | --- |
| ami | `utt_overlay_sdm` / `sdm_overlay_window` | adapted |
| chime_6 | `ref_array_window` / `session_window` | 与连续轨对齐的裁剪 |

#### 样例

| 听这个文件 | protocol |
| --- | --- |
| `smoke_diar_all/clips/ami/ami_diar_en2002a_w0000.wav` | utt overlay 30s |
| `smoke_diar_all/clips/chime_6/chime_6_diar_S01_w0000.wav` | ref_array 连续裁剪 30s |

#### 影响

1. AMI 分数含 **重建失真**，不宜当作「真实 SDM 连续轨」能力上界。  
2. CHiME crop 后仍难（rescored cpCER **~71%**），偏模型/场景难度，而非单纯 overlay 伪影。  
3. CHiME Job 16421 曾有 **ATTR-05** 空 ref 崩评测（已修）；failure 与归属能力需分开看。

#### 建议方向

1. AMI：若补齐连续 Mix，改为与 CHiME 相同的 window crop。  
2. CHiME：保持 `ref_array_window`；attr/diar 共用 clips。  
3. 报告分列 protocol，禁止 overlay 与 crop 混平均成「CHiME 官方分」。

#### 收口结果

- CHiME attr：**16421** 已跑通 crop；ATTR-04/05 修后离线 **10/10** 可评，cpCER ≈ **73.4%**。  
- AMI：仍 overlay；cpCER **61.0%**（rescored 不变）。

---

### 甲-ATTR-03. cn_celeb 错挂为 attribution（实为 SID）


| 字段 | 内容 |
| --- | --- |
| 发现时间 | 2026-09-07 |
| 场景 | `smoke_attr_all` → cn_celeb |
| 相关产物 | `smoke_attr_all/cn_celeb/manifest.jsonl`（`protocol=sid_mislabeled_as_attribution`）；`results/cn_celeb.jsonl` |
| 严重程度 | 高（任务定义错误；分数不可解释为归属） |
| 状态 | ✅ **本轮不评 attribution** |


#### 现象

- 样本为单说话人短音频 + gold `speaker_id`（如 `id10001`）。  
- 这是 **说话人识别（SID）**，不是「多人转写 + 匿名 speaker_1/2 归属」。  
- Prompt 用 `mesh-sid-diagnostic`；模型 10/10 输出占位 `{"speaker_id":"idXXXXX"}`（复读形态），**accuracy=0**。

#### 样例

| id | ref | pred（摘要） | accuracy |
| --- | --- | --- | ---: |
| 0 | `id10001` | `{"speaker_id":"idXXXXX"}`（带 fence） | 0 |
| 5 | `id10019` | 同上 | 0 |

音频示例：manifest 中 `CN-Celeb2_flac/.../id10001/live_broadcast-01-001.flac`。

#### 影响

1. 若把 cn_celeb 算进 attribution 总评，会系统性污染结论。  
2. 0 分反映的是 **错任务 + 占位输出**，不是会议归属能力。

#### 建议方向

1. ~~继续放在 attribution suite 当主分~~ → **本轮已剔除**。  
2. 若要评 SID，单列 capability / suite，换闭集或检索式协议与 prompt。

#### 收口结果

- 状态总览与汇总表均标 **不评**。  
- `eval_task` 对该集 `post_process: []`，不走 attribution normalize。

---

### 甲-ATTR-04. 输出格式不稳定：非法 JSON / fence / 截断抬高 cpCER


| 字段 | 内容 |
| --- | --- |
| 发现时间 | 2026-09-07 |
| 修复时间 | 2026-09-09 |
| 场景 | `smoke_attr_all` 会议集 + chime_6 |
| 相关产物 | Job 15278/16421 原始 `results*`；抢救代码与 `results_rescored/` |
| 严重程度 | 高（单条直接 cpCER=100 / attribution_valid=0，拉高均值） |
| 状态 | ✅ **已收口（评测侧抢救 + prompt 约束）**；推理侧 guided / 加长 token **未做** |


#### 现象

Omni 常能写出「像 attribution」的内容，但 JSON 不稳定：

1. **顶层裸数组** `[{speaker,text},...]` 而非 `{"utterances":[...]}`（多数集；AMI 原先已能兼容解析）。  
2. **markdown fence**：\`\`\`json ... \`\`\`。  
3. **漏键名**（主因）：前几条合法后写成 `{"speaker_1","text":"..."}`（缺 `"speaker":`）→ 整段 `json.loads` 失败 → **整条按无效计 cpCER=100**。  
4. **中途截断**：如 aishell_5 id=1 连发「嗯？」写到半个 object 断开（~2.6k 字符）。

Job **15278** 跑时 **尚未**挂 normalize：`post_process` 内容与 raw 相同；故目录内 overall 是修复前分数。

#### 样例

| 集 | id | 音频（clips 名） | 问题 | 原 cpCER |
| --- | --- | --- | --- | ---: |
| alimeeting | 4 | `alimeeting_diar_R8004_M8006_w0100.wav` | 中段起 `{"speaker_1","text":...}` | 100 |
| aishell_4 | 6 | `aishell4_diar_M_R003S04C01_w0780.wav` | 同上 | 100 |
| aishell_5 | 1 | `aishell5_attr_003_DX01C01_w0440.wav` | fence + 截断在 `"speaker": "speaker_1",` | 100 |
| chime_6 | 6 | `chime_6_diar_S21_w1720.wav` | fence + 漏键名 | 100 |

非法片段形态：

```text
{"speaker_1", "text": "是送个装备什么的"}
```

应变为：

```text
{"speaker": "speaker_1", "text": "是送个装备什么的"}
```

#### 影响

1. 工程解析失败被算成「完全不会做」，**高估错误率**（alimeeting 原 56.8 → 抢落后 45.9）。  
2. 截断后若内容已是胡言（连发「嗯？」），抢救只能保住**已写完整**的 utterance，**补不回**语义——仍属模型能力。  
3. `failure_rate` 仍可能为 0（评价器返回 cpCER=100 而非抛错），易误判「全成功」。

#### 修复（已落地）

| 项 | 位置 |
| --- | --- |
| `normalize_speaker_attribution_prediction` | `mesh_eval/evaluator/speaker.py`：去 fence → 补 `"speaker":` → `extract_json` → 截断正则捞完整对象 → 包 `utterances` |
| `SpeakerAttributionNormalize` | `mesh_eval/process/attribution.py` |
| smoke `post_process` | `mesh-speaker-attribution-normalize`（`registry/eval_task/smoke_attr_all.yaml`，除 cn_celeb） |
| Prompt | `smoke_attr_all/registry/prompt/mesh-speaker-attribution.yaml`（强制键名、禁 fence、禁裸数组） |
| 离线重打 | `python smoke_attr_all/rescore_results.py` → `results_rescored/` |

**未做（无效果数据）**

| 项 | 说明 |
| --- | --- |
| Guided decoding / JSON schema 约束 | `Qwen3Omni` / smoke **未实现** |
| 提高 attr `max_new_tokens` | 模型 yaml **未设**；`thinker-max-new-tokens` 默认 0=模型默认。错键名是主因，加长对 ATTR-04 主体帮助有限 |

#### 收口结果（相对 Job 15278 原始 overall）

| Benchmark | 原 cpCER% | rescored | invalid | salvaged 行 |
|-----------|----------:|---------:|--------:|------------:|
| aishell_4 | 61.6 | **59.8** | 1→0 | 1 |
| aishell_5 | 77.6 | **66.1** | 3→0 | 3 |
| alimeeting | 56.8 | **45.9** | 3→0 | 3 |
| ami | 61.0 | 61.0 | 0→0 | 0 |
| chime_6（16421） | 74.0（8 条） | **~73.4**（10/10） | ATTR-04+05 | 含非法 JSON 抢救 + 事件-only 过滤 |

- **确定**：评分解析层必须可抢救；prompt 保留但不依赖模型自觉。  
- 新 GPU 重跑应走已挂 `post_process` 的 `eval_task`；旧 `results/` overall **不要**当主报。  
- 截断胡言、归属错误、转写错误 → 仍记入 cpCER，不算本工程项。

---

### 甲-ATTR-05. CHiME 英文文本归一化后空 ref → 评测契约失败


| 字段 | 内容 |
| --- | --- |
| 发现时间 | 2026-09-09（Job 16421） |
| 修复时间 | 2026-09-09 |
| 场景 | `smoke_attr_all` → chime_6 crop |
| 相关产物 | `results_chime6_crop/chime_6.jsonl` error 记录；`mesh_eval/evaluator/speaker.py` |
| 严重程度 | 中（2/10 直接 failure；拉高 failure_rate，且与归属能力无关） |
| 状态 | ✅ **已收口**（评测侧） |


#### 现象

**不是**整段 30s 都是 noise。窗内多数人有实词；个别 gold speaker 在窗内**只有**事件标记，例如：

- id4 `S01_w7460`：P04 仅 `[laughs]`（其余 P01–P03 有 Yolo/Mustache…）  
- id7 `S21_w3580`：P48 仅 `[noise]`（其余有 nachos 对话）

`compute_wer` 按 speaker 拼接后做英文归一化（会删 `[laughs]`/`[noise]`/`[inaudible…]`），该 speaker ref 变空 → 抛出：

`ValueError: WER reference corpus is empty after normalization`

| id | sample_id | failure_stage | failure_type |
| --- | --- | --- | --- |
| 4 | `chime_6_attr_S01_w7460` | evaluation | contract_violation |
| 7 | `chime_6_attr_S21_w3580` | evaluation | contract_violation |

→ 原 overall **failure_rate=0.2**；cpCER 只在其余 8 条上平均。

#### 影响

1. 失败条不进 cpCER 均值，但又抬高 failure_rate。  
2. 与归属能力无关，属评测对「事件-only 说话人」未设防。

#### 处理口径（已落地）

**推荐且已采用：评测侧过滤（对齐 ASR filler）**

1. 拼 cpCER 前，对每条 gold utt 调 `is_filler_only_reference`（归一化后无词则丢弃，含纯 `[laughs]`/`[noise]`）。  
2. 若某 speaker 丢光，则不再参与置换。  
3. 调用 `compute_wer` 前再兜一层：normalize-empty 当作空 ref（代价 0/1），**禁止抛 contract_violation**。  
4. 元数据：`attribution_ref_utt_dropped`。

**不推荐**：把整窗标 invalid / 重裁「无笑声的窗」——会误伤大量真实对话，且笑声叠在别人话上很常见。

**可选（未做）**：`build_manifests` 写 gold 时就去掉事件-only utt，使 `n_speakers` 与可计分说话人一致；评测侧已够用。

#### 收口结果

- 离线重评原 16421 pred：原失败 id4/7 → **valid=1**（cpCER 78.3 / 83.0）；全 10 条均值 ≈ **73.4%**，无 exception。  
- 代码：`SpeakerAttributionEvaluator`（`mesh_eval/evaluator/speaker.py`）。

---

## 统一 template

- **任务形态**：单段音频（本轮 30s 窗）→ JSON  
  `{"utterances":[{"speaker":"speaker_1","text":"..."}, ...]}`（时序；匿名 speaker id）  
- **Prompt**：`mesh-speaker-attribution`  
- **post_process**：`mesh-speaker-attribution-normalize`（会议集 / chime；cn_celeb 除外）  
- **evaluator**：`mesh-speaker-attribution`  
- **主指标**：`cpcer%`↓；辅：`attribution_acc`↑  
- **模型配置**：`smoke_attr_all/registry/model/qwen3_omni_local.yaml`（未设 `max_new_tokens`）

---

## 三类问题（本轮结论骨架）

### 1. 模型能力

- **能做**：输出带 speaker 标签的转写；低重叠会议窗相对最好（alimeeting rescored **45.9%**）。  
- **弱项**：车载/远场/多人（aishell_5、chime）；截断后崩溃成「嗯？」循环；SID 占位复读（错任务场景）。  
- **分层（rescored cpCER↓）**：alimeeting **45.9** ≪ aishell_4 **59.8** ≈ ami **61.0** ≪ aishell_5 **66.1** ≪ chime **~73.4**。

### 2. 工程 / 评测

- **ATTR-04**：非法 JSON → ✅ 评测侧已收口（rescored 主报）。  
- **ATTR-05**：事件-only gold speaker → ✅ 评测侧已过滤。  
- **未做**：guided decoding、attr 专用加长 `max_tokens`（无 A/B）。

### 3. 数据 / 协议

- **ATTR-01**：30s adapted，非整场。  
- **ATTR-02**：AMI overlay 仍在；CHiME crop 已对齐。  
- **ATTR-03**：cn_celeb **不评**。

---

## 附录

| 项 | 路径 / 命令 |
| --- | --- |
| 构建 | `smoke_attr_all/build_manifests.py` |
| Suite | `smoke_attr_all/suite.yaml`、`suite_chime6.yaml` |
| 重打 | `python smoke_attr_all/rescore_results.py` |
| 抢救实现 | `mesh_eval/evaluator/speaker.py` → `normalize_speaker_attribution_prediction` |
| Job | **15278**（主）、**16421**（CHiME crop） |
| 相关 | diar 窗与 clips 见 `Capability_diarization_检验记录.md`；英文空 ref 同类见 `Capability_asr_检验记录.md` #2 |
