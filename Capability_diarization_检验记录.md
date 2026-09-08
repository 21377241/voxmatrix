# Capability：diarization（说话人分离）

负责人：甲（T3）  
日期：2026-09-08（更新；初稿 2026-09-03）  
模型：Qwen3-Omni-30B-A3B-Instruct（`qwen3-omni-local`）  
Smoke Job：  
- **12988**：manifest 配置错误全挂  
- **13148**：prompt-v1（含可复制示例）→ 复读示例，能力不评  
- **16253**（SUCCEEDED，`smoke-diar-all-10-prompt-v2`）：去掉可复制数字示例 → 五集主依据  
- **16297**（SUCCEEDED，`smoke-diar-trunc-dur`）：时长约束 prompt，仅 **alimeeting + chime_6** 验证截断  
产物：`VoxMatrix/smoke_diar_all/results/`；截断重跑：`results_trunc/`（v1 归档：`results_archive_20260908_170228/`）

> **协议 vs 本轮实际**  
> 正式 diarization = 整段会议 → speaker segments + **DER**。  
> 本轮因 Omni 无法稳定处理数十分钟长会，统一切 **30s 多说话人窗** 做 smoke；AMI/CHiME 原生索引是句级 ASR，用 utt 时间线 **overlay 重建** 短窗（adapted）。

---

## 状态总览（2026-09-08，prompt-v2）

| # | 问题 | 状态 | 说明 |
| --- | --- | --- | --- |
| 甲-DIAR-00 | manifest 同时写 `text`+`reference` | ✅ 已修 | Job 12988 |
| 甲-DIAR-01 | 长会不可整段喂 Omni | 📝 | 统一 30s 窗 |
| 甲-DIAR-02 | AMI/CHiME 句级 ASR | 📝 | overlay adapted |
| 甲-DIAR-03 | **MISP 角色层非具名说话人** | ✅ **本轮不评 misp** | gold=`主说话人`/`OVERCLEAR`/…，与 spk diar 错位；见下「三类问题」 |
| 甲-DIAR-04 | prompt-v1 复读示例 `[0,1]` | ✅ 已用 v2 规避 | Job 13148 本轮不评；v2 已出真实时间轴 |
| 甲-DIAR-05 | **模型能力：多人/重叠弱** | 📝 可评但偏弱 | aishell 尚可；ami/alimeeting 中等；chime 差 |
| 甲-DIAR-06 | **工程：JSON 截断（窗外时间戳类）** | ✅ 评测已修；推理时长约束已验证 | Job 16297：alimeeting 截断 2→0；见下 |
| 甲-DIAR-07 | **模型碎切循环 → 撞长度上限** | 📝 残留 | **主因=模型**；难数据为诱因；时长约束挡不住 |

### Smoke 汇总（Job **16253**，每集 10×30s）

| Benchmark | failure_rate | **DER**（时长加权）↓ | 结论 |
|-----------|-------------:|--------------------:|------|
| aishell_4 | 0% | **0.281** | 可评；双人清晰轮替较强 |
| alimeeting | 0% | **0.510** | 可评；含 2 条 JSON 截断拉高 |
| ami | 0% | **0.543** | 可评；低重叠好、高重叠差 |
| chime_6 | 0% | **0.806** | 可评但弱；多人重叠 + 2 条截断 |
| misp | 0% | **1.130** | **本轮不评**（数据协议，非纯能力） |

人核：**链路通；prompt-v2 后可出真实 segments。**  
主分建议看 **aishell_4 / alimeeting / ami**；chime 作难例；**misp 不进总评**。

对比 Job 13148（v1 复读）：五集 DER≈0.97–1.03 → 当时「本轮不评能力分」成立；v2 后改为分层结论。

---

## 三类问题（本轮结论骨架）

### 1. 模型能力问题（甲-DIAR-05）

- **能做**：低重叠、大段轮替（aishell 最好样本 DER≈0.06；ami 最好≈0.09）
- **弱项**：≥3–4 人、far、**重叠重** → 压成 2 人、漏短插话、几乎不建模重叠、1s 碎切
- **分层**：aishell **0.28** ≫ alimeeting **0.51** ≈ ami **0.54** ≫ chime **0.81**

### 2. 工程侧 JSON 截断问题（甲-DIAR-06）

- **现象**：生成陷入等长碎切并写出 **>音频时长** 的时间戳 → 输出半截（约 2k 字符切断）→ `extract_json` 失败 → `_segments=[]` → **DER=1**，且 `failure_rate` 仍为 0
- **出现**：alimeeting **2/10**、chime_6 **2/10**、misp **3/10**；ami/aishell 本轮无
- **已修（2026-09-08）**：
  - prompt 注入 `audio_duration_seconds`，要求 `0≤start<end≤T`，禁止窗外时间戳与 1s 碎切
  - manifest 写 `audio_duration_seconds`
  - `normalize_diarization_prediction`：残缺 JSON 抢救 + 按时长裁剪 + 相邻同说话人合并
  - post_process：`mesh-diarization-normalize`（写入 smoke `eval_task`）
  - 空 pred（相对非空 ref）→ `failure=1` / `diarization_valid=0` / `failure_type=diarization_empty_pred`
  - MeshAgg 增加 `diarization_truncated/salvaged` 计数，以及 `der_valid`（仅 valid 样本）
  - 离线重打：`python smoke_diar_all/rescore_results.py` → `*-overall.rescored.json`
- **离线重打摘要（相对 Job 16253 原始 overall）**：

  | Benchmark | 原 DER | 重打 DER | failure_rate | trunc/salv |
  |-----------|-------:|---------:|-------------:|------------|
  | aishell_4 | 0.281 | 0.281 | 0 | 0/0 |
  | alimeeting | 0.510 | **0.418** | 0 | 2/2 |
  | ami | 0.543 | 0.543 | 0 | 0/0 |
  | chime_6 | 0.806 | **0.775** | 0 | 2/2 |
  | misp | 1.130 | 1.439 | 0.10（1 条空 pred） | 3/3 |

- **推理侧重跑（Job 16297，跳过 misp）**：时长约束 prompt + normalize；产物 `results_trunc/`

  | Benchmark | 16253 DER | 16297 DER | 截断条数 | pred≥2k | max_end>30 |
  |-----------|----------:|----------:|---------:|--------:|-----------:|
  | alimeeting | 0.510 | **0.451** | 2→**0** | 2→0 | 4→1 |
  | chime_6 | 0.806 | **0.779** | 2→**1** | 2→1 | 1→0 |

  结论：时长约束对「写出窗外时间戳」类截断 **有效**；alimeeting 截断清零。

- 方案是否确定：**工程侧已落地**；窗外时间戳类截断 **推理侧已验证改善**

### 2b. 残留截断 / 碎切（甲-DIAR-07）— 主因是模型

Job 16297 后 chime_6 仍剩 **1/10** 截断：`S01_w7460`（id4）。

- **不是**忘了音频时长：`max_end≈23.8 < 30`，prompt 已写明 ≤30s  
- **是**陷入约 **0.4s 等长碎切 + spk0/1/2 轮换** 的自回归循环，写到 ~31 段 / ~2100 字符撞上生成长度上限，JSON 半截断开  
- 耗时≈127s；后处理可抢救段，但 DER 仍≈0.99（碎切与真实边界无关）

**归因**：

| | 作用 |
|--|------|
| **模型（主因）** | 边界不清时改用细片轮换「假装 diar」，并自回归放大；清晰轮替样本极少碎切 |
| **数据（诱因）** | chime 多人重叠/嘈杂更容易触发；gold **并未**要求 0.4s 一切，不是标注协议问题 |

与甲-DIAR-03（misp 数据错位）不同：碎切是 **模型生成行为**，难数据只是暴露得更明显。

**后续（可选）**：段数上限 / 更强「禁止等长碎切」、repetition penalty、早停；单靠「≤T 秒」不够。

### 3. misp 数据问题（甲-DIAR-03）

- TextGrid 仅有 **内容层 + 角色层**，无具名说话人层
- gold 的 `speaker` 实为角色/事件：`主说话人`、`OVERCLEAR`、`OVERLAP`、`cough`、`moving sound`
- 标注稀疏（大量 `<NOISE>` 不进 gold）→ pred 在静音区假阳 → **DER 可 >1**（样本最高≈3.3）
- 与 prompt「输出 spk0/spk1…」任务定义不一致 → **本轮不评**；若要坚持需换 speaker-level 参考后再跑

---

## 统一 template

- **任务形态**：单段音频（本轮 30s 窗）→ JSON `segments[{start,end,speaker}]`（秒，相对窗起点）
- **Prompt**（`smoke_diar_all/registry/prompt/mesh-diarization.yaml`，v2）：无具体可抄数字示例；含时长约束、禁止碎切/窗外时间戳
- **选择题是否改为问答**：**否**
- **多音频 / 多轮约定**：单音频；无多轮

---

## 主指标

### 推荐主指标

| 指标 | 角色 | 原因 |
|------|------|------|
| **DER** | **推荐主分** | 协议标准；`mesh-diarization` + `MeshAgg` 参考时长加权 |

### 算分说明

1. 解析 pred/ref 的 `segments`（失败时可抢救完整对象；再按音频时长裁剪）
2. 边界切时间片 + 最优 speaker 置换
3. 样本：`DER = error_duration / reference_duration`（pred 在 gold 静音上说话也可计入 error → **DER 可 >1**）
4. 语料：Σ error / Σ reference（`reference_duration_weighted`）
5. **越小越好**

### 数据流

```
30s 窗 + Prompt → Qwen3-Omni → JSON segments
  → DiarizationEvaluator → der
  → MeshAgg → overall/der
```

---

## 各 benchmark 核验（Job 16253）

协议/抽样见 `smoke_diar_all/*/manifest.jsonl` 与 `clips/`。

### aishell_4（meeting，RTTM 窗）— DER **0.281**
- 相对最强；双人清晰对话可到很低 DER；≥3 人/重叠仍差
- 问题：甲-DIAR-05（能力分层上端）

### alimeeting（meeting，far TextGrid 窗）— DER **0.510** → 时长约束重跑 **0.451**（16297）
- 双人少插话可到 0.14–0.30；多人/重叠到 0.5–0.7
- Job 16253：**2/10 截断**；Job 16297：**0/10 截断**（甲-DIAR-06 缓解）
- 问题：甲-DIAR-05；甲-DIAR-06（窗外截断已改善）

### ami（meeting，SDM overlay adapted）— DER **0.543**
- 两极分化：最好≈0.09，高重叠窗 0.7–0.8；**无截断**
- 问题：甲-DIAR-02、甲-DIAR-05

### chime_6（home，ref_array overlay adapted）— DER **0.806** → 时长约束重跑 **0.779**（16297）
- 无样本 DER&lt;0.5；多人重叠为主
- Job 16253：2/10 截断；Job 16297：**1/10** 残留（`S01_w7460` 碎切撞长度，甲-DIAR-07）
- 问题：甲-DIAR-02、甲-DIAR-05、甲-DIAR-07

### misp（home，角色层窗）— DER **1.130** → **本轮不评**
- gold 非具名说话人；稀疏标注导致 DER>1 常见；另有截断与空 `segments`
- 问题：甲-DIAR-03（主因）、甲-DIAR-06

---

## 问题列表（详述）

### 甲-DIAR-00. `reference` 字段与 EvalTask 参数冲突（✅）

- Job 12988 全挂：`got multiple values for argument 'reference'`
- 修复：manifest 只保留 `text={"segments":...}`

### 甲-DIAR-04. prompt-v1 复读示例（✅ 已用 v2 替代）

- Job 13148：约 48/50 条 pred=`[{start:0,end:1,speaker:speaker_1}]`，DER≈1 无区分度 → 当时「本轮不评」
- Job 16253：去掉可抄数字示例后，模型输出真实多段时间轴

### 甲-DIAR-05. 模型能力：清晰轮替可、重叠/多人弱（📝）

见上文「三类问题 §1」与各集 DER。正式榜仍建议更大样本 + 专用 diar 管线对照。

### 甲-DIAR-06. JSON 截断（窗外时间戳类）（✅ 工程已修 + 推理已验证）

见上文「三类问题 §2」。  
- 评测：`results/*-overall.rescored.json`  
- 推理：Job **16297** / `results_trunc/`（alimeeting 截断清零）

### 甲-DIAR-07. 碎切循环导致长度截断（📝 模型主因）

见上文「三类问题 §2b」。  
样例：`chime_6_diar_S01_w7460`（Job 16297 id4）— 0.4s 轮换碎切，未超 30s，撞 ~2k 字符上限。  
**结论：主因=模型生成失控；难数据为诱因；非标注协议问题。**

### 甲-DIAR-03. MISP 角色层 gold（✅ 本轮不评该集）

见上文「三类问题 §3」。方案是否确定：**确定（misp 本轮不评）**；换 speaker-level 参考前不报能力分。

### 甲-DIAR-01 / 02

窗长、AMI/CHiME adapted——数据侧记录，不单独挡结论。

---

## 附录：smoke 配置

| 项 | 路径 / Job |
|----|------------|
| suite | `smoke_diar_all/suite.yaml`（五集）/ `suite_trunc.yaml`（截断验证） |
| prompt | `smoke_diar_all/registry/prompt/mesh-diarization.yaml`（含时长约束） |
| 构建 | `python smoke_diar_all/build_manifests.py` |
| 运行 | `run_suite.sh` → Job **16253**；`run_trunc.sh` → Job **16297** |
| 切片 | `smoke_diar_all/clips/<benchmark>/` |
| 结果 | `results/`（16253）；`results_trunc/`（16297） |
| v1 归档 | `smoke_diar_all/results_archive_20260908_170228/` |
