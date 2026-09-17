# Capability：speaker_count（说话人计数）

负责人：甲（T3）  
日期：2026-09-03（smoke 回填）  
模型：Qwen3-Omni-30B-A3B-Instruct（`qwen3-omni-local`）  
Smoke Job：**12750**（MMSU MCQ）、**12751**（MMSU open）  
产物：`VoxMatrix/smoke_sc_all/results/`、`results_mmsu_open/`

> **协议 vs 本轮实际**  
> 框架定义 speaker_count 主能力是 **音频 → 说话人个数（整数）**，指标 **speaker_count_acc**。  
> MMSU 官方形态为 **四选一 MCQ**（`N people`）；本轮另设 **open 整数** 轨作端侧推荐口径（adapted）。

---

## 状态总览（2026-09-03）

| # | 问题 | 状态 | 说明 |
| --- | --- | --- | --- |
| 甲-SC-01 | MMSU 仅 MCQ，端侧宜 open 整数 | ✅ **已收口** | open 主分 **80%**；MCQ diagnostic **100%** |
| — | smoke GPU 推理 | ✅ | Job 12750 / 12751，`failure_rate=0` |
| — | 协议主分 | ✅ **定 open** | `mesh-speaker-count` → `speaker_count_acc` |

### Smoke 汇总

| Benchmark | 指标 | 分数 | 角色 |
|-----------|------|-----:|------|
| **mmsu open** | `speaker_count_acc` | **80%**（8/10） | **推荐主分** |
| mmsu MCQ | Accuracy (`match`) | **100%**（10/10） | diagnostic only |

---

## 统一 template

- **任务形态**：单段音频 → **说话人个数**（整数 1–6）。
- **Prompt 骨架**：
  - **MMSU MCQ（diagnostic）**：`mmsu` 模板，四选一 A/B/C/D，选项形如 `3 people`（官方）
  - **MMSU open（推荐）**：`qwen3-omni-speaker-count` — *Listen to the audio and count how many distinct speakers… Output only a single integer.*
- **选择题是否改为问答**：**是（推荐轨）**——端侧少见选 A/B/C/D；open 轨 gold 为整数（从 MCQ 正确选项解析，如 `3 people` → `3`）。MCQ 保留作 diagnostic / 官方对照。
- **多音频 / 多轮约定**：无。

---

## 主指标

### 推荐主指标

| 数据形态 | 主指标 | 原因 |
|----------|--------|------|
| **MMSU open（推荐）** | **speaker_count_acc**（`mesh-speaker-count`） | 与协议一致：pred/ref 均为整数，exact match |
| MMSU MCQ（diagnostic） | Accuracy（`mmsu-choices`） | 官方 MMSU 形态，仅对照 |

### 算分说明

- **open**：`SpeakerCountEvaluator` 将 pred 解析为 `int`，与 ref 整数比较；`speaker_count_acc` = 0/1，语料级 **mean**。
- **MCQ**：pred 字母 A/B/C/D 与 ref 选项文本对应字母比；`match` mean = Accuracy。
- pred 为 JSON `{"count": 3}` 时 open evaluator 亦支持；当前 prompt 要求 **纯整数**。

### 数据流

```
音频 + Prompt → Qwen3-Omni → pred
  ├─ MMSU open：pred 整数 ↔ ref 整数 → speaker_count_acc
  └─ MMSU MCQ：pred 字母 ↔ ref 选项 → Accuracy
```

### Benchmark 算分

| Benchmark | 实现 | smoke | 备注 |
|-----------|------|------:|------|
| mmsu open | `mesh-speaker-count` + agg `speaker-count` | **80%** | **推荐主分** |
| mmsu MCQ | `mmsu-choices` + agg `mean` | **100%** | diagnostic only |

---

## MMSU 数据核验（`total_speaker_counting`）

- **官方 subtask**：`total_speaker_counting`（文件名 / `task_name` 一致）
- **来源**：`MMSU_Perception`；`atomic_task_id`: `audio_grounded_qa`；metadata `language: en`
- **全集**：120 条；问题固定为 *How many different speakers are in the audio?*
- **选项**：`1 people` … `6 people`（全集分布：1×10, 2×18, 3×26, 4×22, 5×21, 6×23）
- **与 code_switch 不同**：本 subtask **确是 Perception 类计数任务**

### 抽样（10 条）+ smoke pred

来源：`smoke_sc_all/mmsu/manifest.jsonl`（等距 stride）；Job 12750 / 12751

| # | sample_id（尾） | gold | MCQ pred | open pred | open✓ | 音频文件 |
|---|----------------|-----:|:--------:|:---------:|:-----:|----------|
| 0 | `d817223b` | 3 | C | 3 | ✓ | `total_speaker_counting_d817223b-...383b.wav` |
| 1 | `ac00fc3e` | 6 | D | 6 | ✓ | `total_speaker_counting_ac00fc3e-...71a0.wav` |
| 2 | `b8f6425f` | 2 | A | 2 | ✓ | `total_speaker_counting_b8f6425f-...ee75.wav` |
| 3 | `8fd47418` | 3 | C | **2** | ✗ | `total_speaker_counting_8fd47418-...b903.wav` |
| 4 | `494969b0` | 3 | D | 3 | ✓ | `total_speaker_counting_494969b0-...69b0.wav` |
| 5 | `1cb17e42` | 5 | D | 5 | ✓ | `total_speaker_counting_1cb17e42-...7e42.wav` |
| 6 | `e9b4615d` | 2 | A | 2 | ✓ | `total_speaker_counting_e9b4615d-...615d.wav` |
| 7 | `9de100ee` | 3 | C | **2** | ✗ | `total_speaker_counting_9de100ee-...00ee.wav` |
| 8 | `e9aed531` | 2 | C | 2 | ✓ | `total_speaker_counting_e9aed531-...d531.wav` |
| 9 | `31c177a4` | 3 | C | 3 | ✓ | `total_speaker_counting_31c177a4-...77a4.wav` |

音频根目录：`/mnt/afs/eval_data/benchmarks/MMSU/audio/`

- 当前指标：MCQ **100%** / open **80%**
- 人核结论：**链路可信**；open 2 条漏计（gold=3 → pred=2），同条 MCQ 选对——选项约束抬高了准确率
- 问题编号：无阻塞；漏计属模型能力边界，非数据/评测坏点

---

## 问题列表

### 甲-SC-01. MMSU MCQ vs open 整数（✅ 已收口）

| 字段 | 内容 |
| --- | --- |
| 发现时间 | 2026-09-02 |
| 关闭时间 | 2026-09-03 |
| 场景 | `smoke_sc_all` |
| 状态 | ✅ open 轨验证通过；主分定为 `speaker_count_acc` |

**原因**

MMSU 官方为 MCQ；框架 `full-mmsu-speaker-count` 用 `mesh-external-classification` + `mesh-speaker-count`，与 MCQ 选项文本不对齐。端侧更合理为 **直接报整数**。

**解决方案（已验证）**

- open：`qwen3-omni-speaker-count` + `mesh-speaker-count`；gold 从 `N people` 解析为 `N` → **80%**
- MCQ：`mmsu` + `mmsu-choices`；diagnostic → **100%**
- 方案确定：**open 为主分**；MCQ 仅对照

**样例（open 错、MCQ 对）**

| sample_id（尾） | gold | MCQ | open |
|----------------|-----:|:---:|-----:|
| `8fd47418` | 3 | C✓ | 2 |
| `9de100ee` | 3 | C✓ | 2 |

---

## 附录：smoke 配置

| 集 | Manifest | Suite | Evaluator | Job | 结果目录 |
|----|----------|-------|-----------|-----|----------|
| mmsu MCQ | `mmsu/` | `suite.yaml` | `mmsu-choices` | **12750** | `results/` |
| mmsu open | `mmsu_open/` | `suite_open.yaml` | `mesh-speaker-count` | **12751** | `results_mmsu_open/` |

- 构建：`python smoke_sc_all/build_manifests.py`
- 运行：`bash smoke_sc_all/run_suite.sh`；`bash smoke_sc_all/run_mmsu_open.sh`
- Registry：`smoke_sc_all/registry/`（含 `mesh-speaker-count`、`speaker-count` agg）
- 索引：`benchmark_subset_manifest.jsonl`（`capability=speaker_count` → mmsu / `total_speaker_counting`）
