# Capability：speaker_attribution（说话人归属）检验记录

负责人：甲（T3）  
日期：2026-09-07；工程修复：2026-09-09  
模型：Qwen3-Omni-30B-A3B-Instruct（`qwen3-omni-local`）  
Smoke Job：**15278**（SUCCEEDED）；CHiME-6 crop：**16421**  
产物：`VoxMatrix/smoke_attr_all/results/`、`results_rescored/`、`results_chime6_crop/`

> 正式协议：多说话人音频 → 带 speaker 的转写 → **cpCER / attribution_acc**。  
> 本轮 30s 窗（复用 diar clips + TextGrid/ASR 文本）。**cn_celeb 为 SID 错挂**，不按 attribution 主分。

---

## 状态总览（2026-09-09）

| # | 问题 | 状态 | 说明 |
| --- | --- | --- | --- |
| 甲-ATTR-01 | 长会切 30s 窗 | 📝 | 同 diar |
| 甲-ATTR-02 | AMI overlay adapted | 📝 | 句级 ASR 重建；CHiME 已改原轨裁剪 |
| 甲-ATTR-03 | cn_celeb = SID | ✅ **本轮不评 attribution** | accuracy=0 |
| 甲-ATTR-04 | pred 非法/截断 JSON 抬高 cpCER | ✅ **已收口** | 键名抢救 + 截断正则 + post_process；见下表 |
| — | smoke GPU | ✅ Job **15278** | 会议四集 failure=0 |

### Smoke 汇总

| Benchmark | Job 15278 cpCER% | **rescored cpCER%** | invalid 行 | 说明 |
|-----------|-----------------:|--------------------:|-----------:|------|
| aishell_4 | 61.6 | **59.8** | 1→0 | 非法 `{"speaker_1","text":...}` 已救 |
| aishell_5 | 77.6 | **66.1** | 3→0 | 同上；截断「嗯？」类仍难 |
| alimeeting | 56.8 | **45.9** | 3→0 | 工程修复收益最大 |
| ami | 61.0 | **61.0** | 0→0 | 本可解析（顶层数组已兼容） |
| cn_celeb | — | — | — | **不评** |
| chime_6（crop） | 74.0（16421） | **~71.5** | 1→0 | 原轨裁剪窗 |

主分仍用 **cpCER%↓**；`attribution_acc` 仅辅分（句数不对齐时虚低）。

---

## 甲-ATTR-04. 输出格式不稳定（工程）


| 字段 | 内容 |
| --- | --- |
| 发现时间 | 2026-09-07 |
| 修复时间 | 2026-09-09 |
| 场景 | `smoke_attr_all` |
| 状态 | ✅ **已收口** |


#### 现象

1. 顶层数组 `[{speaker,text}]` 而非 `{"utterances":[...]}`（多数集）  
2. markdown \`\`\`json fence  
3. 写几条合法后漏键名：`{"speaker_1","text":"..."}` → JSON 整段解析失败 → **cpCER=100 / attribution_valid=0**  
4. 输出中途截断（aishell_5）

#### 修复

| 项 | 位置 |
| --- | --- |
| `normalize_speaker_attribution_prediction` | `mesh_eval/evaluator/speaker.py`（键名修复、截断正则、数组包装） |
| `SpeakerAttributionNormalize` | `mesh_eval/process/attribution.py` |
| smoke post_process | `mesh-speaker-attribution-normalize`（除 cn_celeb） |
| Prompt 约束 | `smoke_attr_all/registry/prompt/mesh-speaker-attribution.yaml` |
| 离线重打 | `python smoke_attr_all/rescore_results.py` → `results_rescored/` |

#### 收口

- **确定**：评分解析层必须可抢救；prompt 约束保留但不依赖模型自觉。  
- 截断后内容本身胡言（如连发「嗯？」）仍属模型能力，不在本工程项。

---

## 统一 template

- **任务形态**：单段音频 → `{"utterances":[{"speaker","text"}]}`  
- **Prompt**：`mesh-speaker-attribution`（强制 utterances 键名、禁 fence）  
- **post_process**：`mesh-speaker-attribution-normalize`  
- **主指标**：cpCER%；辅：attribution_acc  

---

## 问题（非工程）

### 甲-ATTR-03. cn_celeb 错挂

SID 任务挂在 attribution 下 → **本轮不评**。

### 甲-ATTR-01/02. 数据窗

30s 窗 / AMI overlay 仍为 adapted；CHiME-6 已可原轨裁剪（Job 16421）。

---

## 附录

- 构建：`smoke_attr_all/build_manifests.py`  
- 重打：`smoke_attr_all/rescore_results.py`  
- Job **15278** / **16421**
