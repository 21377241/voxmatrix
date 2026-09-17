# Capability：spoof_detection（伪造检测）

负责人：甲（T3）  
日期：2026-09-07  
模型：Qwen3-Omni-30B-A3B-Instruct（`qwen3-omni-local`）  
Smoke Job：**15276**（SUCCEEDED）  
产物：`VoxMatrix/smoke_spoof_all/results/`

> 任务清单仅 **mmsu**；ASVspoof gated → 跳过。MMSU 109 条不够稳算 EER → 主分 **accuracy**（对齐 full_eval spoof diagnostic）。

---

## 状态总览

| # | 问题 | 状态 | 说明 |
| --- | --- | --- | --- |
| 甲-SPOOF-01 | ASVspoof 不可用 | 📝 | 正式 EER 缺数 |
| 甲-SPOOF-02 | Omni 几乎全判 real | ✅ 已记 | 5/5 real 对；5/5 fake→real；acc=**50%**（随机基线） |
| — | smoke GPU | ✅ Job **15276** | `failure_rate=0` |

### Smoke 汇总

| Benchmark | failure_rate | **accuracy** | 结论 |
|-----------|-------------:|-------------:|------|
| mmsu `synthetic_speech_detection` | 0% | **0.50** | 链路通；假音检出失败，分数不可用于模型排序 |

人核：链路 pass；能力弱 → 记录极限，**不因数据改口径**。

---

## 统一 template

- **任务形态**：单音频 → `{"label":"real"|"fake"}`
- **Prompt 骨架**（`mesh-spoof-open`）：*Is this speech real (spoken by a human) or fake (synthesized)? Return JSON only as {"label":"real"} or {"label":"fake"}.*
- **选择题是否改为问答**：**是**（MCQ → 开放 binary）
- **多音频**：无

---

## 主指标

| 指标 | 角色 | 说明 |
|------|------|------|
| **accuracy** | **本轮推荐** | pred/ref ∈ {real,fake} exact match（`mesh-classification`） |
| EER / AUC | 协议名义 | 需 ASVspoof trial；本轮 N/A |

数据流：Omni JSON → ClassificationEvaluator → MeshAgg `overall/accuracy`。

---

## 各 benchmark 核验

### mmsu
- 抽样：5 real + 5 fake（stride）
- 指标：accuracy **0.50**；failure_rate **0**
- 人核：有问题（假音全漏检）
- 结果是否可信：链路可信；能力分接近随机，**不宜排行**
- 问题：甲-SPOOF-01、甲-SPOOF-02

---

## 问题

### 甲-SPOOF-01. ASVspoof 不可用
- 建议：授权后补 LA/DF trial 再报 EER  
- 方案是否确定：**确定**

### 甲-SPOOF-02. Omni 对 MMSU 假音全判 real
样例：ref=`fake`，pred=`real`（5/5）  
原因：模型未形成可用伪造检测决策边界  
解决方案：记录能力极限；正式安全评需专用反欺骗模型  
方案是否确定：**确定**

---

## 附录

- Job **15276**；`smoke_spoof_all/`
