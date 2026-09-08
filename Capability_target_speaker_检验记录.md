# Capability：target_speaker（目标说话人）

负责人：甲（T3）  
日期：2026-09-07  
模型：Qwen3-Omni-30B-A3B-Instruct（`qwen3-omni-local`）  
Smoke Job：**15277**（SUCCEEDED）  
产物：`VoxMatrix/smoke_ts_all/results/`

> AISHELL-5 无现成 enrollment/command；本轮自切短 utt：**5 正 + 5 负**。

---

## 状态总览

| # | 问题 | 状态 | 说明 |
| --- | --- | --- | --- |
| 甲-TS-01 | 无原生 target_speaker 对 | 📝 | adapted 自建 |
| 甲-TS-02 | Omni 偏「应执行」 | ✅ | 负例 4/5 误判 true；acc=**40%** |
| — | smoke GPU | ✅ Job **15277** | `failure_rate=0` |

### Smoke 汇总

| Benchmark | failure_rate | **target_speaker_acc** | mis_execution | 结论 |
|-----------|-------------:|-----------------------:|--------------:|------|
| aishell_5 | 0% | **0.40** | **0.60** | 链路通；正例 3/5、负例 1/5 |

人核：链路 pass；端侧误执行风险高，分数仅作 smoke。

---

## 统一 template

- **任务形态**：双音频（注册 + 指令）→ `{"should_execute": true|false}`
- **Prompt 骨架**（`mesh-target-speaker`）
- **选择题是否改为问答**：**否**
- **多音频**：WavPath=enroll，WavPath2=command

---

## 主指标

| 指标 | 角色 |
|------|------|
| **target_speaker_acc** | 推荐主分 |
| mis_execution | 辅（实现为 pred≠ref；文档「仅误执行」口径略不同，已注明） |

---

## 各 benchmark 核验

### aishell_5
- 抽样：10 对；`target_speaker_acc=0.40`，`mis_execution=0.60`
- 人核：有问题（负例易误执行）
- 结果是否可信：链路可信；能力偏弱
- 问题：甲-TS-01、甲-TS-02

明细：pos 对 3/5；neg 对 1/5（4 条 false→true）。

---

## 问题

### 甲-TS-01. AISHELL-5 需自建双音频对
- 正式榜应协议化注册/指令切段与 should_execute  
- 方案是否确定：**确定（smoke adapted）**

### 甲-TS-02. 负例易误判应执行
- 样例：`aishell5_ts_neg_01` ref=false pred=true  
- 原因：Omni 对「是否同一说话人」不稳定（与 SV 轮类似）  
- 解决方案：记录能力极限；端侧安全场景勿依赖 Omni 裸判  
- 方案是否确定：**确定**

---

## 附录

- Job **15277**；`smoke_ts_all/`
