# Capability：speaker_attribution（说话人归属）

负责人：甲（T3）  
日期：2026-09-07  
模型：Qwen3-Omni-30B-A3B-Instruct（`qwen3-omni-local`）  
Smoke Job：**15278**（SUCCEEDED）  
产物：`VoxMatrix/smoke_attr_all/results/`

> 正式协议：多说话人音频 → 带 speaker 的转写 → **cpCER / attribution_acc**。  
> 本轮 30s 窗（复用 diar clips + TextGrid/ASR 文本）。**cn_celeb 为 SID 错挂**，不按 attribution 主分。

---

## 状态总览

| # | 问题 | 状态 | 说明 |
| --- | --- | --- | --- |
| 甲-ATTR-01 | 长会切 30s 窗 | 📝 | 同 diar |
| 甲-ATTR-02 | AMI overlay adapted | 📝 | 句级 ASR 重建 |
| 甲-ATTR-03 | cn_celeb = SID | ✅ **本轮不评 attribution** | accuracy=0（无法猜 id） |
| 甲-ATTR-04 | pred 格式偶发非标准 JSON | 📝 | 仍多可解析；cpCER 可出 |
| — | smoke GPU | ✅ Job **15278** | 五集 `failure_rate=0` |

### Smoke 汇总（Job 15278）

| Benchmark | failure_rate | **cpCER%** ↓ | attribution_acc ↑ | 结论 |
|-----------|-------------:|-------------:|------------------:|------|
| aishell_4 | 0% | **61.6** | 0.55 | 链路通；有一定归属能力 |
| aishell_5 | 0% | **77.6** | 0.32 | 车载嘈杂，更难 |
| alimeeting | 0% | **56.8** | 0.26 | 相对最好（cpCER） |
| ami | 0% | **61.0** | 0.48 | EN overlay |
| cn_celeb | 0% | —（SID accuracy **0.0**） | — | **本轮不评 attribution** |

人核：会议四集链路与指标成立；Omni 能产出带 speaker 的转写（优于 diar 时间戳）。分数中等偏弱，可作 smoke，正式榜需更大样本。cn_celeb 协议错位。

---

## 统一 template

- **任务形态**：单段音频 → `{"utterances":[{"speaker","text"}]}`
- **Prompt 骨架**（`mesh-speaker-attribution`）
- **选择题是否改为问答**：**否**（cn_celeb 诊断用 speaker_id JSON）
- **多音频**：无

---

## 主指标

| 指标 | 角色 |
|------|------|
| **cpCER%** | **推荐主分**（越小越好） |
| attribution_acc | 辅（依赖 utterance id 对齐，易偏低） |

各会议集同一 evaluator；cn_celeb 用 classification accuracy（diagnostic）。

---

## 各 benchmark 核验

### aishell_4
- cpCER **61.6%**，attr_acc **0.55**；人核：pass（链路）
- 问题：甲-ATTR-01

### aishell_5
- cpCER **77.6%**，attr_acc **0.32**；人核：pass（链路）；车载更难
- 问题：甲-ATTR-01

### alimeeting
- cpCER **56.8%**，attr_acc **0.26**；人核：pass

### ami
- cpCER **61.0%**，attr_acc **0.48**；人核：pass（adapted）
- 问题：甲-ATTR-02

### cn_celeb
- accuracy **0.0**；人核：**本轮不评**（SID≠归属）
- 问题：甲-ATTR-03

---

## 问题

### 甲-ATTR-03. cn_celeb 错挂为 speaker_attribution
- 样例：ref=`id10001`，Omni 无法输出正确说话人 ID  
- 建议：从 attribution 榜剔除或单列 SID  
- 方案是否确定：**确定（本轮不评）**

### 甲-ATTR-04. 输出格式不完全稳定
- 偶发 markdown fence / 顶层数组；evaluator 多仍能抽 utterances  
- 建议：后处理加强 extract_json；不改数据  
- 方案是否确定：**确定**

---

## 附录

- Job **15278**；`smoke_attr_all/`
