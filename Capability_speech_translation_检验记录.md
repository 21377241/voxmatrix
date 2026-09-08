# Capability：speech_translation（语音翻译）检验记录

记录 `speech_translation` capability 抽样 smoke 中发现的问题、口径拍板与分数可信度，便于对齐「评的是什么」。

负责人：甲（T1）  
日期：2026-09-02（本稿 2026-09-09 按实验结果重写）  
模型：Qwen3-Omni-30B-A3B-Instruct（`qwen3-omni-local`）  
Smoke Job：**11931**（CoVoST + MMSU MCQ）、**12251**（MMSU open S2TT）  
产物：`VoxMatrix/smoke_st_all/results/`、`results_mmsu_open/`

> **协议 vs 本轮实际**  
> 框架主能力是 **S2TT（语音→目标语文本）+ BLEU / chrF**（COMET 为可选辅指标，本轮未接）。  
> CoVoST2 为标准双向翻译；MMSU 原始为 **MCQ**，已改编为 **开放式 S2TT**（gold = 正确选项文本）作为 capability 主形态；MCQ 仅作 diagnostic。

---

## 状态总览（2026-09-09）

| # | 问题 | 状态 | smoke 当前指标（各 10 条） | 下一步 |
| --- | --- | --- | --- | --- |
| 甲-ST-01 | MMSU 不宜用 MCQ 作 capability 主形态 | ✅ **已收口** | open BLEU **42.9** / chrF **62.1**；MCQ match **90%** | MCQ 仅 diagnostic |
| 甲-ST-02 | 协议可选 COMET 未接入 smoke | 📝 **已知** | N/A | 需 Unbabel COMET 离线环境；正式榜可选 |
| 甲-ST-03 | zh→en BLEU 偏低、与听感不完全一致 | 📝 **观察** | BLEU **22.5** / chrF **47.1** | 报告分列 BLEU+chrF；勿只报 BLEU |
| — | 链路 failure | ✅ | 四集 **failure_rate=0** | 无 |

**各集分数与可信度**

| 子集 | 任务形态 | 主指标 | 分数 | 辅指标 | 能否当 capability 主分 |
| --- | --- | --- | ---: | --- | --- |
| CoVoST2 zh→en | S2TT | BLEU (13a) | **22.46** | chrF **47.12** | ✅ 可（标准翻译集；BLEU 偏严） |
| CoVoST2 en→zh | S2TT | BLEU-zh | **49.40** | chrF **43.50** | ✅ 可 |
| MMSU open | adapted S2TT | BLEU (13a) | **42.87** | chrF **62.11** | ✅ 可（adapted；推荐主看） |
| MMSU MCQ | 四选一 | match(%) | **90.0** | — | ❌ diagnostic only |

**构建 / 运行**

- 构建：`smoke_st_all/build_manifests.py`
- 主 suite：Job **11931** → `results/`（含 CoVoST 两向 + MMSU MCQ）
- MMSU open：`run_mmsu_open.sh` → Job **12251** → `results_mmsu_open/`
- 元数据快照：`smoke_st_all/summary.json`、`chrf_summary.json`

---

## 评测口径（capability template）

| 层 | 评什么 | 主指标 | 状态 |
| --- | --- | --- | --- |
| A 协议主榜 | CoVoST / 标准 S2TT | BLEU + chrF | 已 smoke |
| B adapted | MMSU 选项文本作 gold 的开放翻译 | BLEU + chrF | 已 smoke（推荐） |
| C diagnostic | MMSU MCQ | Accuracy / match | 已跑，不作主分 |
| D 可选 | 语义质量 | COMET | **未接** |

**Prompt（分轨）**

| 子集 | prompt | 输出要求 |
| --- | --- | --- |
| CoVoST zh→en | `qwen3-omni-s2tt-zh2en`（suite 配置） | 仅英文译文 |
| CoVoST en→zh | `qwen3-omni-s2tt-en2zh` | 仅中文译文 |
| MMSU open | `qwen3-omni-s2tt-to-en` | Listen… translate into English. Output only the translation… |
| MMSU MCQ | `mmsu` | 只输出 A/B/C/D |

Registry：`smoke_st_all/registry/`（当前磁盘上至少保留 `qwen3-omni-s2tt-to-en.yaml`）。

---

## 问题列表

### 甲-ST-01. MMSU 原始为 MCQ，不宜作 speech_translation 主形态


| 字段 | 内容 |
| --- | --- |
| 发现时间 | 2026-09-01～09-02 |
| 场景 | `smoke_st_all`（MMSU speech_translation 子集，各 10 条） |
| 相关产物 | `mmsu/manifest.jsonl` + `results/mmsu*`；`mmsu_open/` + `results_mmsu_open/` |
| 严重程度 | 中（MCQ 高分不能代表翻译能力） |
| 状态 | ✅ **已收口**：改编为 open S2TT（Job **12251**）；MCQ 标为 diagnostic |


#### 现象

1. MMSU 索引里该子集是「听音频 + 题干 + 四选项」，gold 为正确选项字母/文本。
2. 直接跑 MCQ：match **90%**（9/10），与「翻译质量」无关，端侧也少见选 A/B/C/D。
3. 改编后：去掉选项，要求输出英文译文；gold = 原正确选项完整文本；用 **BLEU/chrF** 打分 → BLEU **42.9**、chrF **62.1**，与 CoVoST 同口径可比。

#### 样例（同题 MCQ vs open）

| sample（截断 id） | MCQ pred | MCQ | open BLEU | open pred（截断） |
| --- | --- | --- | ---: | --- |
| `…1920c2d8` | C | ✗ | 21.3 | This is common sense worldwide… |
| `…5c42d299` | A | ✓ | 60.0 | Lapental slope, sixty-eight… |
| `…1eae53d4` | C | ✓ | 100.0 | There, in that office. |
| `…4284983b` | C | ✓ | 6.3 | The recital took place at the Spanish Association. |

#### 建议 / 收口

- **确定**：capability 主报告用 **MMSU open + CoVoST**；MCQ 只作 diagnostic。
- 报告不得把 MCQ 90% 与 BLEU 合成一个「翻译 accuracy」。

---

### 甲-ST-02. COMET 未接入本轮 smoke


| 字段 | 内容 |
| --- | --- |
| 发现时间 | 2026-09-02 |
| 场景 | 协议 / `benchmark_evaluator_map` / smoke suite |
| 严重程度 | 低（辅指标缺失，不阻塞 BLEU/chrF 主链路） |
| 状态 | 📝 **已知**：框架与 smoke 均未挂 COMET |


#### 现象

- `capability_protocols` / full eval 路由主指标为 **BLEU + chrF**。
- COMET 需离线模型（Unbabel / `wmt22-comet-da` 等），标注为 `requires_offline_model`，smoke 默认不可跑。
- 本轮 suite **只配了 bleu / bleu-zh**；chrF 由后处理/汇总写入 overall（见 `chrf_summary.json`）。

#### 建议 / 收口

- **确定（本轮）**：主分 BLEU+chrF；COMET 正式榜有环境再加，不阻塞 capability 检验。

---

### 甲-ST-03. CoVoST zh→en：BLEU 明显低于 en→zh，语义常仍可接受


| 字段 | 内容 |
| --- | --- |
| 发现时间 | 2026-09-01（Job 11931） |
| 场景 | CoVoST2 双向各 10 条 |
| 严重程度 | 中（单看 BLEU 易低估） |
| 状态 | 📝 **观察**：分列 BLEU 与 chrF；人核见抽样表 |


#### 现象

| 方向 | BLEU | chrF | 说明 |
| --- | ---: | ---: | --- |
| zh→en | **22.46** | **47.12** | paraphrase / 术语差异拉低 n-gram；chrF 相对更高 |
| en→zh | **49.40** | **43.50** | 短句多、可字面重合；有 2 条 BLEU=100 |

人核印象：zh→en 多条「意思接近但用词不同」（如 dryness 例、台风升级例），BLEU 个位数～二十不代表完全听错。en→zh 偶发严重跑题（如 Cheryl/诗歌朗诵 → 波特兰酒店，BLEU 9.0）。

#### 建议 / 收口

- **确定**：对外同时报 **BLEU + chrF**；zh→en 不以 BLEU 单独定优劣。
- 可选：正式评加 COMET 复核语义。

---

## 各 benchmark 核验

### CoVoST2 zh→en

- 抽样：10 条；协议：标准 S2TT；failure=0  
- 主分：BLEU **22.46**；chrF **47.12**  
- 人核：链路可信；分数受 paraphrase 影响 → 见甲-ST-03  
- 源：`results/covost2_zh_en.jsonl`

| # | sample_id（截断） | BLEU | ref（截断） | pred（截断） |
| --- | --- | ---: | --- | --- |
| 0 | `…c3bce91a` | 8.5 | Progressive dryness samples… | Examples of dry symptoms include… |
| 1 | `…3f08b883` | 20.2 | Vitelline artery is one of the branches. | One of the branches of the ovarian artery. |
| 2 | `…dab20d1d` | 27.1 | Si Wei Qinjun was one among… | The Shiwai Qinjun was a kind of… |
| 3 | `…18b34966` | 8.1 | Red Mountaintop Heritage… | The historical age of the Hongshanding… |
| 4 | `…7984a573` | 34.2 | Bayern Munich won the competition… | Bayern Munich won the championship again. |
| 5 | `…b3e76c50` | 4.5 | Pei Yiye passed away… | On the 27th day of the 8th month… |
| 6 | `…c71d2725` | 52.0 | But when Clayton gave the draft… | But when Clayton gave the manuscript… |
| 7 | `…2255bb10` | 27.1 | The Central Weather Bureau upgraded… | The Central Meteorological Bureau… |
| 8 | `…185d3445` | 5.8 | Aung San Thuriya… Street… | The central Yangon street of Sule Pagoda… |
| 9 | `…f542703f` | 17.4 | The music video made for the single… | The music video produced for this single… |

### CoVoST2 en→zh

- 抽样：10 条；BLEU-zh **49.40**；chrF **43.50**；failure=0  
- 人核：整体可用；#2/#4 明显错译需留意  
- 源：`results/covost2_en_zh.jsonl`

| # | sample_id（截断） | BLEU | ref | pred（截断） |
| --- | --- | ---: | --- | --- |
| 0 | `…0b2e1afe` | 100.0 | 她会没事的。 | 她会没事的。 |
| 1 | `…51316417` | 70.6 | 潜水员在湖里发现了一具尸体。 | 潜水员在湖里发现了尸体。 |
| 2 | `…66555787` | 7.9 | “我想它们会持续很长时间，”他对和尚说。 | "我想他们会笑很久,"他对妈妈说。 |
| 3 | `…3ddda60f` | 46.9 | 航班延误了 22 分钟。 | 航班延误了二十二分钟。 |
| 4 | `…322979e5` | 9.0 | Cheryl 问我关于明天和她一起去诗歌朗诵比赛的事。 | Cheryl 问我明天是否要去波特兰酒店。 |
| 5 | `…3643e669` | 47.2 | 我们必须奉献自己，而不仅仅是金钱。 | 我们必须付出我们的服务,而不仅仅是金钱。 |
| 6 | `…b7ccca9f` | 100.0 | 不要孤注一掷。 | 不要孤注一掷。 |
| 7 | `…86d0e318` | 35.8 | 一个穿着橙色裙子的女人在狂欢节上骑马。 | 一个穿着橙色连衣裙的女人在乘坐… |
| 8 | `…eb5f743a` | 64.3 | 这所大学承认它是一个由国会创建的… | 该大学承认自己是一个国会创建的… |
| 9 | `…e299a486` | 63.2 | 我们有必要停下来喝杯咖啡。 | 我们必须停下来喝杯咖啡。 |

### MMSU open（推荐）

- 抽样：10 条；BLEU **42.87**；chrF **62.11**；failure=0  
- 人核：pass（adapted S2TT 链路成立）  
- 源：`results_mmsu_open/mmsu_open.jsonl`

| # | sample_id（截断） | BLEU | ref（截断） | pred（截断） |
| --- | --- | ---: | --- | --- |
| 0 | `…1920c2d8` | 21.3 | This is common sense around the world… | This is common sense worldwide… |
| 1 | `…5c42d299` | 60.0 | Côteaux de l’Appenthal… | Lapental slope… |
| 2 | `…2704eb3d` | 15.6 | It belonged to a lower-middle class… | He belonged to a lower middle class… |
| 3 | `…1eae53d4` | 100.0 | There, in that office. | There, in that office. |
| 4 | `…312df408` | 29.8 | It is now Mr. Charles de Courson’s turn… | The floor is now open to Mr. Charles… |
| 5 | `…01bd72d3` | 79.1 | We welcome the practice of submitting… | We welcome the practice of presenting… |
| 6 | `…035dae01` | 40.9 | Guadalupe “Lupita” Carías… | Guadalupe "Lupita" Carias… |
| 7 | `…4284983b` | 6.3 | The concert was held in the Asociación Española. | The recital took place at the Spanish Association. |
| 8 | `…ce583d19` | 55.9 | It is necessary to put into use all human… | We need to mobilize all human resources… |
| 9 | `…a8dac13b` | 18.2 | I also recognize the service of his predecessor… | I also note the work of his predecessor… |

### MMSU MCQ（diagnostic）

- match **90%**（9/10）；唯一错误 `#0` pred=`C`  
- **不作** capability 主分  
- 源：`results/mmsu.jsonl`

---

## 附录

| 项 | 路径 / 值 |
| --- | --- |
| Job | **11931**（v2 主 suite）、**12251**（MMSU open）；11930 为早期尝试 |
| 结果 | `smoke_st_all/results/`、`results_mmsu_open/` |
| 汇总 | `summary.json`、`chrf_summary.json` |
| 索引来源 | `benchmark_subset_manifest.jsonl`（summary 记录） |

**一句话结论**：speech_translation smoke **链路通、failure=0**；主分看 **CoVoST BLEU+chrF + MMSU open**；MMSU MCQ 与 COMET 分别作 diagnostic / 未接辅指标；zh→en 勿只信 BLEU。
