# Capability：code_switch（语码转换）检验记录

记录 `code_switch` capability 抽样 smoke 中发现的问题、口径拍板与修复进度，便于对齐「评的是什么、分数可不可信」。

负责人：甲（T1）  
模型：Qwen3-Omni-30B-A3B-Instruct（`qwen3-omni-local`）  
Smoke Job：12021 / 12284（v1）→ **12329 / 12330**（v2）  
产物：`VoxMatrix/smoke_cs_all/results/`、`results_mmsu_open/`

> **协议 vs 本轮实际**  
> 框架定义 code_switch 主能力是 **混语转写 + MER**。本地 **无 SEAME 等转写 gold**，MMSU / VocalBench-zh 实为 **混语 QA（adapted）**。  
> 报告须 **分列** auxiliary 与 diagnostic，**不得**与 MER 主榜合成一个 accuracy。

---

## 状态总览（2026-09-02 更新）

| # | 问题 | 状态 | smoke 当前指标（v2，各 10 条） | 下一步 |
| --- | --- | --- | --- | --- |
| 甲-CS-01 | 指标 / Prompt 分轨；EM 把分做崩 | ⚠️ **部分解决** | 见下表各集 | MMSU open 主分仍不可信；归一化 contain 未入库；跨语言实体 |
| 甲-CS-02 | VocalBench `Audio` 路径相对 `audio/` 未声明 | ✅ **已收口** | VB 两集 failure=0 | 无 |
| 甲-CS-03 | MMSU 不宜 MCQ 作主形态 | ✅ **已收口** | open 链路跑通 | 无 |
| — | 协议主榜 **MER** 无数据 | 🚫 **阻塞** | N/A | 需混语转写集（SEAME 等） |
| — | 索引 `protocol.metrics=accuracy` 过粗 | ⏳ **未改** | — | 正式 manifest 映射分轨 evaluator |

**各集 smoke 分数与可信度**

| 子集 | evaluator | v1 → v2 | 能否当 capability 主分 |
| --- | --- | --- | --- |
| mmsu MCQ | `mmsu-choices` | 100% → **100%** | ❌ diagnostic only |
| mmsu open | `qa-exist-match` | 10% → **20%** | ⚠️ auxiliary；**低估**（同批 MCQ 100%，token_f1 ≈ 53%） |
| vocalbench_zh knowledge | `qa-exist-match` | 0% → **70%** | ⚠️ auxiliary；3/10 为英/中实体 |
| vocalbench_zh open_ended | `rouge-l` | 0% → **25.2%** | ⚠️ auxiliary；口径可用 |

**重跑**：`bash smoke_cs_all/run_suite.sh`；`bash smoke_cs_all/run_mmsu_open.sh`  
**构建**：`smoke_cs_all/build_manifests.py`

---

## 评测口径（capability template）

| 层 | 评什么 | 主指标 | 状态 |
| --- | --- | --- | --- |
| A 协议主榜 | 混语转写 | MER | N/A |
| B auxiliary QA | MMSU open、VB knowledge、VB open_ended | 短答 contain 族 / 长答 ROUGE-L | v2 已 smoke |
| C diagnostic | MMSU MCQ | Accuracy | 已跑，不作主分 |

**Prompt 分轨**（四集 **不共用** 同一 prompt；不可横比 pred）：

| 子集 | prompt | 输出要求 |
| --- | --- | --- |
| MMSU open | `qwen3-omni-cs-open-qa` | 一句英文 |
| MMSU MCQ | `mmsu` | A/B/C/D |
| VB knowledge | `qwen3-omni-cs-short-qa` | 最短词/短语 |
| VB open_ended | `qwen3-omni-cs-open-ended` | 2–4 句 |

Registry：`smoke_cs_all/registry/prompt/`；MCQ 另见 `registry/prompt/mmsu.yaml`。

---

## 问题列表

### 甲-CS-01. 指标与 Prompt 分轨；v1 指标把 auxiliary 分数做崩


| 字段 | 内容 |
| --- | --- |
| 发现时间 | 2026-09-01；指标拍板：2026-09-02 |
| 场景 | `smoke_cs_all`（Qwen3-Omni，MMSU + VocalBench-zh code_switch 子集，各 10 条） |
| 相关产物 | `results/`、`results_mmsu_open/`；suite `suite.yaml`、`suite_mmsu_open.yaml` |
| 严重程度 | 高（v1 分数不可代表模型能力；v2 改善但仍有多处低估） |
| 状态 | ⚠️ **部分解决**：v2 evaluator + 分轨 prompt 已落地（Job 12329/12330）；**MMSU open 主分仍不可信**；归一化 contain / 跨语言实体 **未闭环** |


#### 现象

1. 协议主指标为 **MER**，但手头 **无混语转写 gold**，只能评 adapted QA。
2. v1 四集共用 **EM / exist-match** 类 0/1 指标 → VB knowledge **0%**、open_ended **0%**，与听音/语义严重不符。
3. 四集 **gold 长度差大**（2 字实体 / 一句英文 / 整段），不能强行一个 accuracy。
4. v2 换 **qa-exist-match**（短答）+ **rouge-l**（长答）+ **分轨 prompt** 后：knowledge **70%**、open_ended **25.2%** 有区分度。
5. **MMSU open 仍异常**：同批音频 MCQ **100%**，qa-exist 仅 **20%** — pred 多为 paraphrase + *The speaker says…*，gold 为 MCQ 选项原文。
6. 离线重算：短答 **0/1 族**（em / qa-exist / mesh-contain / loose-order）在 MMSU open 上 **同分 20%**；**token_f1 ≈ 53%** 更接近「部分对」。


| 子集 | v1 指标 | v1 分 | v2 指标 | v2 分 |
| --- | --- | ---: | --- | ---: |
| mmsu open | exist-match | 10% | qa-exist-match | **20%** |
| VB knowledge | em | 0% | qa-exist-match | **70%** |
| VB open_ended | exist-match | 0% | rouge-l | **25.2%** |


#### 所用指标说明

本问题涉及 **v1 → v2 换指标**；四集 **不共用同一 evaluator**（与 Prompt 分轨一致）。下表仅列 smoke 实际用过或离线比选过的项；registry 名见 `registry/evaluator/`、`smoke_cs_all/registry/evaluator/`。

| 指标 | registry | 怎么算 | 用在哪 | v1 / v2 |
| --- | --- | --- | --- | --- |
| **EM** | `em` | pred 与 ref **字符串完全相等**（strip 后） | VB knowledge（v1） | v1 ❌ → 弃 |
| **exist-match** | `exist-match` | ref **小写 substring** 出现在 pred 中 | MMSU open、VB open（v1） | v1 ❌ → 弃 |
| **qa-exist-match** | `qa-exist-match` | ref 切成 token 后，须在 pred 中 **连续出现**（SQuAD 式 contain） | 短答轨（v2 **当前主分**） | v2 ✅ |
| **rouge-l** | `rouge-l` | pred 与 ref 的 token **最长公共子序列** F1（0–100%） | VB open_ended（v2 **当前主分**） | v2 ✅ |
| **mmsu-choices** | `mmsu-choices` | pred 字母 A/B/C/D 是否与 ref 选项一致 | MMSU MCQ（**diagnostic**，见甲-CS-03） | 一直用 |
| **token_f1** | `mesh-text-f1` | pred/ref token 集合重叠 F1；paraphrase **有部分分** | 离线辅指标；MMSU open 建议辅报 | 未入主 suite |
| **归一化 contain**（计划） | 待入库 | NFKC、去标点、去 *The speaker…* 后 ref⊂pred | 短答 **计划主分** | 离线试算 **与 qa-exist 同分** |
| **MER** | `MixedErrorRateEvaluator` | 混语转写 token 编辑错误率 | 协议主榜（**无转写 gold**） | N/A |

**读分时注意**：

- **短答**（MMSU open、VB knowledge）报的是 **Accuracy %** = 10 条里 match=1 的比例；换词 paraphrase 通常 **整句判 0**。
- **长答**（VB open_ended）报的是 **语料级 ROUGE-L 均值**（非 0/1 accuracy）；exist-match 在长答上 **恒为 0%**，故 v2 改用 rouge-l。
- 索引里写的 `protocol.metrics: ["accuracy"]` **分辨不出**上表哪一列；正式 manifest 须登记具体 evaluator 名（见状态总览「未改」行）。


#### 样例

**MMSU open — paraphrase 判错（MCQ 却对）**

| sample_id | ref | pred（截断） | qa-exist | MCQ |
| --- | --- | --- | ---: | --- |
| `9cb5b2dd` | Watched videos. | The speaker watched a video. | 0 | B✓ |
| `df3de0c6` | Because they don't have to deliberately change. | Because it means they do not need to deliberately change something. | 0 | B✓ |
| `6496444c` | After the school sent an email. | After the school sent an email. | 1 | A✓ |

来源：`results_mmsu_open/mmsu_open.jsonl`；MCQ 对照：`results/mmsu.jsonl`。

**VB knowledge — 英/中实体不一致**

| sample_id | ref | pred | match |
| --- | --- | --- | ---: |
| `knowledge-000` | 盐湖城 | Salt Lake City | 0 |
| `knowledge-072` | 母女情深 | The Women | 0 |
| `knowledge-288` | 马克·哈米尔 | Mark Hamill | 0 |

来源：`results/vocalbench_zh_knowledge.jsonl`。

**VB open_ended — ROUGE 有区分；语言不一致 outlier**

| sample_id | ROUGE-L | 备注 |
| --- | ---: | --- |
| `open_ended-100` | 50.3% | 要点覆盖较好 |
| `open_ended-140` | 1.2% | pred 英文 / ref 中文 |

来源：`results/vocalbench_zh_open_ended.jsonl`。

**备选指标离线重算（同批 pred，不 rerun 模型）**

| 子集 | qa-exist | token_f1 | rouge_l | chrF |
| --- | ---: | ---: | ---: | ---: |
| MMSU open | 20 | 53.5 | 50.1 | 58.1 |
| VB knowledge | 70 | 70.0 | 70.0 | 70.0 |
| VB open_ended | 0 | 41.0 | **25.2** | 15.9 |

MER/WER **不适用于** adapted QA，故未列入。


#### 影响

1. v1 分数 **不能用于** 模型对比或 capability 结论。
2. v2 短答/长答 **分列** 后，VB 两集可作 **auxiliary 参考**；**不能**合成单一 accuracy。
3. MMSU open **20% 会严重误导**；须附 MCQ 100% 或 token_f1 等说明。
4. 四集 **Prompt 不同**，pred **不可**当作同一指令下的能力排序。
5. 在 MER 数据到位前，**整个 code_switch capability 无协议主分**。


#### 建议方向

1. **已做**：短答 `qa-exist-match` + 长答 `rouge-l`；Prompt 分轨（`qwen3-omni-cs-*`）；Job 12329/12330 重跑。
2. **MMSU open**：改 prompt 去 *The speaker says…*；或 gold 改为 natural answer；或 **token_f1 辅** / LLM judge；勿单独报 qa-exist 当能力分。
3. **VB knowledge**：evaluator 增加 **跨语言实体归一**或允许多 ref（中/英译名）。
4. **VB open_ended**：prompt 约束与 ref **同语言**；chrF 作辅指标 **待定**。
5. **计划主分**「归一化 contain」：离线试算与 qa-exist **同分**（MMSU open 20%）— 需与 2 一并推进，非仅加归一化层。
6. 正式报告：**分列 + 标注 adapted**；索引 `protocol.metrics` 改为具体 evaluator 名。


#### 收口结果（2026-09-02，部分）

| 项 | 结果 |
| --- | --- |
| 指标/Prompt 拍板 | 短答 contain 族 + 长答 ROUGE-L；四集 Prompt 分轨 |
| v2 smoke | Job **12329/12330**；failure=0 |
| VB knowledge | 0% → **70%**（短答 prompt 生效） |
| VB open_ended | 0% → **25.2%**（ROUGE 有区分度） |
| MMSU open | 10% → **20%**（仍低估；**未收口**） |
| 归一化 contain registry | **未落地** |
| 跨语言实体 | **未做** |

**MMSU open 逐条（附录）**

| # | sample_id | match | 人核 |
| --- | --- | ---: | --- |
| 0 | `9d5edb56` | 0 | MCQ=D✓ |
| 1 | `f98f977f` | 0 | MCQ=D✓ |
| 2 | `df3de0c6` | 0 | MCQ=B✓ |
| 3 | `e4eb0d86` | 0 | MCQ=B✓ |
| 4 | `3076b823` | 0 | MCQ=D✓ |
| 5 | `d6799770` | 0 | MCQ=D✓ |
| 6 | `6496444c` | 1 | 字面匹配 |
| 7 | `9cb5b2dd` | 0 | MCQ=B✓ |
| 8 | `de81dc9d` | 0 | MCQ=C✓ |
| 9 | `c2b8d10a` | 1 | 字面匹配 |

---

### 甲-CS-02. VocalBench 音频路径：`Audio` 相对 `dataset_root/audio/`


| 字段 | 内容 |
| --- | --- |
| 发现时间 | 2026-09-01；收口：2026-09-02 |
| 场景 | `benchmark_subset_manifest.jsonl` 中 vocalbench / vocalbench_zh；smoke `build_manifests.py` |
| 相关产物 | `benchmark_subset_manifest.jsonl`；`native_v2_pipeline/config/resources.json`；`adapters.py` |
| 严重程度 | 高（下游按 `dataset_root + Audio` 拼接会找不到文件） |
| 状态 | ✅ **已收口** |


#### 现象

样本 JSON 中 **`Audio`** 形如 `code_switching/knowledge/000.wav`，相对 **`dataset_root/audio/`**，不是相对 `dataset_root`。  
索引 `subset.selector` 原先 **未声明** `audio_root`，与官方 VocalBench 格式不一致时易拼错路径。

smoke 脚本一直用 `VB_ROOT / "audio" / item["Audio"]`，故 smoke 能跑；问题在 **索引 / 再生 manifest 的下游**。


#### 样例

| 字段 | 值 |
| --- | --- |
| `source.dataset_root` | `.../VocalBench-zh`（正确） |
| 样本 `Audio` | `code_switching/knowledge/000.wav` |
| **正确绝对路径** | `{dataset_root}/audio/code_switching/knowledge/000.wav` |
| **错误拼接** | `{dataset_root}/code_switching/knowledge/000.wav` |


#### 影响

1. 依赖索引拼路径的流水线 **会批量找不到音频**。
2. 与 ASR 轮 **#5 SBCSAE dict 路径** 同类：元数据规则未写清导致 runtime 失败。


#### 建议方向

1. 在索引 `selector` 声明 `audio_root: "audio"`、`audio_field: "Audio"`。
2. **不改** VocalBench 原始 JSON 里的相对路径。


#### 收口结果（2026-09-02）

- `benchmark_subset_manifest.jsonl`：**13 条** vocalbench / vocalbench_zh 已补 selector。
- 同步 `resources.json`、`adapters.py`（解析优先 `dataset_root/audio/<Audio>`）。
- smoke VB 两集 Job 12329：**failure=0**。

---

### 甲-CS-03. MMSU code_switch 从 MCQ 改编为开放式 QA


| 字段 | 内容 |
| --- | --- |
| 发现时间 | 2026-09-02；收口：2026-09-02 |
| 场景 | `smoke_cs_all` → MMSU `code_switch_question_answering` |
| 相关产物 | `mmsu_open/manifest.jsonl`；`suite_mmsu_open.yaml`；prompt `qwen3-omni-cs-open-qa` |
| 严重程度 | 中（形态与端侧场景不一致；非评测链路 bug） |
| 状态 | ✅ **已收口**（开放 QA 链路；计分问题归 **甲-CS-01**） |


#### 现象

端侧少见「听音频选 A/B/C/D」。同一 capability 下 **MCQ Accuracy** 与 **开放 QA** 不可合并为一分。  
需与 speech_translation 轮 MMSU 改编做法对齐：去选项、保留 **同一批 sample_id**、gold = 原正确选项文本。


#### 样例

| 形态 | prompt | pred 示例 | 指标 |
| --- | --- | --- | --- |
| MCQ（diagnostic） | `mmsu` | `D` | mmsu-choices **100%** |
| open（主形态） | `qwen3-omni-cs-open-qa` | The speaker watched a video. | qa-exist **20%**（见 CS-01） |

同一音频 `9cb5b2dd`：MCQ 选 B 正确；open pred 语义对但 qa-exist=0。


#### 影响

1. 若不改编，MMSU 分数 **不能代表** 端侧开放问答能力。
2. 改编后须 **保留 MCQ** 作 diagnostic，且 open 计分仍受 CS-01 约束。


#### 建议方向

1. MMSU CS 主记录以 **open QA** 为准；MCQ 不删索引、不作 capability 主分。
2. manifest：`protocol: open_qa`；suite 独立 `results_mmsu_open/`。


#### 收口结果（2026-09-02）

- `build_manifests.py` 生成 `mmsu_open/`（10 条与 MCQ **同 sample_id**）。
- Job **12284 / 12330** open 链路跑通（failure=0）。
- MCQ Job 12021 结果保留于 `results/mmsu.jsonl`（100% diagnostic）。

---

## 附录：Manifest / Suite 路径

| 集 | Manifest | Suite | 结果 |
| --- | --- | --- | --- |
| mmsu open | `mmsu_open/` | `suite_mmsu_open.yaml` | `results_mmsu_open/` |
| mmsu MCQ | `mmsu/` | `suite.yaml` | `results/` |
| VB knowledge / open_ended | `vocalbench_zh_*` | `suite.yaml` | `results/` |

索引来源：`benchmark_subset_manifest.jsonl`（`capability=code_switch`）。
