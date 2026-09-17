# VoxMatrix 评测问题记录

记录在使用 / 冒烟 / 正式评测过程中发现的问题，便于后续对齐口径与修复。

> **如何听取样例音频**  
> 原始数据在 Cursor 里经常打不开或没声音，原因包括：文件过大（几十～几百 MB）、8 声道 / WAVEX、AMI 为 32-bit float。  
> **请听已切好的 16-bit 单声道短 wav**（合计约 1MB），目录：  
> `VoxMatrix/问题记录_听音样例/`  
> 在文件树点开对应 `.wav` 即可；不要打开下面「原始文件」那一列的长录音。

---

## 状态总览（2026-09-02 更新）

| # | 问题 | 状态 | smoke 当前指标 | 下一步 |
| --- | --- | --- | --- | --- |
| 1 | 长音频未切片 | ✅ **已收口** | 见下表各集 CER/WER | 无（MISP/SBCSAE 子项见 #4/#5） |
| 2 | 英文语气词 / 空 ref 评测失败 | ✅ **已收口** | AMI WER **6.9%**，skip 1/10 | 无（`YEAH`→`Right.` 等按转写错误计分） |
| 3 | 长音频 Instruct ASR 早停（EasyCom 等） | ⏳ **待处理** | EasyCom WER **~83%**；原生推理已证实 **~60s 即早停** | 句级切分；或换 Qwen3-ASR 专用模型 |
| 4 | MISP pcm / 通道 / 前端 | ⚠️ **部分解决** | CER **~11.4%**（7 条有效，skip 3/10） | 拍板口径 A/B；可选 ch0 对比 |
| 5 | SBCSAE dict 路径 | ✅ **已收口** | WER **~21%**（7 条有效，skip 3/10） | 无 |
| 6 | common_voice 语言配置错 | ✅ **已修** | WER **~2.6%** | 无 |

**主 suite 可直接参考的 smoke 集**：aishell_1/4、alimeeting、magicdata_ramc、librispeech、chime、common_voice、wenetspeech、ami。

**仅 diagnostic / 待口径**：aishell_5（极短口语 smoke ~44%）、misp（#4）、easycom（#3）。sbcsae 已收口，可作 diagnostic（口语/unlabelable）。

产物目录：`smoke_asr_all/results/`（ami/sbcsae/misp 已于 8/31 合并 filler+placeholder skip 重评；旧版备份 `*.bak_filler_skip`）。

---

## 问题列表

### 1. 长音频未按时间戳切片，多条样本共用整段录音


| 字段   | 内容                                                                                                  |
| ---- | --------------------------------------------------------------------------------------------------- |
| 发现时间 | 2026-08-30                                                                                          |
| 场景   | `smoke_asr_all`（Qwen3-Omni，每集 10 条）                                                                 |
| 相关产物 | `smoke_asr_all/{aishell_4,aishell_5,alimeeting,magicdata_ramc,misp}/manifest.jsonl` 及对应 `results/*` |
| 严重程度 | 高（指标基本不可信）                                                                                          |
| 状态   | **已收口（切片链路）**：6 集全量 `utt_clips` 已切完；smoke `manifest.jsonl` 已切到独立短 wav；`results/` 已合并切片重测结果。MISP 另见 **#4**（pcm 通道），SBCSAE 另见 **#5**（空 ref） |


#### 现象

会议 / 车载 / 对话类数据在 `full_eval/*/final/manifest.jsonl` 中是「**同一条长音频 +** `start_time`**/*`*end_time` **+ 短句 ref**」。  
冒烟抽样本时只写了 `WavPath=audio_path` 与 `text`，**丢掉了时间戳**，导致：

1. 多条样本 `WavPath` 完全相同（按文件顺序取前 10 条时，往往落在同一场录音开头）。
2. 模型每次听整场 / 整段长音频，输出几乎相同的长转写。
3. 却用不同短句 ref 算 CER → 语料级分数飙到几百～上千 %。


| 数据集            | smoke 中 WavPath 情况             | overall（不可信）          |
| -------------- | ------------------------------ | --------------------- |
| aishell_4      | 10 条同一 `L_R003S01C02.flac`     | ~164.9%               |
| aishell_5      | 10 条同一 `DA03.wav`              | ~173.2%               |
| alimeeting     | 10 条同一 `R8001_M8004_MS801.wav` | ~598.8%               |
| magicdata_ramc | 10 条同一对话 wav                   | ~1260%                |
| misp           | 10 条同一 `.pcm` 长会               | 失败率 100%，见 **#4**（pcm 打不开 + 未切片） |


对照：`aishell_1` / `librispeech` / `ami` / `chime` 每条已是独立短 wav，无此问题。

#### 样例

短句已抽出到 `VoxMatrix/问题记录_听音样例/`（16-bit 单声道 PCM）。在文件树打开该目录下的 wav。

| 听这个文件 | 对应 ref | 从原始文件切出的时间 |
| --- | --- | --- |
| `问题记录_听音样例/aishell4_0000_零零六.wav` | 零零六 | 4.835–6.445 s |
| `问题记录_听音样例/aishell4_0001_今天啊把各位都叫过来.wav` | 今天啊把各位都叫过来啊… | 23.475–28.485 s |
| `问题记录_听音样例/aishell5_读吧.wav` | 读吧。 | 3.966–4.466 s |
| `问题记录_听音样例/aishell5_是聊天儿还是读啊.wav` | 是聊天儿还是读啊。 | 13.436–14.616 s |
| `问题记录_听音样例/alimeeting_0000_研讨会开场.wav` | 好嗯，咱们今天针对… | 6.9–18.29 s |
| `问题记录_听音样例/alimeeting_0001_首先谈一下看法.wav` | 嗯，首先谈一下大家伙的一个看法啊嗯，那。 | 18.73–22.43 s |
| `问题记录_听音样例/magicdata_0000_采集片头.wav` | 爱数智慧语音采集二零一九年十一月八日 | 0.52–5.088 s |
| `问题记录_听音样例/magicdata_0001_聊高中生活.wav` | 咱们来聊一下高中生活呗 | 6.688–8.32 s |

原始长文件（不必打开）：AISHELL-4 约 273MB/8ch、AliMeeting 约 385MB/8ch WAVEX、MagicData 约 31 分钟、AISHELL-5 约 7 分钟。



#### 影响

1. 上述数据集的 smoke / 未切片评测分数**不能用于模型对比**。
2. 若不修复，正式评测会系统性虚高 CER/WER。
3. 仅把 `{path, start_time, end_time}` 写进 jsonl 不够：Prompt 会把 dict 字符串化当路径打开，见 **#5 SBCSAE**。应**预切成独立短 wav** 再评。



#### 建议方向

1. 对带 `start_time`/`end_time` 的样本，评测前用 soundfile/ffmpeg 切出单句 wav，`WavPath` 指向切片文件。
2. 抽样时尽量跨会话 / 跨文件取，避免 10 条全挤在同一场录音开头（切片后即使同场也有效，但多样性更好）。
3. `misp` 见 **#4**：裸 `.pcm` 需先转 wav，再按时间戳切。
4. 在问题未修复前，报告中将相关数据集标为 **invalid / diagnostic only**。

#### 收口结果（2026-08-31）

全量切片目录：`/mnt/afs/eval_data/utt_clips/<benchmark_id>/`（`clips/` + `indexes/`）。  
smoke manifest 已从 `manifest_unsliced.jsonl` 备份并切换为 `utt_clips` 短 wav；重测产物已写回 `smoke_asr_all/results/`。

| 数据集 | utt_clips 句数 | 切片前 overall | 切片后 overall | 切片前音频时长 | 切片后音频时长 | 结论 |
| --- | ---: | ---: | ---: | --- | --- | --- |
| aishell_4 | 10541 | ~165% | **~6.8%** | 10 条均 ~2363 s | 1.6–16 s | ✅ **可用**，#1 已解决 |
| aishell_5 | 20337 | ~173% | **~44%** | 10 条均 ~403 s | 0.5–3.3 s | ✅ 切片有效；分数偏高因 smoke 多为极短口语句（「读吧」「嗯」），非流水线 bug |
| alimeeting | 45630 | ~599% | **~25%** | 10 条均 ~1574 s | 0.2–11 s | ✅ **可用**；id=5/6 短句 outlier 拉高均值 |
| magicdata_ramc | 36407 | ~1260% | **~6.8%** | 10 条均 ~1894 s | 0.4–4.6 s | ✅ **可用**，#1 已解决 |
| misp | 2663 | 100% 失败 | **~11.4% CER**（skip 3/10） | 裸 pcm 打不开 | 0.2–7 s | ⚠️ **#1+#4 切片+重切已做**；diagnostic 基线；声道/前端口径见 **#4** |
| sbcsae | 14157 | 100% 失败 | **~21% WER**（skip 3/10，7 条有效） | dict 路径打不开 | 0.4–2.3 s | ✅ **#1+#5+#2 已收口**（`um`/`mhm`/`xxx` skip） |

**逐集要点**

- **aishell_4**：10 条 pred 与 ref 语义对齐；最长句 ~16 s，CER 个位数。典型修复：同 pred 对 10 个不同 ref → 每句独立短 wav + 独立 pred。
- **aishell_5**：切片后不再「10 条同一 pred（今天气温不正常啊）」；剩余误差来自 ASR 对超短句（0.5–1 s）识别困难，如 id=0「读吧」→「对吧」、id=3「聊天儿」→「那这」。
- **alimeeting**：会议远场 8ch 切片后 CER ~25%；id=6 ref「就针对性」pred 偏到「彩色」属重叠/难句，非未切片。
- **magicdata_ramc**：车载对话切片后与 aishell_4 同级；id=8「诶」→「唉」单字误差可接受。
- **misp**：8/31 s32le 重切后 CER ~13%→合并 filler-skip 后 **~11.4%**（skip「唉」「嗯。」×2）；含实词短句「多了。」「随便啊。」仍计分。
- **sbcsae**：7/10 条有效 WER ~21%；id=2/9 语气词、id=4 整句 `xxx` 已 **skipped**。

---



### 2. 英文语气词 / 填充词归一化导致评测失败或漏计


| 字段   | 内容                                                   |
| ---- | ---------------------------------------------------- |
| 发现时间 | 2026-08-30                                           |
| 场景   | `smoke_asr_all` → AMI（IHM 已切好的短 wav）                 |
| 相关产物 | `smoke_asr_all/results/ami.jsonl`、`ami-overall.json` |
| 严重程度 | 中（纯语气词曾导致评测失败；AMI 整体可跑）                    |
| 状态   | ✅ **已收口**（2026-08-31）：纯语气词 skip；`results/ami.jsonl` 已更新，**failure_rate=0**，skip 1/10，WER **~6.9%**（9 条有效） |




#### 现象

AMI **不是**长音频未切片问题：每条已是独立短 wav。问题出在 **英文** `EnglishTextNormalizer` **会去掉口头语**：

```text
ignore_patterns ≈ hmm | mm | mhm | mmm | uh | um
```

当 reference 本身几乎全是这类语气词（如 `MM-HMM`）时，归一化后 **ref 变成空串**，`compute_wer` 抛出：

`WER reference corpus is empty after normalization`

导致该条无 `eval`、overall `failure_rate=10%`，并出现 `"error": "'pred'"`。

另有 ref=`YEAH`、pred=`Right.` 等 **转写错误**（非近义等价），按 WER 正常计 100%——属模型/识别问题，不是评测口径争议。

#### 样例

AMI 原始文件是 **32-bit float wav**，Cursor / 常见播放器常打不开或没声。请听已转成 16-bit 的短文件：

| id | 听这个文件 | ref | pred | 结果 |
| --- | --- | --- | --- | --- |
| 0 | `问题记录_听音样例/ami_00_YEAH.wav` | `YEAH` | `Right.` | wer%=100（**转写错误**） |
| 1 | `问题记录_听音样例/ami_01_MM-HMM.wav` | `MM-HMM` | `Uh huh.` | **评测失败**：ref 归一化后为空 |
| 6 | `问题记录_听音样例/ami_06_MM_MONDAY_AFTERNOON.wav` | `MM MONDAY AFTERNOON` | `Monday afternoon.` | wer%=0（`MM` 被剥掉后对齐） |

相关代码：`audio_evals/lib/text_normalization/en.py`（`EnglishTextNormalizer.ignore_patterns`）→ `audio_evals/lib/wer.py` → `compute_wer`。

#### 影响

1. 纯语气词样本会让评测链路报错，而不是给出合理的 0/1 分（已通过 skip 修复）。
2. 含前缀 `MM`/`UH` 的句子可能被静默删掉填充词再计分（如 id=6）——归一化行为，与纯语气词 skip 无关。
3. ref 与 pred 词面不一致（如 `YEAH` vs `Right.`）按 **转写错误** 计 WER，不做近义放宽。
4. smoke 若大量抽 `MM-HMM` 等极短条，分数波动大，不能代表会议 ASR 能力。



#### 建议方向

1. ~~归一化后 ref 为空时：跳过并记 `skipped`~~ → **已实现并合并**（`is_filler_only_reference`；replay 验证见 `results_filler_skip/` → 已写回 `results/`）。
2. **口径**（#2 扩展）：单独语气词 → `filler_only` skip；**整句占位**（`xxx` 等）→ `placeholder_ref` skip；含实词句子正常计分。
3. **转写错误**（如 id=0 `YEAH`→`Right.`）**正常计 WER**，不做近义匹配或放宽。
4. 抽样可提高最短时长 / 最少词数阈值，减少纯应答条占比。

---

### 3. 长音频 Instruct ASR 早停（EasyCom 等）：整段已喂入，pred 仍只覆盖部分

| 字段 | 内容 |
| --- | --- |
| 发现时间 | 2026-08-30 |
| 场景 | `smoke_asr_extra` / 已合并到 `smoke_asr_all/results/easycom.jsonl`（Qwen3-Omni，10 条） |
| 相关产物 | `smoke_asr_extra/easycom/manifest.jsonl`、`results/easycom.jsonl`、`easycom-overall.json` |
| 严重程度 | 高（分数不可当句级 ASR）；流水线本身跑通 |
| 状态 | ⏳ **待处理**（prompt / 原生推理对照已完成，见下） |

#### 与 #1 的区别

**不是**「丢掉时间戳、10 条共用一条超长会」。EasyCom 每条已是独立约 **60 秒** 片段（10 条不同 wav），6 声道已转成单声道 16-bit。  
`audio_duration_seconds` 均为 **60.0**，人耳听完整 60 秒也正常。问题在 **输出覆盖不全 + ref 口径**。

#### 现象

1. ref = 该 60 秒内 **所有说话人转写拼成一大段**。
2. pred 往往只覆盖前半 / 某一说话人，甚至半句就停。语料 WER **~82.9%**，10 条均无 0 分。
3. 不是音频只喂了一半：同一 60 秒，有的 pred ~96 词，有的只有 6 词，更像 **Instruct ASR 早停（EOS）**；多人场景还会叠加「只跟住一条说话线」。
4. 当前 prompt 仅为 `Transcribe the English audio into text.`；对照实验表明 **改 prompt / 加大 `max_new_tokens` 均无效**（见下）。
5. **2026-09-02 补充**：干净 **单人 60s** LibriSpeech 拼接也会早停（覆盖率 ~50%），说明根因不只在「多人重叠」，而是 **Qwen3-Omni-Instruct 对 ~60s 长段 verbatim ASR 的生成上限**。

id=0 示例：pred 写到 Sophie / trucks / drive all over 后停在 **`I was in.`**；ref 后面还有 Tennessee、去 Alaska、啤酒钱等。

| id | 音频时长 | pred 词数 | ref 词数 | wer% |
| --- | --- | --- | --- | --- |
| 0 | 60s | 96 | 200 | ~76 |
| 2 | 60s | 6 | 258 | ~98 |
| 9 | 60s | 59 | 140 | ~63 |

#### 样例（请听完整 60 秒，不要只听 20 秒预览）

| 说明 | 听这个文件 |
| --- | --- |
| id=0 完整评测音频（推荐） | `smoke_asr_extra/easycom/audio/easycom_test_10_00_00_000.wav` |
| id=2 完整评测音频 | `smoke_asr_extra/easycom/audio/easycom_test_10_03_00_332.wav` |
| id=9 完整评测音频 | `smoke_asr_extra/easycom/audio/easycom_test_10_10_00_423.wav` |
| id=0 仅前 20 秒预览（不完整） | `问题记录_听音样例/easycom_00_first20s.wav` |
| id=2 仅前 20 秒预览 | `问题记录_听音样例/easycom_02_first20s.wav` |
| id=9 仅前 20 秒预览 | `问题记录_听音样例/easycom_09_first20s.wav` |

#### 影响

1. EasyCom smoke 的 ~83% WER **不能代表模型句级识别能力**。
2. 人耳觉得「音频没问题」仍可能 pred 很短：输入完整 ≠ 生成写满。
3. 若协议是句级 ASR，应按 `Start_Frame`/`End_Frame` 再切单句；若协议是长段转写，**不宜**继续用 Qwen3-Omni-Instruct 直接评（见原生推理对照）。
4. **~30s 以内单人朗读**仍可正常用 Omni-Instruct ASR（LibriSpeech smoke 已验证 100% 词覆盖）。

#### 建议方向

1. 句级：按转写时间戳切单句 wav 再评 WER（**推荐**）。
2. 段级：换 **Qwen3-ASR 专用模型** 或 VAD 切段后再评；勿指望 Instruct + prompt/`max_new_tokens` 修早停。
3. 报告中在修好前将 EasyCom 标为 diagnostic only。

#### 对照实验（2026-08-31）

已用加强 prompt 重跑 10 条（Job 11317，`smoke_asr_extra/results_full_prompt/`）：

- Prompt 要求：转写整段、全部说话人、不要中途停止。
- 结果：WER **~78.5%**（原短 prompt **~82.9%**），略好但仍不可用。
- 结论：**仅改 prompt 不够**；须 **句级切分** 或换长段 ASR 专用模型。

#### 对照实验（2026-09-01）：EasyCom 原生推理（排除 VoxMatrix 封装）

脚本：`smoke_asr_extra/scripts/native_qwen3_easycom_test.py`（直接调 `transformers` Qwen3OmniMoe，不经 VoxMatrix subprocess）。  
样本：`easycom_test_10_00_00_000.wav`（60s）。Job **11810**。  
产物：`smoke_asr_extra/results_native_qwen3/easycom_test_10_00_00_000.json`

| 配置 | pred 词数 | 生成 token | 覆盖率 |
| --- | --- | --- | --- |
| VoxMatrix 现有结果 | 96 | — | 48%（96/200） |
| 原生 / 默认 prompt | **96** | **126** | 48% |
| 原生 / 加强 prompt | 98 | 129 | 49% |
| 原生 / 加强 prompt + max_tokens=4096 | 98 | 129 | 49% |

- 原生默认 prompt 与 VoxMatrix **逐字一致**，说明流水线无额外截断。
- 模型约 **129 token** 后主动 EOS，**不是** `max_new_tokens` 限制。

#### 对照实验（2026-09-02）：LibriSpeech 60s 单人朗读（验证是否仅多人问题）

脚本：`smoke_asr_extra/scripts/prepare_librispeech_60s_clip.py` + `native_qwen3_asr_test.py`。  
样本：LibriSpeech dev-clean 说话人 `1272-128104`，5 条连续朗读拼接（**62.5s，单人**）。Job **12253**。  
产物：`smoke_asr_extra/results_native_qwen3/librispeech_60s_single.{wav,meta.json,json}`

| 场景 | 时长 | 说话人 | pred/ref 词数 | 生成 token | 覆盖率 |
| --- | --- | --- | --- | --- | --- |
| LibriSpeech 单句（smoke） | 29.4s | 单人 | 68/68 | — | **100%** |
| LibriSpeech 拼接（本次） | 62.5s | 单人 | 76/151 | **93** | **50%** |
| EasyCom id=0（09-01） | 60.0s | 多人 | 96/200 | 126 | 48% |

- pred 停在 `...and can discover.`，后半段（含 29.4s 的第 5 条朗读）未输出。
- 加强 prompt / `max_new_tokens=4096` 仍只生成 **93 token**。
- **结论**：~60s 长段 verbatim ASR 的早停是 **Qwen3-Omni-Instruct 模型行为**，与是否多人重叠无关；EasyCom 低分叠加了「多人 ref 拼接」口径，但早停本身在干净单人 60s 上同样存在。

---

### 4. MISP：裸 `.pcm` 无法推理，且未按时间戳切片


| 字段 | 内容 |
| --- | --- |
| 发现时间 | 2026-08-30 |
| 场景 | `smoke_asr_all` → MISP（Qwen3-Omni，10 条） |
| 相关产物 | `smoke_asr_all/misp/manifest.jsonl`、`results/misp.jsonl`、`misp-overall.json`；对照 `vocal_bench/work/full_eval/misp/final/manifest.jsonl` |
| 严重程度 | 高（无分数；修好切片前也不应当句级 ASR） |
| 状态 | 部分解决：#1 切片 + **8/31 s32le 重切**；smoke **CER ~11.4%**（skip 3/10）；**`dataset.yaml` 已改 8ch s32le**。**声道/前端口径 A/B 待拍板** |


#### 与 #1 的关系

**同时踩了 #1**（丢掉 `start_time`/`end_time`，10 条共用一场长会），但 MISP **连 pred 都没有**。  
MagicData 等同属 #1 的集是合法 wav，能听、能出长转写，只是用短句 ref 把 CER 抬到上千 %。  
MISP 原始文件是 **无头裸 PCM**，模型侧按普通音频打开即失败。

#### 现象

1. 10 条全部 `failure_stage=inference`，`failure_type=model_runtime`。错误仅为：

   `qwen3-omni failed: Error:`

   子进程只回了 `Error:`，几乎没有细节。overall：`failure_rate=1.0`，`"error": "'pred'"`，**没有 CER**。

2. 延迟约 **0.53–0.61 s**（MagicData 同类长音频成功推理约 15 s）。说明在读音频阶段就退出，没有做完整转写。

3. smoke 的 `WavPath` 10 条完全相同：

   `/mnt/afs/oss_data/datasets/03_multi_speaker/MISP/dev-CSOBx3/M014/M014-CSOBx3/M014-CSOBx3.pcm`

   文件约 **1.3 GB**。Cursor / 常见播放器打不开裸 pcm。

4. `full_eval` 实际是句级：家居远场、会议阵列、多人重叠；每条有 `start_time`/`end_time` + 短句 `reference.text`。smoke 只写了路径和 text。

| id | sample_id（节选） | 应切区间 | ref |
| --- | --- | --- | --- |
| 0 | `...0b96e943d9f8` | 68.677–70.083 s | 一共是四十分钟是吧？ |
| 1 | `...cbb9051a27fe` | 72.002–74.062 s | 那我们五个人，一个人讲十分钟呗。 |
| 2 | `...5149bb120441` | 74.632–74.854 s | 唉 |
| 3 | `...96f9d4ea6f85` | 75.117–75.538 s | 多了。 |
| 6 | `...23dccab63712` | 82.128–83.148 s | 不是有主题吗？ |
| 8 | `...673448dc4697` | 392.518–399.4 s | 嗯我认为个性化无论是教育还是说我们生活中的方方面面其实都是非常重要的。 |
| 9 | `...cdc64617ca39` | 399.67–406.11 s | 就比如说我们通过一些个性化的手段，比如说呃 |

前几条挤在约 68–83 s，后面跳到约 391–406 s，路径仍是同一条长会。

**听音：** 请勿直接打开 1.3GB 裸 `.pcm`。请听 `问题记录_听音样例/` 或 `eval_data/utt_clips/misp/clips/` 下 16-bit 单声道短 wav。

| 说明 | 听这个文件 |
| --- | --- |
| **重切后生产 clip**（评测实际喂入，8 路混音） | `eval_data/utt_clips/misp/clips/de/misp_MISP_Meeting_0b96e943d9f8.wav` |
| 重切前错误解码（16ch 当 s16，响但是噪声） | `问题记录_听音样例/misp_错切_16ch当s16_白噪音.wav` |
| 正确解码 ch0 单通道（id=0） | `问题记录_听音样例/misp_00_正确解码_ch0_四十分钟.wav` |
| 8ch s32 混音 vs ch0 对比 | `misp_正确_8ch_s32混音.wav` / `misp_正确_8ch_s32_ch0.wav` |

#### 进展（2026-08-31）

1. **PCM 格式**：CSOBx3 实际为 **8 声道 s32le @ 16kHz**（非 `dataset.yaml` 写的 16ch s16le）。误按 16ch s16le 解码会得到「白噪音」；脚本 `slice_utt_clips.py` 已改为 `8ch s32le`。
2. **全量重切**：Job **11287**（8/31 00:13）`--no-skip-existing` 覆盖 2663 句，`verify_ok: true`。
3. **听感确认**：生产 clip（8 路混音）**有人声、略小声**；不再是 decode 完全错误。
4. **ASR 分数**：Job **11525** smoke CER ~13.2%；合并 filler-skip 后 **~11.4%**（skip id=2/4/7 纯语气词「唉」「嗯。」）。

#### segment adapter（mesh 运行时切句）与 utt_clips 不一致

mesh 正式路径若走 `AudioSegmentPrompt` + `materialize_audio_segment()`（`audio_evals/audio_segment.py`），对 `.pcm` 的假设是：

| 参数 | segment adapter | utt_clips（当前正确做法） |
| --- | --- | --- |
| 格式 | **mono s16le** | **8ch s32le** |
| 混音 | 仅 soundfile 误读多声道时 `mean(axis=1)`；ffmpeg 路径输入即 `-ac 1` | ffmpeg 读 8ch 后 **`-ac 1` 混音** |
| 与 MISP 匹配 | ❌ 格式假设错误 | ✅ 已修正 |

`benchmark_evaluator_map.yaml` 写「segment adapter 解码为 16k mono s16le」描述的是代码字面行为，**不适用于 MISP CSOBx3**。正式评测应走 **`utt_clips` 预切**，或扩展 adapter 支持 dataset 级 pcm 参数。

#### 混音 vs 固定单通道（待决策）

当前 `utt_clips` 切片：`8ch s32le` 读取 → **8 路算术平均混成 mono**（`-ac 1`）。实测 id=0（ref「一共是四十分钟是吧？」）：

| 方式 | RMS | 说明 |
| --- | ---: | --- |
| 8 路混音（当前生产） | 0.0095 | 人耳可听清，偏小 |
| 固定 **ch0** | 0.0126 | 约 **1.33×** 电平 |
| 固定 **ch3**（该句最响） | 0.0130 | 约 **1.37×** 电平 |

id=8 长句：混音 0.0120，ch3 0.0188（约 **1.57×**）。**最响通道随句子变化**（环形阵列 + 多人位置不同），无全局最优固定通道。

**结论**：

1. **固定单通道（如 ch0）大概率优于 8 路混音**（信噪比更高、电平更大），值得重切 smoke 10 条 + 重测验证。
2. 固定 ch0 是简单可复现的第一步；更优可选 **每句选 RMS 最大通道** 或官方 beamforming。
3. 简单混音会把非目标方向环境声/他人语音平均进来，稀释目标说话人——这是「能听但偏小」的主要原因之一。

#### 待讨论：F8N、官方前端、与 benchmark 口径

**F8N 近场参考麦是什么**

MISP 每场会同时录两路音频：

| 路线 | 设备 | 在我们项目里的角色 |
| --- | --- | --- |
| **CSOBx3** | 桌上讯飞 8 麦远场阵列（`.pcm`） | **ASR 评测输入**（`far_field` / `meeting_array`） |
| **F8N** | 每人头戴麦 → Zoom F8N 录音机（`.wav`） | **不直接评测**；标注员听这路写 ref，SNR >15dB |

同一句 ref 在转写 JSON 里**同时有 CSOBx3 与 F8N 的时间戳**，但文本是按近场听写的。因此评测本质是：**用难听的远场阵列音频，去对齐近场标出来的 ref**——这是 MISP 会议 ASR 的设定。

**官方对阵列音频怎么处理**

官方 baseline（[misp2022_baseline / track2_AVDR](https://github.com/mispchallenge/misp2022_baseline/tree/main/track2_AVDR)）**不是**简单混音或固定 ch0，而是：

```
8ch CSOBx3 pcm → WPE 去混响（nara_wpe）→ BeamformIt 波束成形 → 按 diarization 切句 → ASR
```

MISP 2025 优胜队进一步用 **WPE + GSS 分离/增强** 等。官方文档与竞赛管线里，**F8N 不参与 ASR 输入**，只保证转写质量。

**讨论要点：官方处理 vs 模型转写能力**

| 口径 | 测什么 | 代表做法 |
| --- | --- | --- |
| **A. 端到端远场 ASR** | 模型（+可选简单解码）对**原始/ lightly 处理**远场阵列的转写能力 | 当前 utt_clips：8ch s32le 解码 + 混音切句；smoke CER ~13% |
| **B. 官方竞赛管线** | **前端增强 + 切句 + ASR** 的系统能力；前端负责把远场「变清楚」 | WPE + BeamformIt（+ 分离/融合）；CER 显著低于裸远场 |
| **C. 近场参考（F8N）** | 几乎测「听清了吗」，**不是**端侧阵列场景 | 仅作 ref 来源，不宜作评测音频 |

**共识待拍板**：官方处理（WPE / 波束成形 / 分离）的**本质是在评测链路里先把音频变清晰**，相当把「阵列信号恢复」从模型侧拆到**明确的前端模块**；若 benchmark 目标是 **「端侧模型对难远场音频的裸转写能力」**，则不宜默认叠官方全套前端；若目标是 **「会议远场 ASR 系统能力（含增强）」**，则应纳入 WPE+波束成形 并写进协议。

**当前暂定**

- 现状 **A** 已跑通（diagnostic 基线，CER ~13%），**不等于**官方竞赛口径 **B**。
- **F8N（C）** 仅解释 ref 来源，不作评测输入。
- 正式方案需明确：VoxMatrix 对 MISP 属于 **A 还是 B**，再决定是否在 `utt_clips` 前增加 WPE/BeamformIt，或维持混音/ch0 简化基线。

**可选方案（供方案评审）**

| 选项 | 贴近官方 | 工程成本 | 测的侧重 |
| --- | --- | --- | --- |
| 维持 8 路混音（现状） | 低 | 已完成 | 简化远场 + 模型 |
| 固定 ch0 / max-RMS 单通道 | 低 | 低 | 简化远场 + 模型 |
| **WPE + BeamformIt**（官方 baseline 前端） | 高 | 中 | 系统 ASR（含增强） |
| WPE + GSS / 分离融合 | 很高 | 高 | 竞赛级系统 |

#### 数据与用法定位（简述）

- 2663 句 / 6 场长会（dev），`use_bucket=diagnostic_evidence`（诊断集，非 clean read speech 主榜）。
- 评测口径：CSOBx3 **远场阵列** + 句级 CER；同场另有 F8N 近场参考麦，当前刻意不用。
- 文档已同步：`eval_data/utt_clips/misp/dataset.yaml` 改为 **8ch s32le + ffmpeg 混音**说明。

#### 影响

1. 重切前 MISP smoke **不能用于模型对比**（无 pred 或 pred 对噪声）。
2. 重切后 smoke **CER ~13.2% 可作 diagnostic 基线**；是否与官方/竞赛口径一致，取决于是否采用 WPE+波束成形等前端。
3. mesh 若直走 segment adapter 而不经 `utt_clips`，会再次 decode 失败或得到错误音频。
4. 正式评测若继续指向裸 pcm 或未修正的 adapter，结果不可信。

#### 建议方向

1. ~~用当前 `utt_clips` manifest 重跑 MISP smoke 10 条~~ → **已完成**（Job 11525，CER ~13.2%）。
2. **方案讨论**：明确 MISP 属于口径 **A（裸远场模型）** 还是 **B（含官方前端）**，再定声道/增强策略。
3. 若维持 **A**：可对比 **固定 ch0 / max-RMS** vs 混音，重切 smoke 验证。
4. 若采用 **B**：在切片前对整段 meeting pcm 做 **WPE + BeamformIt**，再按时间戳切句（参考官方 baseline）。
5. ~~**必做**：修正 `dataset.yaml`~~ → **已完成**（2026-08-31）。
6. 报告中将 MISP 标为 **diagnostic_evidence**；口径未拍板前不与 aishell_1 同权重。

---

### 5. SBCSAE：`WavPath` 写成 dict，模板字符串化后当路径打开失败


| 字段 | 内容 |
| --- | --- |
| 发现时间 | 2026-08-30 |
| 场景 | `smoke_asr_extra` → SBCSAE（Qwen3-Omni，10 条；结果已进 `smoke_asr_all/results/`） |
| 相关产物 | `smoke_asr_extra/sbcsae/manifest.jsonl`、`smoke_asr_all/results/sbcsae.jsonl`、`sbcsae-overall.json` |
| 严重程度 | 高（无分数） |
| 状态 | ✅ **已收口**（2026-08-31）：路径已修；`um`/`mhm`/`整句xxx` 评测 skip；smoke WER **~21%**（7 条有效） |


#### 与 #1 / #4 的区别

| | #1（如 MagicData） | #4 MISP | **#5 SBCSAE** |
| --- | --- | --- | --- |
| 时间戳 | **丢掉了**，只写文件路径 | 丢掉了 | **写进了 manifest** |
| 喂给模型的路径 | 合法 wav 字符串 | 裸 `.pcm` 字符串 | **dict 的 `str()`** |
| 结果 | 能推理，CER 虚高 | 打不开 pcm | **文件不存在**（去找那整段字符串） |

SBCSAE 的真实 wav 存在：`/mnt/afs/eval_data/benchmarks/SBCSAE_Public_Speech/raw/wav/SBC003.wav`（约 132 MB）。不是磁盘上没文件，是 **路径字段类型错了**。

#### 现象

1. 10 条全部 `failure_stage=inference`，overall `failure_rate=1.0`，`"error": "'pred'"`，**没有 WER**。
2. `prompt` 里 `audio.value` 已是 Python dict 的字符串，例如：

   `{'path': '.../SBC003.wav', 'start_time': 0.0, 'end_time': 1.01}`

3. 模型按这个**整串**去 `open()`，报错：

   `No such file or directory: "{'path': '.../SBC003.wav', 'start_time': 0.0, 'end_time': 1.01}"`

4. 延迟约 **7–12 ms**（id=0 约 162 ms），读文件立刻失败。

manifest 里 `WavPath` 是对象而不是字符串：

```json
{"WavPath": {"path": ".../SBC003.wav", "start_time": 0.0, "end_time": 1.01}, "text": "okay", ...}
```

10 条都来自 `SBC003.wav` 开头约 0–9.5 s，ref 为口语短句（含 `um` / `mhm` / 转写标记 `xxx`）。`sample_id` 带 `unlabelable`。`um`/`mhm`/`整句xxx` 已评测 skip（#2/#5）。

| id | 区间 | ref |
| --- | --- | --- |
| 0 | 0.00–1.01 s | okay |
| 1 | 0.30–1.65 s | do you have a par ticular |
| 2 | 1.65–2.10 s | um |
| 3 | 2.10–4.00 s | use for the red peppers |
| 4 | 2.15–3.05 s | xxx |
| 5 | 4.00–6.31 s | as opposed to the yellow or green pepp ers |
| 9 | 9.06–9.51 s | mhm |

**听音：** 请勿把 dict 字符串当路径。原始 `SBC003.wav` 约 132 MB，宜按上表切 16-bit 短句后再听。

#### 影响

1. ~~当前 SBCSAE smoke 不能用于模型对比~~ → 路径与 skip 已修，**可作 diagnostic**；WER ~21%（7 条有效，含口语难例）。
2. 任何把 `{path, start, end}` 直接塞进 `WavPath` 的集都会同样 100% 失败（Prompt 模板只做字符串替换，不会切片）。
3. 若改成只写 `path`、丢掉时间戳，会退化成 **#1**。
4. ~~ref 含 `xxx` 不可辨转写（id=4）会虚高 WER~~ → **整句 `xxx` 已 skip**（`placeholder_ref`），不进 overall。

#### 建议方向

1. ~~评测前按 `start_time`/`end_time` 切出独立短 wav~~ → **已完成**（`utt_clips/sbcsae`，14157 句）。
2. Dataset / Prompt 侧若收到 dict，应拒绝或自动切片，不要 `str(dict)`（防回归）。
3. ~~抽样注意 `xxx` 等不可辨转写~~ → **整句 `xxx` 评测 skip**（与 #2 语气词 skip 并列）；句内 `xxx` 仍正常计分。
4. 报告中将 SBCSAE 标为 **diagnostic**（unlabelable 子集占比高）。

---

### 6. common_voice：manifest 语言与 prompt 不匹配（已修）

| 字段 | 内容 |
| --- | --- |
| 发现时间 | 2026-08-30 |
| 场景 | `smoke_asr_all` → common_voice |
| 相关产物 | `smoke_asr_all/common_voice/manifest.jsonl`、`results/common_voice.jsonl`；旧备份 `manifest_zhHK_bak.jsonl` |
| 严重程度 | 中（英文 prompt + 粤语 ref → 指标无意义） |
| 状态 | ✅ **已修**（2026-08-31） |

#### 现象

早期 smoke 从 `full_eval` 抽到 **zh-HK** 样本，但 prompt 为英文 `Transcribe the English audio into text.`，模型输出英文、ref 为粤语，WER 不可 interpret。

#### 修复

manifest 切换为 **CommonVoice_26 英文**（`cv-corpus-26.0/en/clips/`）；重测 WER **~2.6%**，0% failure。

#### 建议

抽 manifest 时 **语言、prompt、evaluator lang 三者一致**；保留 `manifest_zhHK_bak.jsonl` 作回归对照。

---
