# vLLM-Omni 语音链路实验报告

> 实验时间：2026-09-25 ～ 09-26；报告整理：2026-09-30。  
> 模型：`Qwen3-Omni-30B-A3B-Instruct`（被测模型），`Qwen3-Omni-30B-A3B-Thinking`（judge）。  
> 性质：**链路 / 协议验证的 smoke**，样本量很小，不作正式榜单。

---

## 1. 结论

1. **vLLM-Omni 语音输出链路已跑通。** LibriTTS（TTS）和 VocalBench IF（语音回答）各 2 条，两轮运行都成功，`failure_rate=0`。VocalBench IF 两轮都是满分；LibriTTS 的汇总 WER 第一轮 9.30%、第二轮 0.00%。逐条 WER 与汇总 WER 原先用两套文本归一化，现已统一为同一实现，第二轮逐条结果已按新口径重算（见 §5.3）。
2. **推理和评测已能拆成两个任务跑。** 推理任务结束、释放 GPU 后，再开一个任务回放已保存的推理结果做评测，没有再出现 judge 启动时显存不足。
3. **整条链路（被测模型 + Thinking judge）都可以只用 vLLM-Omni 环境。** 第二轮的 judge 也走 Omni（只起 thinker），2 卡完成。
4. **AISHELL-4 长音频转写在 Omni 下精度与 stock vLLM 一致**（12 个窗口 CER 逐窗差异 ≤0.34 个百分点）。最初的 Omni 延迟约为 stock vLLM 的 12 倍，原因是只起 thinker 的部署配置关掉了 CUDA Graph（`enforce_eager: true`），解码退化到约 17 token/s。**关掉 eager 后 12 窗合计从 500s 降到 57s（约 8.7 倍），与 stock vLLM（42s）差距缩小到 1.4 倍**，thinker-only 默认值已改为 `enforce_eager: false`（见 §4.3）。三段语音部署一直是 CUDA Graph 模式，不受影响。
5. **之前 judge OOM 的根因不是 Omni 没释放显存**，而是评测器每条样本都新起一个 Thinking judge 进程。已改为进程内单例。

---

## 2. 背景与目标

- 此前语音输出（TTS / 语音回答）走 HF 实现，速度慢；想换成 vLLM-Omni（thinker → talker → code2wav 三段）。
- 要确认两件事：Omni 能否产出可评测的音频；Omni 的 thinker 做纯文本转写时，精度是否与 stock vLLM 一致。
- 首次 smoke（`vox-omni-speech-smoke-0925-222335064`）推理成功，但 VocalBench IF 第 2 条在评测阶段挂掉：Thinking judge 启动时只剩约 5 GiB 显存。因此改为推理、评测两个任务分开跑。

---

## 3. 环境与配置

| 项 | 值 |
|---|---|
| vLLM-Omni / vLLM | 0.28.0 / 0.28.0（Omni 环境 `qwen3_omni_long_audio/envs/vllm_omni`） |
| transformers / torch | 5.14.1 / 2.13.0+cu130 |
| stock vLLM 对照 | 0.30.0（环境 `qwen3_omni_long_audio/envs/vllm`） |
| 语音输出部署 | `audio_evals/lib/qwen3-omni-vllm-omni/deploy_qwen3_omni_speech_2gpu.yaml`：thinker 在 cuda:0，talker + code2wav 在 cuda:1；显存占比 0.88 / 0.55 / 0.12；thinker `temperature=0.6`，`max_tokens=1200` |
| 转写部署 | `deploy_thinker_only_tp4.yaml`：只起 thinker，TP=4，`temperature=0`。首轮为 `enforce_eager: true`（关闭 CUDA Graph），对照轮为 `false`（见 §4.3） |
| Judge | `qwen3-omni-thinking`，`max_new_tokens=2048`，`temperature=0`（两轮的后端不同，见 §5.1） |

---

## 4. 实验一：AISHELL-4 长音频转写（Omni thinker-only）

- 数据：会议 `L_R003S01C02`，从同一起点截取 30s～600s 的 12 个窗口（`windows.jsonl`）。
- Prompt：“请将这段中文音频完整转写成文字。”；`temperature=0`，`max_tokens=8192`，处理器不截断。
- 指标：CER%（`audio_evals.lib.wer.compute_wer(language=zh)`）。
- 产物：`qwen3_omni_long_audio/results/aishell4_omni0.28.0_g4_u0.88_truncation_false/results.json`（eager）、`..._truncation_false_graph/results.json`（CUDA Graph，任务 25331）

### 4.1 CER% 对比（↓）

HF 两列只跑到 300s。

| 窗口 | HF tf4.57.1 | HF tf5.17.0 | vLLM 0.30（4 卡） | **Omni eager（4 卡）** | **Omni graph（4 卡）** |
|---|---:|---:|---:|---:|---:|
| 30s | 11.76 | 11.76 | 11.76 | 11.76 | 11.76 |
| 60s | 8.93 | 8.93 | 8.93 | 8.93 | 8.93 |
| 90s | 8.18 | 8.18 | 7.81 | 7.81 | 7.81 |
| 120s | 6.98 | 6.98 | 7.54 | 7.54 | 7.26 |
| 180s | 8.04 | 7.89 | 8.04 | 7.89 | 8.04 |
| 240s | 8.83 | 8.72 | 8.95 | 8.72 | 8.72 |
| 300s | 9.56 | 9.36 | 9.36 | 9.36 | 9.36 |
| 360s | — | — | 9.87 | 9.87 | 9.63 |
| 420s | — | — | 10.42 | 10.35 | 10.48 |
| 480s | — | — | 10.98 | 10.92 | 10.87 |
| 540s | — | — | 12.54 | 12.35 | 12.20 |
| 600s | — | — | 13.17 | 13.08 | 13.08 |

各后端的 CER 基本一致（Omni graph 与 vLLM 逐窗差异 ≤0.34 个百分点，属于 temperature=0 下不同 kernel 带来的数值差异）；窗口越长 CER 越高（600s 约 13%）。

### 4.2 单窗延迟（秒）

HF 两列都是单卡（`device_map=auto`，`flash_attention_2`），只跑到 300s。

| 窗口 | HF tf4.57.1（1 卡） | HF tf5.17.0（1 卡） | vLLM 0.30（1 卡） | vLLM 0.30（4 卡） | **Omni eager（4 卡）** | **Omni graph（4 卡）** |
|---|---:|---:|---:|---:|---:|---:|
| 30s | 10.48 | 5.86 | 2.51 | 2.59 | 10.54 | 4.86 |
| 60s | 17.61 | 8.70 | 1.72 | 1.74 | 9.51 | 1.95 |
| 90s | 24.65 | 11.96 | 1.07 | 0.83 | 11.85 | 1.17 |
| 120s | 32.09 | 15.35 | 1.38 | 2.61 | 16.71 | 3.01 |
| 180s | 55.24 | 26.13 | 2.29 | 3.27 | 28.09 | 3.93 |
| 240s | 69.99 | 34.32 | 2.95 | 3.36 | 34.85 | 4.25 |
| 300s | 88.68 | 40.33 | 3.61 | 2.74 | 40.88 | 3.94 |
| 360s | — | — | — | 4.80 | 51.92 | 6.27 |
| 420s | — | — | — | 3.89 | 59.16 | 5.48 |
| 480s | — | — | — | 4.54 | 67.53 | 6.40 |
| 540s | — | — | — | 5.40 | 79.88 | 7.66 |
| 600s | — | — | — | 5.96 | 89.12 | 8.44 |
| **30～300s 合计（7 窗）** | **298.7** | **142.6** | **15.5** | **17.1** | **152.4** | **23.1** |
| **12 窗合计** | — | — | — | **41.7** | **500.0** | **57.4** |
| 整体吞吐（30～300s，含固定开销） | 7.9 token/s | 16.4 token/s | 152 token/s | 138 token/s | 15.4 token/s | 102 token/s |

- Omni eager（4 卡）30～300s 合计 152s，与单卡 HF tf5.17（143s）基本持平；关掉 eager 后降到 23s，12 窗合计从 500s 降到 57s（约 8.7 倍）。
- Omni graph 仍比 stock vLLM 慢约 1.4 倍（57s 对 42s）：解码 3.9 ms/token 对 2.8 ms/token，固定开销 2.1s 对 1.6s；30s 窗的 4.86s 是首条请求的预热。
- HF tf4.57.1 比 tf5.17 慢一倍（299s 对 143s），CER 基本相同（见 §4.1）。
- stock vLLM 单卡和 4 卡耗时相近：batch=1 时瓶颈不在算力，多卡对单条延迟帮助不大。

### 4.3 为什么 Omni 慢

**延迟几乎全花在逐 token 解码上，而 Omni 的解码速度只有 stock vLLM 的约 1/15。** 对 12 个窗口做“延迟 = 固定开销 + 每 token 耗时 × 生成 token 数”的线性拟合：

| 后端 | 固定开销（音频编码 + prefill 等） | 解码速度 |
|---|---:|---:|
| HF tf5.17（1 卡） | 1.1s | 17.3 token/s |
| vLLM 0.30（4 卡） | 1.6s | 约 250～360 token/s |
| Omni 0.28 eager（4 卡） | 2.9s | 17.6 token/s |
| Omni 0.28 graph（4 卡） | 2.1s | 254 token/s |

音频编码和 prefill 三者相差不大；差距在解码。Omni 的解码速度和 HF 一模一样，说明它没用上 vLLM 的解码加速。

**原因是只起 thinker 的部署配置里 `enforce_eager: true`。** 从引擎启动日志看两边配置：

| | stock vLLM 0.30 | Omni 0.28 只起 thinker |
|---|---|---|
| `enforce_eager` | False | **True** |
| CUDA Graph | `FULL_AND_PIECEWISE` | **`NONE`** |
| torch.compile | `VLLM_COMPILE` | **`NONE`** |
| async scheduling | 开 | 关 |

Qwen3-Omni 是 MoE 模型（30B 总参、3B 激活），batch=1 解码时每个 token 要发起大量小 kernel，外加 TP=4 的跨卡通信。不用 CUDA Graph 时，CPU 逐个发 kernel 的开销成为瓶颈，GPU 大部分时间在等，于是速度退化到与 HF 同一水平。

这个配置不是有意选的：vLLM-Omni 自带的 thinker-only 模板 `vllm_omni/deploy/qwen3_omni_moe_thinking.yaml` 就写了 `enforce_eager: true`，AISHELL 脚本和 VoxMatrix 的 thinker-only 模式都照抄了它。

**影响范围：**

- 受影响：所有只起 thinker 的 Omni 用法，包括 AISHELL 转写、默认的 `qwen3-omni-local`（纯文本输出）、第二轮切到 Omni 的 Thinking judge。
- 不受影响：三段语音部署（TTS / 语音回答）。第二轮推理日志显示三个阶段都是 `enforce_eager=False`、CUDA Graph `FULL_AND_PIECEWISE`，所以 §5 的语音延迟和 RTF 是正常水平。

**验证（已确认）：** 对照任务 25331（AISHELL-4 同 12 窗，4 卡，`enforce_eager: false`）解码速度从 17.6 升到 254 token/s，12 窗合计从 500s 降到 57.4s，CER 与 eager 和 stock vLLM 一致（§4.1）。日志确认 CUDA Graph 为 `FULL_AND_PIECEWISE`。

- 代价是启动更慢：Omni 就绪用了 714s（含 torch.compile 约 78s、warmup 和 CUDA Graph 捕获）。首次提交的 25330 就是因为 AISHELL 脚本写死 600s 启动超时而失败；该超时已改为可配置、默认 2400s。VoxMatrix 的 Omni 封装默认就是 2400s。
- thinker-only 的默认值已改为 `enforce_eager: false`（VoxMatrix `write_thinker_deploy` 和环境变量 `QWEN3_OMNI_THINKER_ENFORCE_EAGER` 默认 `false`；AISHELL 脚本 `--enforce-eager` / `ENFORCE_EAGER` 默认 `false`）。需要回到 eager 时显式设为 `true`。

---

## 5. 实验二：TTS + VocalBench IF 两阶段评测

### 5.1 两轮运行

两轮样本相同（LibriTTS 前 2 条、VocalBench IF 前 2 条），区别在卡数和 judge 后端。

| | 第一轮 | 第二轮 |
|---|---|---|
| 推理任务 | 24136 `vox-omni-speech-infer`，4 卡 | 24163 `vox-speech-a2a-omni`，2 卡 |
| 推理时间 | 09-26 01:25–01:46（21 分钟，1.37 GPU·h） | 09-26 12:40–13:01（21 分钟，0.69 GPU·h） |
| 评测任务 | 24139 `vox-omni-speech-judge`，4 卡 | 24166 `vox-speech-a2a-omni`，2 卡 |
| 评测时间 | 09-26 01:46–01:59（13 分钟，0.87 GPU·h） | 09-26 13:01–13:14（13 分钟，0.44 GPU·h） |
| Judge 后端 | stock vLLM 0.30，TP=4，CUDA Graph 开 | vLLM-Omni 0.28（只起 thinker），TP=2，`enforce_eager` |

两轮的被测模型都是 Omni 三段（2 卡部署）。部署 YAML 把三段固定在 cuda:0 / cuda:1（日志：Stage-0 → 0，Stage-1、Stage-2 → 1），所以第一轮虽然申请了 4 卡，cuda:2 / cuda:3 在推理阶段完全闲置，两轮推理实际用的是同样的 2 张卡，延迟和 RTF 相同是预期结果。推理提交脚本的默认卡数已从 4 改为 2。推理时间主要花在三段启动上（约 18 分钟），4 条样本本身的推理合计约 22s；评测时间主要花在 judge 启动上（约 10 分钟）。

### 5.2 结果

| 槽 | 指标 | 第一轮 | 第二轮 |
|---|---|---:|---:|
| `speech_generation__libritts`（n=2） | WER% ↓（corpus） | 9.30（4/43 词错） | 0.00（0/42 词错） |
| | content_acc ↑ | 0.907 | 1.000 |
| | 延迟 p50 / RTF | 2.7s / 0.62 | 2.7s / 0.61 |
| `instruction_following__vocalbench`（n=2） | constraint_satisfaction ↑ | 1.00 | 1.00 |
| | rubric_score ↑ | 1.00 | 1.00 |
| | 延迟 p50 / RTF | 8.6s / 0.20 | 8.5s / 0.21 |

两轮 LibriTTS 的 WER 不同，是因为 thinker 用 `temperature=0.6` 采样，每次生成的语音不同；2 条样本下几个词的差别就会让 WER 变化好几个百分点。延迟和 RTF 两轮基本一致。

### 5.3 第二轮逐条

第一轮的逐条 jsonl 已被第二轮覆盖，只保留了汇总指标。

| 样本 | 推理延迟 | 结果 |
|---|---:|---|
| LibriTTS #0（长句，“He hoped there would be stew…”） | 3.18s | WER 0% |
| LibriTTS #1（“It would be a gloomy secret night.”） | 2.13s | WER 0% |
| VocalBench `instruction-200`（给 5 条衣柜抽屉建议） | 11.85s | 通过：恰好 5 条 |
| VocalBench `instruction-800`（给诗起标题并说明理由） | 5.09s | 通过 |

**逐条与汇总的 WER 归一化已统一。** 原先两者口径不同：

- 逐条的 `wer%` 由评测器 `SeedTTSEvalASRWER` 当场用 jiwer 算：只去标点、转小写。
- 汇总由 `mesh` 聚合器用逐条保存的 ASR 转写（`transcription`）和参考文本（`label_text`）重新算 corpus WER，走 `audio_evals.lib.wer.compute_wer`（Whisper `EnglishTextNormalizer`：连字符拆成空格、英式拼写转美式等）。

LibriTTS #0 的 ASR 转写和参考只差 `flour-fattened` / `flour fattened`、`counselled` / `counseled` 两处书写形式，旧逐条口径记 3/35 = 8.57%，汇总口径记 0%。

现在的做法：

- `SeedTTSEvalASRWER` 的逐条 `wer%` / `cer%`、`content_acc`、`intelligibility` 改为用 `compute_wer` 计算，并在结果里写入 `wer_language`（评测器的语言代码）。
- 聚合器重算 corpus WER 时优先用 `wer_language`，保证逐条和汇总用的是同一语言、同一套归一化和分词。`WER` / `CER` 评测器同样写入 `wer_language`。
- 对齐过程中发现 `compute_wer` 对日 / 韩 / 泰语有问题：`EnglishTextNormalizer` 会删掉组合字符（泰语元音符号、日语浊音符号），且按空格分词会把整句当成 1 个词。现改为：这三种语言跳过英文归一化，只做 NFKC、转小写、去标点，然后按字符算 CER。中文、粤语、英文的计算方式不变（对历史结果的影响见 §8）。
- 第二轮 LibriTTS 逐条结果已按新口径重算（原文件备份在 `suites/native/output/tts/_backup_pre_wer_unify_0930/`）：两条均为 0.00，与汇总一致，汇总文件数值不变。

### 5.4 与 HF 语音输出的对比

**速度：** 对照是 HF 语音输出的 LibriTTS smoke（`VoxMatrix/smoke_tts_all/results_fix/`，n=10，1 卡）。它的第 1 条与本次 LibriTTS #0 是同一句。

| | HF（1 卡） | Omni 三段（2 卡，第二轮） |
|---|---:|---:|
| 同一句（“He hoped there would be stew…”）延迟 / 生成音频时长 | 33.4s / 10.7s | 3.2s / 14.1s |
| RTF | 2.6～3.4（n=10，p50 2.88） | 0.25（长句）、0.98（1.9s 的短句） |

- 长句上 Omni 快约 10 倍，RTF 从约 3（比实时慢）降到 0.25。
- 短句提升小一些：Omni 每条约有 2s 固定开销，1.9s 音频要 2.1s。
- 代价是启动：三段启动约 18 分钟，样本少时总耗时反而由启动主导；样本量上去后才划算。

**质量：**

HF 的 LibriTTS smoke（n=10）为 WER 3.96%、content_acc 0.960。Omni 两轮分别是 9.30% 和 0.00%，但只有 2 条样本，**无法判断 Omni 与 HF 孰优孰劣**。需要在同一批 n≥10 的样本上对比。

### 5.5 产物

- 当前结果（第二轮）：`suites/native/output/tts/`，包括 `session.json`、`suite-overall.json`、各 benchmark 的 jsonl / overall / xlsx。
- wav：`suites/native/output/tts/speech_wavs/`，共 8 个；01:45 生成的 4 个属于第一轮，13:00 生成的 4 个属于第二轮。
- 任务日志：`outputs/native_speech_vllm_omni_{infer,judge}/jobs/<任务号>-*/`。

---

## 6. 过程中遇到的问题与处理

| 问题 | 现象 | 处理 |
|---|---|---|
| Omni 0.28 与 Python 3.10 不兼容 | `StrEnum` 导入失败；flashinfer `array.array[int]` 类型注解报错 | 加 `sitecustomize` 补丁；flashinfer `fd_exchange` 加 `from __future__ import annotations` |
| 设备 / TP 参数不生效 | 命令行传入的 TP、devices 被忽略 | 改为通过 `deploy_config` YAML 指定每段的设备 |
| 启动超时 | 三段启动超过默认 600s | `init_timeout` 调到 2400s |
| 引擎逐请求重建 | 每条样本都重新初始化 | `omni.generate(..., py_generator=False)`，引擎常驻 |
| **Judge OOM** | 第 2 条样本启动 Thinking judge 时只剩约 5 GiB 显存 | 根因：`AgentRubricEvaluator` 每条都调用 `registry.get_model()`，新起一个 judge 子进程。改为 `get_judge_model()` 单例，并缓存到 `self.judge` |
| 回放缺音频 | 首次 smoke 的 wav 写在计算节点的 `/tmp`，任务结束后已删除，无法单独重评 | `speech_output_dir` 改为 AFS 路径 |

---

## 7. 代码改动

| 文件 | 改动 |
|---|---|
| `audio_evals/session.py` | 新增 `--phase all\|inference\|evaluation`（也可用环境变量 `VOXMATRIX_SESSION_PHASE`）。`evaluation` 阶段自动从 benchmark 的 jsonl 续跑，用 `ReplayOnlyModel` 代替被测模型；`inference` 阶段跳过评测和汇总，只写 overall 占位 |
| `mesh_eval/evaluator/agent.py` | 本地 `qwen3-omni*` judge 改为进程内单例 |
| `audio_evals/evaluator/seed_tts_eval_asr_wer.py`、`audio_evals/evaluator/wer.py`、`mesh_eval/agg/mesh.py` | 逐条 WER / CER 改用 `compute_wer`，结果写入 `wer_language`；聚合器重算 corpus WER 时优先用该字段，逐条与汇总口径一致 |
| `audio_evals/lib/wer.py` | 日 / 韩 / 泰语跳过英文归一化（避免删掉组合字符），按字符分词算 CER |
| `audio_evals/models/qwen3_omni_vllm_omni.py`、`audio_evals/lib/qwen3-omni-vllm-omni/` | Omni 模型封装和子进程 worker（新增），支持三段语音输出和只起 thinker 两种模式。thinker-only 的 `enforce_eager` 可用环境变量 `QWEN3_OMNI_THINKER_ENFORCE_EAGER` 切换，默认已改为 `false`（CUDA Graph） |
| `qwen3_omni_long_audio/scripts/aishell4_duration_sweep_vllm_omni.py`、`jobs/*aishell4_vllm_omni_sweep.sh` | 新增 `ENFORCE_EAGER` / `ASYNC_SCHEDULING` 开关；关闭 eager 时结果目录加 `_graph` 后缀 |
| `suites/native/registry/model/qwen3_omni_local.yaml` | 语音模型写 wav 到 AFS（`speech_output_dir`）。第二轮前，默认的 `qwen3-omni-local` / `qwen3-omni-speech-local` 已切到 vLLM-Omni，原 HF 语音实现保留为 `qwen3-omni-speech-hf-local` |
| `registry/model/qwen3-omni.yaml` | `qwen3-omni-thinking` 切到 vLLM-Omni（只起 thinker）；原 stock vLLM 版本保留为 `qwen3-omni-thinking-plain-vllm`，HF 版本为 `qwen3-omni-thinking-hf` |
| `scripts/run_native_speech_vllm_omni_{infer,judge}.sh` | 推理 / 评测各自的运行脚本；评测脚本启动前检查回放用的 wav 是否都存在 |
| `scripts/submit_native_speech_vllm_omni_{infer,judge,split}.sh` | 提交脚本；`split` 等推理任务成功后自动提交评测任务 |

用法：

```bash
# 分两步提交
scripts/submit_native_speech_vllm_omni_infer.sh
# 推理任务 SUCCEEDED 后再提交
scripts/submit_native_speech_vllm_omni_judge.sh

# 调整样本数 / 槽位 / 卡数
GPUS=2 NATIVE_LIMIT=10 \
NATIVE_SLOT_FILTER=speech_generation__libritts,instruction_following__vocalbench \
  scripts/submit_native_speech_vllm_omni_infer.sh
```

---

## 8. 局限与后续

1. **样本量太小**：TTS 和 IF 各 2 条，只能说明链路通了。下一步按 n=10 重跑，并与 HF 语音输出放在同一批样本上对比。
2. **WER 口径变更的影响**：逐条与汇总已统一。`compute_wer` 对中文、粤语、英文及其他空格分词语言的算法没变；汇总值只在两种情况下会变，这两种情况下的结果不能直接和 09-30 之前的比：
   - 数据行的 `language` 字段和评测器语言不一致（字段缺失或写全称，如 `chinese`）。之前按该字段或默认值算，现在按评测器语言算。
   - 日、韩、泰语（如 `minimax.yaml`、`cv3.yaml` 的对应评测器）。现在按字符算 CER，且不再删掉组合字符。

   本次 LibriTTS 不属于以上情况，汇总值不变。未被引用的 `NaiveWER`（`registry/evaluator/naive_wer.yaml`）仍是 jiwer 口径，后续如启用需要同样对齐。
3. **Omni 只起 thinker 仍比 stock vLLM 慢约 1.4 倍**：关掉 eager 后已从 12 倍降到 1.4 倍，默认值已改。剩余差距来自解码（3.9 对 2.8 ms/token）和固定开销（2.1s 对 1.6s），可以再试 `async_scheduling`。启动也更慢（约 12 分钟）。第二轮 Thinking judge 是在 eager 下跑的，后续 judge 会自动用上 CUDA Graph，n 变大后评测耗时会明显下降。
4. **语音回答的 thinker 输出上限 1200 tokens**：长回答可能被截断，需要确认 VocalBench 中长回答的样本是否受影响。
5. **`vocalbench_zh` 未纳入本次运行**：`tts/` 下的 `native-instruction-following--vocalbench-zh*` 是 09-24 的旧结果（failure_rate=1.0），与本次无关。
6. **结果会被覆盖、wav 会累积**：benchmark 的 jsonl 按固定文件名写，重跑会覆盖上一轮（第一轮逐条结果已丢）；`speech_wavs/` 按请求 UUID 命名，不会覆盖、会一直累积。建议按 run 分目录输出。

---

## 9. 单引擎多并发（2026-10-08）

### 9.1 改动

文本被测模型（thinker-only）的每个副本现在可以同时处理多条请求，用上 vLLM 的合批。提交时设 `NATIVE_CONCURRENCY=N` 开启，默认 1 时行为不变。评委和三段式语音模型不受影响，仍是一次一条。

- 后台进程 `audio_evals/lib/qwen3-omni-vllm-omni/main.py`：`max_num_seqs > 1` 时改用 `AsyncOmni`，每条请求一个协程；回复按完成顺序带前缀返回，报错也带前缀，不再等 close 确认。部署配置的 `max_num_seqs` 不再写死为 1。
- 封装 `audio_evals/models/qwen3_omni.py`：并发模式下由一个分发线程按前缀把回复送回各自的请求，写管道加锁。`OfflineModel.inference` 外面原有一把实例锁，同一模型同一时间只放行一个请求；并发模式下跳过这把锁。
- 模型池 `audio_evals/models/model_pool.py`：按副本的 `max_inflight` 放置槽位；`run_context` 记录 `concurrent_slots`。
- 预检 `suites/native/preflight.py`：`NATIVE_CONCURRENCY` 生成 `qwen3-omni-local--cN` 模型登记和对应的线程数；全量模式新增 `NATIVE_FULL_LIMIT`，只用于截取前 N 条做速度对照。

第一轮对照（Job 28149、28155）几乎没有提速，输出也和串行完全一致。原因就是上面那把实例锁：8 个请求实际是一条一条发出的。去掉后（Job 28171、28172）才是真正的并发结果。

### 9.2 对照结果（1 卡、1 副本，串行对并发 8）

| 数据 | 串行用时 | 并发 8 用时 | 提速 | 指标（串行 → 并发） | 逐条输出相同 |
|---|---:|---:|---:|---|---:|
| AISHELL-1 全量前 500 条（短音频） | 90s | 44s | 2.05 倍 | WER 0.897% → 0.870% | 494 / 500 |
| AISHELL-4 长音频转写前 64 窗 | 189s | 96s | 1.97 倍 | WER 33.50% → 33.41% | 24 / 64 |

- 用时为评测会话从开始到结束，不含约 12 分钟的模型加载。
- 并发 8 下 KV 缓存仍为 7.27 GiB（约 7.9 万 token），与串行相同，未出现显存问题。
- 输出不同的样本来自合批带来的细微数值差异，temperature 0 下也会出现。短音频大多是个别同音字（如"盛世 / 盛事"）；长输出因为差异会逐步累积，不同的比例更高，多为标点和个别字词。两组的 WER 差异都在 0.1 个百分点以内，方向不固定。
- 提速只有 2 倍左右，没有接近 8 倍。第一阶段输入处理（音频特征提取、分词）在 vLLM-Omni 里是单线程同步完成的，是剩下的瓶颈。

### 9.3 结论

- 并发 8 能让单卡吞吐翻倍，指标基本不变，但逐条输出不再和串行完全相同。跨批次对比逐条输出时，要用同一并发设置下的结果。
- 并发模式下每条结果里的 `latency_ms` / `rtf` 包含在引擎里排队的时间，不能当单条延迟用；延迟类比较只用串行结果。
