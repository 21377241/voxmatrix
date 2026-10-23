# Native 全量评测：多模型结果对比

- **评测套件**：VoxMatrix-clean `suites/native` 全量（与 Qwen3-Omni 同协议）
- **对比模型**：Qwen3-Omni、Step-Audio-2-mini、Kimi-Audio-7B-Instruct、MiniCPM-o 4.5
- **结果目录**：`outputs/native_full/`、`native_full_step_audio_2_mini/`、`native_full_kimi_audio_7b/`、`native_full_minicpmo_4_5/`
- **汇总日期**：2026-10-21
- **箭头**：↑ 越高越好；↓ 越低越好；加粗为该行最优（已出结果中）；`—` 表示尚未跑出
- **官方文献**：Qwen3-Omni [2509.17765](https://arxiv.org/abs/2509.17765)；Step-Audio 2 [2507.16632](https://arxiv.org/abs/2507.16632)；Kimi-Audio [2504.18425](https://arxiv.org/abs/2504.18425)；MiniCPM-o 4.5 [2604.27393](https://arxiv.org/abs/2604.27393)。逐能力详表见 [`native_full_official_report_alignment.md`](native_full_official_report_alignment.md)。

---

## 1. 完成度

| 模型 | 完成度 | 说明 |
|------|--------|------|
| Qwen3-Omni | 9/9 | SV 在 `sv_c8_*` |
| Step-Audio-2-mini | 9/9 | 齐全 |
| Kimi-Audio-7B | 9/9 | 齐全（早期失败组已重跑） |
| MiniCPM-o 4.5 | **7/9** | 缺 asr、speaker_verification（33898/33899 排队，等全局 FIFO 33870） |

---

## 2. 表 A：能力级（capability）聚合

同一能力下多个 benchmark 再汇总。口语理解（spoken_slu）在 suite 里挂在 IF/数据集路径下，**无独立 capability 行**，见基准表 FSC/SLURP。

| 能力 | 指标 | 含义 | Qwen3-Omni | Step-Audio-2-mini | Kimi-Audio-7B | MiniCPM-o-4.5 |
|------|------|------|------------|-------------------|---------------|---------------|
| `acoustic_scene` | `accuracy`↑ | 声学场景分类正确率 | 0.363 | **0.372** | 0.278 | 0.336 |
| `sound_event` | `accuracy`↑ | 声音事件识别正确率 | **0.715** | 0.639 | 0.408 | 0.512 |
| `speaker_count` | `speaker_count_acc`↑ | 说话人数量估计正确率 | 0.592 | 0.425 | 0.200 | **0.625** |
| `paralinguistic_recognition` | `accuracy`↑ | 副语言属性分类正确率 | **0.485** | 0.397 | 0.366 | 0.485 |
| `code_switch` | `match`↑ | 语码转换匹配率 | **0.812** | 0.533 | 0.552 | 0.663 |
| `speech_translation` | `bleu`↑ | 语音翻译 corpus BLEU | 21.14 | 21.51 | 1.522 | **24.51** |
| `target_speaker` | `target_speaker_acc`↑ | 目标说话人任务正确率 | 0.515 | **0.517** | 0.497 | 0.514 |
| `audio_caption` | `cider`↑ | 音频描述 CIDEr | 0.374 | **0.594** | 0.435 | 0.570 |
| `qa` | `match`↑ | 问答与参考匹配率 | **0.724** | 0.638 | 0.641 | 0.680 |
| `diarization` | `der`↓ | 说话人日志错误率 DER | **0.582** | 0.894 | 0.721 | 0.615 |
| `long_form_asr` | `cer%`↓ | 长音频转写字错误率 % | **30.46** | 68.25 | 60.14 | 36.66 |
| `long_form_asr` | `wer%`↓ | 长音频转写词错误率 % | 47.06 | 72.49 | 89.68 | **46.61** |
| `speaker_attribution` | `attribution_acc`↑ | 说话人归因正确率 | **0.464** | 0.385 | 0.423 | 0.435 |
| `multi_turn_dialogue` | `multi_round_acc`↑ | 多轮对话轮次通过率 | **0.871** | 0.647 | 0.626 | 0.740 |
| `multi_turn_dialogue` | `mtbench_score`↑ | 多轮 MT-Bench 分数(约1–10) | **8.950** | 6.000 | 6.089 | 8.477 |
| `multi_turn_dialogue` | `rubric_score`↑ | 语音多轮 mtalk rubric | **0.744** | 0.518 | 0.532 | 0.528 |
| `instruction_following` | `constraint_satisfaction`↑ | 指令约束满足率 | **0.788** | 0.256 | 0.333 | 0.490 |
| `instruction_following` | `strict_prompt`↑ | 严格 prompt 通过率 | **0.788** | 0.406 | 0.386 | 0.710 |
| `speech_generation` | `content_acc`↑ | 生成语音内容正确率 | 0.965 | 0.725 | 0.535 | **0.966** |
| `speech_generation` | `wer%`↓ | 生成语音内容 WER % | 3.475 | 27.47 | 46.53 | **3.387** |
| `asr` | `wer%`↓ | 多 ASR 子集汇总 WER % | **10.23** | 183.9 | 30.16 | — |
| `asr` | `cer%`↓ | 多 ASR 子集汇总 CER % | 18.93 | **17.60** | 18.25 | — |
| `speaker_verification` | `auc`↑ | 声纹验证 ROC-AUC | **0.815** | 0.503 | 0.689 | — |
| `speaker_verification` | `eer`↓ | 声纹验证等错误率 | **0.254** | 0.497 | 0.312 | — |
| `speaker_verification` | `speaker_verification_acc`↑ | 声纹验证准确率 | **0.744** | 0.503 | 0.688 | — |

## 3. 表 B：Benchmark 级（单数据集）

每个 benchmark 取该集主指标（表中「指标」列）。`—` = 该模型尚未跑出。

| Benchmark | 含义 | 指标 | Qwen3-Omni | Step-Audio-2-mini | Kimi-Audio-7B | MiniCPM-o-4.5 |
|-----------|------|------|------------|-------------------|---------------|---------------|
| `native-acoustic-scene-dcase2013-scenes` | 场景 DCASE2013 | `accuracy`↑ | **0.650** | 0.470 | 0.530 | 0.540 |
| `native-acoustic-scene-mmsu` | 场景 MMSU | `accuracy`↑ | **0.636** | 0.309 | 0.464 | 0.382 |
| `native-acoustic-scene-tut-sound-events-2017` | 场景 TUT2017 | `accuracy`↑ | **1.000** | 0.938 | 0.844 | 0.969 |
| `native-acoustic-scene-tut-urban-acoustic-scenes-2018-mobile` | 场景 TUT Urban2018 | `accuracy`↑ | 0.335 | **0.365** | 0.256 | 0.320 |
| `native-sound-event-esc50` | 事件 ESC-50 acc | `accuracy`↑ | **0.820** | 0.713 | 0.482 | 0.637 |
| `native-sound-event-mmsu` | 事件 MMSU acc | `accuracy`↑ | **0.915** | 0.846 | 0.726 | 0.735 |
| `native-sound-event-tut-sound-events-2017` | 事件 TUT2017 acc | `accuracy`↑ | **0.449** | 0.421 | 0.138 | 0.202 |
| `native-sound-event-mavd-traffic` | 事件 MAVD-traffic F1 | `f1`↑ | **0.453** | 0.087 | 0.375 | 0.378 |
| `native-sound-event-sonyc-ust-v23` | 事件 SONYC-UST F1 | `f1`↑ | **0.352** | 0.099 | 0.233 | 0.236 |
| `native-speaker-count-mmsu` | 说话人数 MMSU | `speaker_count_acc`↑ | 0.592 | 0.425 | 0.200 | **0.625** |
| `native-paralinguistic-recognition-mmsu` | 副语言 MMSU | `accuracy`↑ | **0.485** | 0.397 | 0.366 | 0.485 |
| `native-code-switch-vocalbench-zh` | 语码转换 VocalBench-zh match | `match`↑ | **0.812** | 0.533 | 0.552 | 0.663 |
| `native-code-switch-mmsu` | 语码转换 MMSU ROUGE-L | `rouge_l`↑ | **0.375** | 0.280 | 0.313 | 0.318 |
| `native-speech-translation-covost2` | 翻译 CoVoST2 中→英 BLEU | `bleu`↑ | 21.14 | 21.51 | 1.522 | **24.51** |
| `native-speech-translation-mmsu` | 翻译 MMSU match | `match`↑ | **0.176** | 0.101 | 0.101 | 0.092 |
| `native-target-speaker-aishell-5` | 目标说话人 AISHELL-5 macro-F1 | `macro_f1`↑ | **0.509** | 0.475 | 0.346 | 0.413 |
| `native-audio-caption-audiocaps` | 描述 AudioCaps CIDEr | `cider`↑ | 0.395 | 0.610 | 0.440 | **0.611** |
| `native-audio-caption-clotho` | 描述 Clotho CIDEr | `cider`↑ | 0.204 | **0.370** | 0.311 | 0.278 |
| `t4-fsc-instruction-following` | 口语理解 FSC intent | `intent_acc`↑ | **0.938** | 0.695 | 0.708 | 0.937 |
| `t4-slurp-instruction-following` | 口语理解 SLURP intent | `intent_acc`↑ | **0.698** | 0.207 | 0.027 | 0.446 |
| `native-qa-ear-wdyl` | 问答 EAR-WDYL match | `match`↑ | **0.200** | 0.055 | 0.055 | 0.098 |
| `native-qa-spoken-squad` | 问答 Spoken-SQuAD match | `match`↑ | **0.858** | 0.818 | 0.839 | 0.845 |
| `native-qa-uro-bench` | 问答 URO-Bench match | `match`↑ | **0.436** | 0.378 | 0.381 | 0.428 |
| `native-qa-vocalbench` | 问答 VocalBench match | `match`↑ | **0.938** | 0.762 | 0.788 | 0.876 |
| `native-qa-vocalbench-zh` | 问答 VocalBench-zh match | `match`↑ | **0.893** | 0.663 | 0.580 | 0.630 |
| `native-qa-voicebench` | 问答 VoiceBench match | `match`↑ | 0.976 | 0.982 | 0.864 | **0.998** |
| `native-diarization-aishell-4` | 日志 AISHELL-4 DER | `der`↓ | **0.427** | 0.750 | 0.603 | 0.446 |
| `native-diarization-alimeeting` | 日志 AliMeeting DER | `der`↓ | **0.614** | 0.951 | 0.746 | 0.638 |
| `native-diarization-ami` | 日志 AMI DER | `der`↓ | **0.619** | 0.925 | 0.751 | 0.676 |
| `native-diarization-chime-6` | 日志 CHiME-6 DER | `der`↓ | 0.894 | 0.994 | 0.929 | **0.883** |
| `native-aishell4-long-form-asr` | 长音频 AISHELL-4 CER% | `cer%`↓ | **17.58** | 56.61 | 48.45 | 27.53 |
| `native-alimeeting-long-form-asr` | 长音频 AliMeeting CER% | `cer%`↓ | **37.41** | 74.52 | 66.45 | 41.58 |
| `native-ami-long-form-asr` | 长音频 AMI WER% | `wer%`↓ | **37.21** | 68.56 | 84.95 | 43.33 |
| `native-chime6-long-form-asr` | 长音频 CHiME-6 WER% | `wer%`↓ | 112.2 | 98.53 | 121.0 | **68.34** |
| `native-speaker-attribution-aishell-4` | 归因 AISHELL-4 acc | `attribution_acc`↑ | **0.557** | 0.441 | 0.467 | 0.480 |
| `native-speaker-attribution-alimeeting` | 归因 AliMeeting acc | `attribution_acc`↑ | **0.439** | 0.368 | 0.401 | 0.432 |
| `native-speaker-attribution-ami` | 归因 AMI acc | `attribution_acc`↑ | **0.441** | 0.375 | 0.423 | 0.425 |
| `native-speaker-attribution-chime-6` | 归因 CHiME-6 acc | `attribution_acc`↑ | **0.351** | 0.295 | 0.345 | 0.321 |
| `audio-agent-bench-dialogue` | 多轮 AudioAgentBench turn_pass | `turn_pass`↑ | **0.872** | 0.233 | 0.479 | 0.485 |
| `t4-vocalbench-multi-turn` | 多轮 VocalBench acc | `multi_round_acc`↑ | **0.901** | 0.640 | 0.682 | 0.767 |
| `t4-vocalbench-zh-multi-turn` | 多轮 VocalBench-zh acc | `multi_round_acc`↑ | **0.845** | 0.655 | 0.570 | 0.713 |
| `t4-voicebench-mtbench` | 多轮 VoiceBench MT-Bench | `mtbench_score`↑ | **8.950** | 6.000 | 6.089 | 8.477 |
| `t4-mtalk-bench` | 多轮 mtalk rubric | `rubric_score`↑ | **0.744** | 0.518 | 0.532 | 0.528 |
| `t4-voicebench-ifeval` | IF VoiceBench constraint | `constraint_satisfaction`↑ | **0.788** | 0.406 | 0.386 | 0.710 |
| `t4-vocalbench-instruction-following` | IF VocalBench constraint | `constraint_satisfaction`↑ | — | 0.403 | 0.461 | **0.602** |
| `t4-vocalbench-zh-instruction-following` | IF VocalBench-zh constraint | `constraint_satisfaction`↑ | — | 0.123 | 0.218 | **0.390** |
| `native-speech-generation-libritts` | TTS LibriTTS content_acc | `content_acc`↑ | 0.965 | 0.725 | 0.535 | **0.966** |
| `native-asr-librispeech` | ASR LibriSpeech WER% | `wer%`↓ | **1.246** | 1.654 | 3.734 | — |
| `native-asr-aishell-1` | ASR AISHELL-1 CER% | `cer%`↓ | 0.916 | 1.004 | **0.766** | — |
| `native-asr-aishell-5` | ASR AISHELL-5 CER% | `cer%`↓ | 51.53 | **39.45** | 45.18 | — |
| `native-asr-wenetspeech` | ASR WenetSpeech CER% | `cer%`↓ | **6.064** | 6.995 | 6.579 | — |
| `native-asr-common-voice` | ASR CommonVoice WER% | `wer%`↓ | **8.166** | 12.18 | 17.00 | — |
| `native-asr-magicdata-ramc` | ASR MagicData-RAMC CER% | `cer%`↓ | **8.226** | 13.55 | 11.12 | — |
| `native-asr-sbcsae` | ASR SBCSAE WER% | `wer%`↓ | **24.12** | 825.6 | 90.72 | — |
| `native-speaker-verification-voxceleb1` | SV VoxCeleb1 AUC | `auc`↑ | **0.815** | 0.503 | 0.689 | — |

---

## 4. 数值异常一览

| 异常数值 | 一句话根因 |
|----------|------------|
| Step SV AUC≈0.50 / EER≈0.50 | 几乎恒输出 prompt 示例形态的 JSON，未做声纹判别 |
| Step sbcsae WER%=826 | 短会话参考上大量拒答/扩写，corpus WER 被长 hyp 撑爆 |
| Step ASR overall=184 | 几乎只被 sbcsae 拉偏，其它子集正常 |
| Kimi 翻译 BLEU=1.52 | CoVoST2 上 ~85% 输出中文转写，未按要求出英文译文 |
| Kimi SLURP intent=0.027 | 87.5% 推理结果为空串；同模型 FSC 几乎不空 |
| MiniCPM asr/SV = — | 任务从未提交成功（排队脚本中断），不是 0 分 |

---

## 5. 逐项根因探究

### 5.1 Step SV：AUC=0.503 / EER=0.497

**现象（数据）**

- 37720 条 eval 中，**36876（97.8%）** 的 `pred` 完全相同：`{"same_speaker": false, "score": 0.0}`
- **没有任何一条**预测含 `same_speaker: true`
- 正样本（ref=同说话人）错误率 97.8%；负样本正确率 98.3% → 典型「全判不同人」常数分类器 → AUC/EER≈0.5

**原因**

1. Prompt 要求：`Return JSON only as {"same_speaker": true, "score": 0.0}`——示例里 **score 写成了 0.0**，模型几乎原样复读模板结构，并把 `same_speaker` 固定成 `false`。
2. 推理 `content` 已是上述 JSON，后处理未改写 → 不是解析坏了，是**模型输出本身失效**。
3. 与「声纹弱」不同：弱模型应有分散 score；这里是**模板复读 + 常数决策**。

**结论**：指标无效；需改 prompt 示例（勿把 `score: 0.0` 当格式范例）并重跑，才能谈 Step 声纹能力。

---

### 5.2 Step ASR sbcsae：WER%=826

**现象（数据）**

- 与 LibriSpeech **同一 prompt**：`Transcribe the English audio into text.`
- Libri 正常（WER≈1.65%，hyp 为完整句子）；sbcsae 上：
  - ref 很短（median 长度≈17 字符，如 `oh` / `isn't it`）
  - hyp 均值长度≈151（约为 ref 的数倍到数十倍）
  - 单条 WER 可达数百～上千（短 ref + 长 hyp）
  - 拒答/澄清类约 6.4%（`I'm not sure what you're asking…`）；另有把短音扩写成场景描述（`whistle` → 长句描述）

**原因**

1. **数据域**：SBCSAE 多为极短会话碎片/语气词，不像 Libri 的朗读句。
2. Step 在短、含糊输入上易切换到**闲聊/澄清/描述**模式，而不是严格转写。
3. Corpus WER 对「短参考 + 长假设」极度敏感 → 出现 **826%** 这种物理上「不像正常 ASR」的数字。

**结论**：该子集分数失效；Step 的 ASR 能力应看 Libri / AISHELL / CommonVoice，不能看 sbcsae 或被它污染的 overall。

---

### 5.3 Step ASR overall WER%=184

**原因（派生）**

- overall 为多子集 corpus 汇总；sbcsae 样本量大且单条错误极大 → 把整体从「约数个点～十几」拉到 **184**。
- 证据：同一次 asr run 内 Libri≈1.65%、AISHELL-1 CER≈1.0%、CommonVoice≈12%，与 184 并存。

**结论**：overall 数字异常是 **4.2 的直接后果**，不是另一次故障。

---

### 5.4 Kimi speech_translation BLEU=1.52

**现象（数据，CoVoST2 中→英）**

- Prompt：`Listen to the provided Chinese speech and produce a translation in English text.`
- 参考均为英文；Kimi 预测中 **约 84.8%（2180/2570）为中文**，英文仅约 15%。
- 典型：音频中文内容被 **ASR 成中文句子**，未翻译（如输出「河内盛产黄色…」而 ref 为英文）。
- 对照：Qwen 同任务 hyp 含中文比例≈0.1%，BLEU 正常（约 21）。
- MMSU（日→英）上 Kimi 多数已是英文，说明不是「完全不会出英文」，而是 **中→英这条上任务跟从失败（译成转写）**。

**原因**

- 模型把「听中文 → 出文本」做成了**转写**，忽略了 “translation in English”。
- BLEU 对英参、中hyp 接近 0 → 能力聚合 BLEU 被 CoVoST2 拉到 **1.52**。

**结论**：异常来自**语言方向/任务指令未执行**，不是平稳的「翻译很差」。修 prompt/解码约束或后验语言检查后重跑，才可比。

---

### 5.5 Kimi SLURP intent=0.027（vs FSC=0.708）

**现象（数据）**

| 子集 | Prompt 规模（json 约） | 推理空输出比例 | intent_acc |
|------|------------------------|----------------|------------|
| SLURP | ~6952 字符（超长 intent/slot 清单） | **2601/2974 = 87.5%** | 0.027 |
| FSC | ~1811 字符 | 13/3793 = 0.3% | 0.708 |
| 同协议 Qwen/Step/Mini 的 SLURP | 同长 prompt | **0% 空** | 0.45–0.70 |

- 空输出时 `pred=""`、`classification_pred=""`，直接 intent_acc=0。
- 少数非空样本能出 JSON，但整体被空结果淹没。

**原因**

1. **主因**：Kimi 在超长 SLURP schema prompt 下大量 **空生成**（非 timeout、failure_rate 仍为 0）。
2. FSC 意图集小、prompt 短，同模型几乎不空 → 说明不是「完全不会 SLU」，而是 **长约束 prompt 触发了空输出**。
3. 其它三模型同 prompt 不空 → 更偏 **Kimi 适配/上下文对长 JSON schema 的耐受问题**，不是评测脚本算错。

**结论**：0.027 主要反映「空输出率」，不能当 SLU 能力真值；应看 FSC，或缩短 schema / 修空生成后重跑 SLURP。

---

### 5.6 MiniCPM asr / SV 为 —

**原因（流程，非指标）**

- 9 组里 asr、speaker_verification 排在队列末尾；`speech_generation` 已成功，但后台 `/tmp/submit_minicpm_queue.sh` 中断，**两任务从未入 jobs**。
- 已补交 `33898`（asr）、`33899`（SV），受全局 FIFO（他人任务 33870）阻塞，仍为 QUEUED。

**结论**：表中「—」= 缺失，不是测出了 0。

---

## 6. 与官方技术报告是否匹配（逐能力摘要）

判定：匹配 / 部分匹配 / 不匹配 / 官方未说明 / 实测缺失。官方与 VoxMatrix 的 prompt、子集常不一致，**以能力是否成立为主，不要求数值逐点相等**。完整逐行表见 [`native_full_official_report_alignment.md`](native_full_official_report_alignment.md)。

### 6.1 按能力横断

| 能力 | Qwen3-Omni | Step-2-mini | Kimi-Audio | MiniCPM-o 4.5 |
|------|------------|-------------|------------|---------------|
| asr | **匹配**（Libri 1.25≈官方 1.22；Wenet≈6） | 主集**部分匹配**；sbcsae/overall **不匹配** | 中文**部分匹配**；Libri 3.73≪官方 1.28 **不匹配** | **实测缺失**（官方 AISHELL-1≈0.9、Libri≈1.4） |
| speech_translation | **部分匹配**（CoVoST 21.1，方向对；官方主报 Fleurs） | **部分匹配**（21.5 vs 官方 zh→en 29.5） | **不匹配**（BLEU 1.52，多为中文转写） | **匹配**（24.5≈官方 zh→en 26.4） |
| speech_generation | **部分匹配**（content_acc 0.965；官方无数对齐） | 官方对话强 / 我们 TTS 中等、IF 弱 → **偏不匹配** | **部分匹配**（0.535，偏弱） | **匹配**（0.966；与官方 SeedTTS/LongTTS 优势一致） |
| multi_turn / IF / qa | **匹配**（IF/多轮四模型最强） | URO 官方高 / 我们 IF 0.26 → **偏不匹配** | QA 尚可；IF 0.33 ≪ VoiceBench-IF 叙事 → **部分匹配** | **匹配～部分匹配**（QA/strict 强） |
| acoustic_scene / sound_event | **部分匹配**（有理解宣称；集不同） | **官方未说明**（无 ESC/TUT 主表） | 官方 Cochl/TUT/Nonspeech 很高 / 我们 Urban·ESC 低 → **不匹配** | **官方未说明** |
| paralinguistic | **官方未说明** | **部分匹配**（官方自建集 Avg 80；我们 MMSU 0.40） | **部分匹配**（MELD 等） | **官方未说明** |
| audio_caption | **官方未说明**（Captioner 另模） | **官方未说明**（实测 CIDEr 最强） | **官方未说明** | **官方未说明** |
| diarization / long_form / attribution / SV / SLU / code_switch / target_speaker / speaker_count | 多数字段 **官方未说明** | 同左；SV 实测失效 | 同左；SLURP 失效 | 同左；asr/SV 缺失 |

### 6.2 分模型要点

**Qwen3-Omni**（[2509.17765](https://arxiv.org/abs/2509.17765)，权重 30B-A3B-Instruct）  
- 官方：En/Zh ASR SOTA（Libri 1.22/2.48 等）、VoiceBench/IF 强、长音频理解。  
- 实测：ASR/IF/多轮/QA 与叙事一致；翻译子集不同但方向正确；diarization/SV/SLU 等官方无对标数。

**Step-Audio-2-mini**（[2507.16632](https://arxiv.org/abs/2507.16632) Appendix B）  
- 官方：ASR（Libri 1.33、AISHELL 0.78）、CoVoST Avg 39、副语言 Avg 80、MMAU 73、URO 对话高分。  
- 实测：ASR 主集、CoVoST 同向可对；**sbcsae/SV 失效**；IF/多轮远弱于 URO「全面很强」的印象。

**Kimi-Audio-7B**（[2504.18425](https://arxiv.org/abs/2504.18425)）  
- 官方：ASR SOTA（Libri 1.28、AISHELL 0.60）、场景/事件分类很高、VoiceBench Avg 76.9。  
- 实测：中文 ASR 接近；**Libri、CoVoST、SLURP、场景/事件**与「全面 SOTA」不一致（后两项含适配失效）。

**MiniCPM-o 4.5**（[2604.27393](https://arxiv.org/abs/2604.27393)）  
- 官方：CoVoST zh→en 26.4 / en→zh 49.9；TTS 优于 Qwen3-Omni；VoiceBench/QA 强；ASR 接近领先系统。  
- 实测：翻译与 TTS **高度吻合**；QA 强；**ASR/SV 尚未出数**；会议/场景等官方未写。

### 6.3 读报告时的用法

1. 写「复现了官方」时，优先引用 **ASR 标准集 / CoVoST / TTS** 等有双向数字的行。  
2. diarization、SV、SLURP、code_switch 等：**不能**写成「违背官方」——多数是 **官方未说明**。  
3. Step SV、Kimi CoVoST/SLURP 等：**先修适配再比官方**，否则比的是失效输出。

---

## 7. 数据来源

```text
outputs/native_full_<model>/<group>_<timestamp>/main|tts/suite-overall.json
→ 表 A：by_capability.<capability>.<metric>
→ 表 B：flat.benchmark_id/<benchmark_id>/<metric>
```

抽样依据：各异常对应 `*.jsonl` 中 `type=prompt|inference|eval` 记录（Step SV / sbcsae、Kimi CoVoST2 / SLURP / FSC）。

Qwen SV：`outputs/native_full/sv_c8_*/main/native-speaker-verification--voxceleb1-overall.json`

MiniCPM 补跑：asr=`33898`，SV=`33899`（排队中）。

官方报告核对详表：[`native_full_official_report_alignment.md`](native_full_official_report_alignment.md)。

---

## 8. 一句话

四模型横向比：Qwen 综合最稳且与官方 ASR/对话叙事一致；MiniCPM 在翻译/TTS 上对齐官方亮点但缺 ASR/SV；Step/Kimi 在部分标准集上可对上报告，但 SV/sbcsae/CoVoST/SLURP/IF 等出现失效或明显弱于官方「全面 SOTA」表述——解读时需分开「协议结果」与「官方同口径复现」。
