# Native Smoke 小样本结果表

> **汇报结构 = 按 capability 分节**；**capability 之间不合并**（不做跨能力总分 / 模型总分）。  
> 每一节内按 Catalog 槽（Benchmark）分列；槽内多指标并列。  
> **暂不写能力总分**（2026-09-28 定）：节内只列槽级指标，不写能力合计行；以后如要汇总，候选口径见文末。  
> 通常每槽 **n=10**；仅链路/协议验证，不作正式榜。  
> ASR：`outputs/native_asr_smoke_rebuilt_2gpu/`；其余主轨：`suites/native/output/{main,tts}/`；Judge：`outputs/native_*_smoke/`。  
> 更新：2026-09-24。  
> **样本已换（2026-09-30 Native/Full 对齐）**：以下 8 个槽的小样本内容已变，本表中它们的数字来自旧样本，不能与新样本结果直接比较：acoustic_scene 的 dcase2013 / tut_sound_events_2017 / tut_urban_2018，audio_caption 的 clotho，sound_event 的 esc50 / sonyc_ust_v23，qa 的 vocalbench_zh / voicebench。原因见指标口径报告开头的"Native / Full 对齐"。

---

## ASR

中文槽主指标 **CER%**，英文 **WER%**（产物 overall 键常仍为 `wer(%)`，中文实为字级）。二者不可合成一个 ASR 总分。

| 槽 | 主指标 | 主分 | 备注 |
|---|---|---:|---|
| `asr__librispeech` | WER% ↓ | 0.00 | 英文 |
| `asr__aishell_1` | CER% ↓ | 1.33 | 中文 |
| `asr__wenetspeech` | CER% ↓ | 2.94 | 中文 |
| `asr__magicdata_ramc` | CER% ↓ | 7.69 | 中文 |
| `asr__common_voice` | WER% ↓ | 11.43 | 英文 |
| `asr__sbcsae` | WER% ↓ | 15.79 | 英文 |
| `asr__aishell_5` | CER% ↓ | 21.62 | 中文 |

---

## Long-form ASR

30–60s 窗；中文 CER% / 英文 WER%。

| 槽 | 主指标 | 主分 | 备注 |
|---|---|---:|---|
| `long_form_asr__aishell_4` | CER% ↓ | 6.77 | 中文 |
| `long_form_asr__ami` | WER% ↓ | 6.90 | 英文 |
| `long_form_asr__alimeeting` | CER% ↓ | 8.30 | 中文 |
| `long_form_asr__chime_6` | WER% ↓ | 10.56 | 英文 |
| `long_form_asr__misp` | CER% ↓ | 11.36 | 中文 |
| `long_form_asr__sbcsae` | WER% ↓ | 21.21 | 英文 |

---

## Speech Translation

| 槽 | 主指标 | 主分 | 并列（不与主分合成） |
|---|---|---:|---|
| `speech_translation__covost2` | BLEU ↑ | 30.80 | Judge en→zh 86.7（6/10）；zh→en 51.4（7/10） |
| `speech_translation__mmsu` | exist-match ↑ | 0.10 | Judge 83.3（6/10；parse_failed 4） |

> 两槽主指标不同名；能力内如何汇总待定。CoVoST 词面 BLEU 尚未按方向拆 overall。

---

## Audio Caption

每槽多指标并列；主辅（含 FENSE）关系未定，不合成单分。

| 槽 | CIDEr | SPICE | SPIDEr | FENSE |
|---|---:|---:|---:|---:|
| `audio_caption__audiocaps` | 0.370 | 0.103 | 0.237 | 0.550 |
| `audio_caption__clotho` | 0.397 | 0.064 | 0.230 | 0.406 |

---

## Code Switch

| 槽 | 主指标 | 主分 | Judge（并列） |
|---|---|---:|---|
| `code_switch__mmsu` | exist-match ↑ | 0.10 | 67.5（8/10；fail 2） |
| `code_switch__vocalbench_zh` | exist-match ↑ | 0.90 | 100（10/10） |
| `code_switch__vocalbench_zh_open_ended` | ROUGE-L% ↑ | 39.28 | 100（10/10） |

> open_ended 与 knowledge 主指标不同名；能力内汇总待定。

---

## QA

| 槽 | exist-match ↑ | Judge 均分（有效条） |
|---|---:|---|
| `qa__ear_wdyl` | 0.20 | 0.0（2/10；fail 8） |
| `qa__spoken_squad` | 0.40 | 64.4（9/10） |
| `qa__uro_bench` | 0.50 | 98.0（10/10） |
| `qa__vocalbench` | 0.90 | 100（7/10） |
| `qa__vocalbench_zh` | 1.00 | 90.0（10/10） |
| `qa__voicebench` | 0.90 | 75.0（8/10） |

> 主轨与 Judge 分轨，不加权成 QA 总分；跨槽宏平均是否采用待定。

---

## Audio Reasoning

| 槽 | 主指标 | 主分 | Judge |
|---|---|---:|---|
| `audio_reasoning__mmsu` | ROUGE-L% ↑ | 32.85 | 100（7/10；fail 3） |
| `audio_reasoning__uro_bench` | exist-match ↑ | 0.80 | 10/10 skipped |
| `audio_reasoning__vocalbench` | ROUGE-L% ↑ | 12.11 | 82.2（9/10） |
| `audio_reasoning__vocalbench_zh` | exist-match ↑ | 0.10 | 10/10 skipped |
| `audio_reasoning__voicebench` | exist-match ↑ | 1.00 | 10/10 skipped；BBH 长 CoT 易虚高 |

> ROUGE-L 与 exist-match 并存；能力内汇总待定。

---

## Acoustic Scene

| 槽 | accuracy ↑ | macro_f1 |
|---|---:|---:|
| `acoustic_scene__dcase2013_scenes` | 0.70 | 0.63 |
| `acoustic_scene__mmsu` | 0.80 | 0.69 |
| `acoustic_scene__tut_sound_events_2017` | 1.00 | 1.00 |
| `acoustic_scene__tut_urban_acoustic_scenes_2018_mobile` | 0.20 | 0.17 |

---

## Sound Event

| 槽 | 主指标 | 主分 | 辅 |
|---|---|---:|---|
| `sound_event__esc50` | accuracy ↑ | 0.70 | macro_f1 0.58 |
| `sound_event__mmsu` | accuracy ↑ | 0.90 | macro_f1 0.78 |
| `sound_event__tut_sound_events_2017` | accuracy ↑ | 0.70 | macro_f1 0.54 |
| `sound_event__mavd_traffic` | multilabel f1 ↑ | 0.48 | — |
| `sound_event__sonyc_ust_v23` | multilabel f1 ↑ | 0.28 | — |

> 单标签 acc 与多标签 f1 不同名；能力内汇总待定。

---

## Paralinguistic Recognition

| 槽 | accuracy ↑ | macro_f1 |
|---|---:|---:|
| `paralinguistic_recognition__mmsu` | 0.60 | 0.43 |

---

## Speaker Count

| 槽 | speaker_count_acc ↑ |
|---|---:|
| `speaker_count__mmsu` | 0.70 |

---

## Speaker Verification

| 槽 | EER ↓ | AUC ↑ | acc@0.5（诊断） |
|---|---:|---:|---:|
| `speaker_verification__voxceleb1` | 0.40 | 0.66 | 0.60 |

> 主分 EER；不与 acc@0.5 合成。

---

## Diarization

30s adapted（非整会官方口径）。

| 槽 | DER ↓ |
|---|---:|
| `diarization__aishell_4` | 0.29 |
| `diarization__alimeeting` | 0.42 |
| `diarization__ami` | 0.51 |
| `diarization__chime_6` | 0.78 |

---

## Speaker Attribution

30s adapted。

| 槽 | cpCER% ↓ | attribution_acc ↑ |
|---|---:|---:|
| `speaker_attribution__aishell_4` | 56.3 | 0.60 |
| `speaker_attribution__aishell_5` | 74.0 | 0.42 |
| `speaker_attribution__alimeeting` | 42.9 | 0.41 |
| `speaker_attribution__ami` | 61.2 | 0.51 |

---

## Target Speaker

| 槽 | target_speaker_acc ↑ | missed_execution | mis_execution |
|---|---:|---:|---:|
| `target_speaker__aishell_5` | 0.40 | ≈0.50 | ≈0.10 |

> 自建数据；三数并列。旧 overall `mis_execution=0.60` 含全部错误，勿单引。

---

## Speech Generation

| 槽 | content_acc ↑ | wer% |
|---|---:|---:|
| `speech_generation__libritts` | 0.960 | 3.96 |

---

## Instruction Following

**VoiceBench 与 VocalBench 分报，不合并为 IF 总分**（已定）。

| 槽 | 主指标 | 主分 | 备注 |
|---|---|---:|---|
| `instruction_following__voicebench` | constraint_satisfaction ↑ | 0.80 | 文本 IFEval |
| `instruction_following__vocalbench` | constraint_satisfaction ↑ | 仅 1 条=1.0（Job 23740） | 其余评测挂：Thinking `flash_attn` 落 CPU；**Job 23746 重跑中**（`main.py` 改 `device_map=cuda:0` + sdpa 回退） |
| `instruction_following__vocalbench_zh` | — | — | Job 23740 failure_rate=1.0；同上，随 **23746** 重测 |

历史（已移出 IF，不进本节主路径）：FSC intent_acc 0.80；SLURP intent_acc 0.50（SLU 归属待讨论）。

---

## Tool Call

| 槽 | tool_acc ↑ | call_exact | parameter_f1 |
|---|---:|---:|---:|
| `tool_call__fluent_speech_commands` | 1.00 | 0.80 | 0.93 |
| `tool_call__slurp` | 0.60 | 0.10 | 0.30 |
| `tool_call__audioagentbench_suite` | 0.90 | 0.30 | 0.70 |
| `tool_call__stepeval_audio_toolcall` | 0.20 | 0.10 | 0.10 |

---

## Clarification

| 槽 | clarification_accuracy ↑ | clar_f1 |
|---|---:|---:|
| `clarification__ear_wdyl` | 0.20 | 0.33 |
| `clarification__audioagentbench_suite` | 0.40 | 0.40 |

---

## Multi-turn Dialogue

官方协议 + 本地 Thinking judge（2048 tokens），Job 24767，产物 `outputs/native_runs/20260928_125701_multi_turn_smoke/`。均值只算判定成功的条目；parse_failed 单列。

| 槽 | 主指标 ↑ | 值 | 有效 / 10 | parse_failed | 辅助 |
|---|---|---:|---:|---:|---|
| `multi_turn_dialogue__vocalbench` | multi_round_acc | 0.89 | 9 | 1 | — |
| `multi_turn_dialogue__vocalbench_zh` | multi_round_acc | 0.80 | 10 | 0 | — |
| `multi_turn_dialogue__voicebench` | mtbench_score（1–10） | 8.75 | 10 | 0 | turn1 9.1 / turn2 8.4 |
| `multi_turn_dialogue__audioagentbench_suite` | turn_pass | 1.00 | 7 | 3 | 前 10 条全是 appointment-bench；各维度全过 |
| `multi_turn_dialogue__mtalk_bench` | rubric_score | 0.56 | 1 | 9 | 前 10 条全是 ambient；Thinking 2048 tokens 内写不完逐条判断 |

parse_failed 全部是 Thinking 推理超出 2048 tokens、没有输出最终标签（部分在 temperature 0 下陷入重复）；与 `待讨论问题.md` §3 同一问题。MTalk 在这次 smoke 里实际不可评；2026-10-08 改成硬约束加强制收尾后，用同一批语音只跑评委，10/10 出分，`rubric_score` 均分 0.57（结果在 `outputs/mtalk_judge_compare/20261008_judge_only/`），但 ambient 类背景事件常被评委漏听，分数仅供参考（见口径报告 §18）。

---

## Spoof Detection

| 槽 | accuracy ↑ |
|---|---:|
| `spoof_detection__mmsu` | 0.00 |

---

## Interruption

| 槽 | 结果 |
|---|---|
| `interruption__ihbench` | failure_rate=1.0（链路未通） |

---

## 开放问题（节内汇总；非跨能力）

**跨 capability 不分、不加权、不报模型总分**——已定。

**单个 capability 节内也暂不写总分**（2026-09-28 定）。以后如要汇总，需先解决下表几类情况：

| 场景 | 例子 | 说明 |
|---|---|---|
| 同名同量纲 | Diar 四槽皆 DER；QA 六槽皆 exist-match | 微平均 / 宏平均 / 只列表？未选 |
| 同能力但指标名因语言而异 | ASR：CER% vs WER% | 不可平均；或分中英两行再议 |
| 同能力主指标不同名 | ST BLEU vs exist；AR ROUGE vs exist；SE acc vs f1 | 量纲不可比 |
| 协议轴不同 | IF：IFEval vs VocalBench FR | 已定节内也分报 |
| 同槽多指标 | Caption；SV；TS | 并列，不捏单分 |
| 主轨 + Judge | QA / ST / CS / AR | 不与主轨加权 |

---

## 阅读约定

1. 对外按 **capability 章节**讲；章节之间不横向合成。  
2. 章内只报各槽；暂不写「本能力总分」。  
3. Judge 均分只对有效条；parse_failed / skipped 单独说。  
4. 样例见 [`待讨论问题.md`](待讨论问题.md)；口径见 [`已完成能力_指标口径报告.md`](已完成能力_指标口径报告.md)。
