# Capability：speech_generation（语音生成）

负责人：甲（T5）  
日期：2026-09-07；工程修复：2026-09-09  
模型：Qwen3-Omni-30B-A3B-Instruct（`qwen3-omni-speech-local`，`speech=true`）  
Smoke Job：**15296**（生成 SUCCEEDED；当时框架 eval failure 100%）；重算 **15309**  
产物：`smoke_tts_all/generated/`（10 wav）；`results/libritts-rescored-*.json`

> LibriTTS test-clean 文本 → Omni TTS → Whisper ASR → **content_acc**（本轮不挂 SIM/MOS）。

---

## 状态总览（2026-09-09）

| # | 问题 | 状态 | 说明 |
| --- | --- | --- | --- |
| 甲-TTS-01 | Omni pred 为 JSON `{"text","audio"}` 非纯路径 | ✅ **已收口** | `ExtractOmniAudioPath` + smoke `post_process: [extract-omni-audio-path]`；evaluator 内再兜一层 |
| 甲-TTS-02 | SeedTTS Whisper dtype（float vs Half） | ✅ **已收口** | `seed_tts_eval.py` 固定 `float32`，features 与 weights 同 dtype |
| — | 生成 | ✅ Job **15296** | 10/10 wav |
| — | content_acc | ✅ Job **15309** 重算 | **0.950**（WER%≈5.0） |

### Smoke 汇总

| Benchmark | 生成 | **content_acc** | WER% | 结论 |
|-----------|------|----------------:|-----:|------|
| libritts test-clean | 10/10 | **0.950** | **5.04** | 链路+内容均可；工程项已修，待下次 suite 直跑验证 failure=0 |

人核：**pass**（生成可懂、内容基本正确）。

---

## 统一 template

- **任务形态**：文本 → 生成音频
- **Prompt**：`smoke-qwen3-omni-tts-en`（*Repeat the following text once…*）
- **模型**：`speech: true` + `speech_output_dir`
- **post_process**：`extract-omni-audio-path`（抽 wav 路径）
- **evaluator**：`smoke-seed-tts-eval-asr-wer-en` → Whisper → WER → content_acc
- **多音频**：无（无参考音色）

---

## 主指标

| 指标 | 角色 | 公式 |
|------|------|------|
| **content_acc** | 主分 | `max(0, 1 − WER)` |
| intelligibility | 同值别名 | 同上 |
| wer% | 辅 | Whisper(hyp) vs 原文，`jiwer` 词错误率 ×100 |
| SIM / MOS | 本轮不评 | — |

数据流：Omni JSON → 抽 `audio` path → Whisper → WER → content_acc。

---

## 各 benchmark 核验

### libritts
- 抽样：10 条；content_acc **0.950**（rescored）
- 人核：pass
- 结果是否可信：**是**
- 问题：甲-TTS-01/02 已修代码；分数仍来自 15309 离线重算

---

## 问题

### 甲-TTS-01. 评测器不接受 Omni JSON pred — ✅ 已收口

| 字段 | 内容 |
| --- | --- |
| 现象 | pred=`{"text":"...","audio":"/...wav"}`；`SeedTTSEvalASRWER` 断言 `os.path.exists(pred)` 失败 → suite failure 100% |
| 修复 | `audio_evals/process/tts.py` → `ExtractOmniAudioPath`；smoke `post_process`；主 registry `extract_audio` 亦指向同类；evaluator 入口再抽一次 |
| 验证 | 对 Job 15296 的 10 条 inference，抽取后路径均存在 |

### 甲-TTS-02. SeedTTSWhisper dtype 报错 — ✅ 已收口

| 字段 | 内容 |
| --- | --- |
| 现象 | `Input type (float) and bias type (c10::Half)` |
| 修复 | `audio_evals/lib/whisper/seed_tts_eval.py`：`torch_dtype=float32`，`input_features.to(device, dtype=float32)` |
| 说明 | 本机当前 shell 无可用 CUDA，dtype 对齐逻辑已合入；下次 GPU suite 直跑确认 |

---

## 附录

- 生成 Job **15296**；重算 **15309**；`smoke_tts_all/`  
- 关键点：`audio_evals/process/tts.py`、`audio_evals/lib/whisper/seed_tts_eval.py`、`smoke_tts_all/registry/eval_task/smoke_tts_all.yaml`
