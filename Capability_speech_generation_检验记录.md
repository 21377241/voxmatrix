# Capability：speech_generation（语音生成）

负责人：甲（T5）  
日期：2026-09-07  
模型：Qwen3-Omni-30B-A3B-Instruct（`qwen3-omni-speech-local`，`speech=true`）  
Smoke Job：**15296**（生成 SUCCEEDED；框架 eval failure 100%）；重算 **15309**  
产物：`smoke_tts_all/generated/`（10 wav）；`results/libritts-rescored-*.json`

> LibriTTS test-clean 文本 → Omni TTS → Whisper ASR → **content_acc**（本轮不挂 SIM/MOS）。

---

## 状态总览

| # | 问题 | 状态 | 说明 |
| --- | --- | --- | --- |
| 甲-TTS-01 | Omni pred 为 JSON `{"text","audio"}` 非纯路径 | ✅ | `SeedTTSEvalASRWER` 断言失败 → 需 post_process 抽 `audio` |
| 甲-TTS-02 | `seed-tts-whisper` registry 重名 / dtype bug | ✅ | smoke 用独立名；重算用 float32 Whisper |
| — | 生成 | ✅ Job **15296** | 10/10 wav |
| — | content_acc | ✅ Job **15309** 重算 | **0.950**（WER%≈5.0） |

### Smoke 汇总

| Benchmark | 生成 | **content_acc** | WER% | 结论 |
|-----------|------|----------------:|-----:|------|
| libritts test-clean | 10/10 | **0.950** | **5.04** | 链路+内容均可；框架需补 path 抽取 |

人核：**pass**（生成可懂、内容基本正确）。

---

## 统一 template

- **任务形态**：文本 → 生成音频
- **Prompt 骨架**（`qwen3-omni-tts-en` / smoke 别名）：*Repeat the following text once without adding any other words...*
- **模型**：必须 `speech: true` + `speech_output_dir`
- **多音频**：无（无参考音色）

---

## 主指标

| 指标 | 角色 |
|------|------|
| **content_acc** | 推荐主分（`1-WER`） |
| intelligibility | 同源 |
| SIM / MOS | 本轮不评 |

数据流（本轮实际）：Omni → wav →（抽 path）→ Whisper → WER → content_acc。

---

## 各 benchmark 核验

### libritts
- 抽样：10 条；content_acc **0.950**
- 人核：pass
- 结果是否可信：**是**（重算口径与协议一致；正式跑需修好 post_process）
- 问题：甲-TTS-01、甲-TTS-02

---

## 问题

### 甲-TTS-01. 评测器不接受 Omni JSON pred
- pred 形如 `{"text":"...","audio":"/...wav"}`  
- 建议：加 post_process 抽取 `audio` 字段后再进 `seed-tts-eval-asr-wer-*`  
- 方案是否确定：**确定**

### 甲-TTS-02. SeedTTSWhisper dtype 报错
- `Input type (float) and bias type (c10::Half)`  
- 本轮用 float32 Whisper 重算绕过；正式需修 `seed_tts_eval.py`  
- 方案是否确定：**确定（待修框架）**

---

## 附录

- 生成 Job **15296**；重算 **15309**；`smoke_tts_all/`
