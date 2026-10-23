# 官方技术报告 vs Native 全量实测：逐能力核对

- **核对日期**：2026-10-21  
- **实测协议**：VoxMatrix-clean `suites/native`（与对比表同一 run）  
- **官方文献**（已下载核对）：
  - Qwen3-Omni：[arXiv:2509.17765](https://arxiv.org/abs/2509.17765)（实测权重：`Qwen3-Omni-30B-A3B-Instruct`）
  - Step-Audio 2 / mini：[arXiv:2507.16632](https://arxiv.org/abs/2507.16632) Appendix B（实测：`Step-Audio-2-mini`）
  - Kimi-Audio：[arXiv:2504.18425](https://arxiv.org/abs/2504.18425)（实测：`Kimi-Audio-7B-Instruct`）
  - MiniCPM-o 4.5：[arXiv:2604.27393](https://arxiv.org/abs/2604.27393)（实测：`MiniCPMo4_5`）

**判定口径**

| 标签 | 含义 |
|------|------|
| **匹配** | 官方有可对标数字/定性，实测同量级或同方向，差异可用协议/子集解释 |
| **部分匹配** | 官方有相关能力宣称，但指标/子集不同，或数字偏差中等 |
| **不匹配** | 官方宣称明显更强，或实测出现失效级异常 |
| **官方未说明** | 报告未给该能力数字或几乎不提 |
| **实测缺失** | 我们尚未跑出（MiniCPM asr/SV） |

> 重要：官方与 VoxMatrix 的 prompt、聚合、子集切分常不一致，**不能要求数值逐点相等**；下列以「能力是否成立」为主。

---

## 1. Qwen3-Omni（30B-A3B-Instruct）

| 能力 | 官方报告描述 | 我们实测（要点） | 判定 |
|------|--------------|------------------|------|
| **asr** | Table 6：Libri clean/other **1.22/2.48**；Wenet net/meeting **4.69/5.89**；CV15-en **6.05**；强调 En/Zh ASR SOTA | Libri WER **1.25**；Wenet CER **6.06**；CV WER **8.17**；AISHELL-1 CER **0.92**；overall WER 10.2（含难子集） | **匹配**（Libri/Wenet 很接近；CV 略差于官方 6.05，仍属正常 ASR 量级） |
| **speech_translation** | Table 6 S2TT：Fleurs 多向 BLEU 约 25–37；宣称翻译强 | CoVoST2 BLEU **21.1**（中→英）；输出基本为英文 | **部分匹配**（子集不同；方向正确、量级合理，低于其 Fleurs 表内高分） |
| **audio_caption** | 另发布 Captioner 变体做详细 caption；Instruct 正文未给 AudioCaps/Clotho CIDEr | AudioCaps CIDEr **0.395**，Clotho **0.204**（中等偏弱） | **官方未说明**（Instruct 无对标 CIDEr） |
| **qa / 口语交互** | VoiceBench：Thinking 均分 **89.5**；Instruct 亦强调语音交互 | VoiceBench QA match **0.976**；VocalBench QA **0.938**；URO match **0.436** | **部分匹配**（强交互方向一致；指标口径不同，难数值对齐） |
| **instruction_following** | 强调强 instruction following（系统人设等） | IF constraint **0.788**，strict_prompt **0.788**（四模型最高） | **匹配**（定性一致） |
| **multi_turn_dialogue** | 语音对话能力在 VoiceBench 等体现 | multi_round_acc **0.871**，MT-Bench **8.95**，mtalk **0.744**（最强） | **匹配** |
| **speech_generation** | Thinker–Talker、低时延流式合成；未在 Table 6 给 LibriTTS content_acc | TTS content_acc **0.965**，WER% **3.47**（很高） | **部分匹配**（有生成能力且实测强；无同指标官方数） |
| **acoustic_scene / sound_event** | 用 MMAU/MMSU、音乐集等做 audio reasoning；非 TUT Urban/ESC 主表 | scene acc **0.363**；ESC-50 **0.820**；sound_event 能力 **0.715** | **部分匹配**（理解向能力有宣称；数据集不对齐） |
| **paralinguistic_recognition** | 未作独立主表（偏 reasoning/交互） | para acc **0.485** | **官方未说明** |
| **speaker_count / target_speaker** | 未报告 | 0.592 / 0.515 | **官方未说明** |
| **code_switch** | 未报告 | match **0.812** | **官方未说明** |
| **diarization / long_form_asr / speaker_attribution** | 强调最长约 40 min ASR/理解，**无 DER/会议 diarization 表** | DER **0.582**；长音频 CER **30.5**（可用但非专用系统） | **官方未说明**（长音频定性可部分呼应；diarization 无数） |
| **speaker_verification** | 未报告 | AUC **0.815** | **官方未说明** |
| **spoken_slu（FSC/SLURP）** | 未报告 SLURP/FSC | FSC **0.938**，SLURP **0.698** | **官方未说明** |

**小结**：与官方重叠最大的 **ASR / 对话 / IF** 整体匹配；翻译子集不同但方向正确；场景/事件/会议/SV 等多数字段官方未给对标数。

---

## 2. Step-Audio-2-mini

| 能力 | 官方报告描述（Appendix B，mini） | 我们实测 | 判定 |
|------|----------------------------------|----------|------|
| **asr** | Table 7：Libri clean/other **1.33/2.86**；AISHELL **0.78**；Wenet net/meeting **4.82/4.87**；CommonVoice-en **6.76**；宣称与 Step-Audio 2 接近、超多数开源 | Libri **1.65**；AISHELL-1 CER **1.00**；Wenet **7.0**；CV **12.2**；**sbcsae 826 / overall 184（失效）** | **部分匹配→ASR 主集基本匹配；sbcsae/overall 不匹配（协议外失效）** |
| **speech_translation** | CoVoST2 S2TT：zh→en BLEU **29.47**，en→zh **49.12**，Avg **39.29** | CoVoST2 BLEU **21.5**（中→英） | **部分匹配**（同方向、同子集族；我们低约 8 BLEU，可能 prompt/解码差异） |
| **paralinguistic_recognition** | StepEval-Audio-Paralinguistic Avg **80.0**（Emotion 82 等） | MMSU para acc **0.397** | **部分匹配**（官方强、我们中等；**基准完全不同**） |
| **audio understanding（MMAU 等）** | MMAU Avg **73.2**（Sound/Speech/Music 76.6/71.5/71.6） | 无 MMAU；ESC-50 **0.713**，caption CIDEr **0.594**（字幕相对强） | **部分匹配**（理解强的定性可对上 caption/事件；无数对齐） |
| **speech_generation / 对话** | URO-Bench 中文 Basic Avg **77.81**；强调端到端语音对话与副语言 | TTS content_acc **0.725**；multi_turn **0.647**；IF **0.256**（弱） | **不匹配偏多**（官方对话分高；我们 IF/多轮明显偏弱，可能缺工具/RAG/评测槽不同） |
| **acoustic_scene / sound_event** | 训练提到 AudioSet/AudioCaps 等；无 ESC/TUT 主表数字 | scene **0.372**；sound_event **0.639** | **官方未说明**（仅有数据配方叙述） |
| **speaker_count / target_speaker / code_switch** | 未报告 | 0.425 / 0.517 / 0.533 | **官方未说明** |
| **qa** | URO/对话基准有理解分；非 Spoken-SQuAD | QA match **0.638** | **官方未说明**（间接相关） |
| **diarization / long_form / attribution** | 未报告 DER | DER **0.894**（很差） | **官方未说明** |
| **speaker_verification** | 未报告 | AUC **0.503（≈随机，输出常数 JSON）** | **官方未说明**；实测任务失效 |
| **spoken_slu** | 未报告 | FSC **0.695**，SLURP **0.207** | **官方未说明** |
| **audio_caption** | 训练用 AudioCaps 改写成 QA；无 CIDEr 表 | CIDEr **0.594**（四模型最高） | **官方未说明**（但与「音频理解强」定性一致） |

**小结**：官方重点 **ASR / CoVoST / 副语言自建集 / MMAU / URO** 中，ASR 主集与翻译大体可对上；**IF/多轮/SV/sbcsae** 与「全面 SOTA」叙事差距大，后几项多为协议失效或未覆盖能力。

---

## 3. Kimi-Audio-7B-Instruct

| 能力 | 官方报告描述 | 我们实测 | 判定 |
|------|--------------|----------|------|
| **asr** | Table 4：Libri **1.28/2.42**；AISHELL-1 **0.60**；Wenet meeting/net **6.28/5.37**；宣称 ASR SOTA | Libri **3.73**；AISHELL-1 **0.766**；Wenet **6.58**；CV **17.0**；sbcsae **90.7** | **部分匹配**（中文 AISHELL/Wenet 接近；**Libri 明显差于官方 1.28**，不匹配） |
| **acoustic_scene** | CochlScene **~80**；TUT2017 **65.25** | TUT Urban2018 **0.256**；scene 能力 **0.278** | **不匹配**（官方场景分类很高；我们 Urban 很低——子集/协议不同，但落差过大需警惕） |
| **sound_event** | Nonspeech7k **93.93**；VocalSound **94.85** | ESC-50 **0.482**；sound_event 能力 **0.408**（偏弱） | **不匹配**（官方非语音事件很强；我们 ESC 中等偏弱） |
| **paralinguistic** | MELD **59.13**；Step 文中 Kimi 副语言 Avg 49.6 | MMSU para **0.366** | **部分匹配**（中等水平，基准不同） |
| **audio_caption** | 训练含 AudioCaps/Clotho；主表偏 AQA（ClothoAQA ~71–73） | AudioCaps CIDEr **0.435** | **官方未说明**（无 CIDEr）；AQA 未直接测 |
| **qa / 对话** | VoiceBench Avg **76.93**；IFEval **61.10**；OpenAudioBench 多项领先 | VoiceBench QA match **0.864**；VocalBench **0.788**；IF constraint **0.333** | **部分匹配→对话 QA 尚可；IF 明显弱于官方 VoiceBench-IFEval 叙事** |
| **speech_translation** | 报告主表几乎不报 CoVoST BLEU（训练有翻译数据叙述） | CoVoST2 BLEU **1.52**（~85% 输出中文转写） | **不匹配**（任务跟从失败；不能印证翻译能力） |
| **speech_generation** | 有 speech conversation 主观评；无 LibriTTS content_acc | TTS content_acc **0.535**，WER **46.5**（偏弱） | **部分匹配**（有生成；质量一般） |
| **multi_turn_dialogue** | 强调 speech conversation | multi_round **0.626**，MT-Bench **6.09** | **部分匹配**（能做，非最强） |
| **speaker_count / target_speaker / code_switch** | 未报告 | 0.200 / 0.497 / 0.552 | **官方未说明**（speaker_count 实测很差） |
| **diarization / long_form / attribution** | 未报告 | DER **0.721**；长音频差 | **官方未说明** |
| **speaker_verification** | 未报告 | AUC **0.689**（弱但非随机） | **官方未说明** |
| **spoken_slu** | 未报告 | FSC **0.708**；SLURP **0.027（87% 空输出）** | **官方未说明**；SLURP **实测失效** |

**小结**：官方最硬的 **ASR（中文）/ 部分理解** 有条件匹配；**Libri、场景/事件、CoVoST、SLURP、IF** 与报告「全面 SOTA」不一致，其中翻译与 SLURP 属适配失效。

---

## 4. MiniCPM-o 4.5

| 能力 | 官方报告描述 | 我们实测 | 判定 |
|------|--------------|----------|------|
| **asr** | Table 4：AISHELL-1 **0.9**；Libri clean/other **1.4/2.8**；Wenet net/meeting **5.9/5.7**；宣称接近领先系统 | **未跑完（—）** | **实测缺失**（待 33898） |
| **speech_translation** | CoVoST2 en→zh **49.9**，zh→en **26.4**；宣称领先 | CoVoST2 BLEU **24.5**（中→英） | **匹配**（与官方 zh→en **26.4** 非常接近） |
| **qa / VoiceBench** | VoiceBench AlpacaEval **4.81**；Speech TriviaQA **75.5** 等 | VoiceBench QA match **0.998**；VocalBench **0.876** | **匹配**（交互/QA 强，口径不同） |
| **speech_generation** | SeedTTS-EN WER **2.38**；LongTTS-EN WER **3.37**；宣称清晰度与表达优于 Qwen3-Omni | LibriTTS content_acc **0.966**，WER% **3.39**（与 Qwen 并列最高） | **匹配**（生成质量优势一致） |
| **multi_turn / IF** | 强调全双工与指令；文本 IFEval-PLS **84.7**；语音侧见 VoiceBench | multi_round **0.74**；IF constraint **0.490** / strict **0.710** | **部分匹配**（strict 较强；constraint 中等；全双工未测） |
| **audio understanding（MMAU/MELD）** | MMAU **76.9**；MELD **60.2** | 无 MMAU；ESC-50 **0.637**；caption CIDEr **0.570** | **部分匹配** |
| **acoustic_scene / sound_event / para / speaker_count** | 未给 ESC/TUT/MMSU 细分表 | scene **0.336**；event **0.512**；para **0.485**；spk_count **0.625** | **官方未说明** |
| **code_switch / target_speaker** | 未报告 | 0.663 / 0.514 | **官方未说明** |
| **diarization / long_form / attribution** | 未报告 DER | DER **0.615**；长音频 CER **36.7**（优于 Step/Kimi） | **官方未说明** |
| **speaker_verification** | 未报告 | **未跑完（—）** | **实测缺失** + **官方未说明** |
| **spoken_slu** | 未报告 | FSC **0.937**，SLURP **0.446** | **官方未说明** |
| **audio_caption** | 未给 CIDEr | **0.570** | **官方未说明** |

**小结**：官方有数且我们已测的 **CoVoST zh→en、TTS、QA** 高度吻合；ASR/SV 待补；会议/场景等官方未写。

---

## 5. 按能力横断（四模型一览）

| 能力 | Qwen | Step-mini | Kimi | MiniCPM-o 4.5 |
|------|------|-----------|------|---------------|
| asr | 匹配 | 主集部分匹配；sbcsae 不匹配 | 中文部分匹配；Libri 不匹配 | 实测缺失 |
| speech_translation | 部分匹配 | 部分匹配 | **不匹配（失效）** | **匹配** |
| speech_generation | 部分匹配 | 部分匹配/偏弱 | 部分匹配 | **匹配** |
| multi_turn / IF / qa | 匹配 | 对话官方强/我们 IF 弱→偏不匹配 | QA 部分匹配；IF 偏弱 | 匹配～部分匹配 |
| acoustic_scene / sound_event | 部分匹配 | 官方未说明 | **不匹配**（官方高/我们低） | 官方未说明 |
| paralinguistic | 官方未说明 | 部分匹配（基准不同） | 部分匹配 | 官方未说明 |
| audio_caption | 官方未说明 | 官方未说明（实测强） | 官方未说明 | 官方未说明 |
| diarization / SV / SLU / code_switch / target_speaker / speaker_count | 多数字段**官方未说明** | 同左；SV 实测失效 | 同左；SLURP 失效 | 同左；asr/SV 缺失 |

---

## 6. 总判断

1. **最可比、也最吻合的能力带**：短音频 **ASR（标准集）**、**语音翻译（CoVoST 族）**、**TTS 可懂度**、**口语 QA/对话（方向）**。  
2. **叙事强但我们对不上或失效的点**：Kimi 的 CoVoST/SLURP/Libri；Step 的 SV/sbcsae/IF；Kimi/Step 的「全面理解 SOTA」在 ESC/TUT Urban 上未复现。  
3. **大量 VoxMatrix 能力（diarization、SV、SLURP、code_switch、target_speaker 等）在四份官方报告中基本没有数字**——只能记「官方未说明」，不能判「违背报告」。  
4. 协议差是常态：官方常用自建副语言集、MMAU、URO、VoiceBench 分项；我们用 native prompt + suite 聚合，**匹配≠数值相等**。

文献本地缓存（HTML/文本抽取）：`/root/.cursor/projects/mnt-afs-users-wangyl/agent-tools/` 下对应 arXiv 抓取文件；PDF 源见上述 arXiv 链接。
