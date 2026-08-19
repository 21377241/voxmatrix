# Third-party notices

VoxMatrix builds on the Apache-2.0-licensed
[UltraEval-Audio](https://github.com/OpenBMB/UltraEval-Audio) execution core.
The repository root [LICENSE](LICENSE) applies to VoxMatrix-owned changes.

`audio_evals/lib/` contains integration code and reduced inference snapshots.
Training recipes, WebUIs, standalone demos, screenshots, generated results,
and model weights are intentionally excluded. Retained upstream license,
disclaimer, and notice files take precedence for their corresponding code.

| Component | Upstream | Terms retained in this repository |
| --- | --- | --- |
| CosyVoice | <https://github.com/FunAudioLLM/CosyVoice> | `audio_evals/lib/CosyVoice/LICENSE` (Apache-2.0) |
| Dolphin | <https://github.com/DataoceanAI/Dolphin> | `audio_evals/lib/Dolphin/LICENSE` (Apache-2.0) |
| Spark-TTS | <https://github.com/SparkAudio/Spark-TTS> | `audio_evals/lib/Spark-TTS/LICENSE` (Apache-2.0) |
| WavTokenizer | <https://github.com/jishengpeng/WavTokenizer> | `audio_evals/lib/WavTokenizer/LICENSE` (MIT) |
| IndexTTS | <https://github.com/index-tts/index-tts> | Apache-2.0 code license, disclaimer, and separate model license under `audio_evals/lib/index-tts/` |
| IndexTTS2 | <https://github.com/index-tts/index-tts> | bilibili Model Use License Agreement, Chinese translation, and disclaimer under `audio_evals/lib/index-tts2/`; review its restrictions before use or redistribution |
| Kimi-Audio | <https://github.com/MoonshotAI/Kimi-Audio> | mixed Apache-2.0/MIT statement in `audio_evals/lib/Kimi-Audio/NOTICE.md` and retained file-level headers |
| 3D-Speaker | <https://github.com/modelscope/3D-Speaker> | Apache-2.0 attribution in `audio_evals/lib/cv3_speaker_sim/NOTICE.md` and retained file-level headers |

Other compact metric/model integration modules retain upstream copyright and
license headers where present.

Model checkpoints, datasets, generated audio, API responses, and other runtime
assets are not distributed. Obtain them from their official sources and follow
their individual licenses, acceptable-use rules, and access terms.
