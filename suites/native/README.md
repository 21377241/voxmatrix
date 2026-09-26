# Native 覆盖入口

干净主体的正式入口。按标注基准覆盖每个可执行 `(能力, 数据集)`，默认每槽 10 条。用来验证 `clean-mainline` 能进数据、能推理、能出指标或明确 blocked。

## 统一口径（与 ASR 一致）

```
annotation/benchmark_subset_manifest.jsonl  →  catalog 选槽
  →  mesh_eval/data/native/<capability>_<benchmark>.jsonl（每任务一份，limit=10）
  →  WavPath 直指 L0，或需裁窗时写入 eval_clips/
```

- **L1**：只用 [`annotation/benchmark_subset_manifest.jsonl`](../../annotation/benchmark_subset_manifest.jsonl)（= audit 产出）选槽。
- **评测入口**：每槽一份 `mesh_eval/data/native/*.jsonl`；`WavPath` 指向 L0 或本仓 `eval_clips/` / `asr_eval_clips/` / `long_form_windows/`。
- **禁止**作为评测入口：FullEval、`utt_clips` 索引、smoke / 8p31 拷贝路径。
- **`utt_clips` / smoke**：可留作切分台账或历史，不进 registry。
- **T4 / AudioAgentBench 等 external Dataset**：已直挂官方包，本轮不重写 loader，视为已合规。

不可执行或已判无效的槽位保持 `blocked`，不提交：AISHELL-2、EasyCom、MASSIVE、ASVspoof、CN-Celeb attribution、MISP diarization、meeting_summary、todo_extraction。

## 构建顺序

在仓库根目录、`audioeval` 环境中：

```bash
cd /mnt/afs/users/wangyl/VoxMatrix-clean
source suites/native/env.sh

# 1) 句级 ASR：annotation → asr_*.jsonl（必要时裁到 asr_eval_clips/）
"$VOXMATRIX_PYTHON" suites/native/build_asr_native_samples.py

# 2) 其余能力：annotation native_index_files → <cap>_<bench>.jsonl
#    diar/attr/target_speaker 等需裁窗写入 eval_clips/
"$VOXMATRIX_PYTHON" suites/native/build_native_capability_samples.py

# 3) long_form：L0 时间戳源 → long_form_timelines/（不读 utt_clips）
"$VOXMATRIX_PYTHON" suites/native/build_long_form_timelines.py

# 4) long_form：timeline → 30–60s 窗 + *_long_form_30_60.jsonl
"$VOXMATRIX_PYTHON" suites/native/build_long_form_windows.py

# 5) 改挂 catalog / registry（native-*，弹出 stale full-*）
"$VOXMATRIX_PYTHON" suites/native/_build_coverage.py
```

## 一键预检 / 提交

本机无 GPU，只做预检：

```bash
cd /mnt/afs/users/wangyl/VoxMatrix-clean
source suites/native/env.sh
"$VOXMATRIX_PYTHON" suites/native/preflight.py
```

预检通过后提交 GPU：

```bash
bash suites/native/submit.sh
```

## 协议备注

- 异构指标不能合成 NativeVoiceBench 总分。
- `ok`：可行性分；`diagnostic`：链路通，不当主榜；`blocked`：无数据或协议无效。
- `code_switch` 当前为混语 QA（adapted）：MMSU / open_ended 用 ROUGE-L，knowledge 用 qa-exist-match；语义辅轨走 LLM-Judge（见 `scripts/submit_code_switch_local_judge.sh`）。无 MER 主榜。
- `audio_reasoning`：统一 open QA；MMSU/VocalBench 用 ROUGE-L，URO/VocalBench-zh/VoiceBench 用 qa-exist-match；语义辅轨 LLM-Judge（见 `scripts/submit_audio_reasoning_local_judge.sh`）。
- `target_speaker` 不是 VoxCeleb：AISHELL-5 注册对，`WavPath`/`WavPath2` 在 `eval_clips/`。
- `long_form_asr` 不是整场会：固定 30–60s 窗，与句级 `asr` 区分；构建输入来自 L0 时间戳 / msswift，不经 utt_clips。
- `diarization` / `speaker_attribution`：30s 窗在 `eval_clips/<cap>/<bench>/`，不再引用 `smoke_diar_all` / `smoke_attr_all`。

8p31 分轨打分器仍注册在 `registry/evaluator/qa.yaml`：`uro-basic-content`、`vocalbench-qa-split`。本 catalog 的 qa 槽位默认 exist-match。
