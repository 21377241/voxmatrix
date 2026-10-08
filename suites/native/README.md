# Native 覆盖入口

干净主体的正式入口。按标注基准覆盖每个可执行 `(能力, 数据集)`，默认每槽 10 条。用来验证 `clean-mainline` 能进数据、能推理、能出指标或明确 blocked。

## 统一口径（与 ASR 一致）

```
annotation/benchmark_subset_manifest.jsonl  →  catalog 选槽
  →  mesh_eval/data/native/<capability>_<benchmark>.jsonl（每任务一份，limit=10）
  →  WavPath 直指 L0，或需裁窗时写入 eval_clips/
```

- **L1**：只用 [`annotation/benchmark_subset_manifest.jsonl`](../../annotation/benchmark_subset_manifest.jsonl)（= audit 产出）选槽。
- **评测入口**：每槽一份 `mesh_eval/data/native/*.jsonl`；`WavPath` 指向 L0 或本仓 `eval_clips/` / `asr_eval_clips/`。会议窗 Full 索引在 `/mnt/afs/eval_data/windowed_ondevice/`。
- **禁止**作为评测入口：FullEval、`utt_clips` 索引、smoke / 8p31 拷贝路径。
- **`utt_clips` / smoke**：可留作切分台账或历史，不进 registry。
- **T4 / AudioAgentBench 等 external Dataset**：已直挂官方包，本轮不重写 loader，视为已合规。

不可执行或已判无效的槽位保持 `blocked`，不提交：AISHELL-2、EasyCom、MASSIVE、ASVspoof、CN-Celeb attribution、AISHELL-5 attribution、MISP diarization、MISP/SBCSAE long_form、meeting_summary、todo_extraction。

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

# 3) 会议四档窗 Full（S/M/L/XL，ASR+diar+attr）→ /mnt/afs/eval_data/windowed_ondevice/
"$VOXMATRIX_PYTHON" scripts/build_windowed_ondevice_indexes.py
#    正式评测集：每数据集 20 个完整嵌套组 × 4 档 → mesh_eval/data/native/*_windowed.jsonl
"$VOXMATRIX_PYTHON" scripts/sample_windowed_ondevice_eval.py

# 4) 改挂 catalog / registry（native-*，弹出 stale full-*）
"$VOXMATRIX_PYTHON" suites/native/_build_coverage.py

# 5) Native 对齐到 Full（Full 为准；Native 改为 Full 的子集），写 mesh_eval/data/full/slot_map.yaml
#    1)/2) 重跑后必须再跑这一步
"$VOXMATRIX_PYTHON" scripts/align_native_to_full.py
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

会议能力（long_form_asr / diarization / speaker_attribution 共 12 槽）单独提交：

```bash
scripts/submit_native_windowed_ondevice.sh                     # 正式集，每槽 80 条
WINDOWED_LIMIT=8 scripts/submit_native_windowed_ondevice.sh    # smoke，每槽 2 组
WINDOWED_SLOTS_FILTER=diarization__ami scripts/submit_native_windowed_ondevice.sh
```

### 输出目录（所有入口统一）

`env.sh` 为每次运行分配独立目录 `outputs/native_runs/<时间戳>_<节点>_<pid>/`（`NATIVE_OUTPUT_DIR`），内含 `suite*.runtime.yaml`、`registry/`、`preflight.json`、`main/`、`tts/`（含 `speech_wavs/`）、`judge/`。重复或并发提交不会追加到同一份结果。旧的 `suites/native/output/` 只保留历史产物，新运行不再写入。

- 显式传 `NATIVE_OUTPUT_DIR=<目录>` 可续跑或复评同一次运行。
- TTS 分阶段：`scripts/submit_native_speech_vllm_omni_split.sh` 生成一个目录并同时传给 infer / judge 作业；单独提交 judge 时必须指定 infer 的目录。
- 后处理（`submit_qa_local_judge.sh`、`submit_audio_reasoning_local_judge.sh`、`run_fense_posthoc.py`）必须指定 `NATIVE_OUTPUT_DIR`，读取 `<run>/main/`，写入 `<run>/judge/`；judge 的 ASR 缓存按输入音频做键，跨运行共享。

```bash
NATIVE_OUTPUT_DIR=outputs/native_runs/<run> scripts/submit_qa_local_judge.sh
```

## 协议备注

- 异构指标不能合成 NativeVoiceBench 总分。
- `ok`：可行性分；`diagnostic`：链路通，不当主榜；`blocked`：无数据或协议无效。
- `code_switch` 当前为混语 QA（adapted）：MMSU / open_ended 用 ROUGE-L，knowledge 用 qa-exist-match；语义辅轨走 LLM-Judge（见 `scripts/submit_code_switch_local_judge.sh`）。无 MER 主榜。
- `audio_reasoning`：统一 open QA；MMSU/VocalBench 用 ROUGE-L，URO/VocalBench-zh/VoiceBench 用 qa-exist-match；语义辅轨 LLM-Judge（见 `scripts/submit_audio_reasoning_local_judge.sh`）。
- `target_speaker` 不是 VoxCeleb：AISHELL-5 注册对，`WavPath`/`WavPath2` 在 `eval_clips/`。
- `long_form_asr` / 会议 diar / attribution：边界对齐嵌套四档（S≈30s / M≈1min / L≈2min / XL≈5min），见 `scripts/build_windowed_ondevice_indexes.py` → `/mnt/afs/eval_data/windowed_ondevice/`。12 槽均为 `ok`，计入主分；指标沿用各能力的统一口径（CER/WER、DER、cpCER），不对长窗做单独截断或归一。
- 正式集按组（family）行序排列，槽位带 `limit_step: 4`，任何 limit 都会取整到完整组，保证各档条数一致。
- 分档指标：样本带 `window_bucket`；ASR 槽 agg 为 `wer(-zh)-windowed`（`window_bucket/<档>/wer(%)`），diar/attr 走 mesh agg（`window_bucket/...`、`task+capability+benchmark_id+window_bucket/...`），suite 级 `suite-overall.json` 同样有分档。

8p31 分轨打分器仍注册在 `registry/evaluator/qa.yaml`：`uro-basic-content`、`vocalbench-qa-split`。本 catalog 的 qa 槽位默认 exist-match。
