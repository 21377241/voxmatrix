# Native 全能力小样本入口

这是干净主体的正式入口。每条能力默认 10 条，用来验证当前 `clean-mainline` 能进数据、能推理、能出指标或明确 blocked。

能力挂载以 `annotation/benchmark_subset_manifest.jsonl` 为准。评测单元跟 smoke_* / 8p31 已收口方案：长会用 utt_clips，目标说话人用 AISHELL-5 注册对，diar/attr 用 30s 窗，FSC 只走 T4。

## 一键

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

## 口径

- 异构指标不能合成 NativeVoiceBench 总分。
- `ok`：可行性分；`diagnostic`：链路通，不当主榜；`blocked`：无数据。
- `code_switch` 只有短答 contain，没有 MER。
- `target_speaker` 不是 VoxCeleb。
- `long_form_asr` 不是整场会。

8p31 分轨打分器仍注册在 `registry/evaluator/qa.yaml`：`uro-basic-content`、`vocalbench-qa-split`。本 catalog 的 qa 槽位用 Spoken-SQuAD exist-match。
