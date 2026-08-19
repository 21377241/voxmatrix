# 评测任务、输出与断点续跑

## 任务 Registry

`registry/eval_task/` 中的任务把数据集、Prompt、模型、后处理、Evaluator 和
聚合策略绑定在一起：

```yaml
custom-asr-task:
  class: audio_evals.base.EvalTaskCfg
  args:
    dataset: custom-asr
    prompt: asr
    model: custom-model
    post_process: []
    evaluator: wer
    agg: wer
```

数据集的 `default_task` 会自动选择任务；`--task custom-asr-task` 可显式
覆盖。命令行中非空的 `--model`、`--prompt`、`--evaluator`、
`--agg` 与 `--post_process` 会覆盖任务字段。

## 单次评测

```bash
voxmatrix \
  --dataset <dataset_name> \
  --model <model_name> \
  --limit 100 \
  --save outputs/run.jsonl
```

主要输出：

- `run.jsonl`：逐样本 Prompt、推理、评测信息和运行时证据。
- `run-overall.json`：聚合结果。
- `log/app-<timestamp>.log`：当前进程日志。

不指定 `--save` 时，框架使用
`outputs/<model>/<dataset>/<timestamp>.jsonl`。结果、日志和生成音频属于运行
产物，不应提交到源码仓库。

## 推理与评测分阶段

当 Evaluator 不能并发、需要另一套模型或会与被测模型竞争显存时，可使用：

```bash
voxmatrix \
  --dataset <dataset_name> \
  --model <model_name> \
  --two_phase \
  --inference-workers 4 \
  --evaluation-workers 1
```

第一阶段完成模型推理，第二阶段再执行后处理和 Evaluator。共享 Session 中只要
有 Benchmark 启用 `two_phase`，Session 会先完成所有 Benchmark 的推理，
释放 Predictor，再进行第二阶段评测。

## 断点续跑

从明确文件恢复：

```bash
voxmatrix \
  --dataset <dataset_name> \
  --model <model_name> \
  --save outputs/run.jsonl \
  --resume outputs/run.jsonl
```

`-r` 或无参数的 `--resume` 会在当前输出目录中查找最新 JSONL。恢复过程会
复用已记录样本，只运行缺失项。请确保 Dataset、Task、Model 和切片参数与原
运行一致，并保留完整的逐样本事件文件。

只有在现有结果已经包含推理时，才可使用 `--replay-only --resume <file>`
跳过推理模型加载。该能力用于重新聚合或验证已有事件，不代表仓库会分发任何
回放结果。

## 使用已有推理文件

`--inf_file <file>` 可把已有推理附加到数据集后再运行 Evaluator。它与
`--resume` 的区别是：前者明确提供推理输入，后者恢复同一次评测的进度。

## 自定义 Registry

外部 Registry 应保持以下结构：

```text
custom_registry/
├── dataset/
├── model/
├── eval_task/
├── prompt/
├── evaluator/
├── process/
└── agg/
```

单次评测使用 `--registry_path /path/to/custom_registry`。共享 Session 则在
配置顶层使用：

```yaml
registry_paths:
  - /path/to/custom_registry
```

提交正式结果时，应同时保存 Registry 版本、数据 Revision、命令或 Session
配置以及依赖版本。
