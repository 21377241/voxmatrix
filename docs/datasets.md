# 数据集与音频准备

## 查看已有数据集

Registry 中的数据集配置位于 `registry/dataset/` 和
`mesh_eval/registry/dataset/`。在源码目录中可运行：

```bash
python cli/list_available.py --datasets
```

每个数据集至少指定加载类、默认评测任务和参考答案字段。默认任务决定 Prompt、
Evaluator 与聚合策略；命令行仍可覆盖这些设置。

## 注册 JSONL 数据集

JSONL 每行是一个样本。单音频任务通常使用 `WavPath`，参考答案列名由
`ref_col` 指定：

```json
{"WavPath": "/data/custom/0001.wav", "Transcript": "hello world"}
{"WavPath": "/data/custom/0002.wav", "Transcript": "another sample"}
```

创建 `registry/dataset/custom.yaml`：

```yaml
custom-asr:
  class: audio_evals.dataset.dataset.JsonlFile
  args:
    f_name: /data/custom/manifest.jsonl
    default_task: asr
    ref_col: Transcript
```

如果不希望修改内置 Registry，可把同样的目录结构放在外部路径，并在单次评测
中传入：

```bash
voxmatrix \
  --registry_path /path/to/custom_registry \
  --dataset custom-asr \
  --model <model_name>
```

相对音频路径可使用 `audio_evals.dataset.dataset.RelativePath` 并配置
`file_path_prefix`。多音频或 Mesh Schema 样本也可以使用 `WavPath1`、
`WavPath2`，或 `input.audio_path`、`input.audio_path_b` 与
`input.audios[].uri`。

## 注册 Hugging Face 数据集

```yaml
custom-hf-asr:
  class: audio_evals.dataset.huggingface.Huggingface
  args:
    name: organization/dataset
    subset: default
    split: test
    default_task: asr
    ref_col: text
    revision: <pinned_revision>
    audio_cache_root: raw/custom-hf-asr
```

`local_path`、`data_files` 和 `cache_dir` 可用于本地快照或自定义数据文件。
正式评测建议固定 `revision`，并保存数据源与预处理元数据。

## 一次性转换为 WAV

Hugging Face Audio 列在首次读取时需要解码。直接重复运行会反复触发数据准备；
`voxmatrix-prepare` 可提前把它转换为稳定资产：

```bash
voxmatrix-prepare \
  --dataset custom-hf-asr \
  --output-dir prepared/custom-hf-asr \
  --checksum sha256
```

可用 `--offset` 和 `--limit` 准备确定范围；`--checksum none` 可跳过每个
音频文件的 SHA-256，但 Manifest 与 Metadata 仍会被校验。外部 Registry 使用
`voxmatrix-prepare --registry-path /path/to/custom_registry ...`。

成功目录包含：

```text
prepared/custom-hf-asr/
├── _SUCCESS
├── manifest.jsonl
├── metadata.json
└── assets/
    └── ... .wav
```

`_SUCCESS` 只有在 Manifest、Metadata 和资产清单完整写入后才生成。推理时
使用：

```bash
voxmatrix \
  --dataset custom-hf-asr \
  --model <model_name> \
  --prepared-manifest prepared/custom-hf-asr/manifest.jsonl
```

加载器会验证 `_SUCCESS` 和 Manifest 摘要，并检查音频是否存在；因此不要只
复制 `manifest.jsonl`。准备目录可以跨多次评测复用。

## 加载顺序和切片

单次评测默认使用 `--dataset-load-order before_model`：先解析、切片并加载
数据，再初始化模型。这样数据错误不会在模型占用 GPU 后才暴露。

- `--offset N`：从固定偏移开始。
- `--limit N`：最多加载 N 个样本，0 表示不限制。
- `--rand N`：在已加载范围内随机抽 N 个样本。
- `--dataset_load_timeout N`：设置数据加载超时。

`legacy_after_model` 只为历史行为保留，新任务不应使用。

## Mesh Manifest 校验

规范化 Manifest 可在评测前检查：

```bash
python -m mesh_eval.scripts.validate_manifest \
  /path/to/manifest.jsonl \
  --ref_col text \
  --check_audio_exists
```

Schema 与 Benchmark 映射的事实来源分别位于 `annotation/schema/` 和
`mesh_eval/config/benchmark_evaluator_map.yaml`。
