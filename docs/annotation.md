# 数据标注与 Manifest 构建

`annotation/` 提供可选的数据映射、预标注、校验和 Manifest 迁移工具；评测
运行时使用的 Taxonomy 与 JSON Schema 也位于该目录。运行产物应写到外部工作
目录，不要提交到源码仓库。

## 离线流水线

准备包含原始样本的 `raw_index.jsonl` 后，可使用版本化 Mapping 在不调用网络
或模型 API 的情况下生成诊断 Manifest：

```bash
python -m annotation.pipeline.run_pipeline \
  --dataset <dataset_id> \
  --work-dir work/<dataset_id> \
  --raw-index /path/to/raw_index.jsonl \
  --offline
```

新增数据集时至少需要：

1. 在 `annotation/mappings/` 添加 Mapping。
2. 在 `annotation/knowledge/datasets/` 添加来源卡片。
3. 运行流水线并检查校验报告与 `run_manifest.json`。

未通过知识核对的样本不会进入 `formal_subscores`；不确定样本保留在诊断或
覆盖缺口分桶中。

## 可选 AI 增强

去掉 `--offline` 可启用知识归纳和标签核验。复制
`annotation/.env.example` 为未跟踪的 `annotation/.env`，配置兼容的模型 API
后再运行。不要把密钥、响应缓存或工作目录提交到仓库。

## Schema 与迁移

- `annotation/schema/taxonomy.v2.yaml`：任务、能力、场景和条件枚举。
- `annotation/schema/evaluation_sample.v2.schema.json`：规范样本 Schema。
- `annotation/pipeline/migrate_v2.py`：旧 Manifest 到 V2 的迁移工具。
- `mesh_eval.scripts.validate_manifest`：Manifest 和音频路径校验入口。
