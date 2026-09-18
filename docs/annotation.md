# 数据标注基准

能力挂载只认 `annotation/benchmark_subset_manifest.jsonl`。该文件是 native v2
子集×协议盘面。`vocal_bench` 的 `full_eval` capability 字段不是协议真相。
路径可用 `VOXMATRIX_SUBSET_MANIFEST` 覆盖。运行时通过
`mesh_eval.core.subset_manifest.load_annotation_baseline()` 读取。

干净主体不再包含旧的 `annotation/pipeline`、`mappings`、`knowledge`、`ai`
流水线。保留 `annotation/schema/` 与 `annotation/prompts/` 作为 Taxonomy /
样本 Schema。

## Schema

- `annotation/schema/taxonomy.v2.yaml`：任务、能力、场景和条件枚举。
- `annotation/schema/evaluation_sample.v2.schema.json`：规范样本 Schema。
- `mesh_eval.scripts.validate_manifest`：Manifest 和音频路径校验入口。
