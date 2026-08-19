# 共享 Session 与多 GPU 推理

## 单次评测的模型池

对支持隔离运行的模型，`IsolatedModelPool` 会创建多个模型副本，并把并发
样本调度到空闲副本。以下三个量相互独立：

- `replicas`：模型副本数。
- `gpus_per_replica`：每个副本占用的可见 GPU 数。
- `inference_workers`：样本调度线程数。

两张卡、每卡一个副本：

```bash
CUDA_VISIBLE_DEVICES=0,1 voxmatrix \
  --dataset <dataset_name> \
  --model <model_name> \
  --use_model_pool on \
  --replicas 2 \
  --gpus-per-replica 1 \
  --inference-workers 2 \
  --model-startup-workers 2 \
  --model-ready-policy required
```

副本分配在启动前一次性确定。默认 `--model-startup-workers 0` 会使用与副本
数相同的启动并发度，因此不会先等待 GPU 0 完整加载后才开始 GPU 1。若磁盘或
模型仓库承受不了并发读取，可显式降低该值。

## 模型池参数

| 参数 | 含义 |
| --- | --- |
| `--use_model_pool auto\|on\|off` | 自动检测、强制启用或禁用模型池 |
| `--replicas N` | 副本数；0 表示按可见 GPU 分组自动计算 |
| `--gpus-per-replica N` | 一个副本使用的 GPU 数，默认 1 |
| `--inference-workers N` | 并发样本调度数；0 表示跟随副本数 |
| `--model-startup-workers N` | 并发启动数；0 表示全部副本并发 |
| `--model-ready-policy auto\|required\|launch` | 就绪协议要求 |
| `--allow-gpu-sharing` | 允许多个副本共享 GPU；默认禁止 |

`CUDA_VISIBLE_DEVICES` 是权威可见范围，也支持 GPU UUID 和 MIG UUID。设置为
空、`-1` 或无设备形式时，框架不会越过调度器去发现宿主机 GPU。

`--use_model_pool auto` 只对构造函数兼容 `gpu_id` 的适配器启用模型池。
API 模型或普通单实例模型仍按单 Predictor 运行。

## 多 Benchmark 共享模型

`voxmatrix-suite` 在一个 Predictor 生命周期中运行多个 Benchmark：

```yaml
schema_version: evaluation-suite-config/1.0
run_id: experiment-001
model: <registered_model>
output_root: outputs/suite/experiment-001

model_pool:
  use_model_pool: auto
  replicas: 0
  gpus_per_replica: 1
  inference_workers: 0
  model_startup_workers: 0
  allow_gpu_sharing: false
  model_ready_policy: auto

benchmarks:
  - benchmark_id: first
    dataset: <dataset_a>
    limit: 100
    evaluation_workers: 1

  - benchmark_id: second
    dataset: <dataset_b>
    prepared_manifest: prepared/dataset_b/manifest.jsonl
    limit: 100
    evaluation_workers: 1
```

运行：

```bash
voxmatrix-suite --config path/to/suite.yaml
```

Session 的顺序是：

1. 校验配置并加载、切片所有数据集。
2. 创建一次 Predictor 或模型池。
3. 依次运行 Benchmark；每个 Benchmark 内部可并行推理。
4. 写入各 Benchmark 结果与 `session.json`。
5. 释放全部模型副本和子进程。

因此，多 Benchmark 不会重复加载同一个模型。各 Benchmark 必须使用 Session
顶层指定的同一模型，但可以覆盖 `task`、`prompt`、`evaluator`、`agg`
和 `post_process`。

Benchmark 还支持 `offset`、`rand`、`dataset_load_timeout`、`resume`、
`inf_file`、`dataset_ref_col`、`save`、`two_phase` 与
`evaluation_workers`。完整可编辑模板见
[examples/suite.example.yaml](../examples/suite.example.yaml)。

## 容量注意事项

- 默认禁止副本数超过 GPU 分组容量。
- 多副本会近似按副本数放大权重显存和主机内存占用。
- `inference_workers` 大于副本数只会增加排队，不会增加同时执行的模型数。
- 多 GPU 单副本应设置 `gpus_per_replica > 1`，并确认 Worker 自身支持该
  设备组。
- 首次运行可能需要准备隔离环境或下载权重；这些步骤有文件锁，但仍应预留
  磁盘和网络时间。
