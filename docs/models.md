# 模型适配器

## 查看已有模型

模型 Registry 位于 `registry/model/`：

```bash
python cli/list_available.py --models
```

一个条目由可导入的类路径和构造参数组成：

```yaml
custom-model:
  class: package.module.CustomModel
  args:
    model_name: organization/model
    sample_params:
      temperature: 0
```

不要把 API Key、个人绝对路径或集群挂载路径写进 Registry。凭据使用环境变量，
模型与环境路径使用可移植变量或用户自己的外部 Registry。

## 适配器接口

所有适配器继承 `audio_evals.models.model.Model`。常见基类是：

- `APIModel`：调用远端服务，`_inference(prompt, **kwargs)` 返回文本、
  音频路径或任务约定的结构。
- `OfflineModel`：本地模型，基类会为同一实例的推理加锁。
- `@isolated(<worker_script>)`：在独立 Python 环境中启动 Worker，通过 IPC
  与主评测进程通信，适合依赖冲突或 GPU 模型。

Prompt 是由角色和内容组成的结构。每个内容项通常形如
`{"type": "text", "value": "..."}` 或
`{"type": "audio", "value": "/path/audio.wav"}`。适配器应验证输入类型，并
把框架异常转换为包含上下文的明确错误。

新增适配器的基本流程：

1. 在自己的包或 `audio_evals/models/` 中实现类。
2. 为模型增加 Registry YAML。
3. 先用小数据和 `--limit 1` 验证 Prompt、返回类型与资源释放。
4. 为初始化、输入校验和输出解析增加测试。

## 隔离环境

`@isolated` 模型的 Registry 通常包含：

```yaml
custom-offline-model:
  class: package.module.CustomOfflineModel
  args:
    path: organization/model
    env_path: evaluation_utils/envs/custom-model
    requirements_path: requirements/custom-model.txt
    sample_params: {}
```

隔离环境的依赖安装使用文件锁和完成标记；多个副本并发启动时，同一环境只会
准备一次。Worker 必须支持明确的就绪检查，才能配合
`--model-ready-policy required`。

## 支持多 GPU 模型池

模型池会检查适配器构造函数是否接受隔离层注入的 `gpu_id`。使用
`@isolated` 的适配器会获得该参数，并为每个子进程设置独立的
`CUDA_VISIBLE_DEVICES`。

适配器要安全支持模型池，应满足：

- 每个实例只拥有自己的模型进程和临时状态。
- Worker 就绪前不接受请求；推荐实现严格的就绪信号。
- `release` 或隔离清理逻辑可终止完整进程组。
- 模型下载与环境安装允许并发调用并使用进程锁。
- 不在模块导入阶段占用 GPU。

运行参数和启动策略见 [共享 Session 与多 GPU 推理](sessions.md)。

## 覆盖模型、Prompt 或任务

命令行中的 `--model` 会覆盖数据集默认任务里配置的模型。也可以同时指定
`--prompt`、`--evaluator`、`--agg` 与 `--post_process`，但正式
Benchmark 更推荐将这组配置登记为一个明确的评测任务，避免遗漏评测口径。

有关任务 Registry，见 [评测任务](tasks.md)。
