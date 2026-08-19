# 指标与 Evaluator

指标路由的事实来源是
`mesh_eval/config/metric_evaluator_catalog.yaml`，Evaluator 注册项位于
`registry/evaluator/` 与 `mesh_eval/registry/evaluator/`。不要在文档中
维护一份独立的手工清单；发布前应直接校验并按需从配置生成表格。

## 状态含义

| 状态 | 含义 |
| --- | --- |
| `ready_cpu` | 当前 CPU 环境可直接运行 |
| `diagnostic_cpu` | 可运行，但属于近似或诊断口径，正式报告需说明 |
| `requires_api` | 需要网络和 API 凭据 |
| `requires_offline_model` | 需要额外离线权重及对应环境 |
| `requires_tool` | 需要额外系统工具或 Python 包 |
| `pending_external_backend` | 已纳入设计，但仓库尚无可用后端 |

## 指标类别

- 文本与识别：WER、CER、BLEU、ChrF、MER、Exact Match、Text F1、
  ROUGE-L、COMET、BERTScore。
- 分类与结构化输出：Accuracy、Macro F1、JSON Validity、Tool/Parameter
  Accuracy、Intent/Slot F1。
- 语音输出：内容正确率、可懂度、说话人相似度、UTMOS、DNSMOS。
- 说话人与边端行为：DER、JER、EER、AUC、归因、授权与误执行。
- 运行时：延迟分位数、首 Token/首音频块延迟、RTF、超时率、失败率、峰值
  内存与设备利用率。
- 安全与 LLM Judge：攻击成功率、开放式问答和带参考答案的 Judge。

实际名称、Evaluator、依赖和状态以 Catalog 为准。

## 校验与导出

```bash
python -m mesh_eval.scripts.validate_metric_catalog
```

生成当前版本的 Markdown 表：

```bash
python -m mesh_eval.scripts.validate_metric_catalog \
  --markdown /tmp/voxmatrix-metrics.md
```

该命令会检查每个 Evaluator 是否已注册、依赖状态是否完整，以及指标是否存在于
`mesh-router-evaluator` 的路由表。

## 选择评测口径

传统任务通过 EvalTask 的 `evaluator` 和 `agg` 选择实现：

```yaml
custom-asr-task:
  class: audio_evals.base.EvalTaskCfg
  args:
    dataset: custom-asr
    prompt: asr
    model: custom-model
    evaluator: wer
    agg: wer
```

Mesh 样本可以在 `metrics` 中声明规范指标名，由 Router 根据语言、任务和
Catalog 选择 Evaluator。正式结果应记录：

- 指标名称和方向。
- Catalog/Registry 版本。
- 文本规范化、语言 Tokenizer 和聚合方式。
- 外部模型、权重 Revision、API 模型版本和系统工具版本。
- 诊断实现或缺失值处理规则。

## 语音质量指标

UTMOS、DNSMOS、说话人相似度与 ASR 可懂度不是零依赖指标。运行前应：

1. 在相应模型 Registry 中配置权重与隔离环境。
2. 检查采样率、声道和参考音频字段。
3. 用少量样本验证输出范围。
4. 对正式报告固定权重版本和聚合规则。

这些权重不随源码仓库分发。
