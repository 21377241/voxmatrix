# VoxMatrix 文档

根目录 [README](../README_zh.md) 只包含安装和最短上手流程；扩展配置与运行
细节放在这里。

- [数据集与一次性 WAV 准备](datasets.md)
- [数据标注与 Manifest 构建](annotation.md)
- [模型适配器与 Registry 配置](models.md)
- [评测任务、输出和断点续跑](tasks.md)
- [共享 Session 与多 GPU 推理](sessions.md)
- [指标及 Evaluator 依赖](metrics.md)
- [常见问题排查](troubleshooting.md)

配置的事实来源是 `registry/`、`mesh_eval/registry/` 与
`mesh_eval/config/`。文档示例使用 VoxMatrix 的规范命令；旧入口仅用于兼容
已有自动化。
