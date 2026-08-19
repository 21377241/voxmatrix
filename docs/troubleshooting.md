# 常见问题排查

## CUDA 动态库符号不匹配

若出现 `libcusparse.so`、`libnvJitLink.so` 或
`__nvJitLinkAddData_*` 符号错误，通常是驱动、PyTorch CUDA 版本和当前
`LD_LIBRARY_PATH` 混用了不同运行时。

先记录以下信息：

```bash
nvidia-smi
python -c "import torch; print(torch.__version__, torch.version.cuda)"
python -c "import nvidia.nvjitlink; print(nvidia.nvjitlink.__path__)"
```

使用与 PyTorch 匹配的 NVIDIA Python 库路径，或重装匹配版本。不要把其他
环境的 CUDA 路径全局写入 Shell 配置；隔离模型应在对应 `env_path` 中修复。

## Hugging Face 数据无法访问

`LocalEntryNotFoundError`、ConnectionError 或空缓存通常表示网络不可达、
数据集 Revision 不存在，或本地快照不完整。

- 先用 `huggingface-cli whoami` 和最小 `datasets.load_dataset` 调用验证。
- 对受限数据集先在 Hugging Face 页面申请权限并登录。
- 使用镜像时显式设置 `HF_ENDPOINT`，并确认镜像包含需要的 Revision。
- 正式运行固定 `revision`，避免远端默认分支变化。

## 受限数据集加载时报空对象

GigaSpeech 等数据集需要单独授权。先在 Python 中直接加载目标配置和 Split；
只有该步骤成功后再运行 VoxMatrix。框架不会绕过数据集许可。

## Prepared Manifest 被拒绝

常见原因是缺少 `_SUCCESS`、Manifest/Metadata 摘要变化，或 WAV 被移动。
应保留整个准备目录，不要手工编辑 `manifest.jsonl`。源数据或转换配置变化
后，在新的输出目录重新运行 `voxmatrix-prepare`。

## 模型池未自动启用

`--use_model_pool auto` 只识别接受隔离层 `gpu_id` 的适配器。先检查日志中的
自动检测结果，再确认：

- 模型类使用兼容的 `@isolated` 包装。
- `CUDA_VISIBLE_DEVICES` 至少暴露一个有效 GPU Token。
- `replicas * gpus_per_replica` 没有超过可见容量。

调试时可用 `--use_model_pool on` 强制验证；不兼容的普通模型不应强制开启。

## 多卡加载看起来仍然串行

默认所有副本并发启动。若日志时间线串行，检查是否显式设置了
`--model-startup-workers 1`，以及模型下载、隔离环境文件锁或存储带宽是否
成为共享瓶颈。环境安装只执行一次，因此首次运行可能有一个受锁保护的准备
阶段；后续模型进程仍会并发启动。

## 断点文件找不到

无参数 `--resume` 只搜索当前 `--save` 所在目录。最可靠的方式是同时指定
相同的 `--save` 与 `--resume <exact.jsonl>`。不要用聚合 JSON 替代逐样本
JSONL。

## Registry 条目找不到

检查外部 Registry 是否包含正确的子目录和 YAML 扩展名。单次运行参数名是
`--registry_path`，数据准备参数名是 `--registry-path`，Session 则使用
顶层 `registry_paths` 列表。

## 提交 Issue 时应附带什么

请提供最小命令、VoxMatrix Commit、Python/PyTorch/CUDA 版本、相关 Registry
条目、完整错误栈，以及不含凭据和个人路径的日志片段。不要上传模型权重、受限
数据或 API Key。
