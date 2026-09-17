![VoxMatrix](assets/voxmatrix_logo.svg)

# VoxMatrix

A unified framework for evaluating speech and audio models.

[中文说明](README_zh.md) · [Documentation](docs/README.md) · [Capability audit notes](docs/capability_audit/README.md) · [Compatibility](COMPATIBILITY.md) · [Third-party notices](THIRD_PARTY_NOTICES.md)

VoxMatrix combines registry-driven datasets, models, prompts, evaluators, and
aggregation policies with a canonical mesh schema. It supports speech
recognition, audio understanding, speech generation, agent behavior, speaker
tasks, safety checks, and runtime measurements.

![VoxMatrix architecture](assets/voxmatrix_framework.png)

## Highlights

- Convert Hugging Face audio datasets to stable WAV assets once and reuse them
  across evaluations.
- Run one isolated model replica per GPU for sample-level data-parallel
  inference. Replica startup is concurrent by default.
- Evaluate several compatible benchmarks in one session while loading the
  predictor only once.
- Load every requested dataset before occupying GPUs, resume interrupted runs,
  and separate inference from evaluator-heavy post-processing when needed.
- Extend datasets, models, tasks, prompts, evaluators, and aggregators through
  YAML registries.

Model checkpoints, full datasets, generated audio, and evaluation results are
not stored in this repository.

## Installation

Python 3.10 or newer is required.

```bash
cd VoxMatrix
conda create -n voxmatrix python=3.10 -y
conda activate voxmatrix
pip install -e .
```

With `uv`:

```bash
uv venv .venv --python 3.10
source .venv/bin/activate
uv pip install -e .
```

List the bundled registry entries from a source checkout:

```bash
python cli/list_available.py --datasets
python cli/list_available.py --models
```

Some offline models and evaluators use isolated environments with additional
requirements. Their registry entries specify the environment, checkpoint, and
requirements paths.

## Quick start

Run one registered dataset/model pair:

```bash
voxmatrix \
  --dataset <dataset_name> \
  --model <model_name> \
  --limit 10
```

By default, per-sample events are written to
`outputs/<model>/<dataset>/<timestamp>.jsonl`, with the aggregate result in the
adjacent `<timestamp>-overall.json` file. Use `--save <path>.jsonl` to
choose another result path.

### Prepare Hugging Face audio once

Materialize decoded audio before requesting GPUs:

```bash
voxmatrix-prepare \
  --dataset <dataset_name> \
  --output-dir prepared/<dataset_name>

voxmatrix \
  --dataset <dataset_name> \
  --model <model_name> \
  --prepared-manifest prepared/<dataset_name>/manifest.jsonl
```

The prepared directory contains WAV assets, `manifest.jsonl`,
`metadata.json`, and a checked `_SUCCESS` marker. See
[Dataset preparation](docs/datasets.md).

### Use one model replica per GPU

Pool-compatible isolated adapters can distribute samples across independent
model replicas:

```bash
CUDA_VISIBLE_DEVICES=0,1 voxmatrix \
  --dataset <dataset_name> \
  --model <model_name> \
  --use_model_pool on \
  --replicas 2 \
  --gpus-per-replica 1 \
  --inference-workers 2 \
  --model-startup-workers 2
```

`--use_model_pool auto` is the default and enables the pool only for a
compatible adapter. GPU sharing is rejected unless explicitly enabled.
Read [Sessions and parallel inference](docs/sessions.md) before changing
replica, GPU-group, or readiness settings.

### Reuse one predictor across benchmarks

Copy and edit [examples/suite.example.yaml](examples/suite.example.yaml), then
run:

```bash
voxmatrix-suite --config examples/suite.example.yaml
```

All suite datasets are prepared first. The model is then constructed once,
reused across benchmarks, and released at the end of the session.

## Documentation

- [Datasets and one-time WAV preparation](docs/datasets.md)
- [Dataset annotation and manifest construction](docs/annotation.md)
- [Model adapters and registry configuration](docs/models.md)
- [Evaluation tasks, results, and resume behavior](docs/tasks.md)
- [Shared sessions and multi-GPU inference](docs/sessions.md)
- [Metrics and evaluator dependencies](docs/metrics.md)
- [Troubleshooting](docs/troubleshooting.md)

## Compatibility

`voxmatrix`, `voxmatrix-prepare`, and `voxmatrix-suite` are the canonical
commands. Legacy command aliases, the `audio_evals` and `mesh_eval` Python
packages, serialized class paths, and matching `ULTRAEVAL_*` environment
variables remain available for existing automation. New integrations should
use the VoxMatrix names. See [COMPATIBILITY.md](COMPATIBILITY.md).

## License and attribution

VoxMatrix is distributed under the [Apache License 2.0](LICENSE). Vendored
inference components retain their upstream notices and may have additional
terms; review [THIRD_PARTY_NOTICES.md](THIRD_PARTY_NOTICES.md) before
redistribution.

VoxMatrix builds on the execution core of UltraEval-Audio. If that work is
relevant to your use, cite the [original paper](https://arxiv.org/abs/2601.01373):

```bibtex
@article{ultraevalaudio,
  title={UltraEval-Audio: A Unified Framework for Comprehensive Evaluation of Audio Foundation Models},
  author={Qundong Shi and Jie Zhou and Biyuan Lin and Junbo Cui and Guoyang Zeng and Yixuan Zhou and Ziyang Wang and Xin Liu and Zhen Luo and Yudong Wang and Zhiyuan Liu},
  year={2026},
  eprint={2601.01373},
  archivePrefix={arXiv},
  primaryClass={cs.SD},
  url={https://arxiv.org/abs/2601.01373}
}
```
