import argparse
import json
import logging
import os
import random
from datetime import datetime

from audio_evals.dataset.dataset import InMemoryDataset
from audio_evals.dataset.prepared import PreparedAudioJsonl
from audio_evals.eval_task import EvalTask, load_dataset_slice
from audio_evals.recorder import Recorder
from audio_evals.registry import registry
from audio_evals.utils import find_latest_jsonl
from audio_evals.models.model_pool import (
    IsolatedModelPool,
    get_available_gpu_ids,
    model_supports_pool,
)
from audio_evals.models.model import Model


class ReplayOnlyModel(Model):
    def __init__(self):
        super().__init__(is_chat=True)

    def _inference(self, prompt, **kwargs):
        raise RuntimeError("replay-only mode found a sample without saved inference")


def preload_dataset(dataset, *, limit=0, rand_size=0, timeout=0, offset=0):
    """Resolve one evaluation shard before any inference model is constructed."""
    rows = load_dataset_slice(
        dataset=dataset,
        limit=limit,
        timeout=timeout,
        offset=offset,
    )
    if rand_size:
        if rand_size > len(rows):
            raise ValueError(
                f"--rand {rand_size} exceeds the loaded sample count {len(rows)}"
            )
        rows = random.sample(rows, rand_size)
    return InMemoryDataset.from_dataset(dataset, rows)


def resolve_pool_config(args, gpu_ids):
    """Resolve independent replica, GPU-group and scheduler concurrency values."""
    gpus_per_replica = int(args.gpus_per_replica)
    if gpus_per_replica < 1:
        raise ValueError("--gpus-per-replica must be >= 1")
    capacity = len(gpu_ids) // gpus_per_replica
    if capacity < 1:
        raise RuntimeError(
            f"{len(gpu_ids)} visible GPUs cannot satisfy "
            f"--gpus-per-replica {gpus_per_replica}"
        )

    legacy_workers = args.workers
    if args.replicas:
        replicas = args.replicas
    elif legacy_workers is not None:
        # Backward compatibility: an explicitly supplied --workers used to
        # control both values.  New runs should use the independent flags.
        replicas = legacy_workers
    else:
        replicas = capacity
    inference_workers = (
        args.inference_workers or legacy_workers or replicas
    )
    if replicas < 1 or inference_workers < 1:
        raise ValueError("replicas and inference_workers must be >= 1")
    if replicas > capacity and not args.allow_gpu_sharing:
        raise ValueError(
            f"{replicas} replicas require {replicas * gpus_per_replica} GPU slots, "
            f"but only {len(gpu_ids)} GPU tokens are visible; use fewer replicas "
            "or explicitly pass --allow-gpu-sharing after a memory-capacity test"
        )
    return replicas, gpus_per_replica, inference_workers


def create_predictor(model_name, args, logger=None):
    """Construct one predictor (or pool) from the shared CLI/session options."""
    logger = logger or logging.getLogger(__name__)
    model_spec = registry._model.get(model_name, {})
    if args.use_model_pool == "auto":
        cls_path = model_spec.get("cls")
        use_pool = bool(cls_path and model_supports_pool(cls_path))
        logger.info(
            "Auto-detected %s model '%s'%s",
            "pool-compatible isolated" if use_pool else "single/API",
            model_name,
            f" ({cls_path})" if cls_path else "",
        )
    else:
        use_pool = args.use_model_pool == "on"

    inference_workers = args.inference_workers or args.workers or 1
    if not use_pool:
        return registry.get_model(model_name), inference_workers, False

    gpu_ids = get_available_gpu_ids()
    replicas, gpus_per_replica, inference_workers = resolve_pool_config(
        args, gpu_ids
    )
    logger.info(
        "Using IsolatedModelPool with replicas=%s, gpus_per_replica=%s, "
        "inference_workers=%s on GPU tokens %s",
        replicas,
        gpus_per_replica,
        inference_workers,
        gpu_ids,
    )
    predictor = IsolatedModelPool(
        model_factory=lambda **kwargs: registry.get_model(model_name, **kwargs),
        model_kwargs=model_spec.get("args", {}),
        gpu_ids=gpu_ids,
        replicas=replicas,
        gpus_per_replica=gpus_per_replica,
        allow_gpu_sharing=args.allow_gpu_sharing,
        startup_workers=args.model_startup_workers or None,
        ready_policy=args.model_ready_policy,
    )
    return predictor, inference_workers, True


def build_run_context(
    *,
    benchmark_id,
    model_name,
    predictor,
    inference_workers,
    evaluation_workers=1,
):
    """Describe execution settings required to interpret runtime metrics."""
    context = {
        "benchmark_id": str(benchmark_id),
        "model": str(model_name),
        "inference_workers": int(inference_workers),
        "evaluation_workers": int(evaluation_workers),
        "warmup_requests": 0,
        "request_mode": "unspecified",
    }
    if hasattr(predictor, "replicas"):
        context["replicas"] = int(predictor.replicas)
    if hasattr(predictor, "gpus_per_replica"):
        context["gpus_per_replica"] = int(predictor.gpus_per_replica)
    if getattr(predictor, "replica_metadata", None):
        context["replica_devices"] = [
            {
                "replica_id": item.get("replica_id"),
                "gpu_tokens": list(item.get("gpu_tokens") or []),
            }
            for item in predictor.replica_metadata
        ]
    return context


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--dataset_ref_col", default="")
    parser.add_argument("--model", required=True)
    parser.add_argument("--task", default="")
    parser.add_argument("--prompt", default="")
    parser.add_argument("--evaluator", default="")
    parser.add_argument("--agg", default="")
    parser.add_argument("--post_process", nargs="+", default=[])
    parser.add_argument("--save", default="")
    parser.add_argument("--registry_path", default="")
    parser.add_argument("--debug_mode", type=int, default=0)
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--offset", type=int, default=0)
    parser.add_argument("--rand", type=int, default=0)
    parser.add_argument(
        "--workers",
        type=int,
        default=None,
        help="Deprecated compatibility option. When explicitly used with a model "
        "pool it sets both replicas and inference workers.",
    )
    parser.add_argument(
        "--replicas",
        type=int,
        default=0,
        help="Model replica count. 0 auto-fills one replica per GPU group.",
    )
    parser.add_argument(
        "--gpus-per-replica",
        "--gpus_per_replica",
        dest="gpus_per_replica",
        type=int,
        default=1,
        help="Number of visible GPU tokens assigned to each model replica.",
    )
    parser.add_argument(
        "--inference-workers",
        "--inference_workers",
        dest="inference_workers",
        type=int,
        default=0,
        help="Concurrent sample scheduler threads. 0 defaults to replica count.",
    )
    parser.add_argument(
        "--evaluation-workers",
        "--evaluation_workers",
        dest="evaluation_workers",
        type=int,
        default=1,
        help="Second-stage evaluator concurrency in --two_phase mode (default 1).",
    )
    parser.add_argument(
        "--allow-gpu-sharing",
        "--allow_gpu_sharing",
        dest="allow_gpu_sharing",
        action="store_true",
        help="Allow multiple replicas to share a GPU token. Disabled by default.",
    )
    parser.add_argument(
        "--model-startup-workers",
        "--model_startup_workers",
        dest="model_startup_workers",
        type=int,
        default=0,
        help="Concurrent model startup calls. 0 starts all replicas concurrently.",
    )
    parser.add_argument(
        "--model-ready-policy",
        "--model_ready_policy",
        dest="model_ready_policy",
        choices=["auto", "required", "launch"],
        default="auto",
        help="required rejects legacy isolated wrappers without a strict ready hook.",
    )
    parser.add_argument("--dataset_load_timeout", type=int, default=0)
    parser.add_argument(
        "--prepared-manifest",
        "--prepared_manifest",
        dest="prepared_manifest",
        default="",
        help="Use a completed prepare-dataset manifest instead of loading the "
        "registry dataset source. The sibling _SUCCESS marker is required.",
    )
    parser.add_argument(
        "--dataset-load-order",
        "--dataset_load_order",
        choices=["before_model", "legacy_after_model"],
        default="before_model",
        help="Load/validate and sample the requested dataset shard before model "
        "initialization (default). legacy_after_model preserves the old order.",
    )
    parser.add_argument(
        "--use_model_pool",
        nargs="?",
        choices=["auto", "on", "off"],
        const="on",
        default="auto",
        help="Whether to use IsolatedModelPool for multi-GPU parallel inference. "
        "Default 'auto': inspect model class signature and use the pool iff the "
        "model accepts a `gpu_id` kwarg (i.e. @isolated offline models). "
        "Passing the flag bare (`--use_model_pool`) forces 'on' for backward "
        "compatibility; `--use_model_pool off` disables it. "
        "Replica count, GPUs per replica, and inference concurrency are configured "
        "independently. GPU sharing is rejected unless explicitly enabled.",
    )
    parser.add_argument(
        "-r",
        "--resume",
        nargs="?",
        type=str,
        const="latest",
        help="Reuse previous outputs & results, and run any "
        "missing jobs presented in the config. If its "
        "argument is not specified, the latest results in "
        "the work_dir will be reused. The argument should "
        "a valid file",
    )
    parser.add_argument("--inf_file", type=str, default="")
    parser.add_argument(
        "--two_phase",
        action="store_true",
        help="Run in two-phase mode: first parallel inference, then sequential evaluation. "
        "Useful when evaluator cannot run concurrently.",
    )
    parser.add_argument(
        "--replay-only",
        action="store_true",
        help="Evaluate saved inference without loading the configured inference model.",
    )

    args = parser.parse_args()
    return args


def main():
    time_id = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    args = get_args()
    for name in (
        "offset",
        "limit",
        "rand",
        "replicas",
        "inference_workers",
        "model_startup_workers",
        "dataset_load_timeout",
    ):
        if getattr(args, name) < 0:
            raise ValueError(f"{name} must be non-negative")
    if args.workers is not None and args.workers < 1:
        raise ValueError("workers must be >= 1 when specified")
    if args.gpus_per_replica < 1:
        raise ValueError("gpus_per_replica must be >= 1")
    if args.evaluation_workers < 1:
        raise ValueError("evaluation_workers must be >= 1")
    if args.replay_only and not args.resume:
        raise ValueError("--replay-only requires --resume")
    os.makedirs("log/", exist_ok=True)
    logging.basicConfig(
        level=logging.DEBUG if args.debug_mode else logging.INFO,
        format="%(asctime)s - %(name)s - %(levelname)s - %(filename)s:%(lineno)d- %(message)s",
        handlers=[
            logging.FileHandler(f"log/app-{time_id}.log"),
            logging.StreamHandler(),
        ],
    )
    logger = logging.getLogger(__name__)
    logger.propagate = False

    default_output_dir = os.path.join("outputs", args.model, args.dataset)
    if not args.save:
        os.makedirs(default_output_dir, exist_ok=True)
        args.save = os.path.join(default_output_dir, f"{time_id}.jsonl")
    else:
        if not args.save.endswith(".jsonl"):
            args.save = os.path.join(default_output_dir, f"{args.save}.jsonl")
        save_dir = os.path.dirname(args.save)
        if save_dir:
            os.makedirs(save_dir, exist_ok=True)
    overall_save = args.save.replace(".jsonl", "-overall.json")

    if args.registry_path:
        paths = args.registry_path.split()
        registry.add_registry_paths(paths)

    dataset = registry.get_dataset(args.dataset)
    if dataset is None:
        raise KeyError(f"dataset is not registered: {args.dataset}")
    if args.prepared_manifest:
        dataset = PreparedAudioJsonl(
            f_name=args.prepared_manifest,
            default_task=dataset.task_name,
            ref_col=dataset.ref_col,
            col_aliases=dataset.col_aliases,
        )
        logger.info("Using prepared dataset manifest: %s", args.prepared_manifest)
    if args.resume:
        if args.resume == "latest":
            if os.path.exists(args.save):
                args.resume = args.save
            else:
                save_path = os.path.dirname(args.save)
                last_file = find_latest_jsonl(save_path)
                if last_file is None:
                    raise ValueError(
                        "No previous results found, make {} exist `jsonl` file".format(
                            save_path
                        )
                    )
                args.resume = last_file
        logger.info(f"Resuming from {args.resume}")
        dataset = dataset.resume_from(args.resume)
    if args.dataset_ref_col:
        dataset.reset_ref_col(args.dataset_ref_col)
        logger.info("reset ref col {}".format(dataset))
    if args.inf_file:
        dataset = dataset.load_inf_file(args.inf_file)
        logger.info("loaded inference file: {}".format(dataset))

    task_cfg = registry.get_eval_task(dataset.task_name)
    if args.task:
        task_cfg = registry.get_eval_task(args.task)

    attrs = dir(task_cfg)
    for attr in dir(args):
        if not attr.startswith("__") and attr in attrs and getattr(args, attr):
            setattr(task_cfg, attr, getattr(args, attr))
    logger.info("task cfg:\n{}".format(task_cfg))

    run_limit = args.limit
    run_rand = args.rand
    run_dataset_timeout = args.dataset_load_timeout
    run_dataset_offset = args.offset
    if args.dataset_load_order == "before_model":
        logger.info(
            "Loading dataset before model initialization (offset=%s, limit=%s)",
            args.offset,
            args.limit,
        )
        dataset = preload_dataset(
            dataset,
            limit=args.limit,
            rand_size=args.rand,
            timeout=args.dataset_load_timeout,
            offset=args.offset,
        )
        logger.info("Dataset is ready in memory with %s rows", len(dataset.rows))
        # The shard and random selection have already been applied.  Keep the
        # global event offset so resume/event IDs remain compatible.
        run_limit = 0
        run_rand = 0
        run_dataset_timeout = 0
        run_dataset_offset = 0

    inference_workers = args.inference_workers or args.workers or 1
    if args.replay_only:
        logger.info("Replay-only mode enabled; skipping inference model loading")
        predictor = ReplayOnlyModel()
    else:
        predictor, inference_workers, _ = create_predictor(
            task_cfg.model, args, logger
        )

    # evaluator = registry.get_evaluator(task_cfg.evaluator)

    if args.two_phase:
        logger.info(
            "Two-phase mode enabled: parallel inference then sequential evaluation"
        )

    try:
        t = EvalTask(
            dataset=dataset,
            prompt=registry.get_prompt(task_cfg.prompt),
            predictor=predictor,
            evaluator=task_cfg.evaluator,
            post_process=[registry.get_process(item) for item in task_cfg.post_process],
            agg=registry.get_agg(task_cfg.agg),
            recorder=Recorder(args.save),
            event_id_offset=args.offset,
            run_context=build_run_context(
                benchmark_id=args.dataset,
                model_name=args.model,
                predictor=predictor,
                inference_workers=inference_workers,
                evaluation_workers=args.evaluation_workers,
            ),
        )
        res = t.run(
            run_limit,
            run_rand,
            inference_workers,
            args.two_phase,
            run_dataset_timeout,
            run_dataset_offset,
            args.evaluation_workers,
        )
        with open(overall_save, "w", encoding="utf-8") as f:
            json.dump(res[0], f, ensure_ascii=False, indent=2)
            f.write("\n")
        with open(args.save, "r") as f:
            print(f.read())
        print(res[0])
        print(f"Results saved to {args.save}")
    finally:
        if hasattr(predictor, "release") and callable(predictor.release):
            predictor.release()


# Press the green button in the gutter to run the script.
if __name__ == "__main__":
    main()
