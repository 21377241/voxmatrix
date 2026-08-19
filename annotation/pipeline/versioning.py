"""标注产物版本化：配置哈希 + run_manifest。"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from annotation.pipeline.common import ANNOTATION_ROOT, MAPPINGS_DIR, SCHEMA_DIR


def file_sha256(path: Path) -> str | None:
    if not path.is_file():
        return None
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(65536), b""):
            h.update(chunk)
    return h.hexdigest()[:16]


def build_config_fingerprint(dataset_id: str) -> dict[str, Any]:
    mapping = MAPPINGS_DIR / f"{dataset_id}.yaml"
    knowledge = ANNOTATION_ROOT / "knowledge" / "datasets" / f"{dataset_id}.yaml"
    prompts = {
        p.name: file_sha256(p)
        for p in sorted((ANNOTATION_ROOT / "prompts").glob("*.txt"))
    }
    return {
        "dataset_id": dataset_id,
        "mapping_sha256_16": file_sha256(mapping),
        "knowledge_src_sha256_16": file_sha256(knowledge),
        "taxonomy_sha256_16": file_sha256(SCHEMA_DIR / "taxonomy.yaml"),
        "spec_131_sha256_16": file_sha256(SCHEMA_DIR / "spec_131.yaml"),
        "manifest_schema_sha256_16": file_sha256(SCHEMA_DIR / "manifest.schema.json"),
        "taxonomy_v2_sha256_16": file_sha256(SCHEMA_DIR / "taxonomy.v2.yaml"),
        "evaluation_sample_v2_schema_sha256_16": file_sha256(
            SCHEMA_DIR / "evaluation_sample.v2.schema.json"
        ),
        "prompts_sha256_16": prompts,
        "annotation_model": os.environ.get(
            "VOXMATRIX_ANNOTATION_MODEL",
            os.environ.get("CHATANYWHERE_MODEL", "gpt-4o-mini"),
        ),
    }


def make_run_id(dataset_id: str, fingerprint: dict[str, Any]) -> str:
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    short = (fingerprint.get("mapping_sha256_16") or "nomap")[:8]
    return f"{dataset_id}_{stamp}_{short}"


def write_run_manifest(
    work_dir: Path,
    dataset_id: str,
    *,
    stages: dict[str, Any] | None = None,
    extra: dict[str, Any] | None = None,
) -> Path:
    """写入 work_dir/run_manifest.json，并归档到 runs/{run_id}/。"""
    from annotation.ai.client import load_dotenv

    load_dotenv()
    fp = build_config_fingerprint(dataset_id)
    run_id = make_run_id(dataset_id, fp)
    manifest = {
        "run_id": run_id,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "dataset_id": dataset_id,
        "config_fingerprint": fp,
        "stages": stages or {},
        "extra": extra or {},
        "policy": {
            "p0": ["prelabel", "collect_knowledge", "verify_prelabel", "assign_bucket"],
            "p1": ["mapping_align", "subset_policy", "cost_control", "versioning"],
        },
    }

    work_dir.mkdir(parents=True, exist_ok=True)
    out = work_dir / "run_manifest.json"
    with open(out, "w", encoding="utf-8") as f:
        json.dump(manifest, f, ensure_ascii=False, indent=2)

    run_dir = work_dir / "runs" / run_id
    run_dir.mkdir(parents=True, exist_ok=True)
    with open(run_dir / "run_manifest.json", "w", encoding="utf-8") as f:
        json.dump(manifest, f, ensure_ascii=False, indent=2)

    changelog = work_dir / "CHANGELOG.md"
    header = "# Annotation Run Changelog\n\n"
    line = (
        f"- `{run_id}` model={fp.get('annotation_model')} "
        f"mapping={fp.get('mapping_sha256_16')} taxonomy={fp.get('taxonomy_sha256_16')}\n"
    )
    prev = changelog.read_text(encoding="utf-8") if changelog.exists() else ""
    if not prev.startswith("# "):
        changelog.write_text(header + prev + line, encoding="utf-8")
    else:
        with open(changelog, "a", encoding="utf-8") as f:
            f.write(line)

    return out


def main() -> None:
    parser = argparse.ArgumentParser(description="生成 run_manifest 版本指纹")
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--work-dir", type=Path, required=True)
    args = parser.parse_args()
    path = write_run_manifest(args.work_dir, args.dataset)
    print(f"run_manifest: {path}")
    print(json.dumps(json.load(open(path)), ensure_ascii=False, indent=2)[:800])


if __name__ == "__main__":
    main()
