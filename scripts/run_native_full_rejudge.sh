#!/usr/bin/env bash
# GPU-node: re-run evaluators on a finished full run, replaying its saved inference.
#   REJUDGE_SRC=<outputs/native_full/<run>>   source run (suite*.runtime.yaml + main/ tts/)
#   REJUDGE_BENCHMARKS=id1,id2                optional; default = every two_phase benchmark
#   REJUDGE_LIMIT=N                           optional; first N rows only (smoke check)
# Writes to $NATIVE_OUTPUT_DIR/{main,tts}; the source run is left untouched.
set -euo pipefail
ROOT="${VOXMATRIX_ROOT:-/mnt/afs/users/wangyl/VoxMatrix-clean}"
HERE="$ROOT/suites/native"
: "${REJUDGE_SRC:?set REJUDGE_SRC to the finished full run directory}"
: "${NATIVE_OUTPUT_DIR:?set NATIVE_OUTPUT_DIR}"
# shellcheck disable=SC1091
source "$HERE/env.sh"
cd "$ROOT"
PY="${VOXMATRIX_PYTHON}"
mkdir -p "$NATIVE_OUTPUT_DIR"

"$PY" - "$REJUDGE_SRC" "$NATIVE_OUTPUT_DIR" "${REJUDGE_BENCHMARKS:-}" "${REJUDGE_LIMIT:-0}" <<'PY'
import os, sys, yaml
src, out, wanted, limit = sys.argv[1], sys.argv[2], sys.argv[3], int(sys.argv[4] or 0)
wanted = {w.strip() for w in wanted.split(",") if w.strip()}
for suite, sub in (("suite.runtime.yaml", "main"), ("suite_tts.runtime.yaml", "tts")):
    path = os.path.join(src, suite)
    if not os.path.isfile(path):
        continue
    cfg = yaml.safe_load(open(path))
    picked = []
    for bench in cfg["benchmarks"]:
        bid = bench["benchmark_id"]
        if (bid not in wanted) if wanted else not bench.get("two_phase"):
            continue
        inf = os.path.join(src, sub, f"{bid}.jsonl")
        if not os.path.isfile(inf):
            sys.exit(f"missing inference jsonl: {inf}")
        # inf_file replays prompt+inference only, so every row is judged again.
        bench = dict(bench, inf_file=inf, two_phase=True)
        if limit:
            bench["limit"] = limit
        picked.append(bench)
    if not picked:
        continue
    cfg["benchmarks"] = picked
    cfg["output_root"] = os.path.join(out, sub)
    cfg["run_id"] = f"{cfg.get('run_id', 'native')}-rejudge"
    dst = os.path.join(out, f"suite_rejudge_{sub}.yaml")
    yaml.safe_dump(cfg, open(dst, "w"), sort_keys=False, allow_unicode=True)
    print(f"[rejudge] {dst}: {[b['benchmark_id'] for b in picked]}")
PY

found=0
for sub in main tts; do
  SUITE="$NATIVE_OUTPUT_DIR/suite_rejudge_${sub}.yaml"
  [[ -f "$SUITE" ]] || continue
  found=1
  echo "[rejudge] running $sub"
  "$PY" -m audio_evals.session --config "$SUITE" --phase evaluation
done
[[ $found == 1 ]] || { echo "[rejudge] no benchmarks selected" >&2; exit 1; }
echo "[rejudge] done"
ls -la "$NATIVE_OUTPUT_DIR/main" "$NATIVE_OUTPUT_DIR/tts" 2>/dev/null || true
