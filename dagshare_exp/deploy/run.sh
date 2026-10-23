#!/usr/bin/env bash
set -euo pipefail
source "$(dirname "$0")/common.sh"

usage() {
  cat <<'USAGE'
Usage: run.sh --preset P [--retention R] [--seed K] [--workload W] [--matrix-name M] [--dry-run]

Runs one experiment (preset x retention x seed) as a Kubernetes Job and waits for it.
  --preset       vllm | continuum | A0 | A1 | A2 | A3 (configs/presets/<P>.yaml)
  --retention    none | ttl[:s] | until_done[:cap]; default: the preset's own retention
                 (ttl means ttl:2, until_done means until_done:60)
  --seed         integer, default 1
  --workload     configs/workloads/<W>.yaml, default smoke
  --matrix-name  results group, default adhoc
  --dry-run      render the Job manifest and print its path, apply nothing
Results land on the results PVC under <matrix>/<preset>_<none|ttl2|until_done60>_seed<k>/.
Env knobs: SERVER_CPUS SERVER_MEM LOADGEN_CPUS LOADGEN_MEM LOADGEN_CMD VLLM_EXTRA_ARGS
           WARMUP_REQUESTS DEADLINE_MARGIN_S LOG_FOLLOW=all|loadgen|none
Exit code is non-zero if the run fails.
USAGE
}

if want_help "$@"; then usage; exit 0; fi
preset=""; retention=""; seed=1; workload="smoke"; matrix="adhoc"; dry_run=0
while (($#)); do
  case "$1" in
    --preset) preset="$2"; shift 2 ;;
    --retention) retention="$2"; shift 2 ;;
    --seed) seed="$2"; shift 2 ;;
    --workload) workload="$2"; shift 2 ;;
    --matrix-name) matrix="$2"; shift 2 ;;
    --dry-run) dry_run=1; shift ;;
    *) usage; die "unknown argument $1" ;;
  esac
done
[[ -n "$preset" ]] || { usage; die "--preset is required"; }

load_env
need_cmd envsubst "brew install gettext"
require_yaml_tool

if ((dry_run)); then
  IMAGE_DIGEST="${IMAGE_DIGEST:-unknown}"
  compute_run_vars "$preset" "$retention" "$seed" "$workload" "$matrix"
  render_run
  exit 0
fi

require_vars NAMESPACE IMAGE
check_context
[[ -n "${IMAGE_DIGEST:-}" ]] || pin_image
compute_run_vars "$preset" "$retention" "$seed" "$workload" "$matrix"
[[ "$RUN_MODEL" == "$MODEL" ]] || warn "workload model $RUN_MODEL != MODEL=$MODEL; was it prefetched?"

apply_base
manifest="$(render_run)"
log "run $RUN_PATH: policy=$POLICY KV_RETENTION=$RETENTION seed=$SEED workload=$WORKLOAD deadline=${ACTIVE_DEADLINE_S}s"
kcn delete job "$JOB_NAME" --ignore-not-found --wait=true >/dev/null
kc apply -f "$manifest" >/dev/null
log "submitted job $JOB_NAME in $NAMESPACE"

logfile="$LOCAL_LOG_DIR/$MATRIX_NAME/$RUN_NAME.log"
if follow_job "$JOB_NAME" "$logfile" "${LOG_FOLLOW:-all}"; then
  log "results: PVC dagshare-results:/$RUN_PATH (fetch with: make fetch)"
  exit 0
fi
log "run $RUN_PATH failed; full log in $logfile; partial outputs under $RUN_PATH/failed on the PVC"
exit 1
