#!/usr/bin/env bash
set -euo pipefail
source "$(dirname "$0")/common.sh"

usage() {
  cat <<'USAGE'
Usage: render.sh [base|run|all] [--preset P] [--retention R] [--seed K] [--workload W] [--matrix-name M]

Renders the k8s templates in k8s/ with envsubst into .rendered/ (gitignored).
  base   namespace, PVCs, HF secret, prefetch job, fetch pod, GPU smoke pod
  run    the run Job for one preset/retention/seed (defaults: A3, preset retention, 1, smoke)
  all    both (default)
Nothing is applied to any cluster.
USAGE
}

if want_help "$@"; then usage; exit 0; fi
what="all"; preset="A3"; retention=""; seed=1; workload="smoke"; matrix="adhoc"
while (($#)); do
  case "$1" in
    base | run | all) what="$1"; shift ;;
    --preset) preset="$2"; shift 2 ;;
    --retention) retention="$2"; shift 2 ;;
    --seed) seed="$2"; shift 2 ;;
    --workload) workload="$2"; shift 2 ;;
    --matrix-name) matrix="$2"; shift 2 ;;
    *) usage; die "unknown argument $1" ;;
  esac
done

load_env
need_cmd envsubst "brew install gettext"
require_yaml_tool
HF_TOKEN="${HF_TOKEN:-}"

if [[ "$what" != run ]]; then
  render_base
  log "rendered base manifests into $RENDER_DIR"
fi
if [[ "$what" != base ]]; then
  IMAGE_DIGEST="${IMAGE_DIGEST:-unknown}"
  compute_run_vars "$preset" "$retention" "$seed" "$workload" "$matrix"
  log "rendered run job: $(render_run)"
fi
