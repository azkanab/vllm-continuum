#!/usr/bin/env bash
set -euo pipefail
source "$(dirname "$0")/common.sh"

usage() {
  cat <<'USAGE'
Usage: prefetch.sh [--model ORG/NAME]

Creates the namespace, PVCs and HF token secret, then runs the prefetch Job that
downloads the model into the hf-cache PVC (once per namespace). It also pulls the
~10 GB image onto the node, so later runs start faster.
Default model: $MODEL from .env (meta-llama/Llama-3.1-8B-Instruct).
USAGE
}

if want_help "$@"; then usage; exit 0; fi
load_env
while (($#)); do
  case "$1" in
    --model) MODEL="$2"; shift 2 ;;
    *) usage; die "unknown argument $1" ;;
  esac
done
export MODEL
require_vars HF_TOKEN MODEL NAMESPACE IMAGE
need_cmd envsubst "brew install gettext"
check_context

apply_base
kcn delete job dagshare-prefetch --ignore-not-found --wait=true >/dev/null
kc apply -f "$RENDER_DIR/prefetch-job.yaml" >/dev/null
log "prefetching $MODEL into $NAMESPACE/dagshare-hf-cache (image pull can take 5-10 min)"
follow_job dagshare-prefetch "$LOCAL_LOG_DIR/prefetch.log" all || die "prefetch failed (see above; 401/403 = HF token or model access)"
log "prefetch done. next: make run PRESET=A3 WORKLOAD=smoke"
