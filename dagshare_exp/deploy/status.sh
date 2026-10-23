#!/usr/bin/env bash
set -euo pipefail
source "$(dirname "$0")/common.sh"

usage() {
  cat <<'USAGE'
Usage: status.sh

Shows the cluster context, GPU node(s), GPU operator health, and the Jobs, pods
and PVCs in $NAMESPACE.
USAGE
}

if want_help "$@"; then usage; exit 0; fi
load_env
check_context
echo "context $KUBE_CONTEXT, namespace $NAMESPACE, image $IMAGE"
echo
echo "== nodes"
kc get nodes -o custom-columns='NAME:.metadata.name,STATUS:.status.conditions[?(@.type=="Ready")].status,GPU:.status.allocatable.nvidia\.com/gpu,CPU:.status.allocatable.cpu,MEM:.status.allocatable.memory,GPU_PRODUCT:.metadata.labels.nvidia\.com/gpu\.product'
echo
echo "== gpu-operator pods not Running/Completed"
kc -n gpu-operator get pods --no-headers 2>/dev/null | awk '$3!="Running" && $3!="Completed"' || true
echo
echo "== jobs in $NAMESPACE"
kcn get jobs -o custom-columns='NAME:.metadata.name,PRESET:.metadata.labels.dagshare/preset,RETENTION:.metadata.labels.dagshare/retention,SEED:.metadata.labels.dagshare/seed,DONE:.status.succeeded,FAILED:.status.failed,START:.status.startTime,END:.status.completionTime' 2>/dev/null || true
echo
echo "== pods in $NAMESPACE"
kcn get pods -o wide 2>/dev/null || true
echo
echo "== pvcs in $NAMESPACE"
kcn get pvc 2>/dev/null || true
echo
echo "== GPU jobs from other namespaces (shared cluster)"
kc get pods -A -l app.kubernetes.io/component=run --field-selector=status.phase!=Succeeded,status.phase!=Failed \
  -o custom-columns='NAMESPACE:.metadata.namespace,POD:.metadata.name,PHASE:.status.phase' 2>/dev/null || true
