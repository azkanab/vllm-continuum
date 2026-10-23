#!/usr/bin/env bash
set -euo pipefail
source "$(dirname "$0")/common.sh"

usage() {
  cat <<'EOF'
Usage: cluster-down.sh [--yes] [--namespace-only] [--all-namespaces]

Stops billing: deletes the DAGShare namespace(s) and their PVCs (Civo volumes),
deletes the Civo cluster, then lists leftover volumes, which keep billing.
Fetch your results first (make fetch): they live on the results PVC.
  --yes              do not ask for confirmation
  --namespace-only   delete only $NAMESPACE, keep the cluster (shared cluster users)
  --all-namespaces   delete every namespace labelled app.kubernetes.io/part-of=dagshare
EOF
}

if want_help "$@"; then usage; exit 0; fi
yes=0; ns_only=0; all_ns=0
for arg in "$@"; do
  case "$arg" in
    --yes | -y) yes=1 ;;
    --namespace-only) ns_only=1 ;;
    --all-namespaces) all_ns=1 ;;
    *) usage; die "unknown option $arg" ;;
  esac
done

load_env
need_cmd kubectl

confirm() {
  ((yes)) && return 0
  local answer
  read -r -p "$1 [type 'yes']: " answer
  [[ "$answer" == "yes" ]] || die "aborted"
}

reachable=0
if kubectl config get-contexts -o name 2>/dev/null | grep -qx "$KUBE_CONTEXT" \
  && kc get --raw /readyz --request-timeout=15s >/dev/null 2>&1; then
  reachable=1
fi

if ((reachable)); then
  namespaces="$NAMESPACE"
  if ((all_ns)); then
    namespaces="$(kc get ns -l app.kubernetes.io/part-of=dagshare -o jsonpath='{.items[*].metadata.name}')"
  fi
  if [[ -n "${namespaces// /}" ]]; then
    confirm "Delete namespace(s) [$namespaces] and their PVCs (results not fetched are lost) on $KUBE_CONTEXT?"
    for ns in $namespaces; do
      log "deleting PVCs and namespace $ns"
      kc -n "$ns" delete jobs,pods --all --ignore-not-found --wait=false >/dev/null 2>&1 || true
      kc -n "$ns" delete pvc --all --ignore-not-found --timeout=300s || warn "PVC delete in $ns timed out"
      kc delete namespace "$ns" --ignore-not-found --timeout=300s || warn "namespace $ns still terminating"
    done
  fi
else
  warn "context $KUBE_CONTEXT not reachable; skipping namespace cleanup"
fi

if ((ns_only)); then
  log "kept cluster $CLUSTER_NAME (--namespace-only)"
  exit 0
fi

need_cmd civo "brew install civo"
require_vars CIVO_TOKEN
civo apikey save dagshare "$CIVO_TOKEN" >/dev/null 2>&1 || true
civo apikey use dagshare >/dev/null

if civo kubernetes show "$CLUSTER_NAME" --region "$CIVO_REGION" >/dev/null 2>&1; then
  confirm "Delete Civo cluster $CLUSTER_NAME in $CIVO_REGION?"
  civo kubernetes remove "$CLUSTER_NAME" --region "$CIVO_REGION" --yes
  log "cluster $CLUSTER_NAME deleted"
else
  log "cluster $CLUSTER_NAME not found in $CIVO_REGION"
fi
if [[ "$KUBE_CONTEXT" == "$CLUSTER_NAME" ]]; then kubectl config delete-context "$KUBE_CONTEXT" >/dev/null 2>&1 || true; fi

log "volumes still on the account (these keep billing \$0.11/GB-month; delete with 'civo volume rm <name>'):"
civo volume ls --region "$CIVO_REGION" || true
