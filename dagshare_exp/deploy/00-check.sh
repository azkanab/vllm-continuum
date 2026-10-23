#!/usr/bin/env bash
set -euo pipefail
source "$(dirname "$0")/common.sh"

usage() {
  cat <<'EOF'
Usage: 00-check.sh [--local] [--gpu-smoke]

Checks that this laptop can drive a DAGShare run on Civo.
  (default)    tools + env vars + kube context reachable + GPU allocatable
  --local      only tools and env vars (use before the cluster exists)
  --gpu-smoke  also run nvidia-smi in a pod and check the driver supports CUDA >= 12.8
EOF
}

if want_help "$@"; then usage; exit 0; fi
local_only=0; gpu_smoke=0
for arg in "$@"; do
  case "$arg" in
    --local) local_only=1 ;;
    --gpu-smoke) gpu_smoke=1 ;;
    *) usage; die "unknown option $arg" ;;
  esac
done

load_env
fail=0
ok() { printf '  ok    %s\n' "$*"; }
bad() { printf '  FAIL  %s\n' "$*"; fail=1; }
note() { printf '  note  %s\n' "$*"; }

echo "tools:"
for t in civo kubectl helm envsubst curl; do
  hint="brew install $t"; [[ "$t" != envsubst ]] || hint="brew install gettext"
  if command -v "$t" >/dev/null 2>&1; then ok "$t"; else bad "$t missing ($hint)"; fi
done
if have_yq; then ok "yq (mikefarah)"
elif have_pyyaml; then ok "python3 + pyyaml"
else bad "yq or python3+pyyaml missing (brew install yq)"; fi

echo "env ($DEPLOY_DIR/.env):"
for v in HF_TOKEN IMAGE NAMESPACE MODEL CLUSTER_NAME CIVO_REGION KUBE_CONTEXT; do
  if [[ -n "${!v:-}" ]]; then ok "$v"; else bad "$v is empty"; fi
done
if [[ -n "${CIVO_TOKEN:-}" ]]; then ok "CIVO_TOKEN"; else note "CIVO_TOKEN empty (fine on a shared cluster; needed for make up/down)"; fi
[[ "$CIVO_REGION" == "LON1" ]] || note "CIVO_REGION=$CIVO_REGION; Civo GPU Kubernetes is only offered in LON1"
note "NAMESPACE=$NAMESPACE KUBE_CONTEXT=$KUBE_CONTEXT IMAGE=$IMAGE"

if ((local_only)); then
  if ((fail)); then die "local checks failed"; fi
  echo "local checks passed"
  exit 0
fi

echo "cluster (context $KUBE_CONTEXT):"
if ! kubectl config get-contexts -o name 2>/dev/null | grep -qx "$KUBE_CONTEXT"; then
  bad "context $KUBE_CONTEXT not in kubeconfig (make up, or set KUBE_CONTEXT)"
elif ! kc get --raw /readyz --request-timeout=15s >/dev/null 2>&1; then
  bad "API server for $KUBE_CONTEXT not reachable"
else
  ok "API server reachable"
  gpus="$(kc get nodes -o jsonpath='{range .items[*]}{.metadata.name}={.status.allocatable.nvidia\.com/gpu}{"\n"}{end}')"
  if grep -qE '=[1-9]' <<<"$gpus"; then ok "allocatable GPUs: $(tr '\n' ' ' <<<"$gpus")"; else bad "no node reports nvidia.com/gpu yet ($(tr '\n' ' ' <<<"$gpus"))"; fi
  if kc get runtimeclass nvidia >/dev/null 2>&1; then ok "runtimeclass nvidia"; else bad "runtimeclass nvidia missing"; fi
  taints="$(kc get nodes -o jsonpath='{range .items[*]}{.spec.taints[*].key}{" "}{end}')"
  if [[ -n "${taints// /}" ]]; then note "node taints: $taints (set GPU_TOLERATION_KEY if the GPU node is tainted)"; fi
  notready="$(kc -n gpu-operator get pods --no-headers 2>/dev/null | awk '$3!="Running" && $3!="Completed"' || true)"
  if [[ -n "$notready" ]]; then note "gpu-operator pods not ready yet:"; printf '%s\n' "$notready" | sed 's/^/          /'; fi
fi

if ((gpu_smoke && fail == 0)); then
  echo "gpu smoke test:"
  render_base
  kc apply -f "$RENDER_DIR/namespace.yaml" >/dev/null
  kcn delete pod dagshare-gpu-smoke --ignore-not-found --wait=true >/dev/null
  kc apply -f "$RENDER_DIR/gpu-smoke-pod.yaml" >/dev/null
  if kcn wait --for=jsonpath='{.status.phase}'=Succeeded pod/dagshare-gpu-smoke --timeout=600s >/dev/null; then
    out="$(kcn logs dagshare-gpu-smoke)"
    printf '%s\n' "$out" | sed 's/^/          /'
    cuda="$(sed -n 's/.*CUDA Version: *\([0-9.]*\).*/\1/p' <<<"$out" | head -1)"
    if [[ -n "$cuda" ]] && awk -v v="$cuda" 'BEGIN{split(v,a,"."); exit !((a[1]>12)||(a[1]==12&&a[2]>=8))}'; then
      ok "driver supports CUDA $cuda (>= 12.8)"
    else
      bad "driver CUDA version '$cuda' < 12.8; vLLM v0.10.2 (torch 2.8) needs >= 12.8"
    fi
  else
    bad "gpu smoke pod did not succeed: kubectl -n $NAMESPACE describe pod dagshare-gpu-smoke"
  fi
  kcn delete pod dagshare-gpu-smoke --ignore-not-found --wait=false >/dev/null
fi

if ((fail)); then die "some checks failed"; fi
echo "all checks passed"
