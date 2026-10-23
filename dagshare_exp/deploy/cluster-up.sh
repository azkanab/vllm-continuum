#!/usr/bin/env bash
set -euo pipefail
source "$(dirname "$0")/common.sh"

usage() {
  cat <<'EOF'
Usage: cluster-up.sh [--print-size]

Creates (or reuses) a one-node Civo k3s cluster with one GPU in $CIVO_REGION,
installs the NVIDIA GPU operator and waits until the node offers nvidia.com/gpu.
Idempotent: if $CLUSTER_NAME exists, only its kubeconfig is fetched.
  --print-size   only print the GPU node size that would be used, then exit
Env: CIVO_TOKEN, CIVO_REGION (LON1), CLUSTER_NAME, GPU_SIZE (empty = auto-pick)
EOF
}

GPU_OPERATOR_VERSION="${GPU_OPERATOR_VERSION:-v25.10.1}"

pick_gpu_size() {
  local sizes_json
  sizes_json="$(mktemp)"
  civo sizes ls -o json >"$sizes_json"
  python3 - "$sizes_json" <<'PY'
import json, re, sys
rows = json.load(open(sys.argv[1]))
if isinstance(rows, dict):
    rows = rows.get("items") or rows.get("sizes") or list(rows.values())
def norm(r):
    return {str(k).lower().replace(" ", "_"): v for k, v in r.items()}
def num(v):
    try:
        return float(str(v).split()[0])
    except Exception:
        return 1e9
cands = []
for r in map(norm, rows):
    name = str(r.get("name", ""))
    text = " ".join(str(v) for v in r.values()).lower()
    if "kube" not in text:
        continue
    big = re.search(r"80\s*gb|a100[-_ ]?80", text)
    if "a100" in text and not big:
        rank = 0
    elif "l40s" in text:
        rank = 1
    elif "a100" in text:
        rank = 2
    else:
        continue
    cpu = num(r.get("cpu_cores", r.get("cpu", 0)))
    ram = num(r.get("ram_mb", r.get("ram", 0)))
    cands.append((rank, cpu, ram, name))
if not cands:
    sys.exit("no A100/L40S Kubernetes size found in `civo sizes ls`; set GPU_SIZE explicitly")
cands.sort()
for c in cands:
    print(f"candidate: {c[3]} (cpu={c[1]:g}, ram_mb={c[2]:g})", file=sys.stderr)
print(cands[0][3])
PY
  rm -f "$sizes_json"
}

if want_help "$@"; then usage; exit 0; fi
print_only=0
for arg in "$@"; do
  case "$arg" in
    --print-size) print_only=1 ;;
    *) usage; die "unknown option $arg" ;;
  esac
done

load_env
require_vars CIVO_TOKEN CLUSTER_NAME CIVO_REGION
need_cmd civo "brew install civo"
need_cmd kubectl
need_cmd helm "brew install helm"

civo apikey save dagshare "$CIVO_TOKEN" >/dev/null 2>&1 || true
civo apikey use dagshare >/dev/null
civo region use "$CIVO_REGION" >/dev/null

if [[ -z "$GPU_SIZE" ]]; then
  GPU_SIZE="$(pick_gpu_size)"
  log "auto-picked GPU size $GPU_SIZE (override with GPU_SIZE in .env)"
fi
if ((print_only)); then echo "$GPU_SIZE"; exit 0; fi

if civo kubernetes show "$CLUSTER_NAME" --region "$CIVO_REGION" >/dev/null 2>&1; then
  log "cluster $CLUSTER_NAME exists; fetching kubeconfig"
  civo kubernetes config "$CLUSTER_NAME" --region "$CIVO_REGION" --save --merge --switch
else
  log "creating cluster $CLUSTER_NAME ($GPU_SIZE, $CIVO_REGION); this bills from now on, remember 'make down'"
  civo kubernetes create "$CLUSTER_NAME" \
    --region "$CIVO_REGION" --cluster-type k3s -n 1 -s "$GPU_SIZE" \
    --create-firewall --firewall-rules 6443 \
    --wait --save --merge --switch
fi
[[ "$KUBE_CONTEXT" == "$CLUSTER_NAME" ]] || warn "KUBE_CONTEXT=$KUBE_CONTEXT differs from CLUSTER_NAME=$CLUSTER_NAME"
check_context

log "installing NVIDIA GPU operator $GPU_OPERATOR_VERSION"
helm repo add nvidia https://helm.ngc.nvidia.com/nvidia >/dev/null 2>&1 || true
helm repo update nvidia >/dev/null
helm upgrade --install gpu-operator nvidia/gpu-operator \
  --kube-context "$KUBE_CONTEXT" \
  --namespace gpu-operator --create-namespace \
  --version "$GPU_OPERATOR_VERSION" \
  --set driver.enabled=true \
  --set toolkit.enabled=false \
  --set devicePlugin.enabled=true \
  --set gfd.enabled=true \
  --set operator.defaultRuntime=containerd \
  --set validator.cuda.runtimeClassName=nvidia \
  --wait --timeout 15m

log "waiting for allocatable nvidia.com/gpu (driver build can take ~10 min)"
deadline=$((SECONDS + 1800))
until kc get nodes -o jsonpath='{.items[*].status.allocatable.nvidia\.com/gpu}' | grep -qE '[1-9]'; do
  ((SECONDS < deadline)) || die "no GPU after 30 min; check: kubectl --context $KUBE_CONTEXT -n gpu-operator get pods"
  kc -n gpu-operator get pods --no-headers 2>/dev/null | awk '$3!="Running" && $3!="Completed"{printf "  pending: %s %s\n", $1, $3}' || true
  sleep 30
done
log "GPU ready: $(kc get nodes -o jsonpath='{range .items[*]}{.metadata.name}={.status.allocatable.nvidia\.com/gpu}{" "}{end}')"
log "next: make check-gpu && make prefetch"
