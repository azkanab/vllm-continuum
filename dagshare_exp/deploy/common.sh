# shellcheck shell=bash
# Shared helpers sourced by every deploy script.

DEPLOY_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
EXP_DIR="$(cd "$DEPLOY_DIR/.." && pwd)"
REPO_ROOT="$(cd "$EXP_DIR/.." && pwd)"
RENDER_DIR="${RENDER_DIR:-$DEPLOY_DIR/.rendered}"
IMAGE_EXP_DIR="/workspace/engine/dagshare_exp"

log() { printf '[%s] %s\n' "$(date +%H:%M:%S)" "$*" >&2; }
warn() { printf '[%s] WARNING: %s\n' "$(date +%H:%M:%S)" "$*" >&2; }
die() { printf '[%s] ERROR: %s\n' "$(date +%H:%M:%S)" "$*" >&2; exit 1; }

want_help() {
  local arg
  for arg in "$@"; do
    if [[ "$arg" == "-h" || "$arg" == "--help" ]]; then return 0; fi
  done
  return 1
}

# Variables already set in the environment win over .env.
load_env() {
  local env_file="${DEPLOY_ENV_FILE:-$DEPLOY_DIR/.env}"
  local line key
  if [[ -f "$env_file" ]]; then
    while IFS= read -r line || [[ -n "$line" ]]; do
      [[ "$line" =~ ^[[:space:]]*(export[[:space:]]+)?([A-Za-z_][A-Za-z0-9_]*)=(.*)$ ]] || continue
      key="${BASH_REMATCH[2]}"
      [[ -z "${!key:-}" ]] || continue
      eval "export $key=${BASH_REMATCH[3]}"
    done <"$env_file"
  fi
  local user_slug
  user_slug="$(printf '%s' "${USER:-user}" | tr '[:upper:]' '[:lower:]' | tr -c 'a-z0-9-' '-' | cut -c1-40)"
  user_slug="${user_slug%-}"
  CIVO_REGION="${CIVO_REGION:-LON1}"
  CLUSTER_NAME="${CLUSTER_NAME:-dagshare}"
  KUBE_CONTEXT="${KUBE_CONTEXT:-$CLUSTER_NAME}"
  NAMESPACE="$(dns_name "${NAMESPACE:-dagshare-$user_slug}")"
  USER_LABEL="${USER_LABEL:-$user_slug}"
  IMAGE="${IMAGE:-ghcr.io/azkanab/vllm-dagshare:dagshare}"
  MODEL="${MODEL:-meta-llama/Llama-3.1-8B-Instruct}"
  GPU_SIZE="${GPU_SIZE:-}"
  SERVER_CPUS="${SERVER_CPUS:-4}"
  SERVER_MEM="${SERVER_MEM:-32Gi}"
  LOADGEN_CPUS="${LOADGEN_CPUS:-2}"
  LOADGEN_MEM="${LOADGEN_MEM:-4Gi}"
  DEADLINE_MARGIN_S="${DEADLINE_MARGIN_S:-1800}"
  WARMUP_REQUESTS="${WARMUP_REQUESTS:-8}"
  VLLM_EXTRA_ARGS="${VLLM_EXTRA_ARGS:-}"
  GPU_TOLERATION_KEY="${GPU_TOLERATION_KEY:-}"
  RESULTS_DIR="${RESULTS_DIR:-$REPO_ROOT/results}"
  LOCAL_LOG_DIR="${LOCAL_LOG_DIR:-$DEPLOY_DIR/logs}"
  # shellcheck disable=SC2016 # variables are expanded inside the loadgen container
  LOADGEN_CMD="${LOADGEN_CMD:-python -m dagshare_exp.loadgen --workload \"\$WORKLOAD_FILE\" --base-url \"\$BASE_URL\" --model \"\$MODEL\" --seed \"\$SEED\" --out \"\$LOADGEN_OUT\"}"
  export CIVO_REGION CLUSTER_NAME KUBE_CONTEXT NAMESPACE USER_LABEL IMAGE MODEL GPU_SIZE
  export SERVER_CPUS SERVER_MEM LOADGEN_CPUS LOADGEN_MEM DEADLINE_MARGIN_S WARMUP_REQUESTS
  export VLLM_EXTRA_ARGS GPU_TOLERATION_KEY RESULTS_DIR LOCAL_LOG_DIR LOADGEN_CMD
}

need_cmd() {
  command -v "$1" >/dev/null 2>&1 || die "'$1' not found. ${2:-}"
}

require_vars() {
  local v missing=()
  for v in "$@"; do
    if [[ -z "${!v:-}" ]]; then missing+=("$v"); fi
  done
  if ((${#missing[@]})); then
    die "missing env vars: ${missing[*]} (set them in $DEPLOY_DIR/.env, see env.example)"
  fi
}

kc() { kubectl --context "$KUBE_CONTEXT" "$@"; }
kcn() { kubectl --context "$KUBE_CONTEXT" -n "$NAMESPACE" "$@"; }

check_context() {
  kubectl config get-contexts -o name 2>/dev/null | grep -qx "$KUBE_CONTEXT" \
    || die "kube context '$KUBE_CONTEXT' not in kubeconfig. Run 'make up' (own cluster) or set KUBE_CONTEXT (shared cluster)."
  kc get --raw /readyz --request-timeout=15s >/dev/null 2>&1 \
    || die "cluster for context '$KUBE_CONTEXT' is not reachable"
}

have_yq() { command -v yq >/dev/null 2>&1 && yq --version 2>&1 | grep -q mikefarah; }

have_pyyaml() { python3 -c 'import yaml' >/dev/null 2>&1; }

require_yaml_tool() {
  have_yq || have_pyyaml || die "need mikefarah yq (brew install yq) or python3 with pyyaml"
}

py_yaml() {
  python3 - "$@" <<'PY'
import sys, yaml
mode, path = sys.argv[1], sys.argv[2]
with open(path) as f:
    d = yaml.safe_load(f) or {}
def s(v):
    return str(v).lower() if isinstance(v, bool) else ("" if v is None else str(v))
if mode == "get":
    print(s(d.get(sys.argv[3])))
elif mode == "env":
    for k, v in (d.get("env") or {}).items():
        print(f"{k}={s(v)}")
elif mode == "pairs":
    for r in d.get("runs") or []:
        for ret in r.get("retentions") or []:
            print(f"{r['preset']} {s(ret)}")
elif mode == "seeds":
    for x in d.get("seeds") or []:
        print(s(x))
PY
}

cfg_get() {
  if have_yq; then yq -r ".$2 // \"\"" "$1"; else py_yaml get "$1" "$2"; fi
}

preset_env_lines() {
  if have_yq; then
    yq -r '(.env // {}) | to_entries | .[] | .key + "=" + (.value | tostring)' "$1"
  else
    py_yaml env "$1"
  fi
}

matrix_pairs() {
  if have_yq; then
    # shellcheck disable=SC2016 # yq variable, not shell
    yq -r '.runs[] | .preset as $p | .retentions[] | $p + " " + (. | tostring)' "$1"
  else
    py_yaml pairs "$1"
  fi
}

matrix_seeds() {
  if have_yq; then yq -r '.seeds[]' "$1"; else py_yaml seeds "$1"; fi
}

preset_file() {
  local f="$EXP_DIR/configs/presets/$1.yaml"
  [[ -f "$f" ]] || die "unknown preset '$1' (no $f)"
  printf '%s\n' "$f"
}

workload_file() {
  local f="$EXP_DIR/configs/workloads/$1.yaml"
  [[ -f "$f" ]] || die "unknown workload '$1' (no $f)"
  printf '%s\n' "$f"
}

matrix_file() {
  if [[ -f "$1" ]]; then printf '%s\n' "$1"; return; fi
  local f="$EXP_DIR/configs/matrices/$1.yaml"
  [[ -f "$f" ]] || die "unknown matrix '$1' (no $f)"
  printf '%s\n' "$f"
}

normalize_retention() {
  case "$1" in
    ttl) echo "ttl:2" ;;
    until_done) echo "until_done:60" ;;
    none | ttl:* | until_done:*) echo "$1" ;;
    *) die "bad retention '$1' (none | ttl[:s] | until_done[:cap])" ;;
  esac
}

retention_token() {
  printf '%s' "$1" | tr -d ':' | sed 's/\./p/g'
}

dns_name() {
  local name
  name="$(printf '%s' "$1" | tr '[:upper:]' '[:lower:]' | tr -c 'a-z0-9-' '-' | sed 's/--*/-/g')"
  if ((${#name} > 52)); then
    name="${name:0:43}-$(printf '%s' "$1" | cksum | cut -d' ' -f1)"
  fi
  name="${name#-}"
  printf '%s' "${name%-}"
}

yaml_quote() {
  printf "'%s'" "$(printf '%s' "$1" | sed "s/'/''/g")"
}

resolve_image_digest() {
  local image="$1" repo tag registry path token digest
  if [[ "$image" == *@sha256:* ]]; then printf '%s\n' "${image##*@}"; return 0; fi
  repo="${image%:*}"; tag="${image##*:}"
  [[ "$repo" == "$image" ]] && tag="latest"
  registry="${repo%%/*}"; path="${repo#*/}"
  [[ "$registry" == "ghcr.io" ]] || return 1
  token="$(curl -fsS "https://ghcr.io/token?scope=repository:${path}:pull" 2>/dev/null \
    | sed -n 's/.*"token":"\([^"]*\)".*/\1/p')" || return 1
  [[ -n "$token" ]] || return 1
  digest="$(curl -fsSI -H "Authorization: Bearer $token" \
    -H 'Accept: application/vnd.oci.image.index.v1+json, application/vnd.docker.distribution.manifest.list.v2+json, application/vnd.docker.distribution.manifest.v2+json, application/vnd.oci.image.manifest.v1+json' \
    "https://ghcr.io/v2/${path}/manifests/${tag}" 2>/dev/null \
    | tr -d '\r' | awk -F': ' 'tolower($1)=="docker-content-digest"{print $2}')" || return 1
  [[ -n "$digest" ]] || return 1
  printf '%s\n' "$digest"
}

pin_image() {
  if [[ "$IMAGE" == *@sha256:* ]]; then
    IMAGE_DIGEST="${IMAGE##*@}"
  elif IMAGE_DIGEST="$(resolve_image_digest "$IMAGE")"; then
    log "image $IMAGE -> $IMAGE_DIGEST (pinned for this run)"
    IMAGE="${IMAGE%:*}@${IMAGE_DIGEST}"
  else
    warn "could not resolve digest of $IMAGE (private package?); using the tag as is"
    IMAGE_DIGEST="unknown"
  fi
  export IMAGE IMAGE_DIGEST
}

render_template() {
  local tmpl="$1" out="$2" v missing=() vars
  vars="$(grep -oE '\$\{[A-Z_][A-Z0-9_]*\}' "$tmpl" | sort -u | tr '\n' ' ')"
  for v in $vars; do
    v="${v#\$\{}"; v="${v%\}}"
    if [[ -z "${!v+x}" ]]; then missing+=("$v"); fi
  done
  if ((${#missing[@]})); then die "rendering $(basename "$tmpl"): unset vars ${missing[*]}"; fi
  mkdir -p "$(dirname "$out")"
  envsubst "$vars" <"$tmpl" >"$out"
}

tolerations_block() {
  local pad
  pad="$(printf '%*s' "$1" '')"
  if [[ -n "$GPU_TOLERATION_KEY" ]]; then
    printf '%stolerations:\n%s  - {key: %s, operator: Exists, effect: NoSchedule}\n' "$pad" "$pad" "$(yaml_quote "$GPU_TOLERATION_KEY")"
  else
    printf '%stolerations: []\n' "$pad"
  fi
}

render_base() {
  export TOLERATIONS_BLOCK POD_TOLERATIONS_BLOCK HF_TOKEN_YAML
  TOLERATIONS_BLOCK="$(tolerations_block 6)"
  POD_TOLERATIONS_BLOCK="$(tolerations_block 2)"
  HF_TOKEN_YAML="$(yaml_quote "${HF_TOKEN:-}")"
  local t
  for t in namespace hf-cache-pvc results-pvc secret-hf prefetch-job fetch-pod gpu-smoke-pod; do
    render_template "$DEPLOY_DIR/k8s/$t.yaml.tmpl" "$RENDER_DIR/$t.yaml"
  done
  chmod 600 "$RENDER_DIR/secret-hf.yaml"
}

# Sets every variable the run Job template needs; args: preset retention(or "") seed workload matrix.
compute_run_vars() {
  local preset="$1" retention="$2" seed="$3" workload="$4" matrix="$5" pf wf duration warmup line key val
  pf="$(preset_file "$preset")"
  wf="$(workload_file "$workload")"
  [[ "$seed" =~ ^[0-9]+$ ]] || die "seed must be an integer, got '$seed'"
  [[ -n "$retention" ]] || retention="$(cfg_get "$pf" retention)"
  [[ -n "$retention" ]] || retention="none"
  RETENTION="$(normalize_retention "$retention")"
  RETENTION_TOKEN="$(retention_token "$RETENTION")"
  PRESET="$preset"; SEED="$seed"; WORKLOAD="$workload"
  MATRIX_NAME="$(dns_name "$matrix")"
  PRESET_LABEL="$(dns_name "$preset")"
  POLICY="$(cfg_get "$pf" policy)"
  [[ -n "$POLICY" ]] || die "preset $preset has no policy"
  RUN_NAME="${preset}_${RETENTION_TOKEN}_seed${seed}"
  RUN_PATH="${MATRIX_NAME}/${RUN_NAME}"
  JOB_NAME="$(dns_name "ds-${MATRIX_NAME}-${preset}-${RETENTION_TOKEN}-s${seed}")"
  RUN_MODEL="$(cfg_get "$wf" model)"
  RUN_MODEL="${RUN_MODEL:-$MODEL}"
  MAX_MODEL_LEN="$(cfg_get "$wf" max_model_len)"
  GPU_MEM_UTIL="$(cfg_get "$wf" gpu_memory_utilization)"
  duration="$(cfg_get "$wf" duration_s)"; warmup="$(cfg_get "$wf" warmup_s)"
  [[ -n "$MAX_MODEL_LEN" && -n "$GPU_MEM_UTIL" && -n "$duration" ]] \
    || die "workload $workload needs model, max_model_len, gpu_memory_utilization, duration_s"
  ACTIVE_DEADLINE_S=$(( ${duration%.*} + ${warmup:-0} + DEADLINE_MARGIN_S ))
  WORKLOAD_FILE="$IMAGE_EXP_DIR/configs/workloads/$workload.yaml"
  PRESET_ENV_BLOCK="$(printf '            - {name: KV_RETENTION, value: %s}' "$(yaml_quote "$RETENTION")")"
  while IFS= read -r line; do
    [[ -n "$line" ]] || continue
    key="${line%%=*}"; val="${line#*=}"
    [[ "$key" != "KV_RETENTION" ]] || continue
    PRESET_ENV_BLOCK+=$'\n'"$(printf '            - {name: %s, value: %s}' "$key" "$(yaml_quote "$val")")"
  done < <(preset_env_lines "$pf")
  LOADGEN_CMD_YAML="$(yaml_quote "$LOADGEN_CMD")"
  VLLM_EXTRA_ARGS_YAML="$(yaml_quote "$VLLM_EXTRA_ARGS")"
  IMAGE_DIGEST="${IMAGE_DIGEST:-unknown}"
  TOLERATIONS_BLOCK="$(tolerations_block 6)"
  export PRESET RETENTION RETENTION_TOKEN SEED WORKLOAD MATRIX_NAME PRESET_LABEL POLICY RUN_NAME RUN_PATH
  export JOB_NAME RUN_MODEL MAX_MODEL_LEN GPU_MEM_UTIL ACTIVE_DEADLINE_S WORKLOAD_FILE
  export PRESET_ENV_BLOCK LOADGEN_CMD_YAML VLLM_EXTRA_ARGS_YAML IMAGE_DIGEST TOLERATIONS_BLOCK
}

render_run() {
  local out="$RENDER_DIR/runs/$JOB_NAME.yaml"
  render_template "$DEPLOY_DIR/k8s/run-job.yaml.tmpl" "$out"
  printf '%s\n' "$out"
}

apply_base() {
  render_base
  kc apply -f "$RENDER_DIR/namespace.yaml" >/dev/null
  kc apply -f "$RENDER_DIR/hf-cache-pvc.yaml" -f "$RENDER_DIR/results-pvc.yaml" >/dev/null
  if [[ -n "${HF_TOKEN:-}" ]]; then kc apply -f "$RENDER_DIR/secret-hf.yaml" >/dev/null; fi
  if [[ -n "${GHCR_TOKEN:-}" ]]; then
    kcn create secret docker-registry ghcr-pull --docker-server=ghcr.io \
      --docker-username="${GHCR_USER:-$USER}" --docker-password="$GHCR_TOKEN" \
      --dry-run=client -o yaml | kc apply -f - >/dev/null
  fi
}

ensure_fetch_pod() {
  render_base
  kc apply -f "$RENDER_DIR/fetch-pod.yaml" >/dev/null
  kcn wait --for=condition=Ready pod/dagshare-fetch --timeout=300s >/dev/null \
    || die "fetch pod did not become ready (kubectl -n $NAMESPACE describe pod dagshare-fetch)"
}

job_condition() {
  kcn get job "$1" -o jsonpath='{range .status.conditions[?(@.status=="True")]}{.type}{" "}{end}' 2>/dev/null || true
}

job_finished() {
  local c
  c="$(job_condition "$1")"
  [[ " $c " == *" Complete "* || " $c " == *" Failed "* ]]
}

job_pod() {
  kcn get pods -l "job-name=$1" --sort-by=.metadata.creationTimestamp \
    -o jsonpath='{.items[-1:].metadata.name}' 2>/dev/null || true
}

explain_pod() {
  local pod="$1"
  [[ -n "$pod" ]] || return 0
  kcn get pod "$pod" -o jsonpath='{range .status.initContainerStatuses[*]}{"  init "}{.name}{": "}{.state.terminated.reason}{" exit="}{.state.terminated.exitCode}{"\n"}{end}{range .status.containerStatuses[*]}{"  "}{.name}{": "}{.state.terminated.reason}{.state.waiting.reason}{" exit="}{.state.terminated.exitCode}{"\n"}{end}' >&2 || true
  kcn get events --field-selector "involvedObject.name=$pod" --sort-by=.lastTimestamp 2>/dev/null | tail -n 8 >&2 || true
}

# Follows a Job's pod logs until the Job finishes; args: job, log file, LOG_FOLLOW (all|loadgen|none).
follow_job() {
  local job="$1" logfile="$2" mode="${3:-all}" pod="" phase="" status="" last="" start=$SECONDS
  mkdir -p "$(dirname "$logfile")"
  while ! job_finished "$job"; do
    pod="$(job_pod "$job")"
    if [[ -n "$pod" ]]; then
      phase="$(kcn get pod "$pod" -o jsonpath='{.status.phase}' 2>/dev/null || true)"
      [[ "$phase" == "Pending" ]] || break
      status="$(kcn get pod "$pod" -o jsonpath='{.status.conditions[?(@.type=="PodScheduled")].reason} {.status.initContainerStatuses[*].state.waiting.reason} {.status.containerStatuses[*].state.waiting.reason}' 2>/dev/null || true)"
    else
      status="no pod yet"
    fi
    if [[ "$status" != "$last" ]]; then log "$job: pending ($((SECONDS - start))s) ${status}"; last="$status"; fi
    sleep 5
  done
  log "$job: pod $pod is $phase; logs -> $logfile"
  while ! job_finished "$job"; do
    case "$mode" in
      none) sleep 15 ;;
      loadgen) kcn logs -f "$pod" -c loadgen 2>&1 | tee -a "$logfile" || sleep 5 ;;
      *) kcn logs -f --all-containers --prefix --max-log-requests 5 "$pod" 2>&1 | tee -a "$logfile" || sleep 5 ;;
    esac
    job_finished "$job" || sleep 3
  done
  if [[ "$mode" == none ]]; then kcn logs --all-containers --prefix "$pod" >>"$logfile" 2>&1 || true; fi
  if [[ " $(job_condition "$job") " == *" Complete "* ]]; then
    log "$job: succeeded in $((SECONDS - start))s"
    return 0
  fi
  log "$job: FAILED after $((SECONDS - start))s ($(kcn get job "$job" -o jsonpath='{.status.conditions[?(@.type=="Failed")].reason}' 2>/dev/null))"
  explain_pod "$pod"
  return 1
}
