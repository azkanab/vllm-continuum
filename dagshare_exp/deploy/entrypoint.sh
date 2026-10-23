#!/usr/bin/env bash
set -euo pipefail

usage() {
  cat <<'EOF'
Usage: entrypoint.sh {server|loadgen|prefetch|prepare|<command...>}

Container entrypoint of the vllm-dagshare image.
  server    run `vllm serve` pinned to this container's CPUs until $DONE_FILE appears
  loadgen   wait for the server, warm up, run $LOADGEN_CMD, copy outputs, touch $DONE_FILE
  prefetch  download $MODEL into the Hugging Face cache
  prepare   empty $RESULTS_DIR (init container of a run)
Anything else is executed as a command.
EOF
}

DONE_FILE="${DONE_FILE:-/run/dagshare/done}"
SERVER_FAILED_FILE="$(dirname "$DONE_FILE")/server_failed"
PORT="${PORT:-8000}"

log() { printf '[entrypoint %s %s] %s\n' "${ROLE:-?}" "$(date -u +%H:%M:%S)" "$*" >&2; }

cpu_affinity() {
  awk '/^Cpus_allowed_list:/{print $2}' /proc/self/status
}

write_info() {
  local out="$1" role="$2" affinity="$3"
  INFO_ROLE="$role" INFO_AFFINITY="$affinity" python3 - "$out" <<'PY'
import json, os, subprocess, sys, time
def sh(cmd):
    try:
        return subprocess.run(cmd, capture_output=True, text=True, timeout=30).stdout.strip()
    except Exception as e:
        return f"unavailable: {e}"
env = os.environ
info = {
    "role": env["INFO_ROLE"],
    "written_at": time.time(),
    "cpus_allowed_list": env["INFO_AFFINITY"],
    "nproc": os.cpu_count(),
    "node": env.get("NODE_NAME"),
    "pod": env.get("POD_NAME"),
    "image": env.get("IMAGE_REF"),
    "image_digest": env.get("IMAGE_DIGEST"),
    "git_sha": env.get("VLLM_DAGSHARE_GIT_SHA"),
    "preset": env.get("PRESET"),
    "policy": env.get("POLICY"),
    "seed": env.get("SEED"),
    "run_path": env.get("RUN_PATH"),
    "model": env.get("MODEL"),
    "max_model_len": env.get("MAX_MODEL_LEN"),
    "gpu_memory_utilization": env.get("GPU_MEM_UTIL"),
    "settings": {k: v for k, v in env.items() if k.startswith(("DAGSHARE_", "KV_RETENTION"))},
}
if info["role"] == "server":
    info["gpu"] = sh(["nvidia-smi", "--query-gpu=name,memory.total,driver_version", "--format=csv,noheader"])
    smi = sh(["nvidia-smi"])
    info["cuda_driver_version"] = next((l.split("CUDA Version:")[1].split()[0] for l in smi.splitlines() if "CUDA Version:" in l), None)
with open(sys.argv[1], "w") as f:
    json.dump(info, f, indent=2)
PY
}

run_server() {
  : "${MODEL:?}" "${POLICY:?}" "${MAX_MODEL_LEN:?}" "${GPU_MEM_UTIL:?}" "${RUN_OUTPUT_DIR:?}"
  mkdir -p "$RUN_OUTPUT_DIR" "$(dirname "$DONE_FILE")"
  rm -f "$DONE_FILE" "$SERVER_FAILED_FILE"
  local affinity
  affinity="$(cpu_affinity)"
  write_info "$RUN_OUTPUT_DIR/deploy_info.json" server "$affinity"
  log "policy=$POLICY KV_RETENTION=${KV_RETENTION:-unset} cpus=$affinity model=$MODEL"

  local -a extra=()
  if [[ -n "${VLLM_EXTRA_ARGS:-}" ]]; then read -r -a extra <<<"$VLLM_EXTRA_ARGS"; fi
  taskset -c "$affinity" vllm serve "$MODEL" \
    --host 0.0.0.0 --port "$PORT" \
    --scheduling-policy "$POLICY" \
    --max-model-len "$MAX_MODEL_LEN" \
    --gpu-memory-utilization "$GPU_MEM_UTIL" \
    "${extra[@]}" > >(tee -a "$RUN_OUTPUT_DIR/server.log") 2>&1 &
  local pid=$! stopping=0
  trap 'stopping=1; kill -TERM "$pid" 2>/dev/null || true' TERM INT

  while kill -0 "$pid" 2>/dev/null; do
    if [[ -e "$DONE_FILE" || "$stopping" == 1 ]]; then
      stopping=1
      log "stop requested, sending SIGTERM to vllm (pid $pid)"
      kill -TERM "$pid" 2>/dev/null || true
      local waited=0
      while kill -0 "$pid" 2>/dev/null && ((waited < 60)); do sleep 1; waited=$((waited + 1)); done
      if kill -0 "$pid" 2>/dev/null; then log "vllm still alive after 60s, SIGKILL"; kill -KILL "$pid" 2>/dev/null || true; fi
      wait "$pid" 2>/dev/null || true
      log "server stopped cleanly"
      exit 0
    fi
    sleep 2
  done

  local rc=0
  wait "$pid" || rc=$?
  if [[ "$stopping" == 1 ]]; then exit 0; fi
  log "vllm exited on its own with code $rc"
  touch "$SERVER_FAILED_FILE"
  exit $((rc == 0 ? 1 : rc))
}

wait_for_server() {
  local timeout="${SERVER_READY_TIMEOUT_S:-1800}" start=$SECONDS
  log "waiting for $BASE_URL/health (timeout ${timeout}s)"
  until python3 -c 'import sys,urllib.request; urllib.request.urlopen(sys.argv[1], timeout=5)' "$BASE_URL/health" 2>/dev/null; do
    [[ ! -e "$SERVER_FAILED_FILE" ]] || { log "server failed before becoming healthy"; return 1; }
    ((SECONDS - start < timeout)) || { log "server not healthy after ${timeout}s"; return 1; }
    sleep 5
  done
  log "server healthy after $((SECONDS - start))s"
}

copy_outputs() {
  local dest="$1"
  mkdir -p "$dest"
  if [[ -d "$LOADGEN_OUT" ]]; then cp -a "$LOADGEN_OUT"/. "$dest"/; fi
}

run_loadgen() {
  : "${BASE_URL:=http://localhost:$PORT}" "${LOADGEN_CMD:?}" "${RESULTS_DIR:?}"
  : "${LOADGEN_OUT:=/run/dagshare/loadgen-out}"
  export BASE_URL LOADGEN_OUT MODEL SEED WORKLOAD_FILE
  mkdir -p "$(dirname "$DONE_FILE")" "$LOADGEN_OUT"
  trap 'touch "$DONE_FILE"' EXIT

  local affinity
  affinity="$(cpu_affinity)"
  write_info "$LOADGEN_OUT/loadgen_info.json" loadgen "$affinity"

  wait_for_server
  if [[ "${WARMUP_REQUESTS:-0}" -gt 0 ]]; then
    python3 "$(dirname "$0")/warmup.py" --base-url "$BASE_URL" --model "$MODEL" --n "$WARMUP_REQUESTS"
  fi

  log "running: $LOADGEN_CMD"
  local rc=0
  taskset -c "$affinity" bash -c "$LOADGEN_CMD" > >(tee -a "$LOADGEN_OUT/loadgen.log") 2>&1 || rc=$?

  if ((rc == 0)) && [[ -s "$LOADGEN_OUT/requests.jsonl" ]]; then
    copy_outputs "$RESULTS_DIR"
    log "loadgen finished, outputs copied to $RESULTS_DIR"
    return 0
  fi
  ((rc != 0)) || { log "loadgen exited 0 but wrote no requests.jsonl"; rc=1; }
  copy_outputs "$RESULTS_DIR/failed"
  log "loadgen failed (rc=$rc); partial outputs in $RESULTS_DIR/failed"
  return "$rc"
}

run_prefetch() {
  : "${MODEL:?}"
  local -a cli
  if command -v hf >/dev/null 2>&1; then cli=(hf download); else cli=(huggingface-cli download); fi
  log "downloading $MODEL with ${cli[*]}"
  if ! "${cli[@]}" "$MODEL" --exclude 'original/*'; then
    log "download failed. A 401/403 means HF_TOKEN is missing/invalid or your HF account has no access to $MODEL (accept the license on huggingface.co)."
    exit 1
  fi
  du -sh /root/.cache/huggingface 2>/dev/null || true
}

run_prepare() {
  : "${RESULTS_DIR:?}"
  log "clearing $RESULTS_DIR"
  find "$RESULTS_DIR" -mindepth 1 -delete
}

ROLE="${1:-}"
case "$ROLE" in
  server) run_server ;;
  loadgen) run_loadgen ;;
  prefetch) run_prefetch ;;
  prepare) run_prepare ;;
  -h | --help | "") usage ;;
  *) exec "$@" ;;
esac
