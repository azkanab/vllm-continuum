#!/usr/bin/env bash
set -euo pipefail
source "$(dirname "$0")/common.sh"

usage() {
  cat <<'USAGE'
Usage: fetch-results.sh [--matrix NAME] [--dest DIR] [--keep-pod]

Copies results from the results PVC to the laptop with kubectl cp through the
dagshare-fetch helper pod (busybox). Default: everything into $RESULTS_DIR
(<engine>/results). Then analyze with:
  python -m dagshare_exp.analyze results --out results/analysis
  --matrix NAME  only copy one matrix dir (e.g. smoke, required, adhoc)
  --dest DIR     local destination
  --keep-pod     leave the helper pod running afterwards
USAGE
}

if want_help "$@"; then usage; exit 0; fi
load_env
matrix=""; dest="$RESULTS_DIR"; keep=0
while (($#)); do
  case "$1" in
    --matrix) matrix="$2"; shift 2 ;;
    --dest) dest="$2"; shift 2 ;;
    --keep-pod) keep=1; shift ;;
    *) usage; die "unknown argument $1" ;;
  esac
done

check_context
ensure_fetch_pod
src="/results"
target="$dest"
if [[ -n "$matrix" ]]; then
  src="/results/$(dns_name "$matrix")"
  target="$dest/$(dns_name "$matrix")"
fi
kcn exec dagshare-fetch -- test -d "$src" || die "$src does not exist on the results PVC"
log "contents of $src on the PVC:"
kcn exec dagshare-fetch -- sh -c "cd '$src' && find . -name requests.jsonl | sort" | sed 's/^/  /' >&2 || true
mkdir -p "$target"
log "copying $NAMESPACE/dagshare-fetch:$src -> $target"
kcn cp --retries=5 "dagshare-fetch:$src" "$target"
((keep)) || kcn delete pod dagshare-fetch --wait=false >/dev/null
log "done: $(find "$target" -name requests.jsonl | wc -l | tr -d ' ') run(s) with requests.jsonl under $target"
log "analyze: (cd $REPO_ROOT && python -m dagshare_exp.analyze $dest --out $dest/analysis)"
