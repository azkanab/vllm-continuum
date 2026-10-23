#!/usr/bin/env bash
set -euo pipefail
source "$(dirname "$0")/common.sh"

usage() {
  cat <<'USAGE'
Usage: logs.sh [JOB | --preset P [--retention R] [--seed K] [--matrix-name M]] [-c server|loadgen|prepare] [-f] [--tail N]

Shows logs of a run Job. With no job selector it uses the newest run Job in $NAMESPACE.
Use `logs.sh prefetch` for the prefetch Job.
USAGE
}

if want_help "$@"; then usage; exit 0; fi
load_env
job=""; preset=""; retention=""; seed=1; matrix="adhoc"; container=""; follow=(); tail=200
while (($#)); do
  case "$1" in
    --preset) preset="$2"; shift 2 ;;
    --retention) retention="$2"; shift 2 ;;
    --seed) seed="$2"; shift 2 ;;
    --matrix-name) matrix="$2"; shift 2 ;;
    -c) container="$2"; shift 2 ;;
    -f) follow=(-f); shift ;;
    --tail) tail="$2"; shift 2 ;;
    -*) usage; die "unknown option $1" ;;
    *) job="$1"; shift ;;
  esac
done

check_context
if [[ "$job" == prefetch ]]; then job="dagshare-prefetch"; fi
if [[ -z "$job" && -n "$preset" ]]; then
  require_yaml_tool
  pf="$(preset_file "$preset")"
  [[ -n "$retention" ]] || retention="$(cfg_get "$pf" retention)"
  job="$(dns_name "ds-$(dns_name "$matrix")-${preset}-$(retention_token "$(normalize_retention "${retention:-none}")")-s${seed}")"
fi
if [[ -z "$job" ]]; then
  job="$(kcn get jobs -l app.kubernetes.io/component=run --sort-by=.metadata.creationTimestamp \
    -o jsonpath='{.items[-1:].metadata.name}')"
  [[ -n "$job" ]] || die "no run jobs in $NAMESPACE"
fi
pod="$(job_pod "$job")"
[[ -n "$pod" ]] || die "job $job has no pod (kubectl -n $NAMESPACE describe job $job)"
log "job $job, pod $pod"
if [[ -n "$container" ]]; then
  kcn logs ${follow[@]+"${follow[@]}"} --tail "$tail" "$pod" -c "$container"
else
  kcn logs ${follow[@]+"${follow[@]}"} --tail "$tail" --all-containers --prefix "$pod"
fi
