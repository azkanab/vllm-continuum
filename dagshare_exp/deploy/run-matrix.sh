#!/usr/bin/env bash
set -euo pipefail
source "$(dirname "$0")/common.sh"

usage() {
  cat <<'USAGE'
Usage: run-matrix.sh <matrix.yaml | matrix-name> [--seeds "1 2"] [--rerun] [--dry-run]

Runs every (preset x retention x seed) of a matrix one after another with run.sh.
A run is skipped when its results dir on the PVC already has requests.jsonl, so an
interrupted matrix resumes where it stopped. Failed runs do not stop the matrix;
a summary is printed at the end and the exit code is non-zero if any run failed.
  --seeds    override the matrix's seed list
  --rerun    do not skip finished runs
  --dry-run  only list the runs and render their manifests
The image tag is resolved to a digest once, so every run uses the same image.
USAGE
}

if want_help "$@"; then usage; exit 0; fi
[[ $# -ge 1 ]] || { usage; exit 1; }
matrix_arg="$1"; shift
seeds_override=""; rerun=0; dry_run=0
while (($#)); do
  case "$1" in
    --seeds) seeds_override="$2"; shift 2 ;;
    --rerun) rerun=1; shift ;;
    --dry-run) dry_run=1; shift ;;
    *) usage; die "unknown argument $1" ;;
  esac
done

load_env
require_yaml_tool
mf="$(matrix_file "$matrix_arg")"
matrix_name="$(basename "$mf" .yaml)"
workload="$(cfg_get "$mf" workload)"
[[ -n "$workload" ]] || die "$mf has no workload"
seeds="${seeds_override:-$(matrix_seeds "$mf" | tr '\n' ' ')}"
pairs="$(matrix_pairs "$mf")"
[[ -n "$pairs" && -n "${seeds// /}" ]] || die "$mf has no runs or seeds"

if ((dry_run)); then
  IMAGE_DIGEST=unknown
else
  check_context
  pin_image
  ensure_fetch_pod
fi

is_done() {
  kcn exec dagshare-fetch -- test -s "/results/$1/requests.jsonl" >/dev/null 2>&1
}

ok=(); skipped=(); failed=()
total=0
for seed in $seeds; do
  while read -r preset retention; do
    [[ -n "$preset" ]] || continue
    total=$((total + 1))
    path="$(dns_name "$matrix_name")/${preset}_$(retention_token "$(normalize_retention "$retention")")_seed${seed}"
    if ((dry_run)); then
      compute_run_vars "$preset" "$retention" "$seed" "$workload" "$matrix_name"
      log "would run $path -> $(render_run)"
      continue
    fi
    if ((!rerun)) && is_done "$path"; then
      log "skip $path (requests.jsonl exists)"
      skipped+=("$path")
      continue
    fi
    log "=== [$total] $path"
    if "$DEPLOY_DIR/run.sh" --preset "$preset" --retention "$retention" --seed "$seed" \
      --workload "$workload" --matrix-name "$matrix_name" </dev/null; then
      ok+=("$path")
    else
      failed+=("$path")
      ensure_fetch_pod
    fi
  done <<<"$pairs"
done

((dry_run)) && exit 0
echo
echo "matrix $matrix_name ($workload): $total runs, ${#ok[@]} ok, ${#skipped[@]} skipped, ${#failed[@]} failed"
for p in ${ok[@]+"${ok[@]}"}; do echo "  ok      $p"; done
for p in ${skipped[@]+"${skipped[@]}"}; do echo "  skipped $p"; done
for p in ${failed[@]+"${failed[@]}"}; do echo "  FAILED  $p"; done
((${#failed[@]} == 0)) || exit 1
echo "next: make fetch"
