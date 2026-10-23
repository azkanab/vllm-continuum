#!/usr/bin/env bash
set -euo pipefail
source "$(dirname "$0")/common.sh"

usage() {
  cat <<'USAGE'
Usage: validate.sh [DIR]

Validates rendered manifests (default .rendered/) without changing any cluster:
kubeconform -strict if installed, else `kubectl apply --dry-run=client` when the
$KUBE_CONTEXT cluster is reachable, else a YAML parse only.
USAGE
}

if want_help "$@"; then usage; exit 0; fi
load_env
dir="${1:-$RENDER_DIR}"
[[ -d "$dir" ]] || die "$dir not found; run render.sh first"
files=()
while IFS= read -r f; do files+=("$f"); done < <(find "$dir" -name '*.yaml' | sort)
((${#files[@]})) || die "no manifests in $dir"

if command -v kubeconform >/dev/null 2>&1; then
  kubeconform -strict -summary -kubernetes-version 1.31.0 "${files[@]}"
elif kubectl config get-contexts -o name 2>/dev/null | grep -qx "$KUBE_CONTEXT" \
  && kc get --raw /readyz --request-timeout=10s >/dev/null 2>&1; then
  for f in "${files[@]}"; do kc apply --dry-run=client -f "$f" >/dev/null; done
  log "kubectl client dry-run ok for ${#files[@]} manifests (context $KUBE_CONTEXT)"
else
  for f in "${files[@]}"; do
    if have_yq; then yq -e 'true' "$f" >/dev/null; else python3 -c 'import sys,yaml; list(yaml.safe_load_all(open(sys.argv[1])))' "$f"; fi
  done
  warn "only parsed YAML (install kubeconform for schema checks: brew install kubeconform)"
fi
