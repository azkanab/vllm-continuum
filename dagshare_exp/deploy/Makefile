SHELL := /usr/bin/env bash
.DEFAULT_GOAL := help
D := $(CURDIR)

PRESET ?= A3
RETENTION ?=
SEED ?= 1
WORKLOAD ?= smoke
MATRIX ?= smoke
MATRIX_NAME ?= adhoc
GIT_SHA := $(shell git -C $(D) rev-parse HEAD 2>/dev/null || echo unknown)
ENGINE := $(abspath $(D)/../..)

.PHONY: help check check-gpu up down image render prefetch run matrix fetch logs status

help: ## list targets
	@grep -E '^[a-z-]+:.*## ' $(MAKEFILE_LIST) | awk -F':.*## ' '{printf "  make %-10s %s\n", $$1, $$2}'
	@echo "  vars: PRESET=$(PRESET) RETENTION=$(RETENTION) SEED=$(SEED) WORKLOAD=$(WORKLOAD) MATRIX=$(MATRIX)"

check: ## tools, env, cluster reachable (add LOCAL=1 before the cluster exists)
	$(D)/00-check.sh $(if $(LOCAL),--local)

check-gpu: ## check plus nvidia-smi smoke pod (CUDA >= 12.8)
	$(D)/00-check.sh --gpu-smoke

up: ## create/reuse the Civo GPU cluster and install the GPU operator
	$(D)/cluster-up.sh

down: ## delete namespace, PVCs and the cluster (ALWAYS run when done)
	$(D)/cluster-down.sh $(if $(YES),--yes) $(if $(NS_ONLY),--namespace-only)

image: ## podman/docker fallback build of the image (no push)
	@set -euo pipefail; source $(D)/common.sh; load_env; \
	engine=$$(command -v podman || command -v docker) || die "need podman or docker"; \
	ignore=(); [[ "$$engine" != *podman* ]] || ignore=(--ignorefile $(D)/Dockerfile.dockerignore); \
	$$engine build --platform linux/amd64 -f $(D)/Dockerfile $${ignore[@]+"$${ignore[@]}"} --build-arg GIT_SHA=$(GIT_SHA) -t "$$IMAGE" $(ENGINE); \
	echo "built $$IMAGE; push with: $$engine push $$IMAGE"

render: ## render manifests into .rendered/ and validate them offline if possible
	$(D)/render.sh all --preset $(PRESET) $(if $(RETENTION),--retention $(RETENTION)) --seed $(SEED) --workload $(WORKLOAD)
	@$(D)/validate.sh

prefetch: ## download the model into the hf-cache PVC (once per namespace)
	$(D)/prefetch.sh

run: ## one run: make run PRESET=A3 RETENTION=ttl SEED=1 WORKLOAD=smoke
	$(D)/run.sh --preset $(PRESET) $(if $(RETENTION),--retention $(RETENTION)) --seed $(SEED) --workload $(WORKLOAD) --matrix-name $(MATRIX_NAME)

matrix: ## whole matrix, resumable: make matrix MATRIX=required
	$(D)/run-matrix.sh $(MATRIX)

fetch: ## copy results from the PVC to <engine>/results
	$(D)/fetch-results.sh

logs: ## logs of the newest run (or PRESET=.. RETENTION=.. SEED=..)
	$(D)/logs.sh $(if $(JOB),$(JOB)) $(if $(FOLLOW),-f)

status: ## nodes, GPU, jobs, pods, PVCs
	$(D)/status.sh
